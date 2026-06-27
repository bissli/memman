"""External, durable backup of the whole `~/.memman/` store layout.

Backups write ONLY to a user-specified external directory (e.g. a
Dropbox path), never into `~/.memman/` itself -- that directory is
per-host and disposable, so an in-place archive dies with it. Each
run snapshots every store online and non-disruptively (SQLite via
the `sqlite3` online backup API, Postgres via `pg_dump -Fc`), bundles
them with the non-secret config into one atomic `.tar.gz`, and
rotates old bundles. `restore` rebuilds a working store from a bundle
after total loss of `~/.memman/`.

Secrets (API keys, the default Postgres DSN, and every per-store
`MEMMAN_POSTGRES_DSN_<store>`) are excluded from the bundle and
re-entered / resolved on the target host at restore.

Notes:
- The manifest carries its own `BACKUP_FORMAT_VERSION` (no global DB
  schema version exists); `embed_fingerprint` is recorded per store.
- The per-store `backend` field in the manifest is authoritative for
  restore routing.
"""

import contextlib
import errno
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memman import config
from memman.embed.fingerprint import Fingerprint, stored_fingerprint
from memman.setup.scheduler import _write_env_keys
from memman.store import factory
from memman.store.db import read_active, store_dir, write_active

logger = logging.getLogger('memman')

BACKUP_FORMAT_VERSION = 1
DB_FILENAME = 'memman.db'
DUMP_FILENAME = 'dump.pgdump'
MANIFEST_NAME = 'manifest.json'
ENV_NONSECRET_NAME = 'env.nonsecret'
BUNDLE_PREFIX = 'memman-backup'
INCOMING_DIRNAME = '.memman-incoming'

_STAMP_RE = re.compile(r'(\d{8}T\d{6}Z)\.tar\.gz$')

# Env keys describing THIS host's backup schedule (not portable config);
# excluded from the bundle so a restore onto another host never silently
# adopts the source host's cron/target/retention.
_HOST_LOCAL_KEYS = frozenset({
    config.BACKUP_CRON, config.BACKUP_TARGET, config.BACKUP_KEEP})


def snapshot_sqlite(store: str, data_dir: str, dst_db_path: Path) -> None:
    """Copy a store's SQLite DB to `dst_db_path` via the online backup API.

    Uses `sqlite3.Connection.backup`, which is consistent against a
    live writer, so no scheduler stop or drain lock is required. Opens
    the source read-only and does NOT run migrations (unlike `open_db`).
    """
    src_db = os.path.join(store_dir(data_dir, store), DB_FILENAME)
    src = sqlite3.connect(f'file:{src_db}?mode=ro', uri=True)
    try:
        dst = sqlite3.connect(str(dst_db_path))
        try:
            src.backup(dst)
            dst.execute('pragma journal_mode=DELETE')
        finally:
            dst.close()
    finally:
        src.close()


def snapshot_postgres(store: str, dsn: str, dst_dump_path: Path) -> None:
    """Dump `store_<store>` to `dst_dump_path` with `pg_dump -Fc`.

    The custom-format dump is MVCC-consistent, so no scheduler stop is
    required. Raises `RuntimeError` on pg_dump failure.
    """
    schema = f'store_{store}'
    cmd = ['pg_dump', '-Fc', '-d', dsn, '-n', schema,
           '-f', str(dst_dump_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'pg_dump failed for {schema}: {exc.stderr.strip()}') from exc


def build_bundle(data_dir: str, target: str) -> dict[str, Any]:
    """Snapshot every store + non-secret config into one `.tar.gz` at `target`.

    Each store is snapshotted in isolation: a store whose snapshot
    raises is recorded with `status='failed'` in the manifest and
    never aborts the whole bundle. The bundle is staged under
    `<target>/.memman-incoming/` and atomically published; a sidecar
    `<bundle>.manifest.json` is written alongside for cheap listing.
    """
    target_path = Path(os.path.expanduser(target))
    target_path.mkdir(parents=True, exist_ok=True)
    incoming = target_path / INCOMING_DIRNAME
    incoming.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    created = datetime.now(timezone.utc)
    stamp = created.strftime('%Y%m%dT%H%M%SZ')
    active = read_active(data_dir)

    staging = incoming / f'staging-{stamp}'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    bundle_name = f'{BUNDLE_PREFIX}-{host}-{stamp}.tar.gz'
    staged_tar = incoming / bundle_name

    def is_secret(key: str) -> bool:
        if key in config.SECRET_VARS:
            return True
        return any(
            secret and key.startswith(prefix)
            for prefix, _validator, secret in config.PER_STORE_KEY_SPECS)

    try:
        stores_meta: list[dict[str, Any]] = []
        for store in factory.list_stores(data_dir):
            backend_kind = factory.resolve_store_backend(store, data_dir)
            store_out = staging / 'stores' / store
            store_out.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {
                'name': store,
                'backend': backend_kind,
                'embed_fingerprint': None,
                'status': 'ok',
                }
            try:
                if backend_kind == 'postgres':
                    if shutil.which('pg_dump') is None:
                        raise RuntimeError('pg_dump not found on PATH')
                    dsn = factory.resolve_store_pg_dsn(store, data_dir)
                    if not dsn:
                        raise RuntimeError(
                            f'no DSN resolved for postgres store {store!r}')
                    snapshot_postgres(store, dsn, store_out / DUMP_FILENAME)
                    backend = factory.open_backend(
                        store, data_dir, read_only=True)
                    try:
                        fp = stored_fingerprint(backend)
                    finally:
                        backend.close()
                    entry['embed_fingerprint'] = fp.to_json() if fp else None
                else:
                    snapshot_sqlite(store, data_dir, store_out / DB_FILENAME)
                    with contextlib.closing(sqlite3.connect(
                            f'file:{store_out / DB_FILENAME}?mode=ro',
                            uri=True)) as copy_conn:
                        row = copy_conn.execute(
                            "select value from meta"
                            " where key = 'embed_fingerprint'").fetchone()
                    entry['embed_fingerprint'] = row[0] if row else None
            except Exception as exc:
                entry['status'] = 'failed'
                entry['error'] = str(exc)
                logger.warning(
                    'backup: store %r snapshot failed: %s', store, exc)
            stores_meta.append(entry)

        nonsecret = {
            k: v
            for k, v in config.parse_env_file(
                config.env_file_path(data_dir)).items()
            if not is_secret(k) and k not in _HOST_LOCAL_KEYS
            }
        manifest = {
            'format_version': BACKUP_FORMAT_VERSION,
            'created_at_utc': created.isoformat(),
            'host': host,
            'active_store': active,
            'stores': stores_meta,
            }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True))
        (staging / ENV_NONSECRET_NAME).write_text(
            '\n'.join(f'{k}={v}' for k, v in nonsecret.items()) + '\n')

        with tarfile.open(staged_tar, 'w:gz') as tar:
            tar.add(staging, arcname='.')
        fd = os.open(staged_tar, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        final_tar = target_path / bundle_name
        try:
            os.replace(staged_tar, final_tar)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                shutil.move(str(staged_tar), str(final_tar))
            else:
                raise
        sidecar = target_path / f'{bundle_name}.manifest.json'
        sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return {
            'bundle': str(final_tar),
            'manifest': str(sidecar),
            'stores': stores_meta,
            'active_store': active,
            }
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if staged_tar.exists():
            try:
                staged_tar.unlink()
            except OSError:
                pass


def prune(target: str, keep: int) -> list[str]:
    """Delete bundles beyond the newest `keep`, by UTC stamp in the name.

    Returns the removed bundle paths. Names without a parseable
    `YYYYMMDDTHHMMSSZ` stamp are skipped; each bundle's sidecar
    manifest is removed with it.
    """
    target_path = Path(os.path.expanduser(target))
    if not target_path.is_dir():
        return []
    dated: list[tuple[datetime, Path]] = []
    for bundle in target_path.glob(f'{BUNDLE_PREFIX}-*.tar.gz'):
        match = _STAMP_RE.search(bundle.name)
        if not match:
            continue
        try:
            ts = datetime.strptime(match.group(1), '%Y%m%dT%H%M%SZ')
        except ValueError:
            continue
        dated.append((ts, bundle))
    dated.sort(key=lambda item: item[0], reverse=True)

    removed: list[str] = []
    for _ts, bundle in dated[max(keep, 0):]:
        try:
            bundle.unlink()
        except OSError:
            continue
        removed.append(str(bundle))
        sidecar = bundle.with_name(bundle.name + '.manifest.json')
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
    return removed


def run_backup(data_dir: str, target: str | None = None) -> dict[str, Any]:
    """Build a bundle then prune, resolving the target if not passed.

    `target` falls back to `MEMMAN_BACKUP_TARGET`. The worker entry
    point (`memman backup worker`) and `backup run` both call this so
    the build+prune logic lives in one place. Raises `RuntimeError`
    when no target is configured.
    """
    target = target or config.get(config.BACKUP_TARGET)
    if not target:
        raise RuntimeError(
            'no backup target configured; set MEMMAN_BACKUP_TARGET'
            " via 'memman backup schedule' or pass a target")
    result = build_bundle(data_dir, target)
    keep_raw = config.get(config.BACKUP_KEEP)
    try:
        keep = int(keep_raw) if keep_raw else 7
    except ValueError:
        keep = 7
    result['pruned'] = prune(target, keep)
    return result


def restore(bundle_path: str, data_dir: str) -> dict[str, Any]:
    """Rebuild stores + non-secret config from a bundle into `data_dir`.

    Validates the manifest format version and every stored
    fingerprint, merges the non-secret env FIRST (so per-store backend
    routing is in place before stores are written and host secrets are
    preserved), then writes each store by its manifest `backend`.
    Postgres stores resolve their DSN on the target host (the DSN is a
    secret and is not in the bundle); a store with no DSN or no
    `pg_restore` is skipped and reported under `pg_restore_skipped`.
    """
    extract_root = Path(tempfile.mkdtemp(prefix='memman-restore-'))
    try:
        with tarfile.open(bundle_path, 'r:gz') as tar:
            tar.extractall(extract_root, filter='data')

        manifest = json.loads((extract_root / MANIFEST_NAME).read_text())
        version = manifest.get('format_version')
        if version != BACKUP_FORMAT_VERSION:
            raise RuntimeError(
                f'unsupported backup format_version {version!r};'
                f' this memman supports {BACKUP_FORMAT_VERSION}')
        for entry in manifest.get('stores', []):
            fingerprint = entry.get('embed_fingerprint')
            if fingerprint:
                Fingerprint.from_json(fingerprint)

        nonsecret = config.parse_env_file(extract_root / ENV_NONSECRET_NAME)
        if nonsecret:
            _write_env_keys(nonsecret, data_dir=data_dir)

        target_env = config.parse_env_file(config.env_file_path(data_dir))
        host_provider = target_env.get(config.EMBED_PROVIDER) or ''
        host_model_key = {
            'voyage': config.VOYAGE_EMBED_MODEL,
            'openai': config.OPENAI_EMBED_MODEL,
            'openrouter': config.OPENROUTER_EMBED_MODEL,
            'ollama': config.OLLAMA_EMBED_MODEL,
            }.get(host_provider)
        host_model = target_env.get(host_model_key) if host_model_key else None

        restored: list[str] = []
        pg_skipped: list[str] = []
        failed: list[dict[str, Any]] = []
        embed_mismatch: list[str] = []
        for entry in manifest.get('stores', []):
            if entry.get('status') == 'failed':
                continue
            name = entry['name']
            src = extract_root / 'stores' / name
            try:
                if entry.get('backend') == 'postgres':
                    dsn = factory.resolve_store_pg_dsn(name, data_dir)
                    if not dsn or shutil.which('pg_restore') is None:
                        pg_skipped.append(name)
                        continue
                    cmd = ['pg_restore', '--clean', '--if-exists',
                           '-d', dsn, str(src / DUMP_FILENAME)]
                    subprocess.run(
                        cmd, check=True, capture_output=True, text=True)
                else:
                    dst_dir = Path(store_dir(data_dir, name))
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    tmp = dst_dir / (DB_FILENAME + '.tmp')
                    shutil.copy2(src / DB_FILENAME, tmp)
                    os.replace(tmp, dst_dir / DB_FILENAME)
            except (OSError, subprocess.CalledProcessError) as exc:
                stderr = (getattr(exc, 'stderr', '') or '').strip()
                failed.append({
                    'store': name,
                    'error': f'{exc}{f": {stderr}" if stderr else ""}'})
                logger.warning('restore: store %r failed: %s', name, exc)
                continue
            restored.append(name)
            fp_json = entry.get('embed_fingerprint')
            if fp_json and host_provider and host_model:
                fp = Fingerprint.from_json(fp_json)
                if fp.provider != host_provider or fp.model != host_model:
                    embed_mismatch.append(name)

        active = manifest.get('active_store')
        if active:
            write_active(data_dir, active)
        config.reset_file_cache()

        return {
            'restored': restored,
            'failed': failed,
            'pg_restore_skipped': pg_skipped,
            'embed_mismatch': embed_mismatch,
            'active_store': active,
            'secret_keys_needed': [
                k for k in sorted(config.SECRET_VARS)
                if not target_env.get(k)],
            }
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)
