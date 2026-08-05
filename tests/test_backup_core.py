"""Unit tests for memman.backup core (snapshot, bundle, restore)."""

import json
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest
from memman import config
from memman.backup import BACKUP_FORMAT_VERSION, build_bundle, restore
from memman.backup import snapshot_sqlite
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.store.db import read_active, store_dir, write_active
from memman.store.sqlite import open_sqlite_backend
from tests.conftest import make_insight

_FP = Fingerprint('voyage', 'voyage-3-lite', 512)


def _data_dir() -> str:
    return os.environ['MEMMAN_DATA_DIR']


def _seed_store(data_dir: str, store: str = 'default', n: int = 3) -> None:
    """Materialize a sqlite store with a fingerprint and `n` insights."""
    backend = open_sqlite_backend(store, data_dir)
    write_fingerprint(backend, _FP)
    for i in range(n):
        backend.nodes.insert(
            make_insight(id=f'{store}-{i}', content=f'memory {i}'))
    backend.close()
    write_active(data_dir, store)


class TestSnapshotSqlite:
    """Online sqlite snapshot fidelity."""

    def test_row_count_parity(self):
        """The snapshot copy has the same insight count as the source."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=3)
        dst = Path(data_dir) / 'snap.db'
        snapshot_sqlite('default', data_dir, dst)
        conn = sqlite3.connect(str(dst))
        count = conn.execute('select count(*) from insights').fetchone()[0]
        conn.close()
        assert count == 3

    def test_succeeds_with_open_writer_connection(self):
        """Snapshot works while a live backend connection stays open."""
        data_dir = _data_dir()
        backend = open_sqlite_backend('default', data_dir)
        write_fingerprint(backend, _FP)
        backend.nodes.insert(make_insight(id='a', content='live'))
        write_active(data_dir, 'default')
        dst = Path(data_dir) / 'snap_live.db'
        snapshot_sqlite('default', data_dir, dst)
        backend.close()
        conn = sqlite3.connect(str(dst))
        count = conn.execute('select count(*) from insights').fetchone()[0]
        conn.close()
        assert count == 1


class TestBuildBundle:
    """Bundle assembly: atomicity, contents, secret exclusion."""

    def test_atomic_and_complete(self, tmp_path):
        """A successful bundle leaves no staging and contains every member."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=2)
        target = tmp_path / 'archive'
        result = build_bundle(data_dir, str(target))
        bundle = Path(result['bundle'])
        assert bundle.exists()
        incoming = target / '.memman-incoming'
        assert list(incoming.iterdir()) == []
        with tarfile.open(bundle) as tar:
            names = set(tar.getnames())
        assert './manifest.json' in names
        assert './env.nonsecret' in names
        assert './stores/default/memman.db' in names
        assert not any(n.endswith(('-wal', '-shm')) for n in names)

    def test_secret_keys_excluded(self, tmp_path, env_file):
        """env.nonsecret keeps per-store backend but strips every secret."""
        data_dir = _data_dir()
        _seed_store(data_dir, 's1', n=1)
        env_file('MEMMAN_BACKEND_s1', 'sqlite')
        env_file('MEMMAN_POSTGRES_DSN_s1', 'postgresql://u:pw@h/db')
        env_file('MEMMAN_DEFAULT_POSTGRES_DSN', 'postgresql://u:pw@h/default')
        target = tmp_path / 'archive_sec'
        result = build_bundle(data_dir, str(target))
        with tarfile.open(result['bundle']) as tar:
            env_text = tar.extractfile('./env.nonsecret').read().decode()
        assert 'MEMMAN_BACKEND_s1=sqlite' in env_text
        assert 'MEMMAN_POSTGRES_DSN_s1' not in env_text
        assert 'MEMMAN_DEFAULT_POSTGRES_DSN' not in env_text
        assert 'MEMMAN_OPENROUTER_API_KEY' not in env_text
        assert 'MEMMAN_VOYAGE_API_KEY' not in env_text

    def test_partial_store_failure_does_not_abort(self, tmp_path, monkeypatch):
        """A snapshot failure on one store marks it failed; bundle still completes."""
        import memman.backup as backup_mod

        data_dir = _data_dir()
        _seed_store(data_dir, 'good', n=2)
        _seed_store(data_dir, 'bad', n=1)
        real = backup_mod.snapshot_sqlite

        def flaky(store, dd, dst):
            if store == 'bad':
                raise RuntimeError('disk full')
            real(store, dd, dst)

        monkeypatch.setattr(backup_mod, 'snapshot_sqlite', flaky)
        result = build_bundle(data_dir, str(tmp_path / 'archive_partial'))
        by_name = {e['name']: e for e in result['stores']}
        assert by_name['good']['status'] == 'ok'
        assert by_name['bad']['status'] == 'failed'
        assert 'disk full' in by_name['bad']['error']
        assert Path(result['bundle']).exists()

    def test_queue_captured_and_restored(self, tmp_path):
        """Pending queue rows are snapshotted into the bundle and restored."""
        import sqlite3 as _sq

        from memman.queue import enqueue, queue_db, queue_db_path

        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        with queue_db(data_dir) as conn:
            enqueue(conn, 'default', 'a pending memory')
        target = tmp_path / 'archive_q'
        result = build_bundle(data_dir, str(target))
        with tarfile.open(result['bundle']) as tar:
            assert './queue.db' in tar.getnames()
        manifest = json.loads(next(target.glob('*.manifest.json')).read_text())
        assert manifest['queue_pending'] == 1

        fresh = str(tmp_path / 'fresh_q')
        res = restore(result['bundle'], fresh)
        assert res['queue_restored'] is True
        conn = _sq.connect(queue_db_path(fresh))
        pending = conn.execute(
            "select count(*) from queue where status = 'pending'").fetchone()[0]
        conn.close()
        assert pending == 1

    def test_queue_snapshotted_before_stores(self, tmp_path, monkeypatch):
        """queue.db is copied before any store DB (the loss-safety ordering)."""
        import memman.backup as backup_mod
        from memman.queue import enqueue, queue_db

        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        with queue_db(data_dir) as conn:
            enqueue(conn, 'default', 'pending')
        order: list = []
        real = backup_mod._online_copy

        def recording(src, dst):
            order.append(Path(dst).name)
            real(src, dst)

        monkeypatch.setattr(backup_mod, '_online_copy', recording)
        build_bundle(data_dir, str(tmp_path / 'archive_ord'))
        assert order[0] == 'queue.db'
        assert 'memman.db' in order[1:]

    def test_host_local_backup_keys_excluded(self, tmp_path, env_file):
        """BACKUP_CRON/TARGET/KEEP are host-local and stay out of the bundle."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        env_file('MEMMAN_BACKUP_CRON', '0 3 * * *')
        env_file('MEMMAN_BACKUP_TARGET', '/some/host/path')
        env_file('MEMMAN_BACKUP_KEEP', '5')
        result = build_bundle(data_dir, str(tmp_path / 'archive_hl'))
        with tarfile.open(result['bundle']) as tar:
            env_text = tar.extractfile('./env.nonsecret').read().decode()
        assert 'MEMMAN_BACKUP_CRON' not in env_text
        assert 'MEMMAN_BACKUP_TARGET' not in env_text
        assert 'MEMMAN_BACKUP_KEEP' not in env_text

    def test_manifest_records_parseable_fingerprint(self, tmp_path):
        """The manifest's per-store fingerprint parses back to a Fingerprint."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        target = tmp_path / 'archive_fp'
        build_bundle(data_dir, str(target))
        sidecar = next(target.glob('*.manifest.json'))
        manifest = json.loads(sidecar.read_text())
        entry = manifest['stores'][0]
        assert entry['backend'] == 'sqlite'
        assert Fingerprint.from_json(entry['embed_fingerprint']).dim == 512


class TestRestore:
    """Restore rebuilds stores + config and reports secrets to re-enter."""

    def test_round_trip_into_fresh_dir(self, tmp_path):
        """Restoring into an empty dir recreates the store and active pointer."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=4)
        target = tmp_path / 'archive_rt'
        bundle = build_bundle(data_dir, str(target))['bundle']
        fresh = str(tmp_path / 'fresh')
        result = restore(bundle, fresh)
        assert result['active_store'] == 'default'
        assert 'default' in result['restored']
        db_path = Path(store_dir(fresh, 'default')) / 'memman.db'
        conn = sqlite3.connect(str(db_path))
        count = conn.execute('select count(*) from insights').fetchone()[0]
        conn.close()
        assert count == 4
        assert read_active(fresh) == 'default'

    def test_reports_missing_secrets(self, tmp_path):
        """A fresh restore lists secret keys the operator must re-enter."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        target = tmp_path / 'archive_sec2'
        bundle = build_bundle(data_dir, str(target))['bundle']
        result = restore(bundle, str(tmp_path / 'fresh2'))
        assert config.VOYAGE_API_KEY in result['secret_keys_needed']

    def test_preserves_existing_host_secret(self, tmp_path):
        """Restore merges non-secret config without clobbering host secrets."""
        from memman.setup.scheduler import _write_env_keys

        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        target = tmp_path / 'archive_keep'
        bundle = build_bundle(data_dir, str(target))['bundle']
        fresh = str(tmp_path / 'fresh3')
        _write_env_keys(
            {config.VOYAGE_API_KEY: 'host-secret'}, data_dir=fresh)
        restore(bundle, fresh)
        env = config.parse_env_file(config.env_file_path(fresh))
        assert env[config.VOYAGE_API_KEY] == 'host-secret'

    def test_partial_failure_isolated(self, tmp_path, monkeypatch):
        """A per-store restore failure is isolated; other stores still restore."""
        import memman.backup as backup_mod

        data_dir = _data_dir()
        _seed_store(data_dir, 'alpha', n=2)
        _seed_store(data_dir, 'beta', n=1)
        bundle = build_bundle(data_dir, str(tmp_path / 'archive_ri'))['bundle']
        real_copy = backup_mod.shutil.copy2

        def flaky_copy(src, dst, *a, **k):
            if 'beta' in str(src):
                raise OSError('copy denied')
            return real_copy(src, dst, *a, **k)

        monkeypatch.setattr(backup_mod.shutil, 'copy2', flaky_copy)
        result = restore(bundle, str(tmp_path / 'fresh_ri'))
        assert 'alpha' in result['restored']
        assert 'beta' not in result['restored']
        assert any(f['store'] == 'beta' for f in result['failed'])

    def test_reports_embed_mismatch(self, tmp_path, env_file):
        """A store whose fingerprint differs from the bundled embed model is flagged."""
        data_dir = _data_dir()
        _seed_store(data_dir, 'default', n=1)
        env_file('MEMMAN_VOYAGE_EMBED_MODEL', 'voyage-3-large')
        bundle = build_bundle(data_dir, str(tmp_path / 'archive_em'))['bundle']
        result = restore(bundle, str(tmp_path / 'fresh_em'))
        assert 'default' in result['embed_mismatch']

    def test_restore_bundle_without_queue(self, tmp_path):
        """A current-format bundle with no queue.db restores cleanly.

        Mutation: making `restore` require queue.db in the archive.
        Oracle: `queue_restored` is False and no exception surfaces.
        """
        staging = tmp_path / 'st_nq'
        staging.mkdir()
        (staging / 'manifest.json').write_text(json.dumps({
            'format_version': BACKUP_FORMAT_VERSION, 'stores': [],
            'active_store': 'default'}))
        (staging / 'env.nonsecret').write_text('\n')
        bundle = tmp_path / 'noqueue.tar.gz'
        with tarfile.open(bundle, 'w:gz') as tar:
            tar.add(staging, arcname='.')
        res = restore(str(bundle), str(tmp_path / 'out_nq'))
        assert res['queue_restored'] is False

    def test_restore_refuses_v1_bundle(self, tmp_path):
        """A pre-0.18.0 v1 bundle is refused, not silently restored.

        A v1 bundle restored onto this build would yield a store
        missing `session_id`/`queue_uuid` that fails at `open_db`.

        Mutation: forgetting the `BACKUP_FORMAT_VERSION` bump (v1
            would then round-trip as current).
        Oracle: `restore` raises naming the unsupported version 1.
        """
        staging = tmp_path / 'st_v1'
        staging.mkdir()
        (staging / 'manifest.json').write_text(json.dumps({
            'format_version': 1, 'stores': [],
            'active_store': 'default'}))
        (staging / 'env.nonsecret').write_text('\n')
        bundle = tmp_path / 'v1.tar.gz'
        with tarfile.open(bundle, 'w:gz') as tar:
            tar.add(staging, arcname='.')
        with pytest.raises(RuntimeError, match='format_version 1'):
            restore(str(bundle), str(tmp_path / 'out_v1'))

    def test_rejects_unknown_format_version(self, tmp_path):
        """A bundle with a newer format_version is refused."""
        staging = tmp_path / 'staging'
        staging.mkdir()
        (staging / 'manifest.json').write_text(json.dumps({
            'format_version': 999, 'stores': [],
            'active_store': 'default'}))
        (staging / 'active').write_text('default\n')
        (staging / 'env.nonsecret').write_text('\n')
        bundle = tmp_path / 'bad.tar.gz'
        with tarfile.open(bundle, 'w:gz') as tar:
            tar.add(staging, arcname='.')
        with pytest.raises(RuntimeError):
            restore(str(bundle), str(tmp_path / 'out'))
