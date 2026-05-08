"""Move post-migrate store state into a dated archive layout.

After a successful `memman migrate <store>`, the source backend's
state is no longer canonical (writes route to the destination via
`MEMMAN_BACKEND_<store>=<target>`). Both directions of `migrate`
preserve the prior backend's state under
`<data_dir>/archive/<store>/<YYYYMMDD>_<NN>/` so the operator-visible
split is unambiguous: `data/` = live SQLite, `archive/` = preserved
post-migrate snapshots (SQLite dirs for forward, `dump.pgdump` for
reverse).

Per-day counter (`_NN`) starts at 01 and increments to support
multiple migrations of the same store on the same day (e.g. flip
to postgres, flip back to sqlite, flip again).
"""

import errno
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger('memman')


def _next_archive_slot(data_dir: str, store: str) -> Path:
    """Return the next free `archive/<store>/<YYYYMMDD>_<NN>/` path.

    Ensures `archive/<store>/` exists; does NOT create the slot dir
    itself (callers either rename a source dir into it or mkdir it
    before writing).
    """
    archive_root = Path(data_dir) / 'archive' / store
    archive_root.mkdir(mode=0o700, exist_ok=True, parents=True)
    date_stem = datetime.now(timezone.utc).strftime('%Y%m%d')
    counter = 1
    while True:
        archive_dest = archive_root / f'{date_stem}_{counter:02d}'
        if not archive_dest.exists():
            return archive_dest
        counter += 1


def archive_store_dir(data_dir: str, store: str) -> Path | None:
    """Move data/<store>/ to archive/<store>/<YYYYMMDD>_<NN>/.

    Returns the archive destination path on success, None when the
    source dir does not exist (no-op for an already-cleaned store).
    Falls back to non-atomic copy+delete on cross-filesystem moves.
    """
    source_dir = Path(data_dir) / 'data' / store
    if not source_dir.exists():
        return None

    archive_dest = _next_archive_slot(data_dir, store)
    try:
        source_dir.rename(archive_dest)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            logger.warning(
                'archive: cross-filesystem rename for store=%s'
                ' (%s -> %s); falling back to non-atomic copy+delete',
                store, source_dir, archive_dest)
            shutil.move(str(source_dir), str(archive_dest))
        else:
            raise
    return archive_dest


def archive_postgres_schema(
        data_dir: str, store: str, dsn: str) -> Path:
    """Dump `store_<store>` schema to archive/<store>/<NN>/dump.pgdump.

    Invokes `pg_dump -Fc -d <dsn> -n store_<store>` and writes the
    binary archive into the next free archive slot. Returns the slot
    path on success. Raises `RuntimeError` on pg_dump failure.
    """
    archive_dest = _next_archive_slot(data_dir, store)
    archive_dest.mkdir(mode=0o700, parents=False, exist_ok=False)
    dump_path = archive_dest / 'dump.pgdump'
    schema = f'store_{store}'
    cmd = [
        'pg_dump', '-Fc', '-d', dsn, '-n', schema,
        '-f', str(dump_path),
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'pg_dump failed for {schema}: {exc.stderr.strip()}'
            ) from exc
    return archive_dest
