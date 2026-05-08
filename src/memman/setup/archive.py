"""Move post-migrate SQLite store dirs into a dated archive layout.

After a successful `memman migrate <store>`, the source SQLite at
`<data_dir>/data/<store>/` is no longer canonical (writes route to
Postgres via `MEMMAN_BACKEND_<store>=postgres`). This module moves
the directory to `<data_dir>/archive/<store>/<YYYYMMDD>_<NN>/` so
the operator-visible split is unambiguous: `data/` = live SQLite,
`archive/` = preserved post-migrate snapshots.

Per-day counter (`_NN`) starts at 01 and increments to support
multiple migrations of the same store on the same day (e.g. flip
to postgres, flip back to sqlite via re-import, flip again).
"""

import errno
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger('memman')


def archive_store_dir(data_dir: str, store: str) -> Path | None:
    """Move data/<store>/ to archive/<store>/<YYYYMMDD>_<NN>/.

    Returns the archive destination path on success, None when the
    source dir does not exist (no-op for an already-cleaned store).
    Falls back to non-atomic copy+delete on cross-filesystem moves.
    """
    source_dir = Path(data_dir) / 'data' / store
    if not source_dir.exists():
        return None

    archive_root = Path(data_dir) / 'archive' / store
    archive_root.mkdir(mode=0o700, exist_ok=True, parents=True)

    date_stem = datetime.now(timezone.utc).strftime('%Y%m%d')
    counter = 1
    while True:
        archive_dest = archive_root / f'{date_stem}_{counter:02d}'
        if not archive_dest.exists():
            break
        counter += 1

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
