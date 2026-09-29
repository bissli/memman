"""Operation logging with auto-trim.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from memman.store.model import format_timestamp

if TYPE_CHECKING:
    from memman.store.db import DB

logger = logging.getLogger('memman')

MAX_OPLOG_ENTRIES = 5000
OPLOG_RETENTION_DAYS = 180


def log_op(db: 'DB', operation: str, insight_id: str,
           detail: str,
           before: dict[str, Any] | None = None,
           after: dict[str, Any] | None = None) -> None:
    """Insert one oplog row; a failed insert logs a warning.

    Insert-only: `maintenance_step` bounds growth once per drain, so
    Postgres `oplog.log` stays a single statement with no delete.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    operation : str
        Operation name.
    insight_id : str
        Id of the insight the operation touched.
    detail : str
        Free-text detail.
    before : dict[str, Any] or None
        Prior insight content on a replace or forget row.
    after : dict[str, Any] or None
        New content on a remember, replace or target-gone row.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    before_s = json.dumps(before) if before is not None else None
    after_s = json.dumps(after) if after is not None else None
    sql = """
insert into oplog (operation, insight_id, detail, created_at,
                   before, after)
values (?, ?, ?, ?, ?, ?)
"""
    try:
        db._exec(
            sql, (operation, insight_id, detail, now,
                  before_s, after_s))
    except Exception as e:
        logger.warning('oplog insert failed: %s', e)


def maintenance_step(db: 'DB') -> None:
    """Cap the oplog at `MAX_OPLOG_ENTRIES`, then reclaim freelist space.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    """
    sql = """
delete from oplog
where id <= (select max(id) from oplog) - ?
"""
    try:
        db._exec(sql, (MAX_OPLOG_ENTRIES,))
    except Exception as e:
        logger.warning('oplog cap trim failed: %s', e)
    db._exec('pragma incremental_vacuum(200)')


def trim_oplog_by_age(db: 'DB') -> int:
    """Delete oplog rows older than `OPLOG_RETENTION_DAYS`.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.

    Returns
    -------
    int
        Rows deleted; 0 when the delete fails, which logs a warning.
    """
    cutoff_dt = datetime.now(timezone.utc) - timedelta(
        days=OPLOG_RETENTION_DAYS)
    cutoff = format_timestamp(cutoff_dt)
    try:
        # idx_oplog_created bounds the delete to the expired rows.
        cur = db._exec(
            'delete from oplog where created_at < ?', (cutoff,))
        return int(cur.rowcount)
    except Exception as exc:
        logger.warning(f'oplog age trim failed: {exc}')
        return 0


def get_oplog(db: 'DB', limit: int = 20,
              since: str = '') -> list[dict[str, Any]]:
    """Return the most recent oplog entries, newest first.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    limit : int, default 20
        Maximum entries returned.
    since : str, default ''
        RFC3339 timestamp; when set, only entries created at or after
        it are returned.

    Returns
    -------
    list[dict[str, Any]]
        One dict per row, with `before` and `after` decoded from JSON
        (None when unset).
    """
    if since:
        sql = """
select id, operation, insight_id, detail, created_at, before, after
from oplog
where created_at >= ?
order by id desc
limit ?
"""
        rows = db._query(sql, (since, limit)).fetchall()
    else:
        sql = """
select id, operation, insight_id, detail, created_at, before, after
from oplog
order by id desc
limit ?
"""
        rows = db._query(sql, (limit,)).fetchall()
    return [{
            'id': row[0],
            'operation': row[1],
            'insight_id': row[2] or '',
            'detail': row[3] or '',
            'created_at': row[4],
            'before': json.loads(row[5]) if row[5] else None,
            'after': json.loads(row[6]) if row[6] else None,
            } for row in rows]


def get_oplog_stats(db: 'DB', since: str = '') -> dict[str, Any]:
    """Return grouped operation counts and the current insight count.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    since : str, default ''
        RFC3339 timestamp; when set, only operations created at or
        after it are counted.

    Returns
    -------
    dict[str, Any]
        `operation_counts` (operation to count, most frequent first)
        and `total_active` (current insights, neither deleted nor
        replaced; the count ignores `since`).
    """
    if since:
        sql = """
select operation, count(*)
from oplog
where created_at >= ?
group by operation
order by count(*) desc
"""
        rows = db._query(sql, (since,)).fetchall()
    else:
        sql = """
select operation, count(*)
from oplog
group by operation
order by count(*) desc
"""
        rows = db._query(sql, ()).fetchall()

    op_counts = {row[0]: row[1] for row in rows}

    total_row = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is null',
        ()).fetchone()
    total_active = total_row[0] if total_row else 0

    return {
        'operation_counts': op_counts,
        'total_active': total_active,
        }
