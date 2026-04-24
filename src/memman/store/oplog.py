"""Operation logging with auto-trim."""

import logging
from datetime import datetime, timedelta, timezone

from memman.model import format_timestamp

logger = logging.getLogger('memman')

MAX_OPLOG_ENTRIES = 5000
OPLOG_RETENTION_DAYS = 180


def log_op(db: 'DB', operation: str, insight_id: str,
           detail: str) -> None:
    """Record an operation to the oplog and trim old entries."""
    now = format_timestamp(datetime.now(timezone.utc))
    try:
        db._exec(
            'INSERT INTO oplog'
            ' (operation, insight_id, detail, created_at)'
            ' VALUES (?, ?, ?, ?)',
            (operation, insight_id, detail, now))
    except Exception as e:
        logger.warning('oplog insert failed: %s', e)

    try:
        db._exec(
            'DELETE FROM oplog WHERE id <='
            ' (SELECT MAX(id) FROM oplog) - ?',
            (MAX_OPLOG_ENTRIES,))
    except Exception as e:
        logger.warning('oplog trim failed: %s', e)


def trim_oplog_by_age(
        db: 'DB', retention_days: int = OPLOG_RETENTION_DAYS) -> int:
    """Delete oplog rows older than retention_days. Returns deleted count.

    Called once per worker drain so the table cannot grow unbounded
    even with sparse writes per day. Bounded by idx_oplog_created for
    an O(expired) delete.
    """
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff = format_timestamp(cutoff_dt)
    try:
        cur = db._exec(
            'DELETE FROM oplog WHERE created_at < ?', (cutoff,))
        return cur.rowcount
    except Exception as exc:
        logger.warning(f'oplog age trim failed: {exc}')
        return 0


def get_oplog(db: 'DB', limit: int = 20,
              since: str = '') -> list[dict]:
    """Return the most recent N oplog entries, optionally filtered by date."""
    if limit <= 0:
        limit = 20
    if since:
        rows = db._query(
            'SELECT id, operation, insight_id, detail, created_at'
            ' FROM oplog WHERE created_at >= ?'
            ' ORDER BY id DESC LIMIT ?',
            (since, limit)).fetchall()
    else:
        rows = db._query(
            'SELECT id, operation, insight_id, detail, created_at'
            ' FROM oplog ORDER BY id DESC LIMIT ?',
            (limit,)).fetchall()
    entries = [{
            'id': row[0],
            'operation': row[1],
            'insight_id': row[2] or '',
            'detail': row[3] or '',
            'created_at': row[4],
            } for row in rows]
    return entries


def get_oplog_stats(db: 'DB', since: str = '') -> dict:
    """Return grouped operation counts and never-accessed insight count."""
    if since:
        rows = db._query(
            'SELECT operation, COUNT(*) FROM oplog'
            ' WHERE created_at >= ? GROUP BY operation'
            ' ORDER BY COUNT(*) DESC',
            (since,)).fetchall()
    else:
        rows = db._query(
            'SELECT operation, COUNT(*) FROM oplog'
            ' GROUP BY operation ORDER BY COUNT(*) DESC',
            ()).fetchall()

    op_counts = {row[0]: row[1] for row in rows}

    never_row = db._query(
        'SELECT COUNT(*) FROM insights'
        ' WHERE deleted_at IS NULL AND access_count = 0',
        ()).fetchone()
    never_accessed = never_row[0] if never_row else 0

    total_row = db._query(
        'SELECT COUNT(*) FROM insights WHERE deleted_at IS NULL',
        ()).fetchone()
    total_active = total_row[0] if total_row else 0

    return {
        'operation_counts': op_counts,
        'never_accessed': never_accessed,
        'total_active': total_active,
        }
