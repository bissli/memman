"""Deferred-write queue for memman.

The synchronous write path appends to this queue; a background worker
(`memman scheduler drain --pending`, hidden) drains it and runs the
full remember pipeline.
Single SQLite file at <data_dir>/queue.db with WAL mode. Atomic claim
via UPDATE-RETURNING; stale claims are reclaimable after the timeout.
"""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger('memman')

QUEUE_FILENAME = 'queue.db'
STALE_CLAIM_SECONDS = 600
MAX_ATTEMPTS = 5
STALE_RESUME_AGE_SECONDS = 7 * 24 * 3600

STATUS_PENDING = 'pending'
STATUS_DONE = 'done'
STATUS_FAILED = 'failed'
STATUS_STALE = 'stale'


@dataclass(slots=True)
class QueueRow:
    """A single queued blob claimed by a worker."""

    id: int
    store: str
    content: str
    hint_cat: str | None
    hint_imp: int | None
    hint_tags: str | None
    hint_source: str | None
    hint_entities: str | None
    hint_replaced_id: str | None
    priority: int
    queued_at: int
    attempts: int


def queue_db_path(base_dir: str) -> str:
    """Return <base_dir>/queue.db."""
    return os.path.join(base_dir, QUEUE_FILENAME)


def open_queue_db(base_dir: str) -> sqlite3.Connection:
    """Open (or create) the queue SQLite database under base_dir."""
    Path(base_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    path = queue_db_path(base_dir)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=5000')
    _migrate(conn)
    return conn


_BASELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    store         TEXT NOT NULL,
    content       TEXT NOT NULL,
    hint_cat      TEXT,
    hint_imp      INTEGER,
    hint_tags     TEXT,
    hint_source   TEXT,
    hint_entities TEXT,
    hint_replaced_id TEXT,
    priority      INTEGER NOT NULL DEFAULT 0,
    queued_at     INTEGER NOT NULL,
    claimed_at    INTEGER,
    worker_pid    INTEGER,
    attempts      INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK(status IN ('pending','done','failed','stale')),
    last_error    TEXT,
    processed_at  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_queue_ready
    ON queue(status, priority DESC, queued_at ASC)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_queue_store
    ON queue(store);

CREATE TABLE IF NOT EXISTS worker_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    INTEGER NOT NULL,
    finished_at   INTEGER,
    worker_pid    INTEGER,
    rows_claimed  INTEGER NOT NULL DEFAULT 0,
    rows_done     INTEGER NOT NULL DEFAULT 0,
    rows_failed   INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS idx_worker_runs_started
    ON worker_runs(started_at DESC);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply the canonical queue schema.

    Single-user tool: one authoritative schema (`_BASELINE_SCHEMA`),
    always the latest. `CREATE TABLE IF NOT EXISTS` creates fresh
    queue databases; existing databases are expected to already match.
    """
    conn.executescript(_BASELINE_SCHEMA)


def enqueue(
        conn: sqlite3.Connection,
        store: str,
        content: str,
        hint_cat: str | None = None,
        hint_imp: int | None = None,
        hint_tags: str | None = None,
        hint_source: str | None = None,
        hint_entities: str | None = None,
        hint_replaced_id: str | None = None,
        priority: int = 0,
        ) -> int:
    """Append a blob to the queue. Returns the new row's id.

    `hint_replaced_id` carries the id of the insight to soft-delete
    when the worker commits this row — used by `remember --defer`
    from the `replace` subcommand.
    """
    now = int(time.time())
    cur = conn.execute(
        'INSERT INTO queue (store, content, hint_cat, hint_imp,'
        ' hint_tags, hint_source, hint_entities, hint_replaced_id,'
        ' priority, queued_at)'
        ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (store, content, hint_cat, hint_imp, hint_tags, hint_source,
         hint_entities, hint_replaced_id, priority, now))
    row_id = cur.lastrowid
    logger.debug(f'queued blob {row_id} for store {store}')
    return row_id


def claim(
        conn: sqlite3.Connection,
        worker_pid: int,
        stale_after_seconds: int = STALE_CLAIM_SECONDS,
        stores: list[str] | None = None,
        ) -> QueueRow | None:
    """Atomically claim one pending (or stale-claimed) row.

    Returns None if nothing is available. Bumps attempts on the claimed
    row; callers must call mark_done or mark_failed to retire it.
    """
    now = int(time.time())
    store_filter = ''
    store_params: list = []
    if stores:
        placeholders = ','.join('?' * len(stores))
        store_filter = f' AND store IN ({placeholders})'
        store_params = list(stores)

    sql = (
        'UPDATE queue'
        '   SET claimed_at = ?,'
        '       worker_pid = ?,'
        '       attempts   = attempts + 1'
        ' WHERE id = ('
        '   SELECT id FROM queue'
        "    WHERE status = 'pending'"
        '      AND (claimed_at IS NULL'
        '           OR claimed_at <= ? - ?)'
        f'{store_filter}'
        '    ORDER BY priority DESC, queued_at ASC'
        '    LIMIT 1'
        ' )'
        ' RETURNING id, store, content, hint_cat, hint_imp, hint_tags,'
        '           hint_source, hint_entities, hint_replaced_id,'
        '           priority, queued_at, attempts')

    params = [now, worker_pid, now, stale_after_seconds, *store_params]
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return QueueRow(
        id=row[0], store=row[1], content=row[2],
        hint_cat=row[3], hint_imp=row[4], hint_tags=row[5],
        hint_source=row[6], hint_entities=row[7],
        hint_replaced_id=row[8],
        priority=row[9], queued_at=row[10], attempts=row[11])


def mark_done(conn: sqlite3.Connection, row_id: int) -> None:
    """Mark a claimed row as successfully processed."""
    now = int(time.time())
    cur = conn.execute(
        'UPDATE queue SET status = ?, processed_at = ?,'
        ' claimed_at = NULL, worker_pid = NULL'
        ' WHERE id = ?',
        (STATUS_DONE, now, row_id))
    if cur.rowcount == 0:
        logger.warning(f'mark_done: queue row {row_id} not found')
    else:
        logger.debug(f'queue row {row_id} marked done')


def mark_failed(
        conn: sqlite3.Connection,
        row_id: int,
        error: str,
        max_attempts: int = MAX_ATTEMPTS,
        ) -> None:
    """Mark a claimed row as failed or release it for retry.

    If attempts >= max_attempts, transitions to status='failed';
    otherwise releases the claim so the next worker can retry.
    """
    row = conn.execute(
        'SELECT attempts FROM queue WHERE id = ?',
        (row_id,)).fetchone()
    if row is None:
        logger.warning(f'mark_failed: queue row {row_id} not found')
        return
    attempts = row[0]
    if attempts >= max_attempts:
        now = int(time.time())
        conn.execute(
            'UPDATE queue SET status = ?, last_error = ?,'
            ' processed_at = ?, claimed_at = NULL, worker_pid = NULL'
            ' WHERE id = ?',
            (STATUS_FAILED, error[:1000], now, row_id))
        logger.warning(
            f'queue row {row_id} failed after {attempts} attempts: {error[:200]}')
    else:
        conn.execute(
            'UPDATE queue SET last_error = ?'
            ' WHERE id = ?',
            (error[:1000], row_id))
        logger.debug(
            f'queue row {row_id} deferred for retry (attempt {attempts});'
            ' will unlock after stale-claim timeout')


def stats(conn: sqlite3.Connection) -> dict:
    """Return counts by status plus oldest-pending age."""
    result = {
        'pending': 0,
        'done': 0,
        'failed': 0,
        'oldest_pending_age_seconds': None,
        }
    rows = conn.execute(
        'SELECT status, COUNT(*) FROM queue GROUP BY status').fetchall()
    for status, count in rows:
        if status in result:
            result[status] = count

    oldest = conn.execute(
        "SELECT queued_at FROM queue WHERE status = 'pending'"
        ' ORDER BY queued_at ASC LIMIT 1').fetchone()
    if oldest is not None:
        result['oldest_pending_age_seconds'] = (
            int(time.time()) - oldest[0])
    return result


def list_rows(
        conn: sqlite3.Connection,
        status: str | None = None,
        limit: int = 50,
        ) -> list[dict]:
    """Return queue rows as dicts, newest first."""
    sql = (
        'SELECT id, store, priority, queued_at, claimed_at,'
        ' attempts, status, processed_at, substr(content, 1, 80),'
        ' last_error'
        ' FROM queue')
    params: tuple = ()
    if status:
        sql += ' WHERE status = ?'
        params = (status,)
    sql += ' ORDER BY queued_at DESC LIMIT ?'
    params = (*params, limit)
    rows = conn.execute(sql, params).fetchall()
    out = [{
            'id': r[0],
            'store': r[1],
            'priority': r[2],
            'queued_at': r[3],
            'claimed_at': r[4],
            'attempts': r[5],
            'status': r[6],
            'processed_at': r[7],
            'content_preview': r[8],
            'last_error': r[9],
            } for r in rows]
    return out


def get_row(
        conn: sqlite3.Connection,
        row_id: int,
        ) -> dict | None:
    """Return full row (including content) as a dict."""
    row = conn.execute(
        'SELECT id, store, content, hint_cat, hint_imp, hint_tags,'
        ' hint_source, hint_entities, priority, queued_at, claimed_at,'
        ' worker_pid, attempts, status, last_error, processed_at'
        ' FROM queue WHERE id = ?',
        (row_id,)).fetchone()
    if row is None:
        return None
    return {
        'id': row[0], 'store': row[1], 'content': row[2],
        'hint_cat': row[3], 'hint_imp': row[4], 'hint_tags': row[5],
        'hint_source': row[6], 'hint_entities': row[7],
        'priority': row[8], 'queued_at': row[9],
        'claimed_at': row[10], 'worker_pid': row[11],
        'attempts': row[12], 'status': row[13],
        'last_error': row[14], 'processed_at': row[15],
        }


def retry_row(conn: sqlite3.Connection, row_id: int) -> bool:
    """Re-queue a failed row. Returns True if a row was updated."""
    cur = conn.execute(
        "UPDATE queue SET status = 'pending', attempts = 0,"
        ' last_error = NULL, claimed_at = NULL, worker_pid = NULL,'
        ' processed_at = NULL'
        ' WHERE id = ? AND status = ?',
        (row_id, STATUS_FAILED))
    return cur.rowcount > 0


def purge_done(conn: sqlite3.Connection) -> int:
    """Delete all rows with status='done'. Returns deleted count."""
    cur = conn.execute("DELETE FROM queue WHERE status = 'done'")
    return cur.rowcount


def mark_stale_on_resume(
        conn: sqlite3.Connection,
        age_seconds: int = STALE_RESUME_AGE_SECONDS) -> int:
    """Move pending never-attempted rows older than age_seconds to stale.

    Called when a paused scheduler is resumed — content queued many days
    ago is unlikely to reconcile cleanly against the current store state,
    so surface it explicitly rather than silently re-enriching.
    """
    cutoff = int(time.time()) - age_seconds
    cur = conn.execute(
        "UPDATE queue SET status = 'stale'"
        " WHERE status = 'pending'"
        ' AND attempts = 0'
        ' AND queued_at < ?',
        (cutoff,))
    return cur.rowcount


def retry_stale(conn: sqlite3.Connection) -> int:
    """Re-queue all stale rows. Returns number of rows updated."""
    cur = conn.execute(
        "UPDATE queue SET status = 'pending', attempts = 0,"
        ' last_error = NULL, claimed_at = NULL, worker_pid = NULL,'
        ' processed_at = NULL'
        ' WHERE status = ?',
        (STATUS_STALE,))
    return cur.rowcount


def purge_stale(conn: sqlite3.Connection) -> int:
    """Delete all stale rows. Returns deleted count."""
    cur = conn.execute("DELETE FROM queue WHERE status = 'stale'")
    return cur.rowcount


def start_worker_run(
        conn: sqlite3.Connection, worker_pid: int) -> int:
    """Record the start of a drain. Returns the worker_runs.id.
    """
    now = int(time.time())
    cur = conn.execute(
        'INSERT INTO worker_runs (started_at, worker_pid)'
        ' VALUES (?, ?)',
        (now, worker_pid))
    return cur.lastrowid


def finish_worker_run(
        conn: sqlite3.Connection,
        run_id: int,
        rows_claimed: int,
        rows_done: int,
        rows_failed: int,
        error: str | None = None) -> None:
    """Stamp finish time, row counts, and duration onto a worker_runs row.
    """
    cur = conn.execute(
        'SELECT started_at FROM worker_runs WHERE id = ?',
        (run_id,)).fetchone()
    now = int(time.time())
    duration_ms: int | None = None
    if cur is not None:
        duration_ms = int((now - cur[0]) * 1000)
    conn.execute(
        'UPDATE worker_runs SET finished_at = ?, duration_ms = ?,'
        ' rows_claimed = ?, rows_done = ?, rows_failed = ?, error = ?'
        ' WHERE id = ?',
        (now, duration_ms, rows_claimed, rows_done, rows_failed,
         error, run_id))


def last_worker_run(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent worker_runs row as a dict, or None.

    Consumed by `memman scheduler status` and (future) `memman doctor`
    to surface scheduler liveness without log parsing.
    """
    row = conn.execute(
        'SELECT id, started_at, finished_at, worker_pid, rows_claimed,'
        ' rows_done, rows_failed, duration_ms, error'
        ' FROM worker_runs ORDER BY started_at DESC LIMIT 1'
        ).fetchone()
    if row is None:
        return None
    return {
        'id': row[0],
        'started_at': row[1],
        'finished_at': row[2],
        'worker_pid': row[3],
        'rows_claimed': row[4],
        'rows_done': row[5],
        'rows_failed': row[6],
        'duration_ms': row[7],
        'error': row[8],
        }
