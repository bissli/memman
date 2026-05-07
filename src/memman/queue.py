"""Deferred-write queue for memman.

The synchronous write path appends to this queue; a background worker
(`memman scheduler drain --pending`, hidden) drains it and runs the
full remember pipeline.
Single SQLite file at <data_dir>/queue.db with WAL mode. Atomic claim
via update-returning; stale claims are reclaimable after the timeout.
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
    hint_source: str | None
    hint_entities: str | None
    hint_replaced_id: str | None
    hint_no_reconcile: bool
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
    conn.execute('pragma journal_mode=wal')
    conn.execute('pragma busy_timeout=5000')
    _migrate(conn)
    return conn


_BASELINE_SCHEMA = """
create table if not exists queue (
    id            integer primary key autoincrement,
    store         text not null,
    content       text not null,
    hint_cat      text,
    hint_imp      integer,
    hint_source   text,
    hint_entities text,
    hint_replaced_id text,
    hint_no_reconcile integer not null default 0,
    priority      integer not null default 0,
    queued_at     integer not null,
    claimed_at    integer,
    worker_pid    integer,
    attempts      integer not null default 0,
    status        text not null default 'pending'
                  check(status in ('pending','done','failed','stale')),
    last_error    text,
    processed_at  integer
);

create index if not exists idx_queue_ready
    on queue(status, priority desc, queued_at asc)
    where status = 'pending';

create index if not exists idx_queue_store
    on queue(store);

create table if not exists worker_runs (
    id            integer primary key autoincrement,
    started_at    integer not null,
    finished_at   integer,
    worker_pid    integer,
    rows_claimed  integer not null default 0,
    rows_done     integer not null default 0,
    rows_failed   integer not null default 0,
    duration_ms   integer,
    error         text
);

create index if not exists idx_worker_runs_started
    on worker_runs(started_at desc);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply the canonical queue schema.

    Single-user tool: one authoritative schema (`_BASELINE_SCHEMA`),
    always the latest. `create table if not exists` creates a fresh
    queue database; pre-existing databases must already match the
    canonical shape -- wipe and recreate on schema change rather than
    carrying `alter` migrations.
    """
    conn.executescript(_BASELINE_SCHEMA)


def enqueue(
        conn: sqlite3.Connection,
        store: str,
        content: str,
        hint_cat: str | None = None,
        hint_imp: int | None = None,
        hint_source: str | None = None,
        hint_entities: str | None = None,
        hint_replaced_id: str | None = None,
        hint_no_reconcile: bool = False,
        priority: int = 0,
        ) -> int:
    """Append a blob to the queue. Returns the new row's id.

    `hint_replaced_id` carries the id of the insight to soft-delete
    when the worker commits this row — used by the `replace` command.
    `hint_no_reconcile` skips the LLM reconciliation pass for fast
    deterministic stores (`remember --no-reconcile`).
    """
    now = int(time.time())
    sql = """
insert into queue (
    store, content, hint_cat, hint_imp,
    hint_source, hint_entities, hint_replaced_id,
    hint_no_reconcile, priority, queued_at
)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
    cur = conn.execute(sql, (
        store, content, hint_cat, hint_imp, hint_source,
        hint_entities, hint_replaced_id,
        1 if hint_no_reconcile else 0, priority, now))
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
        store_filter = f' and store in ({placeholders})'
        store_params = list(stores)

    sql = f"""
update queue
set claimed_at = ?,
    worker_pid = ?,
    attempts   = attempts + 1
where id = (
    select id from queue
    where status = 'pending'
      and (claimed_at is null or claimed_at <= ? - ?)
      {store_filter}
    order by priority desc, queued_at asc
    limit 1
)
returning id, store, content, hint_cat, hint_imp,
          hint_source, hint_entities, hint_replaced_id,
          hint_no_reconcile, priority, queued_at, attempts
"""
    params = [now, worker_pid, now, stale_after_seconds, *store_params]
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return QueueRow(
        id=row[0], store=row[1], content=row[2],
        hint_cat=row[3], hint_imp=row[4],
        hint_source=row[5], hint_entities=row[6],
        hint_replaced_id=row[7],
        hint_no_reconcile=bool(row[8]),
        priority=row[9], queued_at=row[10], attempts=row[11])


def mark_done(conn: sqlite3.Connection, row_id: int) -> None:
    """Mark a claimed row as successfully processed."""
    now = int(time.time())
    sql = """
update queue
set status = ?,
    processed_at = ?,
    claimed_at = null,
    worker_pid = null
where id = ?
"""
    cur = conn.execute(sql, (STATUS_DONE, now, row_id))
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
    """Mark a claimed row as failed or release it for retry with backoff.

    If attempts >= max_attempts, transitions to status='failed'.
    Otherwise rewrites `claimed_at` to a past timestamp so the existing
    stale-claim reclaim predicate (`claimed_at <= now - STALE_CLAIM_SECONDS`)
    unlocks the row exactly `backoff_seconds` from now -- exponential
    backoff (60, 120, 240, 480, capped at STALE_CLAIM_SECONDS=600) on
    top of the existing claim arithmetic, no new column needed.
    Permanent-failure rows (bad creds, 429 storms) thus get a gentle
    retry curve instead of hammering upstream every drain tick.
    """
    row = conn.execute(
        'select attempts from queue where id = ?',
        (row_id,)).fetchone()
    if row is None:
        logger.warning(f'mark_failed: queue row {row_id} not found')
        return
    attempts = row[0]
    if attempts >= max_attempts:
        now = int(time.time())
        fail_sql = """
update queue
set status = ?,
    last_error = ?,
    processed_at = ?,
    claimed_at = null,
    worker_pid = null
where id = ?
"""
        conn.execute(
            fail_sql, (STATUS_FAILED, error[:1000], now, row_id))
        logger.warning(
            f'queue row {row_id} failed after {attempts} attempts: {error[:200]}')
    else:
        backoff_seconds = min(
            60 * (2 ** (attempts - 1)), STALE_CLAIM_SECONDS)
        reclaim_at = (
            int(time.time()) - STALE_CLAIM_SECONDS + backoff_seconds)
        conn.execute(
            'update queue set last_error = ?, claimed_at = ? where id = ?',
            (error[:1000], reclaim_at, row_id))
        logger.debug(
            f'queue row {row_id} deferred for retry (attempt {attempts});'
            f' unlocks in {backoff_seconds}s')


def stats(conn: sqlite3.Connection) -> dict:
    """Return counts by status plus oldest-pending age."""
    result = {
        'pending': 0,
        'done': 0,
        'failed': 0,
        'oldest_pending_age_seconds': None,
        }
    rows = conn.execute(
        'select status, count(*) from queue group by status').fetchall()
    for status, count in rows:
        if status in result:
            result[status] = count

    oldest = conn.execute(
        "select queued_at from queue where status = 'pending'"
        ' order by queued_at asc limit 1').fetchone()
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
    sql = """
select id, store, priority, queued_at, claimed_at,
       attempts, status, processed_at, substr(content, 1, 80),
       last_error
from queue
"""
    params: tuple = ()
    if status:
        sql += ' where status = ?'
        params = (status,)
    sql += ' order by queued_at desc limit ?'
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
    sql = """
select id, store, content, hint_cat, hint_imp,
       hint_source, hint_entities, priority, queued_at, claimed_at,
       worker_pid, attempts, status, last_error, processed_at
from queue
where id = ?
"""
    row = conn.execute(sql, (row_id,)).fetchone()
    if row is None:
        return None
    return {
        'id': row[0], 'store': row[1], 'content': row[2],
        'hint_cat': row[3], 'hint_imp': row[4],
        'hint_source': row[5], 'hint_entities': row[6],
        'priority': row[7], 'queued_at': row[8],
        'claimed_at': row[9], 'worker_pid': row[10],
        'attempts': row[11], 'status': row[12],
        'last_error': row[13], 'processed_at': row[14],
        }


def retry_row(conn: sqlite3.Connection, row_id: int) -> bool:
    """Re-queue a failed row. Returns True if a row was updated."""
    sql = """
update queue
set status = 'pending',
    attempts = 0,
    last_error = null,
    claimed_at = null,
    worker_pid = null,
    processed_at = null
where id = ? and status = ?
"""
    cur = conn.execute(sql, (row_id, STATUS_FAILED))
    return cur.rowcount > 0


DONE_RETENTION_SECONDS = 60


def purge_done(
        conn: sqlite3.Connection,
        keep_seconds: int = DONE_RETENTION_SECONDS) -> int:
    """Delete `done` queue rows older than `keep_seconds`.

    The grace window avoids racing concurrent readers that may still
    inspect a freshly-completed row via `memman queue list`.
    """
    cutoff = int(time.time()) - keep_seconds
    cur = conn.execute(
        "delete from queue where status = 'done' and processed_at <= ?",
        (cutoff,))
    return cur.rowcount


def purge_store(conn: sqlite3.Connection, store: str) -> int:
    """Delete all queue rows for the named store. Returns deleted count.

    Called from `memman store remove` so that removing a store also
    drops its in-flight queue rows; otherwise stale rows survive the
    rmtree and the worker re-attempts them against a missing data dir.
    """
    cur = conn.execute(
        'delete from queue where store = ?', (store,))
    return cur.rowcount


def purge_worker_runs(
        conn: sqlite3.Connection, keep_days: int = 7) -> int:
    """Drop worker_runs rows older than `keep_days`. Returns deleted count.

    The serve loop writes a heartbeat row every iteration (including
    empty drains) -- at 60 s cadence that is ~525 k rows/year without
    pruning. The maintenance phase calls this once per drain.
    """
    cutoff = int(time.time()) - keep_days * 86400
    cur = conn.execute(
        'delete from worker_runs where started_at < ?', (cutoff,))
    return cur.rowcount


def mark_stale_on_resume(
        conn: sqlite3.Connection,
        age_seconds: int = STALE_RESUME_AGE_SECONDS) -> int:
    """Move pending never-attempted rows older than age_seconds to stale.

    Called when a paused scheduler is resumed -- content queued many days
    ago is unlikely to reconcile cleanly against the current store state,
    so surface it explicitly rather than silently re-enriching.
    """
    cutoff = int(time.time()) - age_seconds
    sql = """
update queue
set status = 'stale'
where status = 'pending'
  and attempts = 0
  and queued_at < ?
"""
    cur = conn.execute(sql, (cutoff,))
    return cur.rowcount


def retry_stale(conn: sqlite3.Connection) -> int:
    """Re-queue all stale rows. Returns number of rows updated."""
    sql = """
update queue
set status = 'pending',
    attempts = 0,
    last_error = null,
    claimed_at = null,
    worker_pid = null,
    processed_at = null
where status = ?
"""
    cur = conn.execute(sql, (STATUS_STALE,))
    return cur.rowcount


def purge_stale(conn: sqlite3.Connection) -> int:
    """Delete all stale rows. Returns deleted count."""
    cur = conn.execute("delete from queue where status = 'stale'")
    return cur.rowcount


def start_worker_run(
        conn: sqlite3.Connection, worker_pid: int) -> int:
    """Record the start of a drain. Returns the worker_runs.id.
    """
    now = int(time.time())
    cur = conn.execute(
        'insert into worker_runs (started_at, worker_pid)'
        ' values (?, ?)',
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
        'select started_at from worker_runs where id = ?',
        (run_id,)).fetchone()
    now = int(time.time())
    duration_ms: int | None = None
    if cur is not None:
        duration_ms = int((now - cur[0]) * 1000)
    update_sql = """
update worker_runs
set finished_at = ?,
    duration_ms = ?,
    rows_claimed = ?,
    rows_done = ?,
    rows_failed = ?,
    error = ?
where id = ?
"""
    conn.execute(update_sql, (
        now, duration_ms, rows_claimed, rows_done, rows_failed,
        error, run_id))


def last_worker_run(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent worker_runs row as a dict, or None.

    Adds `in_progress` (true when finished_at is null), `elapsed_seconds`
    (time since started_at, useful for both in-progress and finished
    runs), and `alive` (true when the worker_pid is still a live process)
    so callers can distinguish healthy in-progress runs from hung or
    orphaned ones without log parsing.
    """
    import os as _os
    import time as _time

    sql = """
select id, started_at, finished_at, worker_pid, rows_claimed,
       rows_done, rows_failed, duration_ms, error
from worker_runs
order by started_at desc
limit 1
"""
    row = conn.execute(sql).fetchone()
    if row is None:
        return None

    started_at = row[1]
    finished_at = row[2]
    worker_pid = row[3]
    in_progress = finished_at is None

    elapsed = None
    if started_at:
        ref = finished_at if finished_at is not None else int(_time.time())
        elapsed = max(0, ref - started_at)

    alive: bool | None = None
    if in_progress and worker_pid:
        try:
            _os.kill(worker_pid, 0)
            alive = True
        except (OSError, ProcessLookupError):
            alive = False

    return {
        'id': row[0],
        'started_at': started_at,
        'finished_at': finished_at,
        'worker_pid': worker_pid,
        'rows_claimed': row[4],
        'rows_done': row[5],
        'rows_failed': row[6],
        'duration_ms': row[7],
        'error': row[8],
        'in_progress': in_progress,
        'elapsed_seconds': elapsed,
        'alive': alive,
        }
