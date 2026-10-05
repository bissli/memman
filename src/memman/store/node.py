"""Insight CRUD, lifecycle, statistics, and embedding operations.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memman.store.model import Insight, format_timestamp, parse_timestamp

if TYPE_CHECKING:
    from memman.store.db import DB

logger = logging.getLogger('memman')


def insert_insight(db: 'DB', i: Insight) -> None:
    """Insert a new insight.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    i : Insight
        The row to insert. Its `created_at` and `updated_at` are
        ignored; both are stamped with the current time.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
insert into insights
    (id, content, created_at, updated_at,
     prompt_version, embedding_model,
     queue_uuid, author)
values (?, ?, ?, ?, ?, ?, ?, ?)
"""
    db._exec(sql, (
        i.id, i.content,
        now, now,
        i.prompt_version, i.embedding_model,
        i.queue_uuid, i.author))


# Must stay byte-identical to postgres.py's _INSIGHT_COLS.
_INSIGHT_COLUMNS = (
    'id, content, created_at, updated_at, deleted_at,'
    ' summary, enrich_attempted_at, enriched_at,'
    ' queue_uuid, replaced_by,'
    ' author')


def get_insight_by_id(db: 'DB', id: str) -> Insight | None:
    """The current insight with this ID; None if deleted or replaced.
    """
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id = ? and deleted_at is null and replaced_by is null
"""
    row = db._query(sql, (id,)).fetchone()
    if row is None:
        return None
    return _scan_insight(row)


def get_insight_by_id_include_deleted(db: 'DB', id: str) -> Insight | None:
    """Return a single insight by ID, including soft-deleted.
    """
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id = ?
"""
    row = db._query(sql, (id,)).fetchone()
    if row is None:
        return None
    return _scan_insight(row)


def query_insights(
        db: 'DB', keyword: str = '', limit: int = 20) -> list[Insight]:
    """Return current insights holding every keyword word, newest first.
    """
    conditions = ['deleted_at is null and replaced_by is null']
    args: list[Any] = []

    if keyword:
        for word in keyword.split():
            escaped = word.replace(
                '\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            conditions.append("content like ? escape '\\'")
            args.append(f'%{escaped}%')

    args.append(limit)

    where_clause = ' and '.join(conditions)
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where {where_clause}
order by created_at desc
limit ?
"""
    rows = db._query(sql, tuple(args)).fetchall()
    return [_scan_insight(r) for r in rows]


def soft_delete_insight(db: 'DB', id: str) -> bool:
    """Set deleted_at on a non-deleted insight.

    Returns True when the row was soft-deleted, False when it is
    missing or already deleted. A replaced row may still be deleted;
    `memman forget` turns False into its not-found error.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set deleted_at = ?, updated_at = ?
where id = ? and deleted_at is null
"""
    cursor = db._exec(sql, (now, now, id))
    return cursor.rowcount != 0


def mark_insight_replaced(
        db: 'DB', predecessor_id: str, successor_id: str) -> bool:
    """Point a current insight at its successor.

    Parameters
    ----------
    predecessor_id : str
        The row being replaced. Must be current: neither deleted nor
        already replaced.
    successor_id : str
        The row that replaces it. Not checked here; the pipeline
        writes the pointer before the successor row exists.

    Returns
    -------
    bool
        True when the pointer was written. False when the predecessor
        is missing, deleted, or already replaced; the caller
        degrades to a plain add.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    # The `replaced_by is null` guard replaces a row at most once,
    # which rules out forks in the chain.
    sql = """
update insights
set replaced_by = ?, updated_at = ?
where id = ? and deleted_at is null and replaced_by is null
"""
    cursor = db._exec(sql, (successor_id, now, predecessor_id))
    return cursor.rowcount != 0


def soft_delete_current_insight(db: 'DB', id: str) -> bool:
    """Set deleted_at on a current insight: neither deleted nor replaced.

    Returns True when the row was soft-deleted, False when it is
    missing, deleted, or replaced.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set deleted_at = ?, updated_at = ?
where id = ? and deleted_at is null and replaced_by is null
"""
    cursor = db._exec(sql, (now, now, id))
    return cursor.rowcount != 0


def unterminated_chains(pointers: dict[str, str]) -> list[str]:
    """Return the rows whose pointer chain never reaches a row without one.

    Parameters
    ----------
    pointers : dict[str, str]
        `{row id: replaced_by}` for every row carrying a pointer.

    Returns
    -------
    list[str]
        Sorted ids of every row on or feeding into a cycle. A chain
        ending at a row with no pointer (current, forgotten, or absent
        from the table) terminates; a middle row of such a chain is
        not reported.
    """
    terminates: dict[str, bool] = {}
    for start in pointers:
        path: list[str] = []
        node = start
        while node in pointers and node not in terminates and node not in path:
            path.append(node)
            node = pointers[node]
        outcome = terminates.get(node, node not in path)
        for row in path:
            terminates[row] = outcome
    return sorted(row for row, ok in terminates.items() if not ok)


def replacement_integrity(db: 'DB') -> dict[str, list[str]]:
    """Return the three populations a well-formed pointer set leaves empty.

    Returns
    -------
    dict[str, list[str]]
        `dangling`: rows whose pointer names an id absent from the
        table (a forgotten target is NOT dangling).
        `self_pointer`: rows pointing at themselves.
        `unterminated`: rows whose chain never reaches a row without a
        pointer (a cycle), which no other population sees and which
        removes every member from the active view. A successor with two
        predecessors passes: stored rows may hold one.
        Each list is sorted by id.
    """
    dangling = db._query("""
select p.id
from insights p
left join insights s on s.id = p.replaced_by
where p.replaced_by is not null and s.id is null
order by p.id
""").fetchall()
    selfp = db._query(
        'select id from insights where replaced_by = id order by id'
        ).fetchall()
    pointers = dict(db._query(
        'select id, replaced_by from insights'
        ' where replaced_by is not null').fetchall())
    return {
        'dangling': [r[0] for r in dangling],
        'self_pointer': [r[0] for r in selfp],
        'unterminated': unterminated_chains(pointers),
        }


def get_predecessors(db: 'DB', successor_id: str) -> list[Insight]:
    """Return every row whose `replaced_by` names `successor_id`.

    Deleted rows are included: the history walk shows a forgotten
    predecessor as forgotten rather than dropping it from the chain.
    """
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where replaced_by = ?
order by created_at, id
"""
    rows = db._query(sql, (successor_id,)).fetchall()
    return [_scan_insight(r) for r in rows]


def update_enrichment(db: 'DB', id: str, summary: str) -> None:
    """Store the enrichment summary for an insight.
    """
    db._exec('update insights set summary = ? where id = ?', (summary, id))


def count_active_insights(db: 'DB') -> int:
    """Return the number of current insights, neither deleted nor replaced.
    """
    row = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is null'
        ).fetchone()
    return int(row[0])


def count_total_insights(db: 'DB') -> int:
    """Return the number of insights, soft-deleted and replaced included.

    A soft-deleted row still carries provenance, so
    `embed.fingerprint.seed_if_fresh` counts it and seeds only an
    empty store.
    """
    row = db._query('select count(*) from insights').fetchone()
    return int(row[0])


def has_row_with_queue_uuid(db: 'DB', queue_uuid: str) -> bool:
    """Return True if any insight, forgotten ones included, carries the uuid.

    Answers "did this write land" for queue replays.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    queue_uuid : str
        The queued write's uuid.

    Returns
    -------
    bool
        True for a replaced row too, since a re-insert would revive a
        fact a later `replace` corrected. True for a forgotten row,
        since a re-insert would collide on its primary key.
    """
    # SQL `= ?` never matches NULL, so a row with a null `queue_uuid`
    # never satisfies it. Keep any default out of Python.
    row = db._query(
        'select 1 from insights where queue_uuid = ? limit 1',
        (queue_uuid,)).fetchone()
    return row is not None


def iter_for_reembed(
        db: 'DB', cursor: str, batch: int
        ) -> list[tuple[str, str, str | None, int | None]]:
    """Return a batch of insights for the reembed sweep.

    Parameters
    ----------
    db : DB
        The store's SQLite connection.
    cursor : str
        Last id of the previous batch; only larger ids are returned.
    batch : int
        Maximum rows returned.

    Returns
    -------
    list[tuple[str, str, str | None, int | None]]
        `(id, content, embedding_model, blob_length)` in id order,
        where `blob_length` is `length(embedding)`.
    """
    sql = """
select id, content, embedding_model, length(embedding)
from insights
where deleted_at is null and replaced_by is null and id > ?
order by id
limit ?
"""
    rows = db._query(sql, (cursor, batch)).fetchall()
    return list(rows)


def provenance_distribution(
        db: 'DB') -> list[tuple[str | None, int]]:
    """Return (prompt_version, count) groups for active rows.

    Sorted by count descending.
    """
    sql = """
select prompt_version, count(*) as n
from insights
where deleted_at is null and replaced_by is null
group by prompt_version
order by n desc
"""
    rows = db._query(sql).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_all_active_insights(db: 'DB') -> list[Insight]:
    """Every current insight (not deleted, not replaced), newest first.
    """
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where deleted_at is null and replaced_by is null
order by created_at desc
"""
    rows = db._query(sql).fetchall()
    return [_scan_insight(r) for r in rows]


def get_stats(db: 'DB') -> dict[str, Any]:
    """Return aggregate statistics.

    The three row counts partition the table: `total_insights` is the
    current rows (neither deleted nor replaced),
    `replaced_insights` the replaced rows not deleted, and
    `deleted_insights` every deleted row, replaced or not.
    """
    stats: dict[str, Any] = {}

    row = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is null'
        ).fetchone()
    stats['total_insights'] = row[0]

    row = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is not null'
        ).fetchone()
    stats['replaced_insights'] = row[0]

    row = db._query(
        'select count(*) from insights where deleted_at is not null'
        ).fetchone()
    stats['deleted_insights'] = row[0]

    row = db._query('select count(*) from oplog').fetchone()
    stats['oplog_count'] = row[0]

    return stats


def iter_for_swap(
        db: 'DB', cursor: str, batch: int) -> list[tuple[str, str]]:
    """Return rows still needing embedding_pending under the swap.

    Returns `(id, content)` for active rows after `cursor`, in id
    order, whose `embedding_pending` is still null.
    """
    sql = """
select id, content
from insights
where deleted_at is null and replaced_by is null
  and embedding_pending is null
  and id > ?
order by id
limit ?
"""
    rows = db._query(sql, (cursor, batch)).fetchall()
    return [(r[0], r[1]) for r in rows]


def write_swap_batch(
        db: 'DB', items: list[tuple[str, bytes]]) -> None:
    """Bulk-update `embedding_pending` for each (id, blob) item.
    """
    sql = 'update insights set embedding_pending = ? where id = ?'
    db._conn.executemany(sql, [(blob, rid) for (rid, blob) in items])


def swap_cutover_sqlite(db: 'DB', model: str) -> None:
    """Copy `embedding_pending` into `embedding`, set model, null shadow.

    Runs as a single statement covering every row whose
    `embedding_pending` is populated. Caller must hold a transaction.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set embedding = embedding_pending,
    embedding_model = ?,
    embedding_pending = null,
    updated_at = ?
where embedding_pending is not null
"""
    db._exec(sql, (model, now))


def swap_abort_sqlite(db: 'DB') -> None:
    """Null `embedding_pending` on every row. Discards in-flight backfill.
    """
    db._exec(
        'update insights set embedding_pending = null'
        ' where embedding_pending is not null')


def update_embedding(db: 'DB', id: str, blob: bytes,
                     model: str) -> None:
    """Store an embedding vector and its model name for an insight.

    The vector and `embedding_model` are written in one statement, so
    the recorded model always matches the stored vector. The reembed
    sweep skips a row by reading this column.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set embedding = ?, embedding_model = ?, updated_at = ?
where id = ?
"""
    db._exec(sql, (blob, model, now, id))


def embedding_stats(db: 'DB') -> tuple[int, int]:
    """Return (total_active, embedded_count).
    """
    total = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is null'
        ).fetchone()[0]
    embedded = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and replaced_by is null and embedding is not null'
        ).fetchone()[0]
    return total, embedded


def stamp_enrich_attempted(db: 'DB', insight_id: str, ts: str) -> None:
    """Set enrich_attempted_at timestamp for an insight.
    """
    db._exec(
        'update insights set enrich_attempted_at = ? where id = ?',
        (ts, insight_id))


def stamp_enriched(
        db: 'DB', insight_id: str, ts: str, *,
        prompt_version: str | None = None) -> None:
    """Set enriched_at, and the staleness key when one is given.

    Parameters
    ----------
    db : DB
        Open store handle.
    insight_id : str
        Row to stamp.
    ts : str
        Formatted `enriched_at` timestamp.
    prompt_version : str or None, default None
        The `compute_prompt_version()` key this enrichment ran under.
        Omitted by the write path, which already set it at insert.
    """
    if prompt_version is None:
        db._exec(
            'update insights set enriched_at = ? where id = ?',
            (ts, insight_id))
        return
    db._exec(
        'update insights set enriched_at = ?, prompt_version = ?'
        ' where id = ?',
        (ts, prompt_version, insight_id))


def get_pending_enrich_ids(db: 'DB', limit: int) -> list[str]:
    """Ids of insights with NULL enrich_attempted_at, oldest first.
    """
    sql = """
select id from insights
where enrich_attempted_at is null and deleted_at is null
  and replaced_by is null
order by created_at asc
limit ?
"""
    rows = db._query(sql, (limit,)).fetchall()
    return [r[0] for r in rows]


def get_all_insight_ids(db: 'DB') -> set[str]:
    """Every insight id, current and retired.
    """
    return {r[0] for r in db._query('select id from insights').fetchall()}


def get_active_insight_ids(db: 'DB') -> list[str]:
    """Return all active insight IDs in creation order.
    """
    sql = """
select id from insights
where deleted_at is null and replaced_by is null
order by created_at asc
"""
    rows = db._query(sql).fetchall()
    return [r[0] for r in rows]


def count_pending_enrich(db: 'DB') -> int:
    """Count current insights with a NULL enrich_attempted_at.
    """
    row = db._query(
        'select count(*) from insights'
        ' where enrich_attempted_at is null and deleted_at is null'
        ' and replaced_by is null').fetchone()
    return row[0] if row else 0


def get_unenriched_attempted_ids(db: 'DB', limit: int) -> list[str]:
    """Return IDs of attempted-but-unenriched insights, oldest first.

    Such a row carries `enrich_attempted_at`, so the pending-enrich
    path skips it, but no `enriched_at`.
    """
    sql = """
select id from insights
where enriched_at is null
  and enrich_attempted_at is not null
  and deleted_at is null and replaced_by is null
order by created_at asc
limit ?
"""
    rows = db._query(sql, (limit,)).fetchall()
    return [r[0] for r in rows]


def iter_stale_insight_ids(
        db: 'DB', active_pv: str) -> list[str]:
    """Return ids of the active insights `enrich --stale-only` replays.

    Parameters
    ----------
    db : DB
        The open SQLite store.
    active_pv : str
        The active `compute_prompt_version()` key.

    Returns
    -------
    list[str]
        Ids oldest first: rows whose `prompt_version` is present and
        differs from `active_pv`, plus stranded rows (attempted,
        never enriched), whatever their key. An enriched row with a
        null key stays out.
    """
    # Keep the key term aligned with `doctor._is_provenance_stale`,
    # and the whole predicate aligned with `count_stale_insights` and
    # the Postgres copies.
    sql = """
select id from insights
where deleted_at is null and replaced_by is null
  and ((prompt_version is not null and prompt_version != ?)
       or (enrich_attempted_at is not null and enriched_at is null))
order by created_at asc
"""
    rows = db._query(sql, (active_pv,)).fetchall()
    return [r[0] for r in rows]


def count_stale_insights(db: 'DB', active_pv: str) -> int:
    """Count the active insights `enrich --stale-only` replays.

    Same predicate as `iter_stale_insight_ids`.
    """
    sql = """
select count(*) from insights
where deleted_at is null and replaced_by is null
  and ((prompt_version is not null and prompt_version != ?)
       or (enrich_attempted_at is not null and enriched_at is null))
"""
    row = db._query(sql, (active_pv,)).fetchone()
    return row[0] if row else 0


def reset_for_rebuild(
        db: 'DB', insight_ids: list[str]) -> None:
    """Clear enriched_at and enrich_attempted_at for given insight IDs.
    """
    if not insight_ids:
        return
    placeholders = ','.join('?' for _ in insight_ids)
    sql = f"""
update insights
set enriched_at = null, enrich_attempted_at = null
where id in ({placeholders})
"""
    db._exec(sql, tuple(insight_ids))


def _scan_insight(row: tuple[Any, ...]) -> Insight:
    """Parse a database row into an Insight dataclass.
    """
    i = Insight()
    i.id = row[0]
    i.content = row[1]
    i.created_at = parse_timestamp(row[2])
    i.updated_at = parse_timestamp(row[3])
    if row[4]:
        i.deleted_at = parse_timestamp(row[4])
    if row[5]:
        i.summary = row[5]
    if row[6]:
        i.enrich_attempted_at = parse_timestamp(row[6])
    if row[7]:
        i.enriched_at = parse_timestamp(row[7])
    if row[8]:
        i.queue_uuid = row[8]
    if row[9]:
        i.replaced_by = row[9]
    if row[10]:
        i.author = row[10]
    return i
