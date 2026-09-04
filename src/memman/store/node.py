"""Insight CRUD, lifecycle, statistics, and embedding operations."""

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memman.store.model import Insight, format_timestamp, parse_timestamp

if TYPE_CHECKING:
    from memman.store.db import DB

logger = logging.getLogger('memman')


def insert_insight(db: 'DB', i: Insight) -> None:
    """Insert a new insight into the database.

    Stamps `created_at` / `updated_at` server-side: caller-passed
    `i.created_at` / `i.updated_at` are IGNORED. Tests that need to
    control insertion time use the `_set_created_at` helper in
    `tests/conftest.py` to issue a raw update after insert. Mirrors
    `PostgresNodeStore.insert` which relies on `DEFAULT now()`.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
insert into insights
    (id, content, category, importance, entities,
     source, access_count, created_at, updated_at,
     prompt_version, model_id, embedding_model,
     session_id, queue_uuid, corroboration_count)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
    db._exec(sql, (
        i.id, i.content, i.category, i.importance,
        i.entities_json(), i.source, i.access_count,
        now, now,
        i.prompt_version, i.model_id, i.embedding_model,
        i.session_id, i.queue_uuid, i.corroboration_count))


# `session_id`, `queue_uuid`, then `corroboration_count`, appended
# last -- must stay byte-identical to postgres.py's _INSIGHT_COLS
# (see test_insight_column_lists_are_identical_across_backends).
_INSIGHT_COLUMNS = (
    'id, content, category, importance, entities,'
    ' source, access_count, created_at, updated_at, deleted_at,'
    ' summary, linked_at, enriched_at, last_accessed_at,'
    ' session_id, queue_uuid, corroboration_count')


def get_insight_by_id(db: 'DB', id: str) -> Insight | None:
    """Return a single insight by ID (excludes soft-deleted)."""
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id = ? and deleted_at is null
"""
    row = db._query(sql, (id,)).fetchone()
    if row is None:
        return None
    return _scan_insight(row)


def get_insight_by_id_include_deleted(db: 'DB', id: str) -> Insight | None:
    """Return a single insight by ID, including soft-deleted."""
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id = ?
"""
    row = db._query(sql, (id,)).fetchone()
    if row is None:
        return None
    return _scan_insight(row)


def query_insights(db: 'DB', keyword: str = '', category: str = '',
                   source: str = '', limit: int = 20) -> list[Insight]:
    """Return insights matching filters, ordered by importance desc, created_at desc."""
    conditions = ['deleted_at is null']
    args: list[Any] = []

    if keyword:
        for word in keyword.split():
            escaped = word.replace(
                '\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            conditions.append(
                "(content like ? escape '\\'"
                " or entities like ? escape '\\'"
                " or keywords like ? escape '\\')")
            args.extend([f'%{escaped}%'] * 3)
    if category:
        conditions.append('category = ?')
        args.append(category)
    if source:
        conditions.append('source = ?')
        args.append(source)

    args.append(limit)

    where_clause = ' and '.join(conditions)
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where {where_clause}
order by importance desc, created_at desc
limit ?
"""
    rows = db._query(sql, tuple(args)).fetchall()
    return [_scan_insight(r) for r in rows]


def soft_delete_insight(
        db: 'DB', id: str, tolerate_missing: bool = False) -> bool:
    """Set deleted_at on an insight and remove all associated edges.

    Returns True if the insight was soft-deleted, False if it was
    already gone and `tolerate_missing=True`. With `tolerate_missing=False`
    (the default) a missing/already-deleted target raises ValueError —
    the right behavior for `memman forget`, where an unknown id is a
    user bug.

    The worker pipeline passes `tolerate_missing=True` so that a queued
    `replace` whose target was concurrently `forget`-ed degrades to a
    plain add instead of crashing the row's transaction.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set deleted_at = ?, updated_at = ?
where id = ? and deleted_at is null
"""
    cursor = db._exec(sql, (now, now, id))
    if cursor.rowcount == 0:
        if tolerate_missing:
            return False
        raise ValueError(f'insight {id} not found or already deleted')
    from memman.store.edge import delete_edges_by_node
    delete_edges_by_node(db, id)
    return True


def update_entities(db: 'DB', id: str, entities: list[str]) -> None:
    """Update the entities field for an insight."""
    seen: set[str] = set()
    deduped: list[str] = []
    for e in entities:
        key = e.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    now = format_timestamp(datetime.now(timezone.utc))
    db._exec(
        'update insights set entities = ?, updated_at = ? where id = ?',
        (json.dumps(deduped, sort_keys=True), now, id))


def update_enrichment(
        db: 'DB', id: str, keywords: list[str],
        summary: str, semantic_facts: list[str]) -> None:
    """Update LLM enrichment columns for an insight."""
    sql = """
update insights
set keywords = ?, summary = ?, semantic_facts = ?
where id = ?
"""
    db._exec(sql, (
        json.dumps(keywords), summary, json.dumps(semantic_facts), id))


def increment_access_count(db: 'DB', id: str) -> None:
    """Bump the access count and refresh last_accessed_at."""
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set access_count = access_count + 1, last_accessed_at = ?
where id = ?
"""
    db._exec(sql, (now, id))


def increment_corroboration(
        db: 'DB', id: str, queue_uuid: str | None = None) -> bool:
    """Bump corroboration_count on a LIVE insight.

    Never touches `access_count` or `last_accessed_at`: those
    record what recall RETURNED, and a restated fact was not
    returned.

    Parameters
    ----------
    id : str
        The corroborated (stored) insight.
    queue_uuid : str | None, default None
        The restating queue row's idempotency key; adopted onto the
        target ONLY when the target carries none.

    Returns
    -------
    bool
        True when a live row was bumped; False when the target is
        missing or soft-deleted, so the caller can degrade instead
        of silently dropping the restated fact.

    Notes
    -----
    - `coalesce(queue_uuid, ?)` never clobbers a populated key: the
      creating queue row's replay guard outranks the restating
      row's. The cost is that a crash-reclaimed all-skips restating
      row may re-bump once (observational only); the alternative --
      adopting over the creator's key -- un-guards the creating row
      and can replay it into a duplicate insert.
    - Adoption is per-target, not per-row: one queue row restating
      several key-less stored facts stamps its uuid onto each
      (`idx_insights_queue_uuid` is non-unique). The replay guard
      only asks "does ANY live row carry it", so this is forensic
      ambiguity, not a correctness hole.
    """
    sql = """
update insights
set corroboration_count = corroboration_count + 1,
    queue_uuid = coalesce(queue_uuid, ?)
where id = ? and deleted_at is null
"""
    cursor = db._exec(sql, (queue_uuid, id))
    return cursor.rowcount == 1


def count_active_insights(db: 'DB') -> int:
    """Return the number of non-deleted insights."""
    row = db._query(
        'select count(*) from insights where deleted_at is null'
        ).fetchone()
    return int(row[0])


def count_total_insights(db: 'DB') -> int:
    """Return the total number of insights (active + soft-deleted).

    Distinct from `count_active_insights`: used by
    `embed.fingerprint.seed_if_fresh` to detect a genuinely empty
    store. A soft-deleted row is still data with provenance, so the
    fingerprint must not be re-seeded against it.
    """
    row = db._query('select count(*) from insights').fetchone()
    return int(row[0])


def has_active_with_queue_uuid(db: 'DB', queue_uuid: str) -> bool:
    """Return True if any active insight carries the given queue uuid.

    The idempotency check for queue replays. SQL `= ?` never matches
    NULL, so legacy rows with a null `queue_uuid` can never satisfy
    it -- do not add a Python-side default that would.
    """
    row = db._query(
        'select 1 from insights where queue_uuid = ?'
        ' and deleted_at is null limit 1',
        (queue_uuid,)).fetchone()
    return row is not None


def get_by_queue_uuid(db: 'DB', queue_uuid: str) -> list[Insight]:
    """Return the active insights one queued write produced.

    Parameters
    ----------
    db : DB
        Open store connection.
    queue_uuid : str
        The write's idempotency key, as returned by `remember` /
        `replace` and by `memman scheduler queue show`.

    Returns
    -------
    list[Insight]
        Active rows carrying this key, oldest first. Empty when the
        write stored nothing.

    Notes
    -----
    - Ordering tiebreaks on `id`, and the tiebreak is load-bearing:
      siblings of one write often share a `created_at`, but nothing
      guarantees it. Both backends stamp server-side, and only
      Postgres is constant across a transaction (`now()` is
      `transaction_timestamp()`); SQLite stamps each row from its own
      clock read, cut to whole seconds. Without the tiebreak the
      order is the query plan's, and Postgres does not sort stably.
    - Active rows only. A fact that a later reconcile merged away is
      a tombstone, not where the write landed, and SQL `= ?` never
      matches the NULL `queue_uuid` of a pre-0.18.0 row.
    - Empty is a real answer, not an error: a write whose extraction
      returned nothing is recorded in `skipped_writes`, and a write
      that only corroborated an existing insight stamps its key on
      that target only when the target carried none.
    """
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where queue_uuid = ?
  and deleted_at is null
order by created_at, id
"""
    rows = db._query(sql, (queue_uuid,)).fetchall()
    return [_scan_insight(r) for r in rows]


def iter_for_reembed(
        db: 'DB', cursor: str, batch: int
        ) -> list[tuple[str, str, str | None, int | None]]:
    """Return a batch of insights for the reembed sweep.

    Returns rows of (id, content, embedding_model, blob_length).
    The blob length is SQLite-specific (`length(blob)`); on Postgres
    the dimension is invariant from the column type.
    """
    sql = """
select id, content, embedding_model, length(embedding)
from insights
where deleted_at is null and id > ?
order by id
limit ?
"""
    rows = db._query(sql, (cursor, batch)).fetchall()
    return list(rows)


def count_orphans(db: 'DB') -> tuple[int, int]:
    """Return (orphan_count, total_active).

    An orphan is an active insight with zero edges. Used by
    `doctor.check_orphan_insights`. Composing this from
    `get_active_insight_ids` + `get_all_edges` is O(N) Python work
    on SQLite but O(N^2) on Postgres at scale; this helper keeps the
    set-difference inside the database.
    """
    total = db._query(
        'select count(*) from insights where deleted_at is null'
        ).fetchone()[0]
    orphan_sql = """
select count(*)
from insights i
where i.deleted_at is null
  and not exists (
      select 1 from edges e
      where e.source_id = i.id or e.target_id = i.id
  )
"""
    orphan_count = db._query(orphan_sql).fetchone()[0]
    return orphan_count, total


def provenance_distribution(
        db: 'DB') -> list[tuple[str | None, str | None, int]]:
    """Return (prompt_version, model_id, count) groups for active rows.

    Used by `doctor.check_provenance_drift` to detect rows enriched
    by older prompt versions or models. Sorted by count descending.
    """
    sql = """
select prompt_version, model_id, count(*) as n
from insights
where deleted_at is null
group by prompt_version, model_id
order by n desc
"""
    rows = db._query(sql).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def review_content_quality(
        db: 'DB', limit: int = 50) -> list[dict[str, Any]]:
    """Review active insights for content quality issues."""
    from memman.search.quality import check_content_quality

    insights = get_all_active_insights(db)
    flagged = []
    for ins in insights:
        warnings = check_content_quality(ins.content)
        if warnings:
            flagged.append({
                'insight': ins,
                'quality_warnings': warnings,
                })
    flagged.sort(
        key=lambda x: len(x['quality_warnings']),  # type: ignore[arg-type]
        reverse=True)
    return flagged[:limit]


def get_recent_insights_in_window(
        db: 'DB', exclude_id: str, window_hours: float,
        limit: int) -> list[Insight]:
    """Return non-deleted insights created within the given time window."""
    cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600
    cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
    cutoff_str = format_timestamp(cutoff_dt)
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id != ? and deleted_at is null and created_at >= ?
order by created_at desc
limit ?
"""
    rows = db._query(sql, (exclude_id, cutoff_str, limit)).fetchall()
    return [_scan_insight(r) for r in rows]


def get_latest_insight_by_session(
        db: 'DB', session_id: str | None,
        exclude_id: str) -> Insight | None:
    """Return the most recent non-deleted insight for a session.

    Notes
    -----
    - A falsy `session_id` (None or '') returns None here, inside the
      backend: `'' = ''` matches in SQL and would fuse every
      unsessioned row into one false chain.
    - Tiebreak is `created_at desc, id desc` so both backends order
      identically (SQLite's old source-keyed verb tiebroke on rowid,
      which Postgres cannot reproduce).
    """
    if not session_id:
        return None
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where session_id = ? and id != ? and deleted_at is null
order by created_at desc, id desc
limit 1
"""
    row = db._query(sql, (session_id, exclude_id)).fetchone()
    if row is None:
        return None
    return _scan_insight(row)


def get_recent_active_insights(
        db: 'DB', exclude_id: str,
        limit: int) -> list[Insight]:
    """Return the N most recent non-deleted insights regardless of source."""
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where id != ? and deleted_at is null
order by created_at desc
limit ?
"""
    rows = db._query(sql, (exclude_id, limit)).fetchall()
    return [_scan_insight(r) for r in rows]


def get_all_active_insights(db: 'DB') -> list[Insight]:
    """Return all non-deleted insights."""
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where deleted_at is null
order by created_at desc
"""
    rows = db._query(sql).fetchall()
    return [_scan_insight(r) for r in rows]


def get_stats(db: 'DB') -> dict[str, Any]:
    """Return aggregate statistics."""
    stats: dict[str, Any] = {'by_category': {}}

    row = db._query(
        'select count(*) from insights where deleted_at is null'
        ).fetchone()
    stats['total_insights'] = row[0]

    row = db._query(
        'select count(*) from insights where deleted_at is not null'
        ).fetchone()
    stats['deleted_insights'] = row[0]

    cat_sql = """
select category, count(*)
from insights
where deleted_at is null
group by category
"""
    rows = db._query(cat_sql).fetchall()
    for cat, count in rows:
        stats['by_category'][cat] = count

    row = db._query('select count(*) from edges').fetchone()
    stats['edge_count'] = row[0]

    row = db._query('select count(*) from oplog').fetchone()
    stats['oplog_count'] = row[0]

    top_entities = []
    try:
        ent_sql = """
select je.value, count(distinct i.id) as cnt
from insights i, json_each(i.entities) je
where i.deleted_at is null
group by je.value
order by cnt desc
limit 20
"""
        erows = db._query(ent_sql).fetchall()
        for entity, count in erows:
            top_entities.append({'entity': entity, 'count': count})
    except Exception:
        pass
    stats['top_entities'] = top_entities

    return stats


def iter_for_swap(
        db: 'DB', cursor: str, batch: int) -> list[tuple[str, str]]:
    """Return rows still needing embedding_pending under the swap.

    Picks active rows where `embedding_pending is null`, ordered by id
    after `cursor`. Self-healing predicate -- a crash mid-backfill
    skips the cursor and the next call still finds whatever rows
    haven't yet been filled.
    """
    sql = """
select id, content
from insights
where deleted_at is null
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

    Both the blob and `embedding_model` are persisted atomically so
    the row's per-row provenance stays in sync with its vector. The
    `embed reembed` loop's idempotency check depends on this column
    being current.
    """
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set embedding = ?, embedding_model = ?, updated_at = ?
where id = ?
"""
    db._exec(sql, (blob, model, now, id))


def get_embedding(db: 'DB', id: str) -> bytes | None:
    """Return the raw embedding blob for an insight."""
    row = db._query(
        'select embedding from insights'
        ' where id = ? and deleted_at is null',
        (id,)).fetchone()
    if row is None or row[0] is None:
        return None
    blob: bytes = row[0]
    return blob


def get_all_embeddings(db: 'DB') -> list[tuple[str, str, bytes]]:
    """Return all active insights that have embeddings as (id, content, blob)."""
    sql = """
select id, content, embedding
from insights
where deleted_at is null and embedding is not null
"""
    rows = db._query(sql).fetchall()
    results = []
    for id, content, blob in rows:
        if blob and len(blob) > 0:
            results.append((id, content, blob))
    return results


def embedding_stats(db: 'DB') -> tuple[int, int]:
    """Return (total_active, embedded_count)."""
    total = db._query(
        'select count(*) from insights where deleted_at is null'
        ).fetchone()[0]
    embedded = db._query(
        'select count(*) from insights'
        ' where deleted_at is null and embedding is not null'
        ).fetchone()[0]
    return total, embedded


def stamp_linked(db: 'DB', insight_id: str, ts: str) -> None:
    """Set linked_at timestamp for an insight."""
    db._exec(
        'update insights set linked_at = ? where id = ?',
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

    Notes
    -----
    - It deliberately does NOT touch `model_id`. That column records
      the model that produced the row's CONTENT, which re-enrichment
      never rewrites; stamping it here attributed every rebuilt row
      to whatever model happened to be configured at rebuild time and
      corrupted `provenance_distribution`.
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


def get_pending_link_ids(db: 'DB', limit: int) -> list[str]:
    """Return IDs of insights with NULL linked_at, ordered by created_at."""
    sql = """
select id from insights
where linked_at is null and deleted_at is null
order by created_at asc
limit ?
"""
    rows = db._query(sql, (limit,)).fetchall()
    return [r[0] for r in rows]


def get_active_insight_ids(db: 'DB') -> list[str]:
    """Return all active insight IDs in creation order."""
    sql = """
select id from insights
where deleted_at is null
order by created_at asc
"""
    rows = db._query(sql).fetchall()
    return [r[0] for r in rows]


def count_pending_links(db: 'DB') -> int:
    """Count insights with NULL linked_at that are not deleted."""
    row = db._query(
        'select count(*) from insights'
        ' where linked_at is null and deleted_at is null').fetchone()
    return row[0] if row else 0


def get_unenriched_linked_ids(db: 'DB', limit: int) -> list[str]:
    """Return IDs of linked-but-unenriched insights, oldest first.

    These rows were stamped `linked_at` (so the pending-link retry
    path skips them) but never stamped `enriched_at` -- e.g. a prior
    enrichment LLM call failed. They are otherwise stranded.
    """
    sql = """
select id from insights
where enriched_at is null
  and linked_at is not null
  and deleted_at is null
order by created_at asc
limit ?
"""
    rows = db._query(sql, (limit,)).fetchall()
    return [r[0] for r in rows]


def count_unenriched_linked(db: 'DB') -> int:
    """Count linked-but-unenriched active insights."""
    row = db._query(
        'select count(*) from insights'
        ' where enriched_at is null and linked_at is not null'
        ' and deleted_at is null').fetchone()
    return row[0] if row else 0


def iter_stale_insight_ids(
        db: 'DB', active_pv: str) -> list[str]:
    """Return ids of active insights whose staleness key has drifted.

    Notes
    -----
    - A row is stale iff `prompt_version` is non-NULL and differs from
      `active_pv`. NULL provenance is deliberately not stale: those
      rows pre-date provenance tracking and need a backfill, not a
      rebuild.
    - There is no `model_id` branch. `active_pv` already folds in the
      `slow_metadata` model, which is the only model
      `link_pending` re-runs; comparing `model_id` as well would fire
      on the CONTENT model, which no rebuild rewrites, so the row
      would report stale forever.
    - Keep this predicate aligned with
      `doctor._is_provenance_stale` and the Postgres copy.
    """
    sql = """
select id from insights
where deleted_at is null
  and prompt_version is not null
  and prompt_version != ?
order by created_at asc
"""
    rows = db._query(sql, (active_pv,)).fetchall()
    return [r[0] for r in rows]


def count_stale_insights(db: 'DB', active_pv: str) -> int:
    """Count active insights whose staleness key has drifted.

    Same predicate as `iter_stale_insight_ids`.
    """
    sql = """
select count(*) from insights
where deleted_at is null
  and prompt_version is not null
  and prompt_version != ?
"""
    row = db._query(sql, (active_pv,)).fetchone()
    return row[0] if row else 0


def reset_for_rebuild(
        db: 'DB', insight_ids: list[str]) -> None:
    """Clear enriched_at and linked_at for given insight IDs."""
    if not insight_ids:
        return
    placeholders = ','.join('?' for _ in insight_ids)
    sql = f"""
update insights
set enriched_at = null, linked_at = null
where id in ({placeholders})
"""
    db._exec(sql, tuple(insight_ids))


def clear_linked_at(db: 'DB') -> None:
    """Set linked_at to NULL for all active insights."""
    db._exec(
        'update insights set linked_at = null'
        ' where deleted_at is null')


def _scan_insight(row: tuple[Any, ...]) -> Insight:
    """Parse a database row into an Insight dataclass."""
    i = Insight()
    i.id = row[0]
    i.content = row[1]
    i.category = row[2]
    i.importance = row[3]
    i.parse_entities(row[4])
    i.source = row[5]
    i.access_count = row[6]
    i.created_at = parse_timestamp(row[7])
    i.updated_at = parse_timestamp(row[8])
    if row[9]:
        i.deleted_at = parse_timestamp(row[9])
    if len(row) > 10 and row[10]:
        i.summary = row[10]
    if len(row) > 11 and row[11]:
        i.linked_at = parse_timestamp(row[11])
    if len(row) > 12 and row[12]:
        i.enriched_at = parse_timestamp(row[12])
    if len(row) > 13 and row[13]:
        i.last_accessed_at = parse_timestamp(row[13])
    if len(row) > 14 and row[14]:
        i.session_id = row[14]
    if len(row) > 15 and row[15]:
        i.queue_uuid = row[15]
    if len(row) > 16 and row[16] is not None:
        i.corroboration_count = int(row[16])
    return i
