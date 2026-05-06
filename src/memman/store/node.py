"""Insight CRUD, lifecycle, statistics, and embedding operations."""

import json
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memman.store.model import MAX_INSIGHTS, Insight, base_weight
from memman.store.model import format_timestamp, insight_to_delta_dict
from memman.store.model import is_immune, parse_timestamp

if TYPE_CHECKING:
    from memman.store.db import DB

logger = logging.getLogger('memman')

HALF_LIFE_DAYS = 30.0
PRUNE_BATCH_SIZE = 10

__all__ = ['MAX_INSIGHTS']


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
     prompt_version, model_id, embedding_model)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
    db._exec(sql, (
        i.id, i.content, i.category, i.importance,
        i.entities_json(), i.source, i.access_count,
        now, now,
        i.prompt_version, i.model_id, i.embedding_model))


_INSIGHT_COLUMNS = (
    'id, content, category, importance, entities,'
    ' source, access_count, created_at, updated_at, deleted_at,'
    ' summary')


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
                   min_importance: int = 0, source: str = '',
                   limit: int = 20) -> list[Insight]:
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
    if min_importance > 0:
        conditions.append('importance >= ?')
        args.append(min_importance)
    if source:
        conditions.append('source = ?')
        args.append(source)

    if limit <= 0:
        limit = 20
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


def compute_effective_importance(
        importance: int, access_count: int,
        days_since_access: float, edge_count: int) -> float:
    """Calculate the current effective importance."""
    base = base_weight(importance)
    access_factor = math.log(1.0 + access_count)
    access_factor = max(access_factor, 1.0)
    decay_factor = math.pow(0.5, days_since_access / HALF_LIFE_DAYS)
    edges = min(edge_count, 5)
    edge_factor = 1.0 + 0.1 * edges
    return base * access_factor * decay_factor * edge_factor


def refresh_effective_importance(db: 'DB', id: str) -> float:
    """Recompute and store effective_importance for one insight."""
    sql = """
select importance, access_count, created_at, last_accessed_at
from insights
where id = ? and deleted_at is null
"""
    row = db._query(sql, (id,)).fetchone()
    if row is None:
        raise ValueError(f'insight {id} not found')

    importance, access_count, created_at_str, last_accessed_at_str = row
    last_access = parse_timestamp(created_at_str)
    if last_accessed_at_str:
        try:
            last_access = parse_timestamp(last_accessed_at_str)
        except ValueError:
            pass

    now = datetime.now(timezone.utc)
    days_since = (now - last_access).total_seconds() / 86400.0

    edge_sql = """
select (select count(*) from edges where source_id = ?)
     + (select count(*) from edges where target_id = ?)
"""
    edge_row = db._query(edge_sql, (id, id)).fetchone()
    edge_count = edge_row[0] if edge_row else 0

    ei = compute_effective_importance(
        importance, access_count, days_since, edge_count)

    db._exec(
        'update insights set effective_importance = ? where id = ?',
        (ei, id))
    return ei


def get_retention_candidates(
        db: 'DB', threshold: float,
        limit: int) -> tuple[list[dict[str, Any]], int]:
    """Return non-immune insights sorted by effective_importance ascending."""
    sql = f"""
select {_INSIGHT_COLUMNS}, last_accessed_at
from insights
where deleted_at is null
"""
    rows = db._query(sql).fetchall()

    insight_rows: list[tuple[Insight, datetime]] = []
    for r in rows:
        ins = _scan_insight(r[:11])
        last_accessed_str = r[11]
        last_access = ins.created_at or datetime.now(timezone.utc)
        if last_accessed_str:
            try:
                last_access = parse_timestamp(last_accessed_str)
            except ValueError:
                pass
        insight_rows.append((ins, last_access))

    ec_sql = """
select id, sum(cnt) from (
    select source_id as id, count(*) as cnt
    from edges
    group by source_id
    union all
    select target_id as id, count(*) as cnt
    from edges
    group by target_id
)
group by id
"""
    ec_rows = db._query(ec_sql).fetchall()
    edge_counts: dict[str, int] = dict(ec_rows)

    now = datetime.now(timezone.utc)
    updates = []
    candidates = []
    for ins, last_access in insight_rows:
        days_since = (now - last_access).total_seconds() / 86400.0
        ec = edge_counts.get(ins.id, 0)
        ei = compute_effective_importance(
            ins.importance, ins.access_count, days_since, ec)
        immune = is_immune(ins.importance, ins.access_count)
        updates.append((ei, ins.id))

        if ei < threshold and not immune:
            candidates.append({
                'insight': ins,
                'effective_importance': ei,
                'days_since_access': days_since,
                'edge_count': ec,
                'immune': immune,
                })

    if updates:
        def apply_ei_updates() -> None:
            for ei_val, uid in updates:
                db._exec(
                    'update insights set effective_importance = ?'
                    ' where id = ?', (ei_val, uid))
        try:
            db.in_transaction(apply_ei_updates)
        except Exception as e:
            logger.warning('batch EI update failed, rolled back: %s', e)

    candidates.sort(
        key=lambda c: float(c['effective_importance']))  # type: ignore[arg-type]
    total = len(insight_rows)
    if limit > 0 and len(candidates) > limit:
        candidates = candidates[:limit]
    return candidates, total


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


def has_active_with_source(db: 'DB', source: str) -> bool:
    """Return True if any active insight exists with the given source."""
    row = db._query(
        'select 1 from insights where source = ?'
        ' and deleted_at is null limit 1',
        (source,)).fetchone()
    return row is not None


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


def auto_prune(db: 'DB', max_insights: int,
               exclude_ids: list[str] | None = None) -> int:
    """Soft-delete the lowest EI non-immune insights when over capacity."""
    if exclude_ids is None:
        exclude_ids = []

    total = count_active_insights(db)
    if total <= max_insights:
        return 0

    excess = min(total - max_insights, PRUNE_BATCH_SIZE)

    args: list[Any] = list(exclude_ids)
    exclude_clause = ''
    if exclude_ids:
        placeholders = ','.join('?' for _ in exclude_ids)
        exclude_clause = f'and id not in ({placeholders})'

    candidate_sql = f"""
select id from insights
where deleted_at is null
  and importance < 4
  and access_count < 3
  {exclude_clause}
order by effective_importance asc
limit {PRUNE_BATCH_SIZE}
"""
    candidate_rows = db._query(candidate_sql, tuple(args)).fetchall()
    for (cid,) in candidate_rows:
        try:
            refresh_effective_importance(db, cid)
        except ValueError:
            pass

    args.append(excess)
    rows_sql = f"""
select id from insights
where deleted_at is null
  and importance < 4
  and access_count < 3
  {exclude_clause}
order by effective_importance asc
limit ?
"""
    rows = db._query(rows_sql, tuple(args)).fetchall()

    now = format_timestamp(datetime.now(timezone.utc))
    pruned = 0
    update_sql = """
update insights
set deleted_at = ?, updated_at = ?
where id = ? and deleted_at is null
"""
    from memman.store.edge import delete_edges_by_node
    from memman.store.oplog import log_op
    for (cid,) in rows:
        before_ins = get_insight_by_id_include_deleted(db, cid)
        cursor = db._exec(update_sql, (now, now, cid))
        if cursor.rowcount > 0:
            delete_edges_by_node(db, cid)
            pruned += 1
            if before_ins is not None:
                log_op(
                    db, 'auto_prune', cid, '',
                    before=insight_to_delta_dict(before_ins))
    return pruned


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


def boost_retention(db: 'DB', id: str) -> None:
    """Boost an insight's retention: access_count +3, refreshes last_accessed_at."""
    now = format_timestamp(datetime.now(timezone.utc))
    sql = """
update insights
set access_count = access_count + 3,
    last_accessed_at = ?,
    updated_at = ?
where id = ? and deleted_at is null
"""
    cursor = db._exec(sql, (now, now, id))
    if cursor.rowcount == 0:
        raise ValueError(f'insight {id} not found or already deleted')


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


def get_latest_insight_by_source(
        db: 'DB', source: str, exclude_id: str) -> Insight | None:
    """Return the most recent non-deleted insight for a given source."""
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where source = ? and id != ? and deleted_at is null
order by created_at desc, rowid desc
limit 1
"""
    row = db._query(sql, (source, exclude_id)).fetchone()
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


def get_insights_without_embedding(
        db: 'DB', limit: int = 100) -> list[Insight]:
    """Return active insights that lack embeddings."""
    if limit <= 0:
        limit = 100
    sql = f"""
select {_INSIGHT_COLUMNS}
from insights
where deleted_at is null and embedding is null
order by importance desc, created_at desc
limit ?
"""
    rows = db._query(sql, (limit,)).fetchall()
    return [_scan_insight(r) for r in rows]


def stamp_linked(db: 'DB', insight_id: str, ts: str) -> None:
    """Set linked_at timestamp for an insight."""
    db._exec(
        'update insights set linked_at = ? where id = ?',
        (ts, insight_id))


def stamp_enriched(db: 'DB', insight_id: str, ts: str) -> None:
    """Set enriched_at timestamp for an insight."""
    db._exec(
        'update insights set enriched_at = ? where id = ?',
        (ts, insight_id))


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
    return i
