"""Postgres + pgvector implementation of the Backend Protocol surface.

Single-file parallel to `store/sqlite.py`. Schema-per-store layout:
each memman store maps to a Postgres schema named `store_<name>`,
holding the four per-store tables (insights, edges, oplog, meta).
Queue tables live in a global `queue` schema, shared across all
stores in the cluster.

Vector storage:
- `embedding vector(512)` (pgvector); pgvector adapter binds
  `list[float]` directly with no per-call serialization.
- HNSW index built `create index concurrently ... vector_cosine_ops
  where deleted_at is null`. Built outside any transaction; reindex
  drops invalid remnants (`pg_index.indisvalid`) before retrying.
- Similarity returned as `1 - (embedding <=> :q)` (cosine in
  [-1, 1]; higher better).

Recall issues one round-trip per anchor subset.
"""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Self

from memman.store.backend import Backend, Cluster, EdgeStore, MetaStore
from memman.store.backend import NodeStore, Oplog, RecallSession
from memman.store.backend import _check_identifier
from memman.store.errors import BackendError, ConfigError
from memman.store.model import Edge, EnrichmentCoverage, Id, Insight
from memman.store.model import IntegrityReport, NodeStats, OpLogEntry
from memman.store.model import OpLogStats, ProvenanceCount, QueueRow
from memman.store.model import QueueStats, ReembedRow, WorkerRun
from memman.store.model import parse_timestamp

if TYPE_CHECKING:
    import psycopg

logger = logging.getLogger('memman')

EMBEDDING_DIM = 512

_PG_SCHEMA_VERSION = 2
_PG_MIGRATIONS: list[tuple[int, str]] = [
    (2,
     ('alter table queue.worker_runs'
      ' add column if not exists last_heartbeat_at timestamptz')),
    ]

_FORBIDDEN_MIGRATION_RE = re.compile(
    r'(?i)\b(?:'
    r'drop\s+column|rename|drop\s+table|truncate'
    r'|alter\s+column\b.*\b(?:not\s+null|type)\b'
    r')')

for _ver, _sql in _PG_MIGRATIONS:
    if _FORBIDDEN_MIGRATION_RE.search(_sql):
        raise AssertionError(
            f'_PG_MIGRATIONS[{_ver}] contains a non-additive operation;'
            ' the ladder is forward-only and additive-only')


def _store_schema(name: str) -> str:
    """Return the Postgres schema name for a memman store."""
    _check_identifier(name)
    return f'store_{name}'


def _advisory_lock_key(schema: str, name: str) -> int:
    """Per-store, per-name int64 key for `pg_advisory_*lock` calls."""
    return abs(hash(f'{schema}:{name}')) & 0x7FFFFFFFFFFFFFFF


_PG_BASELINE_SCHEMA = """
create table if not exists {schema}.insights (
    id          text primary key,
    content     text not null,
    category    text default 'general',
    importance  integer default 3,
    entities    jsonb default '[]'::jsonb,
    source      text default 'user',
    access_count integer default 0,
    keywords    jsonb,
    summary     text,
    semantic_facts jsonb,
    last_accessed_at timestamptz,
    embedding   vector({dim}),
    effective_importance double precision default 0.5,
    linked_at   timestamptz,
    enriched_at timestamptz,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    deleted_at  timestamptz,
    prompt_version text,
    model_id    text,
    embedding_model text
);

create table if not exists {schema}.edges (
    source_id   text not null,
    target_id   text not null,
    edge_type   text not null,
    weight      double precision default 1.0,
    metadata    jsonb default '{{}}'::jsonb,
    created_at  timestamptz not null default now(),
    primary key (source_id, target_id, edge_type),
    foreign key (source_id) references {schema}.insights(id) on delete cascade,
    foreign key (target_id) references {schema}.insights(id) on delete cascade
);

create table if not exists {schema}.oplog (
    id          bigserial primary key,
    operation   text not null,
    insight_id  text,
    detail      text default '',
    created_at  timestamptz not null default now()
);

create table if not exists {schema}.meta (
    key   text primary key,
    value text not null
);

create index if not exists idx_insights_category_{schema}
    on {schema}.insights(category);
create index if not exists idx_insights_importance_{schema}
    on {schema}.insights(importance);
create index if not exists idx_insights_created_{schema}
    on {schema}.insights(created_at);
create index if not exists idx_insights_deleted_{schema}
    on {schema}.insights(deleted_at);
create index if not exists idx_insights_source_{schema}
    on {schema}.insights(source);
create index if not exists idx_insights_eff_imp_{schema}
    on {schema}.insights(effective_importance);
create index if not exists idx_insights_pending_link_{schema}
    on {schema}.insights(linked_at)
    where linked_at is null and deleted_at is null;
create index if not exists idx_edges_source_{schema}
    on {schema}.edges(source_id);
create index if not exists idx_edges_target_{schema}
    on {schema}.edges(target_id);
create index if not exists idx_edges_type_{schema}
    on {schema}.edges(edge_type);
create index if not exists idx_edges_source_type_{schema}
    on {schema}.edges(source_id, edge_type);
create index if not exists idx_edges_target_type_{schema}
    on {schema}.edges(target_id, edge_type);
create index if not exists idx_oplog_created_{schema}
    on {schema}.oplog(created_at);
"""

_PG_QUEUE_SCHEMA = """
create schema if not exists queue;

create table if not exists queue.queue (
    id          bigserial primary key,
    store       text not null,
    op          text not null,
    payload     text not null,
    status      text not null default 'pending',
    attempts    integer not null default 0,
    error       text,
    created_at  timestamptz not null default now(),
    claimed_at  timestamptz,
    finished_at timestamptz
);

create index if not exists idx_queue_status_id
    on queue.queue(status, id);
create index if not exists idx_queue_store
    on queue.queue(store);

create table if not exists queue.worker_runs (
    id            bigserial primary key,
    started_at    timestamptz not null default now(),
    ended_at      timestamptz,
    rows_processed integer not null default 0,
    error         text not null default '',
    last_heartbeat_at timestamptz
);
"""


_MAX_OPLOG_ENTRIES = 5000


_REINDEX_CREATED_BY_FILTER = {
    'semantic': "metadata->>'created_by' = 'auto'",
    'entity': ("(metadata->>'created_by' is null"
               " or metadata->>'created_by'"
               " not in ('claude', 'manual'))"),
    'causal': ("(metadata->>'created_by' is null"
               " or metadata->>'created_by'"
               " not in ('llm', 'claude', 'manual'))"),
    }

_PER_NODE_CREATED_BY_FILTER = {
    'entity': ("(metadata->>'created_by' is null"
               " or metadata->>'created_by'"
               " not in ('claude', 'manual'))"),
    'semantic': ("(metadata->>'created_by' is null"
                 " or metadata->>'created_by' = 'auto')"),
    'causal': "metadata->>'created_by' = 'llm'",
    }


def _open_connection(
        dsn: str, *, autocommit: bool = False,
        keepalives: bool = False) -> psycopg.Connection:
    """Open a fresh psycopg connection with pgvector adapters.

    `keepalives=True` adds `keepalives_idle=30` for the drain-lock
    connection so a hung worker is detected by the kernel rather
    than holding the lock indefinitely.

    Returns a bare connection; lock-holding paths (`drain_lock`,
    `reembed_lock`) and long-lived backend connections own the
    lifecycle directly. One-shot helpers should use `_connection()`
    below for guaranteed close-on-exit semantics.
    """
    import psycopg
    from pgvector.psycopg import register_vector
    kwargs: dict[str, Any] = {'autocommit': autocommit}
    if keepalives:
        kwargs['keepalives'] = 1
        kwargs['keepalives_idle'] = 30
    conn = psycopg.connect(dsn, **kwargs)
    register_vector(conn)
    return conn


@contextmanager
def _connection(
        dsn: str, *, autocommit: bool = False,
        keepalives: bool = False) -> Iterator[psycopg.Connection]:
    """Context-manager wrapper around `_open_connection`.

    Closes on exit (psycopg3's `with conn:` is transaction-scoped, not
    close-scoped, so wrapping `_open_connection` is the way to get
    deterministic close-on-exit semantics for one-shot helpers
    without losing the `register_vector` adapter setup).
    """
    conn = _open_connection(
        dsn, autocommit=autocommit, keepalives=keepalives)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _datetime_or_none(v: Any) -> datetime | None:
    """Coerce a psycopg timestamp value to a UTC-aware datetime.

    psycopg returns TIMESTAMPTZ as `datetime`; this helper normalizes
    naive datetimes (defensive: pgvector / older drivers may strip
    tzinfo) to UTC and passes through None.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    if isinstance(v, str):
        try:
            return parse_timestamp(v)
        except ValueError:
            return None
    return None


def _row_to_insight(row: tuple[Any, ...]) -> Insight:
    """Map a select row into an Insight dataclass."""
    i = Insight()
    i.id = row[0]
    i.content = row[1]
    i.category = row[2] or 'general'
    i.importance = row[3] if row[3] is not None else 3
    ents = row[4]
    if isinstance(ents, list):
        i.entities = ents
    elif isinstance(ents, str):
        i.parse_entities(ents)
    else:
        i.entities = []
    i.source = row[5] or 'user'
    i.access_count = row[6] or 0
    i.created_at = _datetime_or_none(row[7])
    i.updated_at = _datetime_or_none(row[8])
    i.deleted_at = _datetime_or_none(row[9])
    if len(row) > 10 and row[10]:
        i.summary = row[10]
    return i


def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    """Map a select row into an Edge dataclass."""
    e = Edge()
    e.source_id = row[0]
    e.target_id = row[1]
    e.edge_type = row[2]
    e.weight = row[3] if row[3] is not None else 1.0
    md = row[4]
    if isinstance(md, dict):
        e.metadata = md
    elif isinstance(md, str):
        e.parse_metadata(md)
    else:
        e.metadata = {}
    e.created_at = _datetime_or_none(row[5])
    return e


_INSIGHT_COLS = (
    'id, content, category, importance, entities,'
    ' source, access_count, created_at, updated_at, deleted_at,'
    ' summary')


class PostgresNodeStore(NodeStore):
    """NodeStore implementation against a per-store Postgres schema."""

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def _q(self, sql: str) -> str:
        """Format SQL with the per-store schema interpolated."""
        return sql.format(s=self._schema)

    @contextmanager
    def write_lock(self, name: str) -> Iterator[None]:
        """Per-store transaction-scoped advisory lock on the same conn.

        Mirrors `PostgresBackend.write_lock` (shared key namespace via
        `_advisory_lock_key`). Used by `auto_prune` to serialize
        prune sweeps against concurrent reindex/insert work without
        going through the Backend object.
        """
        key = _advisory_lock_key(self._schema, name)
        with self._conn.cursor() as cur:
            cur.execute('select pg_advisory_xact_lock(%s)', (key,))
        yield

    def insert(self, ins: Insight) -> None:
        sql = self._q("""
insert into {s}.insights
    (id, content, category, importance, entities,
     source, access_count, prompt_version, model_id, embedding_model)
values (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                ins.id, ins.content, ins.category, ins.importance,
                ins.entities_json(), ins.source, ins.access_count,
                ins.prompt_version, ins.model_id, ins.embedding_model))

    def get(self, id: Id) -> Insight | None:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id = %s and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()
            return _row_to_insight(row) if row else None

    def get_include_deleted(self, id: Id) -> Insight | None:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()
            return _row_to_insight(row) if row else None

    def get_many(self, ids: Sequence[Id]) -> list[Insight]:
        if not ids:
            return []
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id = any(%s) and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (list(ids),))
            rows = cur.fetchall()
        by_id = {r[0]: _row_to_insight(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def query(
            self, *, keyword: str = '', category: str = '',
            min_importance: int = 0, source: str = '',
            limit: int = 20) -> list[Insight]:
        conditions = ['deleted_at is null']
        args: list[Any] = []
        if keyword:
            for word in keyword.split():
                conditions.append(
                    '(content ilike %s or entities::text ilike %s'
                    ' or keywords::text ilike %s)')
                pat = f'%{word}%'
                args.extend([pat, pat, pat])
        if category:
            conditions.append('category = %s')
            args.append(category)
        if min_importance > 0:
            conditions.append('importance >= %s')
            args.append(min_importance)
        if source:
            conditions.append('source = %s')
            args.append(source)
        if limit <= 0:
            limit = 20
        args.append(limit)
        where_clause = ' and '.join(conditions)
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where {where_clause}
order by importance desc, created_at desc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def soft_delete(
            self, id: Id, *, tolerate_missing: bool = False) -> bool:
        update_sql = self._q("""
update {s}.insights
set deleted_at = now(), updated_at = now()
where id = %s and deleted_at is null
""")
        delete_sql = self._q("""
delete from {s}.edges
where source_id = %s or target_id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(update_sql, (id,))
            if cur.rowcount == 0:
                if tolerate_missing:
                    return False
                raise ValueError(
                    f'insight {id} not found or already deleted')
            cur.execute(delete_sql, (id, id))
        return True

    def update_entities(self, id: Id, entities: list[str]) -> None:
        seen: set[str] = set()
        deduped: list[str] = []
        for e in entities:
            key = e.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        sql = self._q("""
update {s}.insights
set entities = %s::jsonb, updated_at = now()
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(
                sql, (Insight(entities=deduped).entities_json(), id))

    def update_enrichment(
            self, id: Id, *, keywords: list[str], summary: str,
            semantic_facts: list[str]) -> None:
        import json as _json
        sql = self._q("""
update {s}.insights
set keywords = %s::jsonb,
    summary = %s,
    semantic_facts = %s::jsonb,
    updated_at = now()
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                _json.dumps(keywords), summary,
                _json.dumps(semantic_facts), id))

    def increment_access_count(self, id: Id) -> None:
        sql = self._q("""
update {s}.insights
set access_count = access_count + 1, last_accessed_at = now()
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))

    def refresh_effective_importance(self, id: Id) -> float:
        select_sql = self._q("""
select importance, access_count, created_at, last_accessed_at
from {s}.insights
where id = %s and deleted_at is null
""")
        edge_sql = self._q("""
select (select count(*) from {s}.edges where source_id = %s)
     + (select count(*) from {s}.edges where target_id = %s)
""")
        update_sql = self._q("""
update {s}.insights
set effective_importance = %s
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(select_sql, (id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f'insight {id} not found')
            importance = row[0]
            access_count = row[1]
            created_at = _datetime_or_none(row[2]) or datetime.now(
                timezone.utc)
            last_access = _datetime_or_none(row[3]) or created_at
            cur.execute(edge_sql, (id, id))
            edge_row = cur.fetchone()
            edge_count = edge_row[0] if edge_row else 0

        from memman.store.node import compute_effective_importance
        now = datetime.now(timezone.utc)
        days_since = (now - last_access).total_seconds() / 86400.0
        ei = compute_effective_importance(
            importance, access_count, days_since, edge_count)
        with self._conn.cursor() as cur:
            cur.execute(update_sql, (ei, id))
        return ei

    def get_retention_candidates(
            self, *, threshold: float,
            limit: int) -> tuple[list[dict[str, Any]], int]:
        from memman.store.model import is_immune
        from memman.store.node import compute_effective_importance
        rows_sql = self._q(f"""
select {_INSIGHT_COLS}, last_accessed_at
from {{s}}.insights
where deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(rows_sql)
            rows = cur.fetchall()
        if not rows:
            return [], 0

        edge_counts: dict[str, int] = {}
        edge_sql = self._q("""
select id, sum(cnt) from (
    select source_id as id, count(*) as cnt
    from {s}.edges
    group by source_id
    union all
    select target_id as id, count(*) as cnt
    from {s}.edges
    group by target_id
) t
group by id
""")
        with self._conn.cursor() as cur:
            cur.execute(edge_sql)
            for rid, cnt in cur.fetchall():
                edge_counts[rid] = int(cnt)

        now = datetime.now(timezone.utc)
        candidates: list[dict[str, Any]] = []
        updates: list[tuple[float, str]] = []
        for r in rows:
            ins = _row_to_insight(r[:11])
            last_access = _datetime_or_none(r[11]) or (
                ins.created_at or now)
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
            update_many_sql = self._q("""
update {s}.insights
set effective_importance = %s
where id = %s
""")
            with self._conn.cursor() as cur:
                cur.executemany(update_many_sql, updates)
        candidates.sort(key=lambda c: c['effective_importance'])
        if limit > 0:
            candidates = candidates[:limit]
        return candidates, len(rows)

    def count_active(self) -> int:
        sql = self._q("""
select count(*) from {s}.insights where deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def count_total(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(self._q('select count(*) from {s}.insights'))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def has_active_with_source(self, source: str) -> bool:
        sql = self._q("""
select 1 from {s}.insights
where source = %s and deleted_at is null
limit 1
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (source,))
            return cur.fetchone() is not None

    def iter_for_reembed(
            self, cursor: Id, batch: int) -> list[ReembedRow]:
        sql = self._q("""
select id, content, embedding_model,
       case when embedding is null then null else %s end
from {s}.insights
where deleted_at is null and id > %s
order by id
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (EMBEDDING_DIM * 4, cursor, batch))
            return [
                ReembedRow(
                    id=r[0], content=r[1], embedding_model=r[2],
                    blob_length=r[3])
                for r in cur.fetchall()
                ]

    def count_orphans(self) -> tuple[int, int]:
        total_sql = self._q("""
select count(*) from {s}.insights where deleted_at is null
""")
        orphan_sql = self._q("""
select count(*) from {s}.insights i
where i.deleted_at is null
  and not exists (
      select 1 from {s}.edges e
      where e.source_id = i.id or e.target_id = i.id
  )
""")
        with self._conn.cursor() as cur:
            cur.execute(total_sql)
            total = int(cur.fetchone()[0])
            cur.execute(orphan_sql)
            orphans = int(cur.fetchone()[0])
        return orphans, total

    def provenance_distribution(self) -> list[ProvenanceCount]:
        sql = self._q("""
select prompt_version, model_id, count(*)
from {s}.insights
where deleted_at is null
group by prompt_version, model_id
order by count(*) desc
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [
                ProvenanceCount(
                    prompt_version=r[0], model_id=r[1], count=int(r[2]))
                for r in cur.fetchall()
                ]

    def auto_prune(
            self, *, max_insights: int,
            exclude_ids: list[Id] | None = None) -> int:
        from memman.store.node import PRUNE_BATCH_SIZE
        with self.write_lock('prune'):
            excludes = list(exclude_ids or [])
            total = self.count_active()
            if total <= max_insights:
                return 0
            excess = min(total - max_insights, PRUNE_BATCH_SIZE)

            select_sql = self._q("""
select id from {s}.insights
where deleted_at is null
  and importance < 4
  and access_count < 3
  and not (id = any(%s))
order by effective_importance asc
limit %s
""")
            with self._conn.cursor() as cur:
                cur.execute(select_sql, (excludes, PRUNE_BATCH_SIZE))
                cand_rows = cur.fetchall()
            for (cid,) in cand_rows:
                try:
                    self.refresh_effective_importance(cid)
                except ValueError:
                    pass

            update_sql = self._q("""
update {s}.insights
set deleted_at = now(), updated_at = now()
where id = %s and deleted_at is null
""")
            delete_edges_sql = self._q("""
delete from {s}.edges
where source_id = %s or target_id = %s
""")
            with self._conn.cursor() as cur:
                cur.execute(select_sql, (excludes, excess))
                target_rows = cur.fetchall()
                pruned = 0
                for (cid,) in target_rows:
                    cur.execute(update_sql, (cid,))
                    if cur.rowcount > 0:
                        cur.execute(delete_edges_sql, (cid, cid))
                        pruned += 1
            return pruned

    def boost_retention(self, id: Id) -> None:
        sql = self._q("""
update {s}.insights
set access_count = access_count + 3,
    last_accessed_at = now(),
    updated_at = now()
where id = %s and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            if cur.rowcount == 0:
                raise ValueError(
                    f'insight {id} not found or already deleted')

    def get_recent_in_window(
            self, *, exclude_id: Id, window_hours: float,
            limit: int) -> list[Insight]:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id <> %s
  and deleted_at is null
  and created_at >= now() - (%s * interval '1 hour')
order by created_at desc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (exclude_id, window_hours, limit))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def get_latest_by_source(
            self, *, source: str, exclude_id: Id) -> Insight | None:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where source = %s and id <> %s and deleted_at is null
order by created_at desc
limit 1
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (source, exclude_id))
            row = cur.fetchone()
            return _row_to_insight(row) if row else None

    def get_recent_active(
            self, *, exclude_id: Id, limit: int) -> list[Insight]:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id <> %s and deleted_at is null
order by created_at desc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (exclude_id, limit))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def get_all_active(self) -> list[Insight]:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where deleted_at is null
order by created_at desc
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [_row_to_insight(r) for r in cur.fetchall()]

    def stats(self) -> NodeStats:
        active_sql = self._q("""
select count(*) from {s}.insights where deleted_at is null
""")
        deleted_sql = self._q("""
select count(*) from {s}.insights where deleted_at is not null
""")
        cat_sql = self._q("""
select category, count(*)
from {s}.insights
where deleted_at is null
group by category
""")
        ent_sql = self._q("""
select je, count(distinct i.id) cnt
from {s}.insights i, jsonb_array_elements_text(i.entities) je
where i.deleted_at is null
group by je
order by cnt desc
limit 20
""")
        with self._conn.cursor() as cur:
            cur.execute(active_sql)
            total = int(cur.fetchone()[0])
            cur.execute(deleted_sql)
            deleted = int(cur.fetchone()[0])
            cur.execute(cat_sql)
            by_category = {r[0]: int(r[1]) for r in cur.fetchall()}
            cur.execute(self._q('select count(*) from {s}.edges'))
            edges = int(cur.fetchone()[0])
            cur.execute(self._q('select count(*) from {s}.oplog'))
            oplog = int(cur.fetchone()[0])
            top_entities: list[dict[str, Any]] = []
            try:
                cur.execute(ent_sql)
                for entity, cnt in cur.fetchall():
                    top_entities.append(
                        {'entity': entity, 'count': int(cnt)})
            except Exception as exc:
                logger.warning(f'top_entities query failed: {exc}')
        return NodeStats(
            total_insights=total, deleted_insights=deleted,
            edge_count=edges, oplog_count=oplog,
            by_category=by_category, top_entities=top_entities)

    def update_embedding(
            self, id: Id, vec: list[float], model: str) -> None:
        sql = self._q("""
update {s}.insights
set embedding = %s::vector,
    embedding_model = %s,
    updated_at = now()
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (vec, model, id))

    def get_embedding(self, id: Id) -> bytes | None:
        sql = self._q("""
select embedding from {s}.insights
where id = %s and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        from memman.embed.vector import serialize_vector
        return serialize_vector(list(row[0]))

    def get_all_embeddings(self) -> list[tuple[Id, str, bytes]]:
        from memman.embed.vector import serialize_vector
        sql = self._q("""
select id, content, embedding
from {s}.insights
where deleted_at is null and embedding is not null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            results: list[tuple[Id, str, bytes]] = []
            for rid, content, vec in cur.fetchall():
                if vec is None:
                    continue
                results.append((rid, content, serialize_vector(list(vec))))
        return results

    def iter_embeddings_as_vecs(
            self) -> Iterator[tuple[Id, list[float]]]:
        sql = self._q("""
select id, embedding
from {s}.insights
where deleted_at is null and embedding is not null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            for rid, vec in cur.fetchall():
                if vec is None:
                    continue
                yield rid, list(vec)

    def embedding_stats(self) -> tuple[int, int]:
        sql = self._q("""
select count(*),
       count(*) filter (where embedding is not null)
from {s}.insights
where deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def enrichment_coverage(self) -> EnrichmentCoverage:
        sql = self._q("""
select count(*),
       count(*) filter (where embedding is null),
       count(*) filter (
           where keywords is null
              or keywords::text = '[]'
              or jsonb_typeof(keywords) is null
       ),
       count(*) filter (
           where summary is null or summary = ''
       ),
       count(*) filter (
           where semantic_facts is null
              or semantic_facts::text = '[]'
              or jsonb_typeof(semantic_facts) is null
       )
from {s}.insights
where deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        if row is None:
            return EnrichmentCoverage()
        return EnrichmentCoverage(
            total_active=int(row[0] or 0),
            missing_embedding=int(row[1] or 0),
            missing_keywords=int(row[2] or 0),
            missing_summary=int(row[3] or 0),
            missing_semantic_facts=int(row[4] or 0))

    def embedding_size_distribution(self) -> dict[int, int]:
        sql = self._q("""
select vector_dims(embedding), count(*)
from {s}.insights
where deleted_at is null and embedding is not null
group by vector_dims(embedding)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return {
                int(size): int(count) for size, count in cur.fetchall()}

    def get_without_embedding(
            self, *, limit: int = 100) -> list[Insight]:
        if limit <= 0:
            limit = 100
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where deleted_at is null and embedding is null
order by importance desc, created_at desc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def stamp_linked(self, id: Id) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._q(
                'update {s}.insights set linked_at = now() where id = %s'),
                (id,))

    def stamp_enriched(self, id: Id) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._q(
                'update {s}.insights set enriched_at = now() where id = %s'),
                (id,))

    def get_pending_link_ids(self, *, limit: int) -> list[Id]:
        sql = self._q("""
select id from {s}.insights
where linked_at is null and deleted_at is null
order by created_at asc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [r[0] for r in cur.fetchall()]

    def get_active_ids(self) -> list[Id]:
        sql = self._q("""
select id from {s}.insights
where deleted_at is null
order by created_at asc
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

    def count_pending_links(self) -> int:
        sql = self._q("""
select count(*) from {s}.insights
where linked_at is null and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        if not ids:
            return
        sql = self._q("""
update {s}.insights
set enriched_at = null, linked_at = null
where id = any(%s)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (ids,))

    def clear_linked_at(self) -> None:
        sql = self._q("""
update {s}.insights
set linked_at = null
where deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)

    def review_content_quality(
            self, *, limit: int = 50) -> list[dict[str, Any]]:
        from memman.search.quality import check_content_quality
        flagged: list[dict[str, Any]] = []
        for ins in self.get_all_active():
            warnings = check_content_quality(ins.content)
            if warnings:
                flagged.append(
                    {'insight': ins, 'quality_warnings': warnings})
        flagged.sort(
            key=lambda x: len(x['quality_warnings']), reverse=True)
        return flagged[:limit]

    def bulk_update_embedding(
            self, rows: list[tuple[Id, list[float], str]]) -> None:
        """Update embeddings in chunks of <= 1000 rows.

        Under autocommit=True each `executemany` is its own implicit
        transaction, keeping WAL bloat bounded and preventing a single
        long-running statement from holding row-level locks for
        unrelated readers.
        """
        if not rows:
            return
        sql = self._q("""
update {s}.insights
set embedding = %s::vector,
    embedding_model = %s,
    updated_at = now()
where id = %s
""")
        chunk = 1000
        for start in range(0, len(rows), chunk):
            batch = rows[start:start + chunk]
            with self._conn.cursor() as cur:
                cur.executemany(
                    sql,
                    [(vec, model, eid) for eid, vec, model in batch])


class PostgresEdgeStore(EdgeStore):
    """EdgeStore implementation against a per-store Postgres schema."""

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def _q(self, sql: str) -> str:
        return sql.format(s=self._schema)

    def upsert(self, edge: Edge) -> None:
        import json as _json
        sql = self._q("""
insert into {s}.edges
    (source_id, target_id, edge_type, weight, metadata)
values (%s, %s, %s, %s, %s::jsonb)
on conflict (source_id, target_id, edge_type) do update set
    metadata = case
        when {s}.edges.metadata->>'created_by' in ('claude', 'manual')
            then {s}.edges.metadata
        when excluded.weight >= {s}.edges.weight
            then excluded.metadata
        else {s}.edges.metadata
    end,
    weight = greatest({s}.edges.weight, excluded.weight)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                edge.source_id, edge.target_id, edge.edge_type,
                edge.weight, _json.dumps(edge.metadata or {})))

    def by_node(self, node_id: Id) -> list[Edge]:
        sql = self._q("""
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
where source_id = %s
union all
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
where target_id = %s and source_id <> %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (node_id, node_id, node_id))
            return [_row_to_edge(r) for r in cur.fetchall()]

    def by_node_and_type(
            self, node_id: Id, edge_type: str) -> list[Edge]:
        sql = self._q("""
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
where source_id = %s and edge_type = %s
union all
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
where target_id = %s and edge_type = %s and source_id <> %s
""")
        with self._conn.cursor() as cur:
            cur.execute(
                sql, (node_id, edge_type, node_id, edge_type, node_id))
            return [_row_to_edge(r) for r in cur.fetchall()]

    def by_source_and_type(
            self, source_id: Id, edge_type: str) -> list[Edge]:
        sql = self._q("""
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
where source_id = %s and edge_type = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (source_id, edge_type))
            return [_row_to_edge(r) for r in cur.fetchall()]

    def find_with_entity(
            self, entity: str, *, exclude_id: Id,
            limit: int) -> list[Id]:
        ent = entity.strip().lower()
        sql = self._q("""
select distinct i.id
from {s}.insights i, jsonb_array_elements_text(i.entities) je
where i.deleted_at is null
  and i.id <> %s
  and lower(trim(je)) = %s
order by i.id
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (exclude_id, ent, limit))
            return [r[0] for r in cur.fetchall()]

    def count_with_entity(
            self, entity: str, *, exclude_id: Id) -> int:
        ent = entity.strip().lower()
        sql = self._q("""
select count(distinct i.id)
from {s}.insights i, jsonb_array_elements_text(i.entities) je
where i.deleted_at is null
  and i.id <> %s
  and lower(trim(je)) = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (exclude_id, ent))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def all(self) -> list[Edge]:
        sql = self._q("""
select source_id, target_id, edge_type, weight, metadata, created_at
from {s}.edges
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [_row_to_edge(r) for r in cur.fetchall()]

    def delete_by_node(self, node_id: Id) -> None:
        sql = self._q("""
delete from {s}.edges
where source_id = %s or target_id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (node_id, node_id))

    def delete_auto_for_node(
            self, node_id: Id, edge_type: str) -> None:
        filt = _PER_NODE_CREATED_BY_FILTER[edge_type]
        sql = self._q(f"""
delete from {{s}}.edges
where (source_id = %s or target_id = %s)
  and edge_type = %s
  and {filt}
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (node_id, node_id, edge_type))

    def delete_auto_by_type(self, edge_type: str) -> None:
        filt = _REINDEX_CREATED_BY_FILTER[edge_type]
        sql = self._q(f"""
delete from {{s}}.edges
where edge_type = %s and {filt}
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (edge_type,))

    def count_auto_by_type(self, edge_type: str) -> int:
        filt = _REINDEX_CREATED_BY_FILTER[edge_type]
        sql = self._q(f"""
select count(*) from {{s}}.edges
where edge_type = %s and {filt}
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (edge_type,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def delete_low_weight_temporal_proximity(
            self, *, min_weight: float) -> None:
        sql = self._q("""
delete from {s}.edges
where edge_type = 'temporal'
  and metadata->>'sub_type' = 'proximity'
  and weight < %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (min_weight,))

    def count_low_weight_temporal_proximity(
            self, *, min_weight: float) -> int:
        sql = self._q("""
select count(*) from {s}.edges
where edge_type = 'temporal'
  and metadata->>'sub_type' = 'proximity'
  and weight < %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (min_weight,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_weight(
            self, source_id: Id, target_id: Id,
            edge_type: str) -> float | None:
        sql = self._q("""
select weight from {s}.edges
where source_id = %s and target_id = %s and edge_type = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (source_id, target_id, edge_type))
            row = cur.fetchone()
            return float(row[0]) if row else None

    def count_dangling_by_type(self) -> dict[str, int]:
        sql = self._q("""
select e.edge_type, count(*)
from {s}.edges e
where not exists (
    select 1 from {s}.insights i
    where i.id = e.source_id and i.deleted_at is null
)
   or not exists (
    select 1 from {s}.insights i
    where i.id = e.target_id and i.deleted_at is null
)
group by e.edge_type
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return {r[0]: int(r[1]) for r in cur.fetchall()}

    def degree_distribution(self) -> dict[Id, int]:
        ids_sql = self._q("""
select id from {s}.insights
where deleted_at is null
""")
        degree_sql = self._q("""
select id, sum(cnt) from (
    select source_id as id, count(*) as cnt
    from {s}.edges
    group by source_id
    union all
    select target_id as id, count(*) as cnt
    from {s}.edges
    group by target_id
) t
group by id
""")
        with self._conn.cursor() as cur:
            cur.execute(ids_sql)
            ids = [r[0] for r in cur.fetchall()]
            if not ids:
                return {}
            cur.execute(degree_sql)
            by_id = {r[0]: int(r[1]) for r in cur.fetchall()}
        return {iid: by_id.get(iid, 0) for iid in ids}

    def get_neighborhood(
            self, seed_id: Id, *, depth: int,
            edge_filter: str = '') -> list[tuple[Id, int, str]]:
        """Bounded BFS via recursive CTE.

        Postgres-native equivalent of the Python deque BFS in
        SqliteEdgeStore. The recursive CTE emits only the bounded
        subgraph (depth <= `depth`); active-node filtering is applied
        in the outer select so deleted nodes do not seed traversal.
        """
        if depth <= 0:
            return []
        edge_filter_join = (
            ' and e.edge_type = %s' if edge_filter else '')
        sql = self._q(f"""
with recursive walk(node_id, hop, via_edge) as (
    select %s::text, 0::int, null::text
    union
    select
        case when e.source_id = w.node_id
             then e.target_id else e.source_id end,
        w.hop + 1,
        e.edge_type
    from walk w
    join {{s}}.edges e
        on (e.source_id = w.node_id or e.target_id = w.node_id)
        {edge_filter_join}
    where w.hop < %s
)
select distinct on (w.node_id) w.node_id, w.hop, w.via_edge
from walk w
join {{s}}.insights i on i.id = w.node_id
where w.hop > 0
  and w.node_id <> %s
  and i.deleted_at is null
order by w.node_id, w.hop asc
""")
        params: list[Any] = [seed_id]
        if edge_filter:
            params.append(edge_filter)
        params.extend([depth, seed_id])
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        triples = [(r[0], int(r[1]), r[2] or '') for r in rows]
        triples.sort(key=lambda t: (t[1], t[0]))
        return triples


class PostgresMetaStore(MetaStore):
    """MetaStore implementation against a per-store Postgres schema."""

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def get(self, key: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f'select value from {self._schema}.meta where key = %s',
                (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        sql = f"""
insert into {self._schema}.meta (key, value)
values (%s, %s)
on conflict (key) do update set value = excluded.value
"""
        with self._conn.cursor() as cur:
            cur.execute(sql, (key, value))


class PostgresOplog(Oplog):
    """Oplog implementation: insert-only writes; trim in maintenance."""

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def log(
            self, *, operation: str, insight_id: Id,
            detail: str) -> None:
        sql = f"""
insert into {self._schema}.oplog (operation, insight_id, detail)
values (%s, %s, %s)
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (operation, insight_id, detail))
        except Exception as exc:
            logger.warning(f'oplog insert failed: {exc}')

    def maintenance_step(self) -> None:
        sql = f"""
delete from {self._schema}.oplog
where id <= (select max(id) from {self._schema}.oplog) - %s
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (_MAX_OPLOG_ENTRIES,))
        except Exception as exc:
            logger.warning(f'oplog cap trim failed: {exc}')

    def trim_by_age(self, *, retention_days: int = 180) -> int:
        sql = f"""
delete from {self._schema}.oplog
where created_at < now() - (%s * interval '1 day')
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (retention_days,))
                return int(cur.rowcount or 0)
        except Exception as exc:
            logger.warning(f'oplog age trim failed: {exc}')
            return 0

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        if limit <= 0:
            limit = 20
        if since:
            since_dt = parse_timestamp(since)
            sql = f"""
select id, operation, insight_id, detail, created_at
from {self._schema}.oplog
where created_at >= %s
order by id desc
limit %s
"""
            with self._conn.cursor() as cur:
                cur.execute(sql, (since_dt, limit))
                rows = cur.fetchall()
        else:
            sql = f"""
select id, operation, insight_id, detail, created_at
from {self._schema}.oplog
order by id desc
limit %s
"""
            with self._conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        return [
            OpLogEntry(
                id=int(r[0]), operation=r[1],
                insight_id=r[2] or '', detail=r[3] or '',
                created_at=_datetime_or_none(r[4])
                or datetime.now(timezone.utc))
            for r in rows
            ]

    def stats(self, *, since: str = '') -> OpLogStats:
        op_counts: dict[str, int] = {}
        with self._conn.cursor() as cur:
            if since:
                since_dt = parse_timestamp(since)
                sql = f"""
select operation, count(*)
from {self._schema}.oplog
where created_at >= %s
group by operation
order by count(*) desc
"""
                cur.execute(sql, (since_dt,))
            else:
                sql = f"""
select operation, count(*)
from {self._schema}.oplog
group by operation
order by count(*) desc
"""
                cur.execute(sql)
            for op, cnt in cur.fetchall():
                op_counts[op] = int(cnt)
            never_sql = f"""
select count(*) from {self._schema}.insights
where deleted_at is null and access_count = 0
"""
            cur.execute(never_sql)
            never = int(cur.fetchone()[0])
            total_sql = f"""
select count(*) from {self._schema}.insights
where deleted_at is null
"""
            cur.execute(total_sql)
            total = int(cur.fetchone()[0])
        return OpLogStats(
            operation_counts=op_counts, never_accessed=never,
            total_active=total)


class PostgresRecallSession(RecallSession):
    """Read-side session bound to a dedicated autocommit connection.

    Owns its own connection (separate from the parent Backend's
    connection) so the session's `search_path` does not leak into
    write traffic, and so the connection can be borrowed from a pool
    without collision.

    On `__exit__` the session resets `search_path` to the default
    (`"$user", public`) before the connection is closed -- so no
    session state leaks when the connection is returned to a pool.

    `snapshot` mirrors `SqliteRecallSession.snapshot` as a sentinel:
    Postgres has no in-memory snapshot (HNSW + pgvector serve that
    role), so the pipeline's `if session.snapshot is not None`
    branching falls through to the Backend-verb path.
    """

    snapshot: None = None

    def __init__(
            self, dsn: str, schema: str,
            *, owns_conn: bool = True) -> None:
        self._dsn = dsn
        self._schema = schema
        self._owns_conn = owns_conn
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> Self:
        self._conn = _open_connection(self._dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(
                f'set search_path = {self._schema}, public')
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute('set search_path = "$user", public')
            except Exception as e:
                logger.warning(f'search_path reset failed: {e}')
            if self._owns_conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = None

    def close(self) -> None:
        self.__exit__(None, None, None)

    def vector_anchors(
            self, query_vec: list[float], *, k: int = 10,
            min_sim: float = 0.0) -> list[tuple[Id, float]]:
        """Return top-k (id, similarity) matches via HNSW.

        Similarity is `1 - (embedding <=> :q)` (cosine in [-1, 1],
        higher is better).
        """
        assert self._conn is not None
        sql = f"""
select id, 1 - (embedding <=> %s::vector) as sim
from {self._schema}.insights
where deleted_at is null and embedding is not null
order by embedding <=> %s::vector
limit %s
"""
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_vec, query_vec, k))
            return [
                (r[0], float(r[1])) for r in cur.fetchall()
                if r[1] is not None and float(r[1]) >= min_sim
                ]


class PostgresBackend(Backend):
    """Per-store Postgres backend: schema-bound connection + sub-stores.

    Single primary connection per backend. `transaction()` uses
    psycopg's nested transaction (BEGIN / SAVEPOINT). `recall_session`
    and `readonly_context` open dedicated autocommit connections so
    long reads don't share a connection with active writes.
    """

    nodes: PostgresNodeStore
    edges: PostgresEdgeStore
    meta: PostgresMetaStore
    oplog: PostgresOplog

    def __init__(
            self, dsn: str, store: str,
            *, conn: psycopg.Connection | None = None,
            owns_conn: bool = True) -> None:
        self._dsn = dsn
        self._store = store
        self._schema = _store_schema(store)
        self._owns_conn = owns_conn
        self._conn = conn if conn is not None else _open_connection(
            dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(f'set search_path = {self._schema}, public')
        self.nodes = PostgresNodeStore(self._conn, self._schema)
        self.edges = PostgresEdgeStore(self._conn, self._schema)
        self.meta = PostgresMetaStore(self._conn, self._schema)
        self.oplog = PostgresOplog(self._conn, self._schema)

    @property
    def path(self) -> str:
        """DSN+schema identifier (Postgres has no filesystem path)."""
        from memman.trace import redact_dsn
        return f'{redact_dsn(self._dsn)}#{self._schema}'

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block in a write transaction.

        Nested calls reuse the outer transaction via SAVEPOINT (psycopg
        emits one when `conn.transaction()` is entered while already in
        a transaction).
        """
        with self._conn.transaction():
            yield

    @contextmanager
    def write_lock(self, name: str) -> Iterator[None]:
        """Acquire a per-store transaction-scoped advisory lock.

        Postgres: `pg_advisory_xact_lock`. Must be called inside an
        active transaction; the lock auto-releases on transaction
        commit/rollback. Reentrant: the same session may acquire the
        same key multiple times safely (used by the
        `apply_all -> auto_prune` nested pattern).
        """
        key = _advisory_lock_key(self._schema, name)
        with self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute('select pg_advisory_xact_lock(%s)', (key,))
            yield

    @contextmanager
    def readonly_context(self) -> Iterator[PostgresBackend]:
        """Yield a Backend bound to a separate autocommit connection.

        Postgres autocommit lets reader threads see commits from the
        main backend connection as they land. The per-call connection
        is closed deterministically on context exit.
        """
        ro_conn = _open_connection(self._dsn, autocommit=True)
        ro = PostgresBackend(
            self._dsn, self._store, conn=ro_conn, owns_conn=False)
        try:
            yield ro
        finally:
            try:
                ro_conn.close()
            except Exception:
                pass

    @contextmanager
    def recall_session(self) -> Iterator[PostgresRecallSession]:
        session = PostgresRecallSession(self._dsn, self._schema)
        with session:
            yield session

    @contextmanager
    def reembed_lock(self, name: str) -> Iterator[bool]:
        """Acquire a per-store session-scoped advisory sweep lock.

        Mirrors `drain_lock`: dedicated `psycopg.connect()` outside
        any pool, autocommit, with `keepalives_idle=30`. Uses
        `pg_try_advisory_lock` (non-blocking) so a second sweep
        agent fails fast with `False` instead of waiting hours.
        Released on connection close (intended crash-recovery
        mechanism). Wrong primitive for `auto_prune` /
        `reindex_auto_edges` -- those want `write_lock`'s
        transaction-scoped variant.
        """
        key = _advisory_lock_key(self._schema, f'reembed:{name}')
        conn = _open_connection(
            self._dsn, autocommit=True, keepalives=True)
        acquired = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'select pg_try_advisory_lock(%s)', (key,))
                row = cur.fetchone()
                acquired = bool(row[0]) if row else False
            yield acquired
        finally:
            try:
                if acquired:
                    with conn.cursor() as cur:
                        cur.execute(
                            'select pg_advisory_unlock(%s)', (key,))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    @contextmanager
    def drain_lock(
            self, store: str | None = None) -> Iterator[bool]:
        """Acquire a per-store advisory drain lock on a dedicated conn.

        Opens a NEW connection outside any pool (psycopg.connect()
        directly) with `keepalives_idle=30` so a hung worker is
        detected by the kernel rather than holding the lock
        indefinitely. The lock auto-releases when the connection
        closes -- the intended crash-recovery mechanism.

        Yields True when the lock was acquired, False otherwise.
        """
        target = store or self._store
        key = abs(hash(f'memman_drain:{target}')) & 0x7FFFFFFFFFFFFFFF
        conn = _open_connection(
            self._dsn, autocommit=True, keepalives=True)
        acquired = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'select pg_try_advisory_lock(%s)', (key,))
                row = cur.fetchone()
                acquired = bool(row[0]) if row else False
            yield acquired
        finally:
            try:
                if acquired:
                    with conn.cursor() as cur:
                        cur.execute(
                            'select pg_advisory_unlock(%s)', (key,))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def storage_summary(self) -> dict[str, Any]:
        sizes: dict[str, Any] = {}
        try:
            with self._conn.cursor() as cur:
                for table in ('insights', 'edges', 'oplog', 'meta'):
                    cur.execute(
                        'select pg_relation_size(%s::regclass)',
                        (f'{self._schema}.{table}',))
                    row = cur.fetchone()
                    sizes[f'{table}_bytes'] = (
                        int(row[0]) if row else 0)
        except Exception as exc:
            logger.warning(f'pg_relation_size failed: {exc}')
        sizes['schema'] = self._schema
        return sizes

    def maintenance_step(self) -> None:
        """Run per-store maintenance: trim oplog cap.

        Autovacuum handles vacuuming on Postgres; no pragma needed.
        """
        self.oplog.maintenance_step()

    def reindex_hnsw(self) -> None:
        """Idempotently ensure the HNSW index is current.

        Drops any invalid remnant first (per `pg_index.indisvalid`),
        then issues `create index concurrently if not exists`. Cannot
        run inside a transaction; opens a dedicated autocommit
        connection.
        """
        _ensure_hnsw_index(self._dsn, self._schema)

    def integrity_check(self) -> dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(
                f'select 1 from {self._schema}.insights limit 1')
            cur.fetchone()
        return {'ok': True, 'detail': 'schema reachable'}

    def introspect_columns(self, table: str) -> set[str]:
        _check_identifier(table)
        sql = """
select column_name from information_schema.columns
where table_schema = %s and table_name = %s
"""
        with self._conn.cursor() as cur:
            cur.execute(sql, (self._schema, table))
            return {row[0] for row in cur.fetchall()}

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


def _resolve_active_dim() -> int:
    """Return the active embedding client's dim, or `EMBEDDING_DIM` fallback.

    Resolved lazily because the active client requires env-resolved
    config that may not be set at module import time. Falls back to
    the historical 512 default when the active client is not yet
    available (the open will then proceed with vector(512), and
    later writes will land on a 512-dim column).
    """
    try:
        from memman.embed.fingerprint import active_fingerprint
        active = active_fingerprint()
        if active.dim > 0:
            return int(active.dim)
    except Exception as exc:
        logger.warning(
            f'active fingerprint resolution failed; '
            f'using {EMBEDDING_DIM}-dim default: {exc}')
    return EMBEDDING_DIM


def _ensure_baseline_schema(
        dsn: str, store: str, *, dim: int = EMBEDDING_DIM) -> None:
    """Create the schema and apply baseline DDL idempotently.

    `dim` is the embedding dimension to bake into `vector(N)` for
    new schemas. Resolved from `active_fingerprint().dim` by
    `PostgresCluster.open()` so a non-Voyage operator (e.g. openai
    1536) gets a correctly-sized column on first deploy. For
    existing schemas the call is idempotent: `create table if not
    exists` does not alter the existing column width, and the
    open-time guard at `_assert_vector_dim_matches` refuses the
    open if the stored width differs from `dim`.
    """
    schema = _store_schema(store)
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute('create extension if not exists vector')
        cur.execute(f'create schema if not exists {schema}')
        cur.execute(_PG_QUEUE_SCHEMA)
        cur.execute(
            _PG_BASELINE_SCHEMA.format(
                schema=schema, dim=dim))


def _assert_vector_dim_matches(
        dsn: str, store: str, expected_dim: int) -> None:
    """Refuse to open if the stored `vector(N)` column width differs.

    pgvector stores `N` directly in `pg_attribute.atttypmod` (no
    VARHDRSZ offset, unlike standard varlena types). Querying via
    the conventional `information_schema.columns` does not work
    because pgvector extension types do not populate
    `character_maximum_length`.

    Raises `BackendError` with an upgrade hint when the operator's
    active embedding fingerprint dim differs from the stored column
    width. This is the parallel of the schema-version skew refusal
    for embedding-dim skew.
    """
    schema = _store_schema(store)
    sql = """
select atttypmod from pg_attribute
where attrelid = (%s || '.insights')::regclass
  and attname = 'embedding'
  and not attisdropped
"""
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql, (schema,))
        row = cur.fetchone()
        if row is None or row[0] is None or int(row[0]) <= 0:
            return
        stored_dim = int(row[0])
    if stored_dim != expected_dim:
        raise BackendError(
            f'store {store!r} has vector({stored_dim}) but the active'
            f' embedding client produces dim={expected_dim}.'
            f" Run 'memman embed reembed' against a fresh store, or"
            f' switch back to a {stored_dim}-dim provider.')


def _apply_pending_migrations(dsn: str, store: str) -> None:
    """Apply forward-only `_PG_MIGRATIONS` entries to a store schema.

    Reads `meta.pg_schema_version` (defaulting to 0 when absent),
    runs every migration with `stored < target_ver <= code_version`
    in one transaction, and writes the new version atomically. If
    `stored > code_version` (older binary, newer store), refuses
    with `BackendError`. Each SQL string interpolates `{schema}`.
    """
    schema = _store_schema(store)
    select_sql = f"""
select value from {schema}.meta
where key = 'pg_schema_version'
"""
    upsert_sql = f"""
insert into {schema}.meta (key, value)
values ('pg_schema_version', %s)
on conflict (key) do update set value = excluded.value
"""
    with _connection(dsn, autocommit=False) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(select_sql)
                row = cur.fetchone()
                stored = int(row[0]) if row else 0
                if stored > _PG_SCHEMA_VERSION:
                    raise BackendError(
                        f'store {store!r} is at schema version {stored};'
                        f' this binary supports {_PG_SCHEMA_VERSION}.'
                        ' Upgrade memman before opening this store.')
                if stored == _PG_SCHEMA_VERSION:
                    return
                for target_ver, sql in _PG_MIGRATIONS:
                    if target_ver <= stored:
                        continue
                    if target_ver > _PG_SCHEMA_VERSION:
                        break
                    cur.execute(sql.format(schema=schema))
                cur.execute(upsert_sql, (str(_PG_SCHEMA_VERSION),))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _ensure_hnsw_index(dsn: str, schema: str) -> None:
    """Create or recreate the HNSW index on `insights.embedding`.

    1. Query `pg_index.indisvalid` for any prior HNSW index on this
       column; drop it if invalid (an aborted CONCURRENTLY build
       leaves an invalid remnant).
    2. `create index concurrently if not exists` with
       `vector_cosine_ops where deleted_at is null`.

    Runs on a dedicated autocommit connection because
    `create index concurrently` cannot run inside a transaction.
    `statement_timeout` is set from `MEMMAN_REINDEX_TIMEOUT` (default
    180 seconds) so a stuck build aborts and the next call's
    invalid-remnant cleanup can recover.
    """
    _check_identifier(schema)
    index_name = f'idx_insights_hnsw_{schema}'
    timeout_s = int(os.environ.get('MEMMAN_REINDEX_TIMEOUT', '180'))
    inspect_sql = """
select i.indexrelid::regclass::text, i.indisvalid
from pg_index i
join pg_class c on c.oid = i.indexrelid
where c.relname = %s
"""
    create_sql = f"""
create index concurrently if not exists {index_name}
on {schema}.insights
using hnsw (embedding vector_cosine_ops)
where deleted_at is null
"""
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"set statement_timeout = '{timeout_s}s'")
        cur.execute(inspect_sql, (index_name,))
        row = cur.fetchone()
        if row and not row[1]:
            logger.warning(
                f'dropping invalid HNSW index {row[0]}')
            cur.execute(f'drop index if exists {row[0]} cascade')
        cur.execute(create_sql)


class PostgresQueueBackend:
    """Cluster-global work queue using `queue.queue` schema.

    Uses `for update skip locked` claim semantics so multiple workers
    can claim disjoint rows safely. Queue tables live in a single
    `queue` schema (not per-store) so cross-store ordering is
    preserved.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _conn(self) -> psycopg.Connection:
        return _open_connection(self._dsn, autocommit=True)

    def enqueue(self, *, store: str, op: str, payload: str) -> int:
        sql = """
insert into queue.queue (store, op, payload)
values (%s, %s, %s)
returning id
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (store, op, payload))
            return int(cur.fetchone()[0])

    def claim_batch(self, *, limit: int) -> list[QueueRow]:
        sql = """
update queue.queue q
set status = 'claimed',
    claimed_at = now(),
    attempts = q.attempts + 1
from (
    select id from queue.queue
    where status = 'pending'
    order by id asc
    for update skip locked
    limit %s
) sel
where q.id = sel.id
returning q.id, q.store, q.op, q.payload, q.attempts, q.created_at
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        return [
            QueueRow(
                id=int(r[0]), store=r[1], op=r[2],
                payload=r[3], attempts=int(r[4]),
                created_at=_datetime_or_none(r[5])
                or datetime.now(timezone.utc))
            for r in rows
            ]

    def mark_done(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        sql = """
update queue.queue
set status = 'done', finished_at = now()
where id = any(%s)
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (list(ids),))

    def mark_failed(
            self, ids: Sequence[int], *, error: str) -> None:
        if not ids:
            return
        sql = """
update queue.queue
set status = 'failed', finished_at = now(), error = %s
where id = any(%s)
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (error, list(ids)))

    def purge_store(self, store: str) -> int:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                'delete from queue.queue where store = %s', (store,))
            return int(cur.rowcount or 0)

    def stats(self) -> QueueStats:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute('select count(*) from queue.queue')
            total = int(cur.fetchone()[0])
            cur.execute(
                'select store, count(*) from queue.queue group by store')
            by_store = {r[0]: int(r[1]) for r in cur.fetchall()}
        return QueueStats(total=total, by_store=by_store)

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        sql = """
select id, started_at, ended_at, rows_processed, error,
       last_heartbeat_at
from queue.worker_runs
order by id desc
limit %s
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        return [
            WorkerRun(
                id=int(r[0]),
                started_at=_datetime_or_none(r[1])
                or datetime.now(timezone.utc),
                ended_at=_datetime_or_none(r[2]),
                rows_processed=int(r[3]),
                error=r[4] or '',
                last_heartbeat_at=_datetime_or_none(r[5]))
            for r in rows
            ]

    def start_run(self) -> int:
        sql = """
insert into queue.worker_runs (last_heartbeat_at)
values (now())
returning id
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    def beat_run(self, run_id: int) -> None:
        sql = """
update queue.worker_runs
set last_heartbeat_at = now()
where id = %s
"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            conn.commit()

    def integrity_report(self) -> IntegrityReport:
        return IntegrityReport()


class PostgresCluster(Cluster):
    """Cluster implementation for Postgres: schema-per-store.

    Reads `MEMMAN_PG_DSN` from config to find the connection target.
    `data_dir` is unused for the Postgres backend (every store maps
    to a schema in one database) but kept on the verb signature so
    the Cluster Protocol stays shared.
    """

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            from memman import config
            dsn = config.get(config.PG_DSN)
        if not dsn:
            raise ConfigError(
                'MEMMAN_PG_DSN is not set; cannot open Postgres cluster')
        self._dsn = dsn

    def open(self, *, store: str, data_dir: str) -> PostgresBackend:
        active_dim = _resolve_active_dim()
        _ensure_baseline_schema(self._dsn, store, dim=active_dim)
        _assert_vector_dim_matches(self._dsn, store, active_dim)
        _apply_pending_migrations(self._dsn, store)
        backend = PostgresBackend(self._dsn, store)
        try:
            _ensure_hnsw_index(self._dsn, _store_schema(store))
        except Exception as exc:
            logger.warning(f'HNSW index ensure failed: {exc}')
        return backend

    def open_read_only(
            self, *, store: str, data_dir: str) -> PostgresBackend:
        """Postgres read-only: fresh autocommit connection."""
        _check_identifier(store)
        ro_conn = _open_connection(self._dsn, autocommit=True)
        return PostgresBackend(
            self._dsn, store, conn=ro_conn, owns_conn=True)

    def list_stores(self, *, data_dir: str) -> list[str]:
        sql = """
select nspname from pg_namespace
where nspname like 'store_%'
order by nspname
"""
        try:
            with _connection(self._dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return sorted(
                        r[0][len('store_'):] for r in cur.fetchall())
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f'cannot connect to postgres: {exc}')

    def drop_store(self, *, store: str, data_dir: str) -> None:
        schema = _store_schema(store)
        with _connection(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
                cur.execute(
                    'delete from queue.queue where store = %s',
                    (store,))

    def close(self) -> None:
        """Postgres clusters do not hold a pool."""


def _bfs_python_fallback(
        edges: list[Edge], active_ids: set[Id], seed_id: Id,
        depth: int, edge_filter: str) -> list[tuple[Id, int, str]]:
    """Python BFS fallback used in tests when CTE shape is awkward.

    Mirrors `SqliteEdgeStore.get_neighborhood` so an integration test
    can compare results when the Python and SQL paths diverge during
    development. Not used at runtime.
    """
    adj: dict[Id, list[Edge]] = {}
    for e in edges:
        adj.setdefault(e.source_id, []).append(e)
        if e.source_id != e.target_id:
            adj.setdefault(e.target_id, []).append(e)
    visited = {seed_id}
    queue: deque[tuple[Id, int]] = deque([(seed_id, 0)])
    out: list[tuple[Id, int, str]] = []
    while queue:
        cur_id, hop = queue.popleft()
        if hop >= depth:
            continue
        for edge in adj.get(cur_id, []):
            if edge_filter and edge.edge_type != edge_filter:
                continue
            neighbor_id = (
                edge.target_id if edge.target_id != cur_id
                else edge.source_id)
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            if neighbor_id not in active_ids:
                continue
            out.append((neighbor_id, hop + 1, edge.edge_type))
            queue.append((neighbor_id, hop + 1))
    return out
