"""Postgres + pgvector implementation of the Backend Protocol surface.

Single-file parallel to `store/sqlite.py`. Schema-per-store layout:
each memman store maps to a Postgres schema named `store_<name>`,
holding the per-store tables (insights, oplog, meta, worker_runs).

Vector storage:
- `embedding vector(N)` (pgvector), `N` sized to the active embed
  model's dim; pgvector adapter binds
  `list[float]` directly with no per-call serialization.
- HNSW index built `create index concurrently ... vector_cosine_ops
  where deleted_at is null and replaced_by is null`. Built outside
  any transaction; reindex drops invalid remnants
  (`pg_index.indisvalid`) before retrying.
- Similarity returned as `1 - (embedding <=> :q)` (cosine in
  [-1, 1]; higher better).

Recall issues one round-trip per anchor subset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import starmap
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Self

from memman import config
from memman.embed.fingerprint import Fingerprint, seed_default_fingerprint
from memman.embed.swap import swap_remedy
from memman.exceptions import ConfigError as RuntimeConfigError
from memman.migrate import Artifact, MigrateError, MigrateInsight
from memman.migrate import MigrateOpLog, MigrationPayload, Migrator
from memman.migrate import sanitize_identifier
from memman.search.keyword import insight_tokens
from memman.setup.archive import archive_postgres_schema
from memman.store.backend import Backend, MetaStore, NodeStore, Oplog
from memman.store.backend import RecallSession, _check_identifier
from memman.store.errors import BackendError, ConfigError, StoreMissingError
from memman.store.errors import SwapCutoverRefused
from memman.store.model import EnrichmentCoverage, Id, Insight, NodeStats
from memman.store.model import OpLogEntry, OpLogStats, WorkerRun
from memman.store.model import format_timestamp, parse_timestamp
from memman.store.node import unterminated_chains
from memman.store.oplog import MAX_OPLOG_ENTRIES, OPLOG_RETENTION_DAYS
from memman.trace import redact_dsn

if TYPE_CHECKING:
    import psycopg

logger = logging.getLogger('memman')

# Postgres NAMEDATALEN is 64; an identifier is truncated to 63
# bytes, silently.
PG_NAME_MAX_CHARS = 63


def _store_schema(name: str) -> str:
    """Return the Postgres schema name for a memman store.

    Parameters
    ----------
    name : str
        Store name, unprefixed.

    Returns
    -------
    str
        `store_<name>`.

    Raises
    ------
    ConfigError
        When `name` is not a SQL identifier, or when the prefixed
        schema would exceed `PG_NAME_MAX_CHARS`.
    """
    _check_identifier(name)
    schema = f'store_{name}'
    # Postgres truncates an over-long identifier silently, so two
    # names differing only past the limit would share one schema.
    if len(schema) > PG_NAME_MAX_CHARS:
        raise ConfigError(
            f'store name {name!r} is too long: schema {schema!r} is'
            f' {len(schema)} characters and postgres truncates past'
            f' {PG_NAME_MAX_CHARS}, which would silently merge it'
            f' with another store')
    return schema


def _lock_id(name: str) -> int:
    """Deterministic signed int8 lock id for `pg_advisory_*lock` calls.

    Built on blake2b. `hash()` is randomized per process by
    PYTHONHASHSEED, so it would give two processes different ids for
    one name.
    """
    digest = hashlib.blake2b(name.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big', signed=True)


def _advisory_lock_key(schema: str, name: str) -> int:
    """Per-store, per-name int8 key for `pg_advisory_*lock` calls.
    """
    return _lock_id(f'{schema}:{name}')


PG_BASELINE_SCHEMA = """
create table if not exists {schema}.insights (
    id          text primary key,
    content     text not null,
    summary     text,
    embedding   vector({dim}),
    enrich_attempted_at timestamptz,
    enriched_at timestamptz,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    deleted_at  timestamptz,
    embedding_model text,
    queue_uuid  text,
    kw_tokens   text[] not null,
    replaced_by text,
    author      text,
    summary_model text
);

create table if not exists {schema}.oplog (
    id          bigserial primary key,
    operation   text not null,
    insight_id  text,
    detail      text default '',
    created_at  timestamptz not null default now(),
    before      jsonb,
    after       jsonb,
    legacy_id   bigint,
    constraint oplog_legacy_id_key_{schema} unique (legacy_id)
);

create table if not exists {schema}.meta (
    key   text primary key,
    value text not null
);

create table if not exists {schema}.worker_runs (
    id            bigserial primary key,
    started_at    timestamptz not null default now(),
    ended_at      timestamptz,
    last_heartbeat_at timestamptz
);

create index if not exists idx_insights_created_{schema}
    on {schema}.insights(created_at);
create index if not exists idx_insights_deleted_{schema}
    on {schema}.insights(deleted_at);
create index if not exists idx_insights_queue_uuid_{schema}
    on {schema}.insights(queue_uuid);
create index if not exists idx_insights_pending_enrich_{schema}
    on {schema}.insights(enrich_attempted_at, created_at)
    where enrich_attempted_at is null and deleted_at is null
      and replaced_by is null;
create index if not exists idx_insights_kw_tokens_{schema}
    on {schema}.insights using gin (kw_tokens)
    where deleted_at is null and replaced_by is null;
create index if not exists idx_insights_current_listing_{schema}
    on {schema}.insights(deleted_at, replaced_by, created_at);

create index if not exists idx_oplog_created_{schema}
    on {schema}.oplog(created_at);
"""


def _open_connection(
        dsn: str, *, autocommit: bool = False,
        keepalives: bool = False,
        connect_timeout: int | None = None,
        register_vector: bool = True) -> psycopg.Connection:
    """Open a fresh psycopg connection with pgvector adapters.

    Parameters
    ----------
    dsn : str
        Connection string.
    autocommit : bool, default False
        Open the connection in autocommit mode.
    keepalives : bool, default False
        Add `keepalives_idle=30` for a lock-holding connection so the
        kernel detects a hung worker instead of the lock being held
        indefinitely.
    connect_timeout : int or None, default None
        Seconds to wait for the server; None leaves the driver default.
    register_vector : bool, default True
        Register the pgvector adapter. Pass False for a probe against
        a database where the `vector` extension may be absent (the
        install wizard's pgvector-presence check): registration raises
        `ProgrammingError` without the extension, and skipping it lets
        the caller detect absence with its own SQL.

    Returns
    -------
    psycopg.Connection
        A bare connection the caller closes. Lock-holding paths
        (`reembed_lock`, `swap_lock`) and long-lived backend
        connections own the lifecycle; one-shot helpers use
        `_connection()` for close-on-exit.

    Raises
    ------
    BackendError
        When the server is unreachable or rejects the connection.
    """
    import psycopg
    from pgvector.psycopg import register_vector as _register_vector
    kwargs: dict[str, Any] = {'autocommit': autocommit}
    if keepalives:
        kwargs['keepalives'] = 1
        kwargs['keepalives_idle'] = 30
    if connect_timeout is not None:
        kwargs['connect_timeout'] = connect_timeout
    # An unreachable or rejecting server is an ordinary operator
    # condition. The lock paths call this directly, bypassing
    # `_connection`, so the translation happens here.
    try:
        conn = psycopg.connect(dsn, **kwargs)
        if register_vector:
            _register_vector(conn)
    except psycopg.Error as exc:
        raise BackendError(
            f'postgres connection failed: {exc}') from exc
    return conn


@contextmanager
def _connection(
        dsn: str, *, autocommit: bool = False,
        connect_timeout: int | None = None,
        register_vector: bool = True
        ) -> Iterator[psycopg.Connection]:
    """Context-manager wrapper around `_open_connection`.

    Takes the parameters of `_open_connection`, less `keepalives`, and
    yields its connection, closed on exit. psycopg3's own `with conn:` scopes a
    transaction and leaves the connection open.

    Raises
    ------
    BackendError
        On a connection failure, or on any driver error raised in the
        `with` body.

    Notes
    -----
    - A caller that branches on a driver exception type nests its own
      handler around the statement, which runs before this wrapper
      sees the error.
    """
    import psycopg as _psycopg

    conn = _open_connection(
        dsn, autocommit=autocommit, connect_timeout=connect_timeout,
        register_vector=register_vector)
    try:
        yield conn
    # One translation here covers every query in the scope, since a
    # backend must raise `BackendError`.
    except _psycopg.Error as exc:
        raise BackendError(f'postgres query failed: {exc}') from exc
    finally:
        try:
            conn.close()
        except _psycopg.Error as exc:
            logger.debug(f'pg connection close failed: {exc}')


def _row_to_insight(row: tuple[Any, ...]) -> Insight:
    """Map a select row into an Insight dataclass.
    """
    i = Insight()
    i.id = row[0]
    i.content = row[1]
    i.created_at = row[2]
    i.updated_at = row[3]
    i.deleted_at = row[4]
    if row[5]:
        i.summary = row[5]
    i.enrich_attempted_at = row[6]
    i.enriched_at = row[7]
    if row[8]:
        i.queue_uuid = row[8]
    if row[9]:
        i.replaced_by = row[9]
    if row[10]:
        i.author = row[10]
    return i


# Must stay byte-identical to node.py's _INSIGHT_COLUMNS.
_INSIGHT_COLS = (
    'id, content, created_at, updated_at, deleted_at,'
    ' summary, enrich_attempted_at, enriched_at,'
    ' queue_uuid, replaced_by,'
    ' author')


_RAW_SELECT = (
    'id, content, summary, embedding::real[],'
    ' enrich_attempted_at, enriched_at, created_at, updated_at,'
    ' deleted_at, embedding_model,'
    ' queue_uuid, replaced_by, author, summary_model')

_RAW_INSERT = (
    'id, content, summary, embedding,'
    ' enrich_attempted_at, enriched_at, created_at, updated_at,'
    ' deleted_at, embedding_model,'
    ' queue_uuid, kw_tokens, replaced_by, author, summary_model')


def _raw_values(row: MigrateInsight) -> tuple:
    """Column values of `row` in `_RAW_INSERT` order.

    Parameters
    ----------
    row : MigrateInsight
        Any row. A deleted one gets empty `kw_tokens`, as
        `soft_delete` leaves it.

    Returns
    -------
    tuple
    """
    return (
        row.id, row.content, row.summary,
        [float(x) for x in row.embedding]
        if row.embedding is not None else None,
        row.enrich_attempted_at, row.enriched_at,
        row.created_at, row.updated_at, row.deleted_at,
        row.embedding_model, row.queue_uuid,
        [] if row.deleted_at else sorted(
            insight_tokens(Insight(content=row.content))),
        row.replaced_by, row.author, row.summary_model)


class PostgresNodeStore(NodeStore):
    """NodeStore implementation against a per-store Postgres schema.
    """

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def _q(self, sql: str) -> str:
        """Format SQL with the per-store schema interpolated.
        """
        return sql.format(s=self._schema)

    def insert(self, ins: Insight) -> None:
        """Insert a new insight.

        Parameters
        ----------
        ins : Insight
            The row to insert. Its `created_at` and `updated_at` are
            ignored; both are stamped with the current time.
        """
        # Notes:
        # - The stamp comes from the Python clock through
        #   `format_timestamp`, the whole-second form the SQLite path
        #   uses, and skips the column's `default now()`.
        # - A row then carries the same stamp on both backends, so
        #   recall's time channel and a migrate round-trip agree.
        now = format_timestamp(datetime.now(timezone.utc))
        sql = self._q("""
insert into {s}.insights
    (id, content, created_at, updated_at,
     embedding_model,
     queue_uuid, kw_tokens, author)
values (%s, %s, %s, %s, %s, %s, %s, %s)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                ins.id, ins.content,
                now, now,
                ins.embedding_model,
                ins.queue_uuid,
                sorted(insight_tokens(ins)),
                ins.author))

    def insert_raw(self, row: MigrateInsight) -> bool:
        sql = self._q(
            f'insert into {{s}}.insights ({_RAW_INSERT})'
            ' values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
            ' %s, %s, %s, %s)'
            ' on conflict (id) do nothing')
        with self._conn.cursor() as cur:
            cur.execute(sql, _raw_values(row))
            return cur.rowcount == 1

    def get_raw(self, id: Id) -> MigrateInsight | None:
        sql = self._q(f'select {_RAW_SELECT} from {{s}}.insights where id = %s')
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()
        return MigrateInsight(*row) if row else None

    def get(self, id: Id) -> Insight | None:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where id = %s and deleted_at is null and replaced_by is null
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

    def resolve_id(self, id_or_prefix: str) -> str:
        """Resolve an exact id or unambiguous prefix to a full id.
        """
        exact_sql = self._q('select id from {s}.insights where id = %s')
        with self._conn.cursor() as cur:
            cur.execute(exact_sql, (id_or_prefix,))
            exact = cur.fetchone()
        if exact is not None:
            return exact[0]
        prefix_sql = self._q(
            'select id from {s}.insights'
            ' where substr(id, 1, length(%s)) = %s')
        with self._conn.cursor() as cur:
            cur.execute(prefix_sql, (id_or_prefix, id_or_prefix))
            rows = cur.fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) == 0:
            return id_or_prefix
        raise ValueError(
            f'prefix {id_or_prefix!r} matches {len(rows)} rows')

    def query(
            self, *, keyword: str = '', limit: int = 20) -> list[Insight]:
        conditions = ['deleted_at is null and replaced_by is null']
        args: list[Any] = []
        if keyword:
            for word in keyword.split():
                escaped = word.replace(
                    '\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                conditions.append('content ilike %s')
                args.append(f'%{escaped}%')
        args.append(limit)
        where_clause = ' and '.join(conditions)
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where {where_clause}
order by created_at desc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def soft_delete(self, id: Id) -> bool:
        # Emptying `kw_tokens` sheds a token set the keyword scan can
        # no longer reach, so a soft-deleted row carries no tokens
        # into the table or its index.
        update_sql = self._q("""
update {s}.insights
set deleted_at = now(), updated_at = now(), kw_tokens = '{{}}'
where id = %s and deleted_at is null
""")
        with self._conn.cursor() as cur:
            cur.execute(update_sql, (id,))
            return cur.rowcount != 0

    def soft_delete_current(self, id: Id) -> bool:
        update_sql = self._q("""
update {s}.insights
set deleted_at = now(), updated_at = now(), kw_tokens = '{{}}'
where id = %s and deleted_at is null and replaced_by is null
""")
        with self._conn.cursor() as cur:
            cur.execute(update_sql, (id,))
            return cur.rowcount != 0

    def mark_replaced(self, predecessor_id: Id, successor_id: Id) -> bool:
        # `kw_tokens` stays populated, unlike `soft_delete`: the GIN
        # predicate and `keyword_counts` already exclude replaced
        # rows.
        update_sql = self._q("""
update {s}.insights
set replaced_by = %s, updated_at = now()
where id = %s and deleted_at is null and replaced_by is null
""")
        with self._conn.cursor() as cur:
            cur.execute(update_sql, (successor_id, predecessor_id))
            return cur.rowcount != 0

    def predecessors(self, successor_id: Id) -> list[Insight]:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where replaced_by = %s
order by created_at, id
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (successor_id,))
            return [_row_to_insight(r) for r in cur.fetchall()]

    def replacement_integrity(self) -> dict[str, list[Id]]:
        dangling_sql = self._q("""
select p.id
from {s}.insights p
left join {s}.insights s on s.id = p.replaced_by
where p.replaced_by is not null and s.id is null
order by p.id
""")
        self_sql = self._q("""
select id from {s}.insights where replaced_by = id order by id
""")
        pointers_sql = self._q("""
select id, replaced_by from {s}.insights where replaced_by is not null
""")
        out: dict[str, list[Id]] = {}
        with self._conn.cursor() as cur:
            for key, sql in (('dangling', dangling_sql),
                             ('self_pointer', self_sql)):
                cur.execute(sql)
                out[key] = [r[0] for r in cur.fetchall()]
            cur.execute(pointers_sql)
            out['unterminated'] = unterminated_chains(dict(cur.fetchall()))
        return out

    def update_enrichment(
            self, id: Id, *, summary: str, summary_model: str) -> None:
        sql = self._q("""
update {s}.insights
set summary = %s,
    summary_model = %s,
    updated_at = now()
where id = %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (summary, summary_model, id))

    def count_active(self) -> int:
        sql = self._q("""
select count(*) from {s}.insights where deleted_at is null and replaced_by is null
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

    def has_row_with_queue_uuid(self, queue_uuid: str) -> bool:
        sql = self._q("""
select 1 from {s}.insights
where queue_uuid = %s
limit 1
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (queue_uuid,))
            return cur.fetchone() is not None

    def get_all_active(self) -> list[Insight]:
        sql = self._q(f"""
select {_INSIGHT_COLS}
from {{s}}.insights
where deleted_at is null and replaced_by is null
order by created_at desc
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [_row_to_insight(r) for r in cur.fetchall()]

    def stats(self) -> NodeStats:
        active_sql = self._q("""
select count(*) from {s}.insights where deleted_at is null and replaced_by is null
""")
        replaced_sql = self._q("""
select count(*) from {s}.insights
where deleted_at is null and replaced_by is not null
""")
        deleted_sql = self._q("""
select count(*) from {s}.insights where deleted_at is not null
""")
        with self._conn.cursor() as cur:
            cur.execute(active_sql)
            total = int(cur.fetchone()[0])
            cur.execute(replaced_sql)
            replaced = int(cur.fetchone()[0])
            cur.execute(deleted_sql)
            deleted = int(cur.fetchone()[0])
            cur.execute(self._q('select count(*) from {s}.oplog'))
            oplog = int(cur.fetchone()[0])
        return NodeStats(
            total_insights=total, replaced_insights=replaced,
            deleted_insights=deleted,
            oplog_count=oplog)

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

    def embedding_stats(self) -> tuple[int, int]:
        sql = self._q("""
select count(*),
       count(*) filter (where embedding is not null)
from {s}.insights
where deleted_at is null and replaced_by is null
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
           where (summary is null or summary = '')
             and enriched_at is null
       ),
       count(*) filter (
           where enrich_attempted_at is not null
             and enriched_at is null
       )
from {s}.insights
where deleted_at is null and replaced_by is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        if row is None:
            return EnrichmentCoverage()
        return EnrichmentCoverage(
            total_active=int(row[0] or 0),
            missing_embedding=int(row[1] or 0),
            missing_summary=int(row[2] or 0),
            stranded=int(row[3] or 0))

    def embedding_size_distribution(self) -> dict[int, int]:
        sql = self._q("""
select vector_dims(embedding), count(*)
from {s}.insights
where deleted_at is null and replaced_by is null and embedding is not null
group by vector_dims(embedding)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return {
                int(size): int(count) for size, count in cur.fetchall()}

    def stamp_enrich_attempted(self, id: Id) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._q(
                'update {s}.insights set enrich_attempted_at = now()'
                ' where id = %s'),
                (id,))

    def stamp_enriched(self, id: Id) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._q(
                'update {s}.insights set enriched_at = now()'
                ' where id = %s'),
                (id,))

    def get_pending_enrich_ids(self, *, limit: int) -> list[Id]:
        sql = self._q("""
select id from {s}.insights
where enrich_attempted_at is null and deleted_at is null
  and replaced_by is null
order by created_at asc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [r[0] for r in cur.fetchall()]

    def get_active_ids(self) -> list[Id]:
        sql = self._q("""
select id from {s}.insights
where deleted_at is null and replaced_by is null
order by created_at asc
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

    def get_all_ids(self) -> set[Id]:
        with self._conn.cursor() as cur:
            cur.execute(self._q('select id from {s}.insights'))
            return {r[0] for r in cur.fetchall()}

    def count_pending_enrich(self) -> int:
        sql = self._q("""
select count(*) from {s}.insights
where enrich_attempted_at is null and deleted_at is null
  and replaced_by is null
""")
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_unenriched_attempted_ids(self, *, limit: int) -> list[Id]:
        sql = self._q("""
select id from {s}.insights
where enriched_at is null
  and enrich_attempted_at is not null
  and deleted_at is null and replaced_by is null
order by created_at asc
limit %s
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [r[0] for r in cur.fetchall()]

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        if not ids:
            return
        sql = self._q("""
update {s}.insights
set enriched_at = null, enrich_attempted_at = null
where id = any(%s)
""")
        with self._conn.cursor() as cur:
            cur.execute(sql, (ids,))


class PostgresMetaStore(MetaStore):
    """MetaStore implementation against a per-store Postgres schema.
    """

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

    def delete(self, key: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f'delete from {self._schema}.meta where key = %s',
                (key,))

    def keys(self) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(f'select key from {self._schema}.meta')
            return [r[0] for r in cur.fetchall()]


class PostgresOplog(Oplog):
    """Oplog implementation: insert-only writes; trim in maintenance.
    """

    def __init__(
            self, conn: psycopg.Connection, schema: str) -> None:
        self._conn = conn
        self._schema = schema

    def log(
            self, *, operation: str, insight_id: Id,
            detail: str,
            before: dict[str, Any] | None = None,
            after: dict[str, Any] | None = None) -> None:
        before_s = json.dumps(before) if before is not None else None
        after_s = json.dumps(after) if after is not None else None
        sql = f"""
insert into {self._schema}.oplog
       (operation, insight_id, detail, before, after)
values (%s, %s, %s, %s::jsonb, %s::jsonb)
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    sql,
                    (operation, insight_id, detail,
                     before_s, after_s))
        except Exception as exc:
            logger.warning(f'oplog insert failed: {exc}')

    def maintenance_step(self) -> None:
        sql = f"""
delete from {self._schema}.oplog
where id <= (select max(id) from {self._schema}.oplog) - %s
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (MAX_OPLOG_ENTRIES,))
        except Exception as exc:
            logger.warning(f'oplog cap trim failed: {exc}')

    def trim_by_age(self) -> int:
        sql = f"""
delete from {self._schema}.oplog
where created_at < now() - (%s * interval '1 day')
"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (OPLOG_RETENTION_DAYS,))
                return int(cur.rowcount or 0)
        except Exception as exc:
            logger.warning(f'oplog age trim failed: {exc}')
            return 0

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        if since:
            since_dt = parse_timestamp(since)
            sql = f"""
select id, operation, insight_id, detail, created_at,
       before, after
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
select id, operation, insight_id, detail, created_at,
       before, after
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
                created_at=r[4],
                before=r[5], after=r[6])
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
            total_sql = f"""
select count(*) from {self._schema}.insights
where deleted_at is null and replaced_by is null
"""
            cur.execute(total_sql)
            total = int(cur.fetchone()[0])
        return OpLogStats(
            operation_counts=op_counts, total_active=total)


class PostgresRecallSession(RecallSession):
    """Read-side session bound to a dedicated autocommit connection.

    Owns its own connection (separate from the parent Backend's
    connection) so the session's `search_path` does not leak into
    write traffic. On `__exit__` the session resets `search_path` to
    the default (`"$user", public`) and closes the connection.

    Vector work stays server-side: `vector_anchors` rides the HNSW
    index, and `similarities` scores with `embedding <=>` so the
    pipeline receives N scalars instead of N x dim floats.
    """

    def __init__(self, dsn: str, schema: str) -> None:
        self._dsn = dsn
        self._schema = schema
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
            import psycopg as _psycopg
            try:
                self._conn.close()
            except _psycopg.Error as exc:
                logger.debug(f'pg connection close failed: {exc}')
            self._conn = None

    def vector_anchors(
            self, query_vec: list[float], *,
            k: int = 10) -> list[tuple[Id, float]]:
        """Return top-k (id, similarity) matches via HNSW.

        Similarity is `1 - (embedding <=> :q)` (cosine in (0, 1],
        higher is better).
        """
        assert self._conn is not None
        sql = f"""
select id, 1 - (embedding <=> %s::vector) as sim
from {self._schema}.insights
where deleted_at is null and replaced_by is null and embedding is not null
order by embedding <=> %s::vector
limit %s
"""
        # Notes:
        # - HNSW is approximate: a search width near `k` returns a
        #   top-k that is only near the true one, and a width well
        #   above `k` matches an exact scan. The width scales with `k`
        #   above pgvector's own default of 40, up to its cap of 1000.
        # - Set on THIS connection: the session opens its own in
        #   `__enter__`, so a width set on the Backend's connection
        #   never reaches this query.
        with self._conn.cursor() as cur:
            cur.execute(
                f'set hnsw.ef_search = {min(1000, max(40, 4 * int(k)))}')
            cur.execute(sql, (query_vec, query_vec, k))
            return [
                (r[0], float(r[1])) for r in cur.fetchall()
                if r[1] is not None and float(r[1]) > 0.0
                ]

    def similarities(
            self, query_vec: list[float]) -> dict[Id, float]:
        """Cosine per id, positives only, computed in the database.
        """
        assert self._conn is not None
        sql = f"""
select id, 1 - (embedding <=> %s::vector) as sim
from {self._schema}.insights
where deleted_at is null and replaced_by is null and embedding is not null
"""
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_vec,))
            return {
                r[0]: float(r[1]) for r in cur
                if r[1] is not None and float(r[1]) > 0.0
                }

    def keyword_counts(
            self, query_tokens: set[str]) -> dict[Id, int]:
        """Match count per active insight id, computed in the database.

        See the Protocol docstring for the contract.
        """
        if not query_tokens:
            return {}
        assert self._conn is not None
        sql = f"""
select i.id, cardinality(array(
        select unnest(%(q)s::text[])
        intersect
        select unnest(i.kw_tokens)
        )) as matched
from {self._schema}.insights i
where i.deleted_at is null and i.replaced_by is null and i.kw_tokens && %(q)s::text[]
"""
        # Notes:
        # - `kw_tokens` holds the row's tokens as
        #   `keyword.insight_tokens` produced them at write time, so
        #   the count matches the Python route exactly and no per-row
        #   tokenizing happens at recall.
        # - Row-side stopword filtering cannot change the count:
        #   `query_tokens` is stopword-filtered too, and `intersect`
        #   sees only tokens present in both.
        # - `kw_tokens && query` is exactly `matched > 0`, so the GIN
        #   index answers the filter and the intersect runs only on
        #   rows that can contribute.
        # - Unnesting the stored array beats walking the query against
        #   it with `= any` once the query holds more than a few
        #   tokens.
        with self._conn.cursor() as cur:
            cur.execute(sql, {'q': sorted(query_tokens)})
            return {r[0]: int(r[1]) for r in cur}


class PostgresBackend(Backend):
    """Per-store Postgres backend: schema-bound connection + sub-stores.

    Single primary connection per backend. `transaction()` uses
    psycopg's nested transaction (BEGIN / SAVEPOINT). `recall_session`
    opens a dedicated autocommit connection so a long read does not
    share a connection with active writes.
    """

    nodes: PostgresNodeStore
    meta: PostgresMetaStore
    oplog: PostgresOplog

    def __init__(
            self, dsn: str, store: str, *, read_only: bool = False) -> None:
        """Open the primary connection, bound to the store's schema.

        Parameters
        ----------
        dsn : str
            Connection string.
        store : str
            Store name.
        read_only : bool, default False
            The server refuses every write on the primary connection
            with `ReadOnlySqlTransaction`.
        """
        self._dsn = dsn
        self._store = store
        self._schema = _store_schema(store)
        self._conn = _open_connection(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(f'set search_path = {self._schema}, public')
            if read_only:
                cur.execute('set default_transaction_read_only = on')
        self.nodes = PostgresNodeStore(self._conn, self._schema)
        self.meta = PostgresMetaStore(self._conn, self._schema)
        self.oplog = PostgresOplog(self._conn, self._schema)

    @property
    def path(self) -> str:
        """DSN+schema identifier (Postgres has no filesystem path).
        """
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
    def recall_session(self) -> Iterator[PostgresRecallSession]:
        """Yield a PostgresRecallSession for one recall request.
        """
        session = PostgresRecallSession(self._dsn, self._schema)
        with session:
            yield session

    @contextmanager
    def _advisory_lock(self, key: int) -> Iterator[bool]:
        """Hold a session-scoped advisory lock on a dedicated connection.

        Yields whether `pg_try_advisory_lock` acquired the key. The
        lock releases on connection close.
        """
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
    def reembed_lock(self, name: str) -> Iterator[bool]:
        """Acquire a per-store session-scoped advisory sweep lock.

        Dedicated `psycopg.connect()` outside any pool, autocommit,
        with `keepalives_idle=30`. Uses `pg_try_advisory_lock`
        (non-blocking) so a second sweep agent fails fast with
        `False` instead of waiting hours. Released on connection
        close (intended crash-recovery mechanism).
        """
        with self._advisory_lock(
                _advisory_lock_key(self._schema, f'reembed:{name}')
                ) as acquired:
            yield acquired

    @contextmanager
    def swap_lock(self) -> Iterator[bool]:
        """Acquire a per-store session-scoped advisory swap lock.

        Mirrors `reembed_lock` but with the dedicated key
        `embed_swap:<schema>` so swaps and reembeds do not contend
        for the same lock. Held continuously across multi-step
        orchestration (swap_prepare -> backfill -> cutover) since
        a swap may span minutes-to-hours and crosses CLI invocations
        on resume. Auto-releases on connection close, surviving
        process crash.
        """
        with self._advisory_lock(
                _advisory_lock_key(self._schema, 'embed_swap')
                ) as acquired:
            yield acquired

    def swap_prepare(self, target_dim: int) -> None:
        """Add the pending vector column and its HNSW index.
        """
        _check_pg_version(self._dsn)
        _swap_prepare_pg(self._dsn, self._schema, int(target_dim))

    def iter_for_swap(
            self, cursor: str, batch: int) -> list[tuple[str, str]]:
        sql = (
            f'select id, content from {self._schema}.insights'
            f' where deleted_at is null and replaced_by is null'
            f'   and embedding_pending is null'
            f'   and id > %s'
            f' order by id limit %s')
        with self._conn.cursor() as cur:
            cur.execute(sql, (cursor, int(batch)))
            return [(r[0], r[1]) for r in cur.fetchall()]

    def write_swap_batch(
            self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        sql = (
            f'update {self._schema}.insights'
            f' set embedding_pending = %s::vector'
            f' where id = %s')
        with self._conn.cursor() as cur:
            cur.executemany(
                sql, [(vec, rid) for (rid, vec) in items])

    def swap_cutover(self, target: Fingerprint) -> None:
        _swap_cutover_pg(self._dsn, self._schema)

    def swap_abort(self) -> None:
        _swap_abort_pg(self._dsn, self._schema)

    def integrity_check(self) -> dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(
                f'select 1 from {self._schema}.insights limit 1')
            cur.fetchone()
        return {'ok': True, 'detail': 'schema reachable'}

    def start_run(self) -> int | None:
        """Insert a per-store `worker_runs` row, return its id.
        """
        sql = (
            f'insert into {self._schema}.worker_runs'
            f' (last_heartbeat_at) values (now()) returning id')
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            self._conn.commit()
        return int(row[0]) if row else None

    def beat_run(self, run_id: int | None) -> None:
        """Advance `last_heartbeat_at = now()` on the per-store run row.
        """
        if run_id is None:
            return
        sql = (
            f'update {self._schema}.worker_runs'
            f' set last_heartbeat_at = now() where id = %s')
        with self._conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            self._conn.commit()

    def finish_run(self, run_id: int | None) -> None:
        """Stamp `ended_at = now()` on the per-store run row.
        """
        if run_id is None:
            return
        sql = (
            f'update {self._schema}.worker_runs'
            f' set ended_at = now() where id = %s')
        with self._conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            self._conn.commit()

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        """Return the per-store recent `worker_runs` rows (newest first).
        """
        sql = (
            f'select id, started_at, ended_at,'
            f' last_heartbeat_at'
            f' from {self._schema}.worker_runs'
            f' order by id desc limit %s')
        with self._conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        return [
            WorkerRun(
                id=int(r[0]),
                started_at=r[1],
                ended_at=r[2],
                last_heartbeat_at=r[3])
            for r in rows
            ]

    def close(self) -> None:
        import psycopg as _psycopg
        try:
            self._conn.close()
        except _psycopg.Error as exc:
            logger.debug(f'pg connection close failed: {exc}')

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        self.close()


def _resolve_active_dim(expected_dim: int | None) -> int:
    """Embedding dim to bake into a fresh schema.

    Parameters
    ----------
    expected_dim : int or None
        Dim read from `meta.embed_fingerprint` on a pre-existing
        schema; None when no stored dim is readable. A positive value
        wins without consulting the env-bound active client, which may
        not match a store whose stored fingerprint differs.

    Returns
    -------
    int
        `expected_dim`, else the active fingerprint's dim.

    Raises
    ------
    BackendError
        When neither is positive: a column built at a guessed width
        would refuse every vector the configured model writes.
    """
    if expected_dim is not None and expected_dim > 0:
        return int(expected_dim)
    try:
        active = seed_default_fingerprint()
    except (RuntimeConfigError, ImportError) as exc:
        raise BackendError(
            f'cannot size the store vector column: {exc}'
            ) from exc
    if active.dim <= 0:
        raise BackendError(
            f'cannot size the store vector column: embed model'
            f' {active.model!r} is not reachable; check {config.API_KEY}'
            f' and {config.ENDPOINT}')
    return int(active.dim)


def _read_stored_dim(dsn: str, store: str) -> int | None:
    """Stored fingerprint dim for `store`, read on a best-effort basis.

    Parameters
    ----------
    dsn : str
        Connection string.
    store : str
        Store name.

    Returns
    -------
    int or None
        None when the schema does not exist yet, the meta row is
        absent, or the value is unparseable.

    Raises
    ------
    BackendError
        On a connection failure (network or auth). A transient outage
        must not pass for a fresh schema and force a fingerprint
        mismatch on the next assert.
    """
    import psycopg

    schema = _store_schema(store)
    sql = f"select value from {schema}.meta where key = 'embed_fingerprint'"
    # The handler sits at the statement, not around the block: it
    # runs before `_connection` translates, so a missing schema stays
    # a None result while every other driver error still becomes a
    # BackendError.
    with _connection(dsn, autocommit=True) as conn, \
            conn.cursor() as cur:
        try:
            cur.execute(sql)
        except psycopg.errors.UndefinedTable:
            return None
        row = cur.fetchone()
    if row is None or not row[0]:
        return None
    try:
        return int(json.loads(row[0]).get('dim') or 0) or None
    except Exception:
        return None


def postgres_store_exists(store: str, dsn: str) -> bool:
    """True when the database at `dsn` holds the store's schema.

    Raises
    ------
    BackendError
        On a connection failure, which answers neither way.
    """
    # A database without pgvector holds no memman schema, and type
    # registration would raise there as if the server were down.
    with _connection(dsn, autocommit=True, register_vector=False) as conn, \
            conn.cursor() as cur:
        cur.execute(
            'select 1 from pg_namespace where nspname = %s',
            (_store_schema(store),))
        return cur.fetchone() is not None


def open_postgres_backend(
        store: str, dsn: str, *,
        read_only: bool = False,
        create: bool = False) -> PostgresBackend:
    """Open the per-store Postgres backend at `dsn`.

    Reads the store's stored fingerprint dim (if any) before
    `_ensure_baseline_schema`, so a freshly discovered Postgres-backed
    store opens at its own dim rather than the env active client's.

    Parameters
    ----------
    store : str
        Store name.
    dsn : str
        Connection string.
    read_only : bool, default False
        Run no schema statement and return a backend whose writes the
        server refuses.
    create : bool, default False
        Create the schema when it is missing. A writable open also
        recreates a baseline or HNSW index dropped by name.

    Returns
    -------
    PostgresBackend

    Raises
    ------
    StoreMissingError
        The schema is missing and `create` is False. Nothing is
        written.
    BackendError
        On a connection failure.
    """
    if not create and not postgres_store_exists(store, dsn):
        raise StoreMissingError(store)
    stored = _read_stored_dim(dsn, store)
    target_dim = _resolve_active_dim(expected_dim=stored)
    if read_only:
        _assert_vector_dim_matches(dsn, store, target_dim)
        return PostgresBackend(dsn, store, read_only=True)
    _ensure_baseline_schema(dsn, store, dim=target_dim)
    _assert_vector_dim_matches(dsn, store, target_dim)
    backend = PostgresBackend(dsn, store)
    try:
        _ensure_hnsw_index(dsn, _store_schema(store))
    except Exception as exc:
        logger.warning(f'HNSW index ensure failed: {exc}')
    return backend


def drop_postgres_store(store: str, dsn: str) -> None:
    """Drop the per-store schema at `dsn`.

    Queue rows are not purged here: the queue is SQLite under the
    per-store routing model and `factory.drop_store` calls
    `queue.purge_store` separately.
    """
    schema = _store_schema(store)
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'drop schema if exists {schema} cascade')


def apply_baseline_schema(
        conn: Any, schema: str, dim: int) -> None:
    """Apply the baseline DDL on an open connection (idempotent).

    Parameters
    ----------
    conn : Any
        Open psycopg connection; the caller controls the transaction.
        The migrator passes one inside its transaction, so the DDL
        rolls back if the import fails.
    schema : str
        Store schema to create.
    dim : int
        Width N of the `vector(N)` embedding column for a new schema.

    Raises
    ------
    BackendError
        When an existing schema lacks a baseline column.
    """
    import psycopg
    with conn.cursor() as cur:
        cur.execute('create extension if not exists vector')
        cur.execute(f'create schema if not exists {schema}')
        try:
            cur.execute(
                PG_BASELINE_SCHEMA.format(schema=schema, dim=dim))
        except psycopg.errors.UndefinedColumn as exc:
            # `create table if not exists` no-ops on an existing
            # table, so a store missing a column trips the first
            # baseline index that names it.
            raise BackendError(
                f'postgres schema {schema} predates the current'
                f' schema ({exc}); add or rename the missing column'
                ' in the live schema, drop by name every index whose'
                ' definition changed, then reopen') from exc


def _ensure_baseline_schema(
        dsn: str, store: str, *, dim: int) -> None:
    """Create the schema and apply baseline DDL idempotently.

    Parameters
    ----------
    dsn : str
        Connection string.
    store : str
        Store name.
    dim : int
        Width N of `vector(N)` for a new schema. An existing schema
        keeps its column width; `_assert_vector_dim_matches` refuses
        the open on a mismatch.
    """
    schema = _store_schema(store)
    with _connection(dsn, autocommit=True) as conn:
        apply_baseline_schema(conn, schema, dim)


def _assert_vector_dim_matches(
        dsn: str, store: str, expected_dim: int) -> None:
    """Refuse to open if the stored `vector(N)` column width differs.

    Reads `N` from `pg_attribute.atttypmod`, where pgvector stores it
    with no VARHDRSZ offset. `information_schema.columns` cannot supply
    it, since pgvector types leave `character_maximum_length` unset.

    Parameters
    ----------
    dsn : str
        Connection string.
    store : str
        Store name.
    expected_dim : int
        Dim of the operator's active embedding fingerprint. While
        `meta.embed_swap_state` is `backfilling` or `cutover`, the dim
        of the in-flight `embedding_pending` column also passes, so a
        process can open the store mid-swap.

    Raises
    ------
    BackendError
        When `expected_dim` differs from the stored column width. The
        message carries an upgrade hint.
    """
    schema = _store_schema(store)
    dim_sql = """
select attname, atttypmod from pg_attribute
where attrelid = (%s || '.insights')::regclass
  and attname in ('embedding', 'embedding_pending')
  and not attisdropped
"""
    state_sql = (
        f"select value from {schema}.meta"
        f" where key = 'embed_swap_state'")
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(dim_sql, (schema,))
        rows = cur.fetchall()
        cur.execute(state_sql)
        state_row = cur.fetchone()
    dims = {r[0]: int(r[1]) for r in rows if r[1] is not None
            and int(r[1]) > 0}
    swap_state = (state_row[0] if state_row else '') or ''
    swap_active = swap_state in {'backfilling', 'cutover'}
    accepted = {dims['embedding']}
    if swap_active and 'embedding_pending' in dims:
        accepted.add(dims['embedding_pending'])
    if expected_dim in accepted:
        return
    stored_dim = dims['embedding']
    raise BackendError(
        f'store {store!r} has vector({stored_dim}) but the active'
        f' embedding client produces dim={expected_dim}.'
        f" Run 'memman embed swap --to <model>' to migrate, or"
        f' switch back to a {stored_dim}-dim provider.')


def _check_pg_version(dsn: str) -> None:
    """Refuse if the server is older than Postgres 12.

    `embed swap` relies on `ADD COLUMN vector(N)` being metadata-only
    (PG 11+) and on `CREATE INDEX CONCURRENTLY` semantics that PG 12
    cleaned up.
    """
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute('show server_version_num')
        row = cur.fetchone()
        if row is None:
            return
        version_num = int(row[0])
    if version_num < 120000:
        raise BackendError(
            f'Postgres {version_num // 10000} is below the swap'
            ' minimum (12).')


def _swap_index_timeout_s() -> int:
    """Read `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` (default 0 = unlimited).
    """
    raw = os.environ.get(config.EMBED_SWAP_INDEX_TIMEOUT)
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _swap_pending_index_name(schema: str) -> str:
    """Name of the in-flight HNSW index on the pending column.

    Cutover renames it to the canonical `idx_insights_hnsw_<schema>`.
    """
    return f'idx_insights_hnsw_pending_{schema}'


def _swap_prepare_pg(
        dsn: str, schema: str, target_dim: int) -> None:
    """Add `embedding_pending vector(N)` and build a new HNSW.

    Parameters
    ----------
    dsn : str
        Connection string.
    schema : str
        Store schema; must be a plain SQL identifier.
    target_dim : int
        Width N of the new vector column.

    Raises
    ------
    BackendError
        When the column cannot be added within 3 attempts.

    Notes
    -----
    - Idempotent on resume: both the column and the index use
      `if not exists`.
    - The index build uses `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT`
      (default 0, unlimited).
    """
    _check_identifier(schema)
    add_sql = (
        f'alter table {schema}.insights add column if not exists'
        f' embedding_pending vector({int(target_dim)})')
    # The `add column` runs under a short `lock_timeout` and retries,
    # so it cannot wedge behind a long-running query.
    retries = 3
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with _connection(dsn, autocommit=True) as conn, \
                    conn.cursor() as cur:
                cur.execute("set lock_timeout = '5s'")
                cur.execute(add_sql)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt + 1 == retries:
                break
            time.sleep(1.0)
    if last_exc is not None:
        raise BackendError(
            f'failed to add embedding_pending column on'
            f' {schema}: {last_exc}')

    index_name = _swap_pending_index_name(schema)
    timeout_s = _swap_index_timeout_s()
    create_idx_sql = (
        f'create index concurrently if not exists {index_name}'
        f' on {schema}.insights using hnsw'
        f' (embedding_pending vector_cosine_ops)'
        f' where deleted_at is null and replaced_by is null')
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"set statement_timeout = '{timeout_s}s'")
        cur.execute(create_idx_sql)


def _swap_cutover_pg(
        dsn: str, schema: str) -> None:
    """Atomic switch from `embedding` to `embedding_pending`.

    One transaction with `statement_timeout=0` replaces the column and
    renames the pending HNSW index to the canonical name. The
    orchestrator records cutover state and writes the fingerprint
    around this call.

    A schema with no `embedding_pending` column already committed its
    cutover, so a resume after a crash past that commit changes
    nothing here and the orchestrator finishes the swap.

    Raises
    ------
    SwapCutoverRefused
        When fewer current rows carry `embedding_pending` than carry
        `embedding` (the backfill is incomplete); nothing changes.
    """
    _check_identifier(schema)
    canonical_idx = f'idx_insights_hnsw_{schema}'
    pending_idx = _swap_pending_index_name(schema)
    pending_sql = (
        'select 1 from information_schema.columns'
        ' where table_schema = %s and table_name = %s'
        " and column_name = 'embedding_pending'")
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(pending_sql, (schema, 'insights'))
        if cur.fetchone() is None:
            logger.debug(f'swap cutover on {schema}: already committed')
            return
    verify_sql = (
        f'select count(*) filter (where embedding is not null),'
        f' count(*) filter (where embedding_pending is not null)'
        f' from {schema}.insights where deleted_at is null and replaced_by is null')
    with _connection(dsn, autocommit=False) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("set local statement_timeout = '0'")
                cur.execute(verify_sql)
                row = cur.fetchone()
                old_count = int(row[0]) if row else 0
                new_count = int(row[1]) if row else 0
                if new_count < old_count:
                    raise SwapCutoverRefused(
                        f'cutover refused: embedding_pending has'
                        f' {new_count} rows but embedding has'
                        f' {old_count}; backfill is incomplete')
                cur.execute(
                    f'drop index if exists'
                    f' {schema}.{canonical_idx} cascade')
                cur.execute(
                    f'alter table {schema}.insights'
                    f' drop column embedding')
                cur.execute(
                    f'alter table {schema}.insights'
                    f' rename column embedding_pending to embedding')
                cur.execute(
                    f'alter index if exists {schema}.{pending_idx}'
                    f' rename to {canonical_idx}')
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _swap_abort_pg(dsn: str, schema: str) -> None:
    """Drop the pending column and any pending HNSW remnant.
    """
    _check_identifier(schema)
    pending_idx = _swap_pending_index_name(schema)
    with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            f'drop index if exists {schema}.{pending_idx}')
        cur.execute(
            f'alter table {schema}.insights'
            f' drop column if exists embedding_pending')


def _ensure_hnsw_index(dsn: str, schema: str) -> None:
    """Create or recreate the HNSW index on `insights.embedding`.

    An invalid prior index (the remnant of an aborted CONCURRENTLY
    build) is dropped first, so the rebuild can proceed.

    Runs on a dedicated autocommit connection because
    `create index concurrently` cannot run inside a transaction.
    `statement_timeout` is set from `MEMMAN_REINDEX_TIMEOUT` (default
    180 seconds) so a stuck build aborts and the next call's
    invalid-remnant cleanup can recover.

    Parameters
    ----------
    dsn : str
        Connection string.
    schema : str
        Store schema; must be a plain SQL identifier.
    """
    _check_identifier(schema)
    index_name = f'idx_insights_hnsw_{schema}'
    timeout_s = int(os.environ.get(config.REINDEX_TIMEOUT, '180'))
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
where deleted_at is null and replaced_by is null
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


class PostgresMigrator(Migrator):
    """Postgres + pgvector implementation of the Migrator surface.

    `gather(store)` reads `store_<store>` schema into a portable
    `MigrationPayload`. `apply(store, payload)` creates the schema
    (idempotently) and inserts rows in one transaction with
    `ON CONFLICT DO NOTHING`. `archive` invokes `pg_dump -Fc` for
    a recoverable filesystem artifact.
    """

    backend_name: ClassVar[str] = 'postgres'

    def __init__(self, *, dsn: str) -> None:
        self.dsn = dsn

    def preflight_source(self, store: str) -> None:
        schema = _store_schema(store)
        with _connection(self.dsn, autocommit=True) as conn, \
                conn.cursor() as cur:
            cur.execute(
                'select 1 from pg_namespace where nspname = %s',
                (schema,))
            if cur.fetchone() is None:
                raise MigrateError(
                    f'source postgres schema {schema!r} does not'
                    f' exist for store {store!r}')
            cur.execute(
                f"select value from {schema}.meta"
                " where key = 'embed_fingerprint'")
            fp_row = cur.fetchone()
            if fp_row is None or not fp_row[0]:
                raise MigrateError(
                    f'source schema {schema!r} has no'
                    f' meta.embed_fingerprint; run `memman doctor`'
                    f' on the source store before migrating')
            cur.execute(
                f"select value from {schema}.meta"
                " where key = 'embed_swap_state'")
            swap_row = cur.fetchone()
            if swap_row is not None and swap_row[0]:
                raise MigrateError(
                    f'store {store!r} has an embed swap in flight'
                    f' (state={swap_row[0]!r});'
                    f' {swap_remedy(store, swap_row[0])}')

    def preflight_target(self, store: str) -> None:
        sanitize_identifier(store)
        _check_identifier(store)
        with _connection(self.dsn, autocommit=True) as conn, \
                conn.cursor() as cur:
            cur.execute('select 1')
            cur.execute(
                "select 1 from pg_extension where extname = 'vector'")
            if cur.fetchone() is None:
                raise MigrateError(
                    'pgvector extension not installed in the'
                    ' target database; run `create extension'
                    ' vector;` as a superuser first')
            cur.execute(
                'select has_database_privilege(current_user,'
                " current_database(), 'CREATE')")
            if not bool(cur.fetchone()[0]):
                raise MigrateError(
                    'current postgres role lacks create schema'
                    ' privilege on the target database')

    def gather(self, store: str) -> MigrationPayload:
        schema = _store_schema(store)
        with _connection(self.dsn, autocommit=True) as conn, \
                conn.cursor() as cur:
            cur.execute(
                'select 1 from pg_namespace where nspname = %s',
                (schema,))
            if cur.fetchone() is None:
                raise MigrateError(
                    f'source postgres schema {schema!r} does not'
                    f' exist for store {store!r}')

            cur.execute(f'select key, value from {schema}.meta')
            meta_dict = dict(cur.fetchall())

            fp_str = meta_dict.get('embed_fingerprint')
            if not fp_str:
                raise MigrateError(
                    f'source schema {schema!r} has no'
                    f' meta.embed_fingerprint')
            fingerprint = Fingerprint.from_json(fp_str)

            cur.execute(
                f'select {_RAW_SELECT} from {schema}.insights order by id')
            insights = list(starmap(MigrateInsight, cur.fetchall()))

            cur.execute(f"""
select coalesce(legacy_id, id) as sqlite_id,
       operation, insight_id, detail, created_at,
       before, after, id
from {schema}.oplog
order by sqlite_id
""")
            oplog = [
                MigrateOpLog(
                    id=int(o[7]), operation=o[1],
                    insight_id=o[2], detail=o[3] or '',
                    created_at=o[4],
                    before=dict(o[5]) if o[5] else None,
                    after=dict(o[6]) if o[6] else None,
                    legacy_id=int(o[0]))
                for o in cur.fetchall()]

        return MigrationPayload(
            fingerprint=fingerprint,
            embedding_dim=fingerprint.dim,
            insights=insights,
            oplog=oplog,
            meta=meta_dict)

    def apply(
            self, store: str, payload: MigrationPayload) -> None:
        schema = _store_schema(store)
        dim = payload.embedding_dim
        with _connection(self.dsn, autocommit=False) as conn:
            try:
                apply_baseline_schema(conn, schema, dim)

                if payload.insights:
                    with conn.cursor() as cur:
                        cur.executemany(
                            f'insert into {schema}.insights ({_RAW_INSERT})'
                            ' values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
                            ' %s, %s, %s, %s)'
                            ' on conflict (id) do nothing',
                            [_raw_values(ins) for ins in payload.insights])

                if payload.oplog:
                    op_rows = [(
                            op.operation, op.insight_id, op.detail,
                            op.created_at,
                            json.dumps(op.before)
                            if op.before is not None else None,
                            json.dumps(op.after)
                            if op.after is not None else None,
                            op.legacy_id) for op in payload.oplog]
                    with conn.cursor() as cur:
                        cur.executemany(
                            f'insert into {schema}.oplog'
                            ' (operation, insight_id, detail,'
                            '  created_at, before, after, legacy_id)'
                            ' values (%s, %s, %s, %s, %s::jsonb,'
                            ' %s::jsonb, %s)'
                            ' on conflict (legacy_id) do nothing',
                            op_rows)

                meta_rows = list(payload.meta.items())
                if meta_rows:
                    with conn.cursor() as cur:
                        cur.executemany(
                            f'insert into {schema}.meta (key, value)'
                            ' values (%s, %s)'
                            ' on conflict (key) do update'
                            ' set value = excluded.value',
                            meta_rows)

                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise MigrateError(
                    f'postgres apply for store {store!r} failed:'
                    f' {type(exc).__name__}: {exc}') from exc

    def archive(self, store: str, data_dir: str) -> Artifact:
        try:
            path = archive_postgres_schema(data_dir, store, self.dsn)
        except RuntimeError as exc:
            raise MigrateError(
                f'archive failed for store {store!r}: {exc}'
                ) from exc
        return Artifact(
            kind='filesystem', location=str(path / 'dump.pgdump'))
