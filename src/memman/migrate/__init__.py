"""Bidirectional store migration shared types and primitives.

Backs the `memman migrate --to <backend>` CLI command. The actual
gather/apply work lives in per-backend `Migrator` subclasses
(`memman.store.sqlite.SqliteMigrator`,
`memman.store.postgres.PostgresMigrator`); this module owns the
backend-agnostic types they exchange (`MigrationPayload`,
`MigrateInsight`, `MigrateEdge`, `MigrateOpLog`,
`PendingReembed`, `SwapState`, `Artifact`) plus orchestration
helpers used by the CLI runner (`held_drain_lock`,
`inspect_target_schemas`, `preflight`,
`_verify_destination_counts`).

The drain.lock is held for the duration of the migrate command so
a scheduler-fired drain cannot race the source reader. Per-backend
schema state is captured into `Artifact` records so the operator
keeps a recoverable snapshot of the source after cutover.
"""

from __future__ import annotations

import abc
import enum
import hashlib
import re
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from memman.embed.fingerprint import Fingerprint

PAYLOAD_VERSION = 1

EmbeddingDtype = Literal[
    'float64', 'float32', 'float16', 'int8', 'binary']


@dataclass(frozen=True)
class BackendFeatures:
    """Capability flags a backend's migrator advertises.

    Drives capability gates in the CLI (e.g. dry-run, recall
    snapshot regeneration) and `apply()` payload-content checks
    (e.g. refusing edge-bearing payloads on edgeless backends).
    """

    supports_edges: bool
    supports_oplog: bool
    supports_recall_snapshot: bool
    supports_reembed: bool
    supports_drain_heartbeat: bool
    supports_filesystem_artifacts: bool
    supports_dry_run: bool
    accepted_embedding_dtypes: frozenset[str] = field(
        default_factory=lambda: frozenset({'float32', 'float64'}))


@dataclass(frozen=True)
class SwapState:
    """Mid-reembed swap state captured from `meta.embed_swap_*`.

    `cursor` is the highest insight id whose pending embedding has
    been written; resuming after a partial swap requires the same
    cursor to avoid re-embedding completed rows.
    """

    target_provider: str
    target_model: str
    target_dim: int
    cursor: str | None
    started_at: datetime | None


@dataclass
class PendingReembed:
    """In-flight pending embedding for a single insight during a swap."""

    insight_id: str
    vector: list[float]


@dataclass
class MigrateInsight:
    """Full insight row for migration round-trips.

    Mirrors the union of SQLite/Postgres column shapes. JSON columns
    arrive as parsed Python objects; timestamps as `datetime`. The
    embedding is carried as a `list[float]`; precision is governed
    by the payload's `embedding_dtype` field.
    """

    id: str
    content: str
    category: str
    importance: int
    entities: list[str]
    source: str
    access_count: int
    keywords: list[str] | None
    summary: str | None
    semantic_facts: list[Any] | None
    last_accessed_at: datetime | None
    embedding: list[float] | None
    effective_importance: float
    linked_at: datetime | None
    enriched_at: datetime | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    prompt_version: str | None
    model_id: str | None
    embedding_model: str | None


@dataclass
class MigrateEdge:
    """Full edge row for migration round-trips."""

    source_id: str
    target_id: str
    edge_type: str
    weight: float
    metadata: dict[str, Any]
    created_at: datetime


@dataclass
class MigrateOpLog:
    """Full oplog row for migration round-trips.

    `legacy_id` carries the original sqlite row id when a row that
    began life on sqlite gets exported to postgres; on the reverse
    direction the sqlite row id is recovered as
    `coalesce(legacy_id, id)` so id continuity is preserved across
    repeated round-trips.
    """

    id: int
    operation: str
    insight_id: str | None
    detail: str
    created_at: datetime
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    legacy_id: int | None = None


@dataclass
class MigrationPayload:
    """Wire-format payload for a single store's migration.

    Backend-agnostic. Produced by `Migrator.gather`, consumed by
    `Migrator.apply`. Round-trip preservation between any two
    backends with matching `BackendFeatures.supports_*` is the
    invariant.
    """

    payload_version: int
    fingerprint: Fingerprint
    embedding_dim: int
    embedding_dtype: EmbeddingDtype
    insights: list[MigrateInsight]
    edges: list[MigrateEdge]
    oplog: list[MigrateOpLog]
    embedding_pending: list[PendingReembed]
    swap_state: SwapState | None
    meta: dict[str, str]

    @property
    def has_edges(self) -> bool:
        return bool(self.edges)

    @property
    def has_oplog(self) -> bool:
        return bool(self.oplog)

    @property
    def has_swap(self) -> bool:
        return self.swap_state is not None or bool(self.embedding_pending)


@dataclass
class Artifact:
    """Where a backend's pre-migration source state was archived.

    `kind='filesystem'` describes a local archive directory or
    file. `kind='none'` is used by backends whose `apply` is a
    full migration (no pre-state to preserve as a snapshot).
    Other kinds (`'object_store'`, `'dump_job'`) are reserved for
    future backends and should be treated as opaque by callers.
    """

    kind: Literal['filesystem', 'object_store', 'dump_job', 'none']
    location: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class Migrator(abc.ABC):
    """Per-backend migration surface.

    Abstract base class for the six migration verbs. Concrete
    implementations live with their backend (`store/sqlite.py`,
    `store/postgres.py`) and inherit from this class. Stateless
    across calls: each method acquires + releases its own
    connection. ABC was chosen over Protocol so missing methods
    fail at instantiation (registry build time) rather than at
    first call from the CLI runner -- the entire point of the
    abstraction is making it cheap to add backends, and immediate
    failure shortens the feedback loop.
    """

    backend_name: ClassVar[str]
    snapshot_features: ClassVar[BackendFeatures]

    @abc.abstractmethod
    def preflight_source(self, store: str) -> None:
        """Verify the store is in a state that can be migrated FROM.

        Raises `MigrateError` on any precondition failure (missing
        store, schema mismatch, broken connection).
        """

    def preflight_target(self, store: str) -> None:
        """Verify the backend can accept a fresh migration INTO `store`.

        Default checks identifier sanity against the backend's
        feature flags; subclasses override to add extension /
        privilege / name-collision checks. Raises `MigrateError`
        on failure.
        """
        sanitize_identifier(
            store, max_len=63, allowed_chars=r'[A-Za-z0-9_]')

    @abc.abstractmethod
    def gather(self, store: str) -> MigrationPayload:
        """Read the full store contents into a portable payload."""

    @abc.abstractmethod
    def apply(self, store: str, payload: MigrationPayload) -> None:
        """Write `payload` into a fresh `store` on this backend."""

    def archive(self, store: str, data_dir: str) -> Artifact:
        """Snapshot the source state to a recoverable artifact.

        Default: `Artifact(kind='none', ...)` for backends whose
        `apply` is a full migration with no pre-state to preserve.
        Subclasses with filesystem state (sqlite dirs, postgres
        pg_dump) override to return a `kind='filesystem'` artifact.
        """
        return Artifact(
            kind='none', location=None,
            metadata={'reason': 'apply is a full migration'})

    @abc.abstractmethod
    def drop(self, store: str) -> None:
        """Remove this backend's storage for `store`."""


def sanitize_identifier(
        name: str, *, max_len: int,
        allowed_chars: str = r'[A-Za-z0-9_]') -> str:
    """Backend-portable identifier sanitizer.

    Postgres/MySQL allow 63/64 chars; SQL Server / Oracle 12.2+
    allow 128; Oracle legacy caps at 30. When `len(name) > max_len`
    a deterministic 8-hex-char sha256 suffix replaces the truncated
    tail so two distinct names with the same prefix don't collide.
    Raises `MigrateError` on illegal characters.
    """
    if not re.fullmatch(rf'{allowed_chars}+', name):
        raise MigrateError(
            f'identifier {name!r} contains characters outside'
            f' the allowed set; expected pattern {allowed_chars}+')
    if len(name) <= max_len:
        return name
    digest = hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]
    suffix = f'_{digest}'
    return name[:max_len - len(suffix)] + suffix


class MigrateError(Exception):
    """Migration aborted because a precondition or invariant failed."""


class SchemaState(enum.Enum):
    """Target Postgres schema state for a memman store."""

    ABSENT = 'absent'
    EMPTY = 'empty'
    POPULATED = 'populated'


@dataclass
class MigrateResult:
    """One store's migration outcome (counts per table)."""

    store: str
    schema: str
    insights: int = 0
    edges: int = 0
    oplog: int = 0
    meta: int = 0
    dry_run: bool = False
    verified: bool = False


def preflight(dsn: str) -> dict[str, bool]:
    """Verify the target Postgres role can run the migration.

    Returns a dict mapping check name to pass/fail. Raises
    `MigrateError` on the first hard failure (connection refused,
    pgvector missing).
    """
    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True)
    except Exception as exc:
        raise MigrateError(
            f'cannot connect to postgres: {type(exc).__name__}: {exc}'
            ) from exc

    checks: dict[str, bool] = {}
    with closing(conn), conn.cursor() as cur:
        cur.execute('select 1')
        checks['select_1'] = cur.fetchone()[0] == 1

        sql = """
select 1 from pg_extension
where extname = 'vector'
"""
        cur.execute(sql)
        row = cur.fetchone()
        if row is None:
            raise MigrateError(
                'pgvector extension is not installed in the target '
                'database; run `create extension vector;` as a '
                'superuser first')
        checks['pgvector_installed'] = True

        sql = """
select has_database_privilege(current_user, current_database(), 'CREATE')
"""
        cur.execute(sql)
        checks['create_schema_privilege'] = bool(cur.fetchone()[0])
        if not checks['create_schema_privilege']:
            raise MigrateError(
                'current postgres role lacks create schema '
                'privilege on the target database')
    return checks


def inspect_target_schemas(
        dsn: str, stores: list[str]) -> dict[str, SchemaState]:
    """Classify each `store_<name>` schema as ABSENT / EMPTY / POPULATED.

    Single round-trip query joining `pg_namespace` with
    `information_schema.tables` filtered to the four memman tables.
    A schema absent from the result is ABSENT; present with no
    memman tables is EMPTY (likely an aborted prior run); present
    with one or more tables is POPULATED. Raises `MigrateError` on
    connection or permission failures so preflight stays fail-closed.
    """
    from memman.store.postgres import _connection, _store_schema

    schema_to_store = {_store_schema(s): s for s in stores}
    schema_names = list(schema_to_store.keys())

    sql = """
select n.nspname, count(t.table_name)
from pg_namespace n
left join information_schema.tables t
  on t.table_schema = n.nspname
  and t.table_name in ('insights', 'edges', 'oplog', 'meta')
where n.nspname = any(%s)
group by n.nspname
"""
    try:
        with _connection(dsn, autocommit=True) as conn, \
                conn.cursor() as cur:
            cur.execute(sql, (schema_names,))
            rows = cur.fetchall()
    except Exception as exc:
        raise MigrateError(
            f'failed to inspect target schemas: '
            f'{type(exc).__name__}: {exc}') from exc

    seen = {row[0]: int(row[1]) for row in rows}
    result: dict[str, SchemaState] = {}
    for schema, store in schema_to_store.items():
        if schema not in seen:
            result[store] = SchemaState.ABSENT
        elif seen[schema] == 0:
            result[store] = SchemaState.EMPTY
        else:
            result[store] = SchemaState.POPULATED
    return result


def _verify_destination_counts(
        pg_conn, schema: str, store: str,
        expected: dict[str, int]) -> None:
    """Compare destination table counts against captured source counts.

    Idempotent re-runs against an already-populated schema may legitimately
    end with destination counts equal to source counts; the destination's
    `ON CONFLICT DO NOTHING` makes the per-call insert count a lower
    bound but the post-commit absolute count is the authoritative check.
    Raises `MigrateError` on any mismatch with the per-table delta.
    """
    sql = (
        f'select '
        f'  (select count(*) from {schema}.insights),'
        f'  (select count(*) from {schema}.edges),'
        f'  (select count(*) from {schema}.oplog),'
        f'  (select count(*) from {schema}.meta)')
    with pg_conn.cursor() as cur:
        cur.execute(sql)
        ins, edges, oplog, meta = cur.fetchone()
    actual = {
        'insights': int(ins), 'edges': int(edges),
        'oplog': int(oplog), 'meta': int(meta),
        }
    diffs = [
        (table, expected[table], actual[table])
        for table in ('insights', 'edges', 'oplog', 'meta')
        if expected[table] != actual[table]
        ]
    if diffs:
        detail = ', '.join(
            f'{t}: source={s} dest={d}' for t, s, d in diffs)
        raise MigrateError(
            f'verify failed for store {store!r}: {detail}')


@contextmanager
def held_drain_lock(data_dir: str) -> Iterator[int]:
    """Acquire the shared drain.lock for the duration of the block."""
    from memman.drain_lock import DrainLockBusy, acquire, release
    try:
        fd = acquire(data_dir)
    except DrainLockBusy:
        raise MigrateError(
            'drain.lock is held by another process; stop the scheduler '
            'with `memman scheduler stop` before running migrate')
    try:
        yield fd
    finally:
        release(fd)
