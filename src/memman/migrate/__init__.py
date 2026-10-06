"""Bidirectional store migration shared types and primitives.

Backs the `memman migrate --to <backend>` CLI command. The actual
gather/apply work lives in per-backend `Migrator` subclasses
(`memman.store.sqlite.SqliteMigrator`,
`memman.store.postgres.PostgresMigrator`); this module owns the
backend-agnostic types they exchange (`MigrationPayload`,
`MigrateInsight`, `MigrateOpLog`, `Artifact`) plus orchestration
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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal

from memman.drain_lock import DrainLockBusy, acquire, release
from memman.embed.fingerprint import Fingerprint


@dataclass
class MigrateInsight:
    """Full insight row for migration round-trips.

    Mirrors the union of SQLite/Postgres column shapes. JSON columns
    arrive as parsed Python objects; timestamps as `datetime`. The
    embedding is carried as a `list[float]`.
    """

    id: str
    content: str
    summary: str | None
    embedding: list[float] | None
    enrich_attempted_at: datetime | None
    enriched_at: datetime | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    prompt_version: str | None
    embedding_model: str | None
    queue_uuid: str | None
    replaced_by: str | None
    author: str | None


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
    legacy_id: int | None


@dataclass
class MigrationPayload:
    """Wire-format payload for a single store's migration.

    Backend-agnostic. Produced by `Migrator.gather`, consumed by
    `Migrator.apply`. Round-trip preservation between any two
    backends is the invariant.
    """

    fingerprint: Fingerprint
    embedding_dim: int
    insights: list[MigrateInsight]
    oplog: list[MigrateOpLog]
    meta: dict[str, str]


@dataclass
class Artifact:
    """Where a backend's pre-migration source state was archived.

    `kind='filesystem'` describes a local archive directory or
    file. `kind='none'` means there was no source state to archive:
    a SQLite store with no directory on disk.
    """

    kind: Literal['filesystem', 'none']
    location: str | None


class Migrator(abc.ABC):
    """Per-backend migration surface.

    Abstract base class for the five migration verbs. Concrete
    implementations live with their backend (`store/sqlite.py`,
    `store/postgres.py`) and inherit from this class. Stateless
    across calls: each method acquires and releases its own
    connection. A missing method fails at instantiation, before the
    CLI runner makes its first call.
    """

    backend_name: ClassVar[str]

    @abc.abstractmethod
    def preflight_source(self, store: str) -> None:
        """Verify the store is in a state that can be migrated FROM.

        Raises
        ------
        MigrateError
            On any precondition failure (missing store, schema
            mismatch, broken connection, or an embed swap in flight).
        """

    @abc.abstractmethod
    def preflight_target(self, store: str) -> None:
        """Verify the backend can accept a fresh migration INTO `store`.

        Raises
        ------
        MigrateError
            On identifier collision or a missing extension or
            privilege.
        """

    @abc.abstractmethod
    def gather(self, store: str) -> MigrationPayload:
        """Read the full store contents into a portable payload.
        """

    @abc.abstractmethod
    def apply(self, store: str, payload: MigrationPayload) -> None:
        """Insert the payload insights `store` lacks, in one transaction.

        Creates the store when it is missing. An insight whose id the
        store already holds keeps its stored columns. Every payload
        oplog row is appended, except on Postgres, which skips a row
        whose `legacy_id` it already holds. Every meta key is written,
        replacing a stored value.

        Raises
        ------
        MigrateError
            The write failed; the transaction rolled back.
        """

    @abc.abstractmethod
    def archive(self, store: str, data_dir: str) -> Artifact:
        """Move or dump the source state into a recoverable archive.

        Returns
        -------
        Artifact
            A `kind='filesystem'` artifact naming the archive: the
            SQLite store directory, moved under `archive/<store>/`, or
            the Postgres `pg_dump` file. `Artifact(kind='none',
            location=None)` when there is no source state to archive.

        Raises
        ------
        MigrateError
            When the Postgres dump fails.
        """


def sanitize_identifier(name: str) -> str:
    """Backend-portable identifier sanitizer.

    Postgres/MySQL allow 63/64 chars; a name over 63 chars gets a
    deterministic 8-hex-char sha256 suffix replacing the truncated
    tail so two distinct names with the same prefix don't collide.

    Parameters
    ----------
    name : str
        Candidate identifier.

    Returns
    -------
    str
        `name`, or its 63-char truncated form with a hash suffix.

    Raises
    ------
    MigrateError
        On characters outside `[A-Za-z0-9_]`.
    """
    allowed_chars = r'[A-Za-z0-9_]'
    max_len = 63
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
    """Migration aborted because a precondition or invariant failed.
    """


class SchemaState(enum.Enum):
    """Target Postgres schema state for a memman store.
    """

    ABSENT = 'absent'
    EMPTY = 'empty'
    POPULATED = 'populated'


def preflight(dsn: str) -> dict[str, bool]:
    """Verify the target Postgres role can run the migration.

    Parameters
    ----------
    dsn : str
        Postgres connection string.

    Returns
    -------
    dict[str, bool]
        Check name to pass/fail.

    Raises
    ------
    MigrateError
        On the first hard failure (connection refused, pgvector
        missing).
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
    `information_schema.tables` filtered to the three memman tables.
    A schema absent from the result is ABSENT; present with no
    memman tables is EMPTY (likely an aborted prior run); present
    with one or more tables is POPULATED.

    Parameters
    ----------
    dsn : str
        Postgres connection string.
    stores : list[str]
        Store names whose schemas to classify.

    Returns
    -------
    dict[str, SchemaState]
        State per store name.

    Raises
    ------
    MigrateError
        On connection or permission failures, so preflight stays
        fail-closed.
    """
    from memman.store.postgres import _connection, _store_schema

    schema_to_store = {_store_schema(s): s for s in stores}
    schema_names = list(schema_to_store.keys())

    sql = """
select n.nspname, count(t.table_name)
from pg_namespace n
left join information_schema.tables t
  on t.table_schema = n.nspname
  and t.table_name in ('insights', 'oplog', 'meta')
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
        pg_conn: Any, schema: str, store: str,
        expected: dict[str, int]) -> None:
    """Compare destination table counts against captured source counts.

    Checks the post-commit absolute counts: on a re-run against a
    populated schema, `ON CONFLICT DO NOTHING` makes the per-call
    insert count only a lower bound.

    Parameters
    ----------
    pg_conn : Any
        Open Postgres connection.
    schema : str
        The store's schema name.
    store : str
        Store name, for the error message.
    expected : dict[str, int]
        Source counts keyed `insights`, `oplog`, `meta`.

    Raises
    ------
    MigrateError
        On any mismatch, naming the per-table counts.
    """
    sql = (
        f'select '
        f'  (select count(*) from {schema}.insights),'
        f'  (select count(*) from {schema}.oplog),'
        f'  (select count(*) from {schema}.meta)')
    with pg_conn.cursor() as cur:
        cur.execute(sql)
        ins, oplog, meta = cur.fetchone()
    actual = {
        'insights': int(ins), 'oplog': int(oplog), 'meta': int(meta),
        }
    diffs = [
        (table, expected[table], actual[table])
        for table in ('insights', 'oplog', 'meta')
        if expected[table] != actual[table]
        ]
    if diffs:
        detail = ', '.join(
            f'{t}: source={s} dest={d}' for t, s, d in diffs)
        raise MigrateError(
            f'verify failed for store {store!r}: {detail}')


@contextmanager
def held_drain_lock(data_dir: str) -> Iterator[int]:
    """Acquire the shared drain.lock for the duration of the block.
    """
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
