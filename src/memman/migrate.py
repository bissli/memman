"""SQLite -> Postgres store migration.

Backs the `memman migrate` CLI command. Wraps the streaming logic in
`scripts/import_sqlite_to_postgres.py` with CLI orchestration: DSN
preflight, drain-lock guard, per-store transaction, dry-run mode,
and an interactive confirmation flow.

Migration is one-way: the SQLite source is read-only. The Postgres
destination uses `ON CONFLICT (id) DO NOTHING` on the `insights`
insert path so an interrupted run can be re-run safely. When a
target schema already exists the operator is shown its state
(EMPTY / POPULATED) and explicit destructive overwrite happens only
on confirmation. The shared drain.lock is held for the duration of
the migrate command so a scheduler-fired drain cannot race the
SQLite reader.

On a successful migrate the CLI flips `MEMMAN_BACKEND=postgres` in
the env file so the next drain routes to the new database. With B3
the drain pipeline dispatches on `MEMMAN_BACKEND`, so this flip
makes the cutover atomic.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger('memman.migrate')


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
    try:
        with conn.cursor() as cur:
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
    finally:
        conn.close()
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
    import psycopg
    from memman.store.postgres import _store_schema

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
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
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


def migrate_store(
        *, source_dir: str, dsn: str, store: str,
        dry_run: bool = False,
        state: SchemaState = SchemaState.ABSENT) -> MigrateResult:
    """Migrate a single SQLite store into Postgres schema `store_<store>`.

    Returns counts of rows moved per table. On `dry_run=True`, reports
    the counts that would be moved without writing. When `state` is
    `EMPTY` or `POPULATED` the existing schema is dropped and
    recreated; when `ABSENT` the schema is created fresh. All inserts
    run inside one Postgres transaction; on any failure the
    transaction rolls back and `MigrateError` is raised. The caller
    is responsible for the outer drain.lock guard and the
    confirmation prompt.
    """
    import sqlite3

    import psycopg
    from memman.store.postgres import _check_identifier, _store_schema

    _check_identifier(store)
    schema = _store_schema(store)

    sqlite_path = Path(source_dir) / 'memman.db'
    if not sqlite_path.exists():
        raise MigrateError(
            f'source SQLite database not found: {sqlite_path}')

    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = None
    try:
        n_insights = sqlite_conn.execute(
            'select count(*) from insights').fetchone()[0]
        n_edges = sqlite_conn.execute(
            'select count(*) from edges').fetchone()[0]
        n_oplog = sqlite_conn.execute(
            'select count(*) from oplog').fetchone()[0]
        n_meta = sqlite_conn.execute(
            'select count(*) from meta').fetchone()[0]
        if dry_run:
            sqlite_conn.close()
            return MigrateResult(
                store=store, schema=schema,
                insights=n_insights, edges=n_edges,
                oplog=n_oplog, meta=n_meta, dry_run=True)

        from scripts.import_sqlite_to_postgres import _ensure_schema
        from scripts.import_sqlite_to_postgres import _import_edges
        from scripts.import_sqlite_to_postgres import _import_insights
        from scripts.import_sqlite_to_postgres import _import_meta
        from scripts.import_sqlite_to_postgres import _import_oplog

        pg_conn = psycopg.connect(dsn, autocommit=False)
        try:
            from pgvector.psycopg import register_vector
            register_vector(pg_conn)

            if state in (SchemaState.EMPTY, SchemaState.POPULATED):
                with pg_conn.cursor() as cur:
                    cur.execute(
                        f'drop schema if exists {schema} cascade')
            _ensure_schema(pg_conn, schema)
            ins_count = _import_insights(sqlite_conn, pg_conn, schema)
            edge_count = _import_edges(sqlite_conn, pg_conn, schema)
            oplog_count = _import_oplog(sqlite_conn, pg_conn, schema)
            meta_count = _import_meta(sqlite_conn, pg_conn, schema)
            pg_conn.commit()
            return MigrateResult(
                store=store, schema=schema,
                insights=ins_count, edges=edge_count,
                oplog=oplog_count, meta=meta_count, dry_run=False)
        except Exception as exc:
            pg_conn.rollback()
            if isinstance(exc, MigrateError):
                raise
            raise MigrateError(
                f'migration of store {store!r} failed: '
                f'{type(exc).__name__}: {exc}') from exc
        finally:
            pg_conn.close()
    finally:
        sqlite_conn.close()
