"""SQLite -> Postgres store migration.

Backs the `memman migrate` CLI command. Wraps the streaming logic in
`scripts/import_sqlite_to_postgres.py` with CLI orchestration: DSN
preflight, drain-lock guard, per-store transaction, dry-run mode, and
a fail-closed confirmation gate.

Migration is intentionally one-way: SQLite source is read-only;
Postgres destination uses `ON CONFLICT (id) DO NOTHING` on the
`insights` insert path so an interrupted run can be re-run safely.
The shared drain.lock is held for the duration of the migrate
command so a scheduler-fired drain cannot race the SQLite reader.

This module does NOT flip `MEMMAN_BACKEND` in the env file; the
operator does that explicitly after verifying the migrate run with
`memman doctor`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

logger = logging.getLogger('memman.migrate')


class MigrateError(Exception):
    """Migration aborted because a precondition or invariant failed."""


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

            cur.execute(
                "select 1 from pg_extension where extname = 'vector'")
            row = cur.fetchone()
            if row is None:
                raise MigrateError(
                    'pgvector extension is not installed in the target '
                    'database; run `create extension vector;` as a '
                    'superuser first')
            checks['pgvector_installed'] = True

            cur.execute(
                "select has_database_privilege(current_user,"
                " current_database(), 'CREATE')")
            checks['create_schema_privilege'] = bool(cur.fetchone()[0])
            if not checks['create_schema_privilege']:
                raise MigrateError(
                    'current postgres role lacks create schema '
                    'privilege on the target database')
    finally:
        conn.close()
    return checks


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
        overwrite_schema: bool = False) -> MigrateResult:
    """Migrate a single SQLite store into Postgres schema `store_<store>`.

    Returns counts of rows moved per table. On `dry_run=True`,
    reports counts that would be moved without writing.

    All inserts run inside one Postgres transaction; on any failure
    the transaction rolls back and `MigrateError` is raised. The
    caller is responsible for the outer drain.lock guard.
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

            if overwrite_schema:
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
