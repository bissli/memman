"""Postgres + pgvector test fixture.

Drives a real pgvector container so tests can validate vector
operations, HNSW behaviour, advisory-lock contention, and
search_path semantics that no SQLite-only mock can exercise.

Container: pgvector/pgvector:pg16 (PostgreSQL 16.x + pgvector 0.8.x,
multi-arch). Tests gate on the `postgres` pytest marker; SQLite-only
`make test` runs are unaffected.

Layout:

- `pgvector_docker` (session): starts the pgvector container, ensures
  the `vector` extension is loaded, returns a connection-info dict.
- `pg_dsn` (session): the libpq DSN string for the running container.
- `pg_conn` (function): yields a fresh psycopg connection. Drops and
  recreates the `store_test` schema per test.
- `terminate_pg_connections`: cleanup helper (mirrors the database/
  reference fixture).
- `wait_for(condition, timeout)`: polling helper (jobsync verbatim).
- `clean_tables(*names)`: decorator that TRUNCATEs named tables in
  the `store_test` schema before the test body runs.
- `drain_connection_pair`: spawns two non-pooled psycopg connections
  for advisory-lock contention tests (memman-native equivalent of
  jobsync's `cluster()` since memman drains are one-shot, not
  long-running workers).
- `simulate_drain_connection_drop`: closes a connection without
  releasing its advisory lock (drives the "released on connection
  close" contract test for `pg_try_advisory_lock`).
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

logger = logging.getLogger('memman.tests.fixtures.postgres')

PGVECTOR_IMAGE = 'pgvector/pgvector:pg16'
SCHEMA = 'store_test'


@pytest.fixture(scope='session')
def pgvector_docker(request) -> dict[str, Any]:
    """Session-scoped pgvector/pgvector:pg16 container.

    Runs `CREATE EXTENSION IF NOT EXISTS vector` once on startup,
    then yields a dict with the DSN + container metadata. All
    function-scoped tests share the same container; per-test
    isolation comes from the `pg_conn` fixture's schema reset.
    """
    container = PostgresContainer(image=PGVECTOR_IMAGE)
    container.start()
    try:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(5432))
        dbname = container.dbname
        user = container.username
        password = container.password
        dsn = (
            f'host={host} port={port} dbname={dbname}'
            f' user={user} password={password}')
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                assert cur.fetchone() is not None, (
                    'pgvector extension failed to install')
        info = {
            'dsn': dsn,
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password,
            }
        logger.info(
            f'pgvector container started at {host}:{port}'
            f' dbname={dbname}')

        def finalizer() -> None:
            try:
                container.stop()
                logger.info('pgvector container stopped')
            except Exception as e:
                logger.warning(f'Error stopping container: {e}')

        request.addfinalizer(finalizer)
        return info
    except Exception:
        try:
            container.stop()
        except Exception:
            pass
        raise


@pytest.fixture(scope='session')
def pg_dsn(pgvector_docker: dict[str, Any]) -> str:
    """Convenience: the libpq DSN string for the session container."""
    return pgvector_docker['dsn']


@pytest.fixture
def pg_conn(pg_dsn: str) -> Iterator[psycopg.Connection]:
    """Function-scoped psycopg connection with a fresh `store_test` schema.

    Drops and recreates the schema per test, then sets `search_path`
    to `store_test, public` so tests don't have to qualify table
    names. The `vector` extension lives in the database (created at
    session start) and stays in place.
    """
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {SCHEMA} CASCADE')
            cur.execute(f'CREATE SCHEMA {SCHEMA}')
            cur.execute(f'SET search_path = {SCHEMA}, public')
        try:
            yield conn
        finally:
            terminate_pg_connections(conn)


def terminate_pg_connections(conn: psycopg.Connection) -> None:
    """Force-terminate all other connections to the test database.

    Mirrors database/tests/fixtures/postgres.py's
    `terminate_postgres_connections`. Useful in teardown to keep a
    leaked test connection from blocking the next session run.
    """
    try:
        if conn.info.transaction_status != 0:
            try:
                conn.rollback()
            except Exception:
                pass
        with conn.cursor() as cur:
            cur.execute(
                'SELECT pg_terminate_backend(pid)'
                ' FROM pg_stat_activity'
                ' WHERE datname = current_database()'
                '   AND pid <> pg_backend_pid()')
    except Exception as e:
        logger.warning(f'Failed to terminate connections: {e}')


def wait_for(
        condition: Callable[[], bool],
        timeout_sec: float = 5.0,
        check_interval: float = 0.1) -> bool:
    """Poll `condition` until it returns True or timeout elapses.

    Borrowed verbatim from jobsync/tests/fixtures.py. Used in
    advisory-lock contention tests to wait for a second connection
    to detect that a lock has been acquired by the first.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(check_interval)
    return False


def clean_tables(*table_names: str) -> Callable[[Callable], Callable]:
    """Decorator that TRUNCATEs named tables in `store_test` before the test.

    Adapted from jobsync's clean_tables; the only change is the
    schema-qualified TRUNCATE (jobsync looked up table names via a
    schema registry).

    Usage:

        @clean_tables('insights', 'edges')
        def test_something(pg_conn):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            conn = kwargs.get('pg_conn')
            if conn is None and args:
                for arg in args:
                    if isinstance(arg, psycopg.Connection):
                        conn = arg
                        break
            if conn is None:
                raise RuntimeError(
                    '@clean_tables requires a pg_conn fixture')
            qualified = ', '.join(f'{SCHEMA}.{t}' for t in table_names)
            with conn.cursor() as cur:
                cur.execute(f'TRUNCATE {qualified} RESTART IDENTITY CASCADE')
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def drain_connection_pair(
        dsn: str) -> Iterator[tuple[psycopg.Connection, psycopg.Connection]]:
    """Open two non-pooled psycopg connections for contention tests.

    memman-native equivalent of jobsync's `cluster()`: memman drains
    are one-shot function calls, not long-running workers, so the
    contention primitive is "two raw connections" rather than "two
    worker objects." Both connections close cleanly on exit.
    """
    a = psycopg.connect(dsn, autocommit=True)
    b = psycopg.connect(dsn, autocommit=True)
    try:
        yield a, b
    finally:
        for c in (a, b):
            try:
                c.close()
            except Exception:
                pass


def simulate_drain_connection_drop(conn: psycopg.Connection) -> None:
    """Close a connection without releasing its advisory locks.

    Drives the `test_advisory_lock_released_on_connection_close`
    contract: Postgres releases session-level advisory locks when
    the underlying connection closes, even without an explicit
    `pg_advisory_unlock`. This is the crash-recovery mechanism the
    drain-lock contract relies on.
    """
    try:
        conn.close()
    except Exception:
        pass
