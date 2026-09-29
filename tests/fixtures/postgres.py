"""Postgres + pgvector test fixture.

Drives a real pgvector container so tests can validate vector
operations, HNSW behavior, advisory-lock contention, and
search_path semantics that no SQLite-only mock can exercise.

The container image is `PGVECTOR_IMAGE`. Tests gate on the `postgres`
pytest marker, so SQLite-only `make test` runs skip it.

Layout:

- `pgvector_docker` (session): starts the pgvector container, ensures
  the `vector` extension is loaded, returns a connection-info dict.
- `pg_dsn` (session): the libpq DSN string for the running container.
- `pg_conn` (function): yields a fresh psycopg connection. Drops and
  recreates the `store_test` schema per test.
- `terminate_pg_connections`: cleanup helper.
- `wait_for(condition, timeout)`: polling helper.
- `connection_pair`: spawns two non-pooled psycopg connections for
  advisory-lock contention tests.
- `simulate_connection_drop`: closes a connection without releasing
  its advisory lock (drives the "released on connection close"
  contract test for `pg_try_advisory_lock`).
"""

from __future__ import annotations

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

    Runs `create extension if not exists vector` once on startup,
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
                cur.execute('create extension if not exists vector')
                cur.execute(
                    "select 1 from pg_extension where extname = 'vector'")
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
        except Exception as e:
            logger.warning(f'Error stopping container: {e}')
        raise


@pytest.fixture(scope='session')
def pg_dsn(pgvector_docker: dict[str, Any]) -> str:
    """The libpq DSN string for the session container.
    """
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
            cur.execute(f'drop schema if exists {SCHEMA} cascade')
            cur.execute(f'create schema {SCHEMA}')
            cur.execute(f'set search_path = {SCHEMA}, public')
        try:
            yield conn
        finally:
            terminate_pg_connections(conn)


def terminate_pg_connections(conn: psycopg.Connection) -> None:
    """Force-terminate all other connections to the test database.

    Teardown uses it so a leaked test connection cannot block the next
    session run.
    """
    try:
        if conn.info.transaction_status != 0:
            conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                'select pg_terminate_backend(pid)'
                ' from pg_stat_activity'
                ' where datname = current_database()'
                '   and pid <> pg_backend_pid()')
    except Exception as e:
        logger.warning(f'Failed to terminate connections: {e}')


def wait_for(
        condition: Callable[[], bool],
        timeout_sec: float = 5.0,
        check_interval: float = 0.1) -> bool:
    """Poll `condition` until it returns True or timeout elapses.

    Used in advisory-lock contention tests to wait for a second
    connection to detect that a lock has been acquired by the first.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            if condition():
                return True
        except psycopg.Error:
            pass
        time.sleep(check_interval)
    return False


@contextmanager
def connection_pair(
        dsn: str) -> Iterator[tuple[psycopg.Connection, psycopg.Connection]]:
    """Open two non-pooled psycopg connections for contention tests.

    The contention primitive is two raw connections. Both close on exit.
    """
    a = psycopg.connect(dsn, autocommit=True)
    b = psycopg.connect(dsn, autocommit=True)
    try:
        yield a, b
    finally:
        for c in (a, b):
            c.close()


def simulate_connection_drop(conn: psycopg.Connection) -> None:
    """Close a connection without releasing its advisory locks.

    Drives the `test_advisory_lock_released_on_connection_close`
    contract: Postgres releases session-level advisory locks when
    the underlying connection closes, even without an explicit
    `pg_advisory_unlock`. `reembed_lock` and `swap_lock` rely on
    this to recover from a crashed holder.
    """
    conn.close()
