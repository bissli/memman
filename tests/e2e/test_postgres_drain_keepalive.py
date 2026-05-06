"""Hung-drain detection via TCP keepalives.

The drain-lock contract: `Backend.drain_lock` opens a dedicated
psycopg connection with `keepalives=1` + `keepalives_idle=30`, holds
a session-scoped `pg_try_advisory_lock`, and relies on connection
close (clean or dropped) to release the lock so another agent can
claim a hung drain.

Two guarantees covered:

- A hung-worker simulation (closing the connection without explicit
  unlock) releases the lock promptly enough that another agent
  acquires within the keepalive window.
- The drain-lock connection has `keepalives_idle=30` set.
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.postgres import PostgresCluster, _open_connection
from memman.store.postgres import _store_schema
from tests.e2e.conftest import _safe
from tests.fixtures.postgres import wait_for

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_hung_worker_releases_lock_within_keepalive_window(
        pg_dsn, request):
    """A dropped drain-lock connection releases the lock to a contender.

    Mirrors the production crash path: drain holder's `drain_lock`
    contextmanager closes its dedicated connection on exit. While
    `pg_try_advisory_lock` would normally also be released by an
    explicit `pg_advisory_unlock`, this test simulates the kernel-
    detected drop case by closing the holder's underlying socket
    without going through the contextmanager's clean unlock path.

    The contender then polls -- under typical CI load the lock is
    available within a couple of seconds, well inside the
    `tcp_keepalives_idle + tcp_keepalives_interval` ceiling.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend_seed = cluster.open(store=store, data_dir='')
    backend_seed.close()

    schema = _store_schema(store)
    lock_key = abs(hash(f'memman_drain:{store}')) & 0x7FFFFFFFFFFFFFFF

    holder = _open_connection(pg_dsn, autocommit=True, keepalives=True)
    with holder.cursor() as cur:
        cur.execute('SELECT pg_try_advisory_lock(%s)', (lock_key,))
        assert cur.fetchone()[0] is True, (
            'expected to acquire drain lock on first try')

    with psycopg.connect(pg_dsn, autocommit=True) as observer:
        with observer.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(%s)', (lock_key,))
            assert cur.fetchone()[0] is False, (
                'observer should not acquire while holder is alive')

    holder.close()

    contender = cluster.open(store=store, data_dir='')
    try:
        def _can_acquire() -> bool:
            with contender.drain_lock(store) as got:
                return bool(got)

        assert wait_for(_can_acquire, timeout_sec=5.0), (
            'contender should acquire within keepalive window after'
            ' holder connection drop')
    finally:
        contender.close()
        cluster.drop_store(store=store, data_dir='')


def test_drain_lock_connection_has_keepalives_idle_30(pg_dsn, request):
    """The drain-lock connection sets `tcp_keepalives_idle=30`.

    Any connection used to hold the drain lock must opt into TCP
    keepalives with `keepalives_idle=30`, otherwise a hung worker on
    a dead network path holds the lock indefinitely. Inspects the
    live connection's settings via `SHOW`.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend = cluster.open(store=store, data_dir='')

    sniffed: dict[str, str] = {}

    try:
        original = _open_connection

        def spy(dsn, *, autocommit=False, keepalives=False):
            conn = original(
                dsn, autocommit=autocommit, keepalives=keepalives)
            if keepalives:
                with conn.cursor() as cur:
                    cur.execute('SHOW tcp_keepalives_idle')
                    sniffed['idle'] = cur.fetchone()[0]
                    cur.execute('SHOW tcp_keepalives_interval')
                    sniffed['interval'] = cur.fetchone()[0]
            return conn

        from memman.store import postgres as pg_mod
        original_attr = pg_mod._open_connection
        pg_mod._open_connection = spy
        try:
            with backend.drain_lock(store) as got:
                assert got is True
        finally:
            pg_mod._open_connection = original_attr
    finally:
        backend.close()
        cluster.drop_store(store=store, data_dir='')

    assert sniffed.get('idle'), (
        'drain_lock did not call _open_connection with keepalives=True')
    assert sniffed['idle'] != '0', (
        f"tcp_keepalives_idle should be non-zero on drain conn;"
        f" got {sniffed['idle']}")
