"""write_lock concurrent reindex on Postgres.

The contract under test: while an agent holds the per-store
`write_lock` advisory key (transaction-scoped via
`pg_advisory_xact_lock`), a concurrent edge insert from a *different*
agent's connection still commits, and the inserted edge survives the
reindex window.

This proves that `write_lock`'s purpose -- serializing two reindex
passes against each other -- does not also serialize plain edge
writes, so no remembered work is lost during reindex.
"""

from __future__ import annotations

import threading
import time

import pytest
from memman.store.model import Edge, Insight
from memman.store.postgres import drop_postgres_store, open_postgres_backend
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_concurrent_remember_during_reindex_preserves_edge(
        pg_dsn, request):
    """A plain edge insert during reindex window survives commit.

    Sequence: agent A opens a transaction and acquires `write_lock`,
    blocks on a release event. Agent B (separate connection) inserts
    a manual edge and commits. Agent A then releases its transaction
    (commit). After both finish, the edge is fetched back via a
    fresh connection and matches.
    """
    store = _safe(request.node.name)
    try:
        drop_postgres_store(store, pg_dsn)
    except Exception:
        pass

    backend_seed = open_postgres_backend(store, pg_dsn)
    try:
        backend_seed.nodes.insert(Insight(
            id='wl-src', content='source node',
            importance=3, source='user'))
        backend_seed.nodes.insert(Insight(
            id='wl-dst', content='destination node',
            importance=3, source='user'))
        backend_seed._conn.commit()
    finally:
        backend_seed.close()

    a_acquired = threading.Event()
    a_release = threading.Event()
    b_done = threading.Event()
    errors: list[Exception] = []

    def agent_a_holds_write_lock() -> None:
        try:
            backend = open_postgres_backend(store, pg_dsn)
            try:
                with backend.transaction():
                    with backend.write_lock('reindex'):
                        a_acquired.set()
                        a_release.wait(timeout=10)
            finally:
                backend.close()
        except Exception as exc:
            errors.append(exc)

    def agent_b_inserts_edge() -> None:
        try:
            assert a_acquired.wait(timeout=5), (
                'agent A did not acquire write_lock')
            backend = open_postgres_backend(store, pg_dsn)
            try:
                backend.edges.upsert(Edge(
                    source_id='wl-src',
                    target_id='wl-dst',
                    edge_type='manual',
                    weight=1.0,
                    metadata={'origin': 'concurrent-remember'}))
                backend._conn.commit()
                b_done.set()
            finally:
                backend.close()
        except Exception as exc:
            errors.append(exc)

    thread_a = threading.Thread(target=agent_a_holds_write_lock)
    thread_b = threading.Thread(target=agent_b_inserts_edge)
    thread_a.start()
    thread_b.start()

    assert b_done.wait(timeout=10), (
        'agent B should commit its edge while A holds write_lock'
        ' -- write_lock must not block plain edge inserts')

    a_release.set()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert not errors, f'thread errors: {errors}'

    verify = open_postgres_backend(store, pg_dsn)
    try:
        edges = verify.edges.get_neighborhood(
            seed_id='wl-src', depth=1, edge_filter='manual')
        target_ids = [hit[0] for hit in edges]
        assert 'wl-dst' in target_ids, (
            f'concurrent edge must survive write_lock window;'
            f' got {target_ids}')
    finally:
        verify.close()
        drop_postgres_store(store, pg_dsn)


def test_two_reindex_holders_serialize_via_write_lock(pg_dsn, request):
    """Two `write_lock('reindex')` waiters must serialize.

    Agent A acquires the xact-scoped advisory lock and parks. Agent B
    tries the same key in a fresh transaction -- this should block
    until A commits. Asserts B is still waiting after a short window
    and completes promptly after A releases.
    """
    store = _safe(request.node.name)
    try:
        drop_postgres_store(store, pg_dsn)
    except Exception:
        pass
    backend_seed = open_postgres_backend(store, pg_dsn)
    backend_seed.close()

    a_acquired = threading.Event()
    a_release = threading.Event()
    b_acquired = threading.Event()
    errors: list[Exception] = []

    def agent_a() -> None:
        try:
            backend = open_postgres_backend(store, pg_dsn)
            try:
                with backend.transaction():
                    with backend.write_lock('reindex'):
                        a_acquired.set()
                        a_release.wait(timeout=15)
            finally:
                backend.close()
        except Exception as exc:
            errors.append(exc)

    def agent_b() -> None:
        try:
            assert a_acquired.wait(timeout=5)
            backend = open_postgres_backend(store, pg_dsn)
            try:
                with backend.transaction():
                    with backend.write_lock('reindex'):
                        b_acquired.set()
            finally:
                backend.close()
        except Exception as exc:
            errors.append(exc)

    thread_a = threading.Thread(target=agent_a)
    thread_b = threading.Thread(target=agent_b)
    thread_a.start()
    thread_b.start()

    a_acquired.wait(timeout=5)
    time.sleep(0.5)
    assert not b_acquired.is_set(), (
        'agent B should be blocked while A holds write_lock')

    a_release.set()
    assert b_acquired.wait(timeout=5), (
        'agent B should acquire promptly after A commits')

    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    try:
        assert not errors, f'thread errors: {errors}'
    finally:
        drop_postgres_store(store, pg_dsn)
