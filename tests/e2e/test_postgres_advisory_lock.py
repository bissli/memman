"""Phase 5 e2e -- advisory-lock contention.

Drives the per-store `Backend.drain_lock` advisory lock from two
threads gated on a `threading.Barrier` and asserts exactly one
acquires the lock. Phase 5 gate item 4 explicitly names this test.

Threads (not processes) are sufficient because `drain_lock` opens
its own dedicated psycopg connection inside the contextmanager;
each thread gets a distinct backend session.
"""

from __future__ import annotations

import threading

import pytest
from memman.store.postgres import PostgresCluster

from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]
def test_only_one_drain_wins_per_store(pg_dsn, request):
    """Two threads racing on `drain_lock`: exactly one acquires.

    Both threads open their own `PostgresBackend`, hit a barrier, and
    call `drain_lock` simultaneously. The first holder keeps the lock
    while the second runs its `pg_try_advisory_lock`, so the second
    must observe `acquired=False`.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass

    setup = cluster.open(store=store, data_dir='')
    setup.close()

    barrier = threading.Barrier(2)
    holder_acquired = threading.Event()
    release_holder = threading.Event()
    results: dict[str, bool] = {}

    def holder() -> None:
        backend = cluster.open(store=store, data_dir='')
        try:
            barrier.wait(timeout=5)
            with backend.drain_lock(store) as got:
                results['holder'] = bool(got)
                if got:
                    holder_acquired.set()
                release_holder.wait(timeout=5)
        finally:
            backend.close()

    def contender() -> None:
        backend = cluster.open(store=store, data_dir='')
        try:
            barrier.wait(timeout=5)
            assert holder_acquired.wait(timeout=5), (
                'holder thread never reported lock acquisition')
            with backend.drain_lock(store) as got:
                results['contender'] = bool(got)
        finally:
            backend.close()

    h = threading.Thread(target=holder)
    c = threading.Thread(target=contender)
    h.start()
    c.start()
    c.join(timeout=10)
    release_holder.set()
    h.join(timeout=10)

    assert not h.is_alive(), 'holder thread did not finish'
    assert not c.is_alive(), 'contender thread did not finish'
    try:
        assert results == {'holder': True, 'contender': False}, (
            f'expected exactly one winner; got {results}')
    finally:
        cluster.drop_store(store=store, data_dir='')


def test_lock_reacquirable_after_holder_exits(pg_dsn, request):
    """Once the holder context exits, a fresh acquire succeeds.

    Asserts the contextmanager's release path (advisory_unlock +
    connection close) leaves the lock available for the next acquirer.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend = cluster.open(store=store, data_dir='')
    try:
        with backend.drain_lock(store) as first:
            assert first is True
        with backend.drain_lock(store) as second:
            assert second is True
    finally:
        backend.close()
        cluster.drop_store(store=store, data_dir='')
