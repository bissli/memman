"""Phase 5 e2e -- wipe-and-recreate (drop schema + queue purge).

Covers Phase 5 e2e scenario "Wipe-and-recreate (drop schema + queue
purge)". `PostgresCluster.drop_store` performs both operations
atomically: `DROP SCHEMA ... CASCADE` for the per-store schema and
`DELETE FROM queue.queue WHERE store = %s` for any pending work.
After a wipe, reopening the same store name yields a fresh schema
with no residue.
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.model import Insight
from memman.store.postgres import PostgresCluster, PostgresQueueBackend
from memman.store.postgres import _store_schema

from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]
def test_drop_store_purges_pending_queue_rows(pg_dsn, request):
    """drop_store removes both schema rows and queue.queue rows.

    Enqueue a row for store A, confirm it lands, drop A, then assert
    `queue.queue WHERE store = 'A'` is empty. A second store B's
    queue rows must be untouched.
    """
    base = _safe(request.node.name)[:36]
    store_a = f'{base}_a'
    store_b = f'{base}_b'

    cluster = PostgresCluster(dsn=pg_dsn)
    for s in (store_a, store_b):
        try:
            cluster.drop_store(store=s, data_dir='')
        except Exception:
            pass

    a = cluster.open(store=store_a, data_dir='')
    b = cluster.open(store=store_b, data_dir='')
    a.close()
    b.close()

    queue = PostgresQueueBackend(dsn=pg_dsn)
    queue.enqueue(store=store_a, op='probe', payload='wipe-a-1')
    queue.enqueue(store=store_a, op='probe', payload='wipe-a-2')
    queue.enqueue(store=store_b, op='probe', payload='wipe-b-1')

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT count(*) FROM queue.queue WHERE store = %s',
                (store_a,))
            assert cur.fetchone()[0] == 2

    cluster.drop_store(store=store_a, data_dir='')

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 0
            cur.execute(
                'SELECT count(*) FROM queue.queue WHERE store = %s',
                (store_a,))
            assert cur.fetchone()[0] == 0, (
                'drop_store must purge pending queue rows for that store')
            cur.execute(
                'SELECT count(*) FROM queue.queue WHERE store = %s',
                (store_b,))
            assert cur.fetchone()[0] == 1, (
                'queue rows for unrelated stores must not be purged')

    cluster.drop_store(store=store_b, data_dir='')


def test_recreate_after_drop_yields_empty_schema(pg_dsn, request):
    """After drop_store + open, the schema has zero data rows.

    Insert an Insight, drop the store, reopen, assert the new
    `insights` table is empty (a wipe-and-recreate cycle leaves no
    residual rows even though the schema name is reused).
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass

    first = cluster.open(store=store, data_dir='')
    try:
        first.nodes.insert(Insight(
            id='pre-wipe', content='will be wiped',
            importance=3, source='user'))
        first._conn.commit()
        assert first.nodes.get('pre-wipe') is not None
    finally:
        first.close()

    cluster.drop_store(store=store, data_dir='')

    second = cluster.open(store=store, data_dir='')
    try:
        assert second.nodes.get('pre-wipe') is None, (
            'recreated schema should not contain pre-wipe rows')
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT count(*) FROM {_store_schema(store)}.insights')
                assert cur.fetchone()[0] == 0
    finally:
        second.close()
        cluster.drop_store(store=store, data_dir='')
