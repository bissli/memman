"""Fresh init, multi-store isolation, cross-backend parity.

Drives the per-store backend factory against the testcontainers
pgvector session container:

- Fresh init: schema applied cleanly on a new store.
- Multi-store isolation: dropping store A does not affect store B.
- Cross-backend parity smoke: a SQLite Backend and a Postgres
  Backend opened against the same fingerprint accept the same
  insert/get/get-by-source verbs.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from memman.store.model import Insight
from memman.store.postgres import _store_schema, drop_postgres_store
from memman.store.postgres import open_postgres_backend
from memman.store.sqlite import open_sqlite_backend
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_fresh_init_creates_schema_with_all_tables(pg_dsn, request):
    """Cluster.open on a never-seen store creates all four tables."""
    store = _safe(request.node.name)
    schema = _store_schema(store)
    try:
        drop_postgres_store(store, pg_dsn)
    except Exception:
        pass
    backend = open_postgres_backend(store, pg_dsn)
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT tablename FROM pg_tables'
                    ' WHERE schemaname = %s ORDER BY tablename',
                    (schema,))
                tables = [r[0] for r in cur.fetchall()]
        assert tables == ['edges', 'insights', 'meta', 'oplog']
    finally:
        backend.close()
        drop_postgres_store(store, pg_dsn)


def test_drop_store_a_does_not_affect_store_b(pg_dsn, request):
    """drop_store(A) leaves store B intact (data + schema)."""
    base = _safe(request.node.name)[:36]
    store_a = f'{base}_a'
    store_b = f'{base}_b'
    for s in (store_a, store_b):
        try:
            drop_postgres_store(s, pg_dsn)
        except Exception:
            pass

    a = open_postgres_backend(store_a, pg_dsn)
    b = open_postgres_backend(store_b, pg_dsn)
    try:
        a.nodes.insert(Insight(
            id='a-1', content='only in A', importance=3, source='user'))
        b.nodes.insert(Insight(
            id='b-1', content='only in B', importance=3, source='user'))
        a._conn.commit()
        b._conn.commit()
    finally:
        a.close()
        b.close()

    drop_postgres_store(store_a, pg_dsn)

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 0, (
                'store_a schema should be dropped')
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_b),))
            assert cur.fetchone()[0] == 1, (
                'store_b schema must survive drop of A')

    b2 = open_postgres_backend(store_b, pg_dsn)
    try:
        survivor = b2.nodes.get('b-1')
        assert survivor is not None, (
            'store B data must survive store A drop')
        assert survivor.content == 'only in B'
    finally:
        b2.close()
        drop_postgres_store(store_b, pg_dsn)


def test_cross_backend_parity_insert_and_get(pg_dsn, tmp_path, request):
    """Same Insight inserted via SQLite + Postgres returns equal content
    on get(). Smoke test for the cross-backend parity matrix.
    """
    sqlite_data = str(tmp_path / 'memman_sqlite')
    Path(sqlite_data).mkdir(parents=True, exist_ok=True)
    sqlite_backend = open_sqlite_backend('parity', sqlite_data)

    pg_store = _safe(request.node.name)
    try:
        drop_postgres_store(pg_store, pg_dsn)
    except Exception:
        pass
    pg_backend = open_postgres_backend(pg_store, pg_dsn)

    try:
        ins = Insight(
            id='parity-1', content='same content both ways',
            importance=4, source='user')
        sqlite_backend.nodes.insert(ins)
        pg_backend.nodes.insert(ins)
        pg_backend._conn.commit()

        sq = sqlite_backend.nodes.get('parity-1')
        pg = pg_backend.nodes.get('parity-1')
        assert sq is not None
        assert pg is not None
        assert sq.content == pg.content == 'same content both ways'
        assert sq.importance == pg.importance == 4
        assert sq.source == pg.source == 'user'
    finally:
        sqlite_backend.close()
        pg_backend.close()
        drop_postgres_store(pg_store, pg_dsn)
