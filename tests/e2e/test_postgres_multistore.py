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

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.store.model import Insight
from memman.store.postgres import _store_schema, drop_postgres_store
from memman.store.postgres import open_postgres_backend
from memman.store.sqlite import open_sqlite_backend
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_fresh_init_creates_schema_with_all_tables(pg_dsn, request):
    """Verify opening a new store creates exactly the per-store tables.

    Mutation: a baseline table (`insights`, `meta`, `oplog`,
        `worker_runs`) missing from the DDL, or one created outside
        the store schema.
    Oracle: the sorted `pg_tables` names in the store schema.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    drop_postgres_store(store, pg_dsn)
    backend = open_postgres_backend(store, pg_dsn)
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'select tablename from pg_tables'
                    ' where schemaname = %s order by tablename',
                    (schema,))
                tables = [r[0] for r in cur.fetchall()]
        assert tables == [
            'insights', 'meta', 'oplog', 'worker_runs']
    finally:
        backend.close()
        drop_postgres_store(store, pg_dsn)


def test_drop_store_a_does_not_affect_store_b(pg_dsn, request):
    """Verify dropping store A leaves store B's schema and rows intact.

    Mutation: a drop that cascades across stores, such as one
        that drops every store schema or a shared one.
    Oracle: `pg_namespace` counts (A gone, B present) and B's
        inserted row read back after the drop.
    """
    base = _safe(request.node.name)[:36]
    store_a = f'{base}_a'
    store_b = f'{base}_b'
    for s in (store_a, store_b):
        drop_postgres_store(s, pg_dsn)

    a = open_postgres_backend(store_a, pg_dsn)
    b = open_postgres_backend(store_b, pg_dsn)
    try:
        a.nodes.insert(Insight(id='a-1', content='only in A'))
        b.nodes.insert(Insight(id='b-1', content='only in B'))
        a._conn.commit()
        b._conn.commit()
    finally:
        a.close()
        b.close()

    drop_postgres_store(store_a, pg_dsn)

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 0, (
                'store_a schema should be dropped')
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
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
    """Verify SQLite and Postgres backends return the same inserted content.

    Mutation: either backend truncating or altering `content` on
        insert or get.
    Oracle: the literal content string inserted through both.
    """
    sqlite_data = str(tmp_path / 'memman_sqlite')
    Path(sqlite_data).mkdir(parents=True, exist_ok=True)
    sqlite_backend = open_sqlite_backend('parity', sqlite_data)

    pg_store = _safe(request.node.name)
    drop_postgres_store(pg_store, pg_dsn)
    pg_backend = open_postgres_backend(pg_store, pg_dsn)

    try:
        ins = Insight(id='parity-1', content='same content both ways')
        sqlite_backend.nodes.insert(ins)
        pg_backend.nodes.insert(ins)
        pg_backend._conn.commit()

        sq = sqlite_backend.nodes.get('parity-1')
        pg = pg_backend.nodes.get('parity-1')
        assert sq is not None
        assert pg is not None
        assert sq.content == pg.content == 'same content both ways'
    finally:
        sqlite_backend.close()
        pg_backend.close()
        drop_postgres_store(pg_store, pg_dsn)
