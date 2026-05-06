"""Wipe-and-recreate on Postgres (drop schema).

`drop_postgres_store(store, dsn)` runs `DROP SCHEMA ... CASCADE`
for the per-store schema. After a wipe, reopening the same store
name yields a fresh schema with no residue. The cross-store work
queue lives in SQLite under the per-store routing model, so its
purge happens via `factory.drop_store` (covered separately).
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.model import Insight
from memman.store.postgres import _store_schema, drop_postgres_store
from memman.store.postgres import open_postgres_backend

from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_drop_store_removes_schema(pg_dsn, request):
    """drop_postgres_store removes the per-store schema entirely.

    Open store A, confirm its schema exists in pg_namespace, drop A,
    confirm the namespace row is gone. A sibling store B is untouched.
    """
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
    a.close()
    b.close()

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 1
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_b),))
            assert cur.fetchone()[0] == 1

    drop_postgres_store(store_a, pg_dsn)

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 0
            cur.execute(
                'SELECT count(*) FROM pg_namespace WHERE nspname = %s',
                (_store_schema(store_b),))
            assert cur.fetchone()[0] == 1, (
                'sibling store schema must not be dropped')

    drop_postgres_store(store_b, pg_dsn)


def test_recreate_after_drop_yields_empty_schema(pg_dsn, request):
    """After drop + open, the schema has zero data rows.

    Insert an Insight, drop the store, reopen, assert the new
    `insights` table is empty (a wipe-and-recreate cycle leaves no
    residual rows even though the schema name is reused).
    """
    store = _safe(request.node.name)
    try:
        drop_postgres_store(store, pg_dsn)
    except Exception:
        pass

    first = open_postgres_backend(store, pg_dsn)
    try:
        first.nodes.insert(Insight(
            id='pre-wipe', content='will be wiped',
            importance=3, source='user'))
        first._conn.commit()
        assert first.nodes.get('pre-wipe') is not None
    finally:
        first.close()

    drop_postgres_store(store, pg_dsn)

    second = open_postgres_backend(store, pg_dsn)
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
        drop_postgres_store(store, pg_dsn)
