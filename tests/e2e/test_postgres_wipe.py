"""Wipe-and-recreate on Postgres (drop schema).

`drop_postgres_store(store, dsn)` runs `drop schema ... cascade`
for the per-store schema. After a wipe, reopening the same store
name yields a fresh schema with no residue. The cross-store work
queue lives in SQLite under the per-store routing model, so its
purge happens via `factory.drop_store` (covered separately).
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.store.model import Insight
from memman.store.postgres import _store_schema, drop_postgres_store
from memman.store.postgres import open_postgres_backend
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_drop_store_removes_schema(pg_dsn, request):
    """Verify `drop_postgres_store` removes only the named store's schema.

    Mutation: a drop that leaves the schema behind, or one that
        removes the sibling store's schema too.
    Oracle: `pg_namespace` counts before and after the drop.
    """
    base = _safe(request.node.name)[:36]
    store_a = f'{base}_a'
    store_b = f'{base}_b'

    for s in (store_a, store_b):
        drop_postgres_store(s, pg_dsn)

    a = open_postgres_backend(store_a, pg_dsn)
    b = open_postgres_backend(store_b, pg_dsn)
    a.close()
    b.close()

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 1
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
                (_store_schema(store_b),))
            assert cur.fetchone()[0] == 1

    drop_postgres_store(store_a, pg_dsn)

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
                (_store_schema(store_a),))
            assert cur.fetchone()[0] == 0
            cur.execute(
                'select count(*) from pg_namespace where nspname = %s',
                (_store_schema(store_b),))
            assert cur.fetchone()[0] == 1, (
                'sibling store schema must not be dropped')

    drop_postgres_store(store_b, pg_dsn)


def test_recreate_after_drop_yields_empty_schema(pg_dsn, request):
    """Verify a store reopened after a drop holds no earlier rows.

    Mutation: the drop leaving the old tables in place, so the
        reused schema name resurrects earlier rows.
    Oracle: `get` of the pre-wipe id is None and `insights` counts
        zero rows.
    """
    store = _safe(request.node.name)
    drop_postgres_store(store, pg_dsn)

    first = open_postgres_backend(store, pg_dsn)
    try:
        first.nodes.insert(Insight(id='pre-wipe', content='will be wiped'))
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
                    f'select count(*) from {_store_schema(store)}.insights')
                assert cur.fetchone()[0] == 0
    finally:
        second.close()
        drop_postgres_store(store, pg_dsn)
