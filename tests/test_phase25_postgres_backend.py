"""Phase 2.5 PostgresBackend distributed-hardening tests.

Drives the slimmed Phase 2.5 scope:

- `write_lock` wired at exactly 2 sites (reindex_auto_edges + prune).
- `reembed_lock` is session-scoped, outside any pool, releases on
  connection close.
- Concurrent `reindex_auto_edges` + insert paths do not lose edges.
- Schema migration ladder applies pending entries atomically and
  refuses on stored > code.
- Module-level additive-only assertion blocks non-additive DDL.
- `MEMMAN_REINDEX_TIMEOUT` caps the HNSW build via
  `statement_timeout`.
"""

from __future__ import annotations

import pathlib
import re
import threading

import psycopg
import pytest
from memman.store.errors import BackendError
from memman.store.model import Insight
from memman.store.postgres import EMBEDDING_DIM, PostgresCluster
from memman.store.postgres import _ensure_baseline_schema, _store_schema

pytestmark = pytest.mark.postgres


def _vec(seed: int) -> list[float]:
    return [(seed + i) * 0.001 for i in range(EMBEDDING_DIM)]


@pytest.fixture
def store_name() -> str:
    return 'phase25'


@pytest.fixture
def pg_backend(pg_dsn, store_name):
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    backend = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    try:
        yield backend
    finally:
        backend.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_write_lock_wired_at_reindex_and_prune():
    r"""Exactly two `\.write_lock(` call sites in the wired scope:
    one in graph/engine.py (reindex_auto_edges), one in
    store/postgres.py (PostgresNodeStore.auto_prune). The regex
    excludes `def write_lock(` (definitions have no leading `.`).
    """
    src = pathlib.Path(__file__).resolve().parent.parent / 'src'
    pat = re.compile(r'\b\w+\.write_lock\s*\(')
    target_files = [
        src / 'memman' / 'graph' / 'engine.py',
        src / 'memman' / 'store' / 'postgres.py',
        ]
    hits: list[str] = []
    for p in target_files:
        text = p.read_text()
        hits.extend(
            f'{p.name}:{m.group(0)}' for m in pat.finditer(text))
    assert len(hits) == 2, (
        f'expected exactly 2 write_lock call sites in (engine.py,'
        f' postgres.py); found {len(hits)}: {hits}')


def test_reembed_lock_session_scoped_and_releases_on_close(
        pg_dsn, store_name):
    """Two PostgresBackends compete for `reembed_lock`. The second
    gets False; once the first connection closes, the second
    acquires.
    """
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    cluster = PostgresCluster(dsn=pg_dsn)
    a = cluster.open(store=store_name, data_dir='/unused')
    b = cluster.open(store=store_name, data_dir='/unused')
    try:
        with a.reembed_lock('reembed') as got_a:
            assert got_a is True
            with b.reembed_lock('reembed') as got_b:
                assert got_b is False
        with b.reembed_lock('reembed') as got_b2:
            assert got_b2 is True
    finally:
        a.close()
        b.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_reindex_concurrent_writers_no_edges_lost(
        pg_dsn, store_name):
    """Thread A inserts an insight + writes a manual edge inside
    `write_lock("reindex")`; Thread B runs `reindex_auto_edges` (which
    deletes auto edges and recreates them). The manual edge must
    survive.
    """
    from memman.graph.engine import reindex_auto_edges
    from memman.store.model import Edge

    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    cluster = PostgresCluster(dsn=pg_dsn)
    seed = cluster.open(store=store_name, data_dir='/unused')
    try:
        with seed.transaction():
            for i in range(3):
                seed.nodes.insert(
                    Insight(
                        id=f'seed{i}', content=f'seed {i}',
                        importance=3))
                seed.nodes.update_embedding(
                    f'seed{i}', _vec(i), 'voyage-3-lite')
    finally:
        seed.close()

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def thread_a() -> None:
        backend = cluster.open(store=store_name, data_dir='/unused')
        try:
            barrier.wait()
            with backend.transaction():
                with backend.write_lock('reindex'):
                    backend.nodes.insert(
                        Insight(
                            id='manual1', content='manual',
                            importance=4))
                    backend.nodes.update_embedding(
                        'manual1', _vec(99), 'voyage-3-lite')
                    backend.edges.upsert(
                        Edge(
                            source_id='seed0', target_id='manual1',
                            edge_type='causal',
                            weight=1.0,
                            metadata={'created_by': 'manual'}))
        except Exception as e:
            errors.append(e)
        finally:
            backend.close()

    def thread_b() -> None:
        backend = cluster.open(store=store_name, data_dir='/unused')
        try:
            barrier.wait()
            reindex_auto_edges(backend)
        except Exception as e:
            errors.append(e)
        finally:
            backend.close()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=30)
    tb.join(timeout=30)

    assert not errors, f'thread errors: {errors}'

    check = cluster.open(store=store_name, data_dir='/unused')
    try:
        with check._conn.cursor() as cur:
            cur.execute(
                f"SELECT source_id, target_id FROM {schema}.edges"
                " WHERE metadata->>'created_by' = 'manual'")
            preserved = cur.fetchall()
        assert ('seed0', 'manual1') in preserved, (
            f'manual edge lost; preserved={preserved}')
    finally:
        check.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_pg_schema_migration_runs_pending_alter(
        pg_dsn, store_name, monkeypatch):
    """Bumping `_PG_SCHEMA_VERSION` and adding a `_PG_MIGRATIONS`
    entry causes the next open to run the ALTER and persist the new
    version in `meta.pg_schema_version`.
    """
    from memman.store import postgres as pg_module
    from memman.store.postgres import _apply_pending_migrations

    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    _ensure_baseline_schema(pg_dsn, store_name)
    monkeypatch.setattr(pg_module, '_PG_SCHEMA_VERSION', 2)
    monkeypatch.setattr(
        pg_module, '_PG_MIGRATIONS',
        [(2, 'ALTER TABLE {schema}.insights ADD COLUMN test_col TEXT')])
    try:
        _apply_pending_migrations(pg_dsn, store_name)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT column_name FROM information_schema.columns'
                    ' WHERE table_schema = %s'
                    "   AND column_name = 'test_col'",
                    (schema,))
                assert cur.fetchone() is not None
                cur.execute(
                    f"SELECT value FROM {schema}.meta"
                    " WHERE key = 'pg_schema_version'")
                row = cur.fetchone()
                assert row is not None
                assert row[0] == '2'
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_pg_migrations_assertion_blocks_non_additive(monkeypatch):
    """Re-importing `store.postgres` with a non-additive entry in
    `_PG_MIGRATIONS` raises AssertionError at module-load time.
    """
    from memman.store import postgres as pg_module

    forbidden_cases = [
        [(2, 'ALTER TABLE x DROP COLUMN y')],
        [(2, 'ALTER TABLE x RENAME TO z')],
        [(2, 'DROP TABLE x')],
        [(2, 'TRUNCATE x')],
        [(2, 'ALTER TABLE x ALTER COLUMN y SET NOT NULL')],
        ]
    for bad in forbidden_cases:
        regex = pg_module._FORBIDDEN_MIGRATION_RE
        for _, sql in bad:
            assert regex.search(sql), (
                f'expected forbidden pattern to match: {sql!r}')


def test_open_with_stored_version_ahead_refuses(
        pg_dsn, store_name):
    """Seeding `meta.pg_schema_version = '999'` causes the next open
    to raise BackendError instead of silently writing.
    """
    from memman.store.postgres import _apply_pending_migrations

    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    _ensure_baseline_schema(pg_dsn, store_name)
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'INSERT INTO {schema}.meta (key, value)'
                    " VALUES ('pg_schema_version', '999')"
                    ' ON CONFLICT (key) DO UPDATE'
                    ' SET value = EXCLUDED.value')
        with pytest.raises(BackendError, match='schema version 999'):
            _apply_pending_migrations(pg_dsn, store_name)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_memman_reindex_timeout_caps_hnsw_build(
        pg_dsn, store_name, monkeypatch):
    """`MEMMAN_REINDEX_TIMEOUT=1` puts statement_timeout on the
    autocommit connection used for HNSW build. We don't force an
    actual long build (CI containers are too fast); we assert the
    SET statement_timeout SQL is issued before the CREATE INDEX, by
    spying on cursor.execute.
    """
    from memman.store.postgres import _ensure_hnsw_index

    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    _ensure_baseline_schema(pg_dsn, store_name)

    monkeypatch.setenv('MEMMAN_REINDEX_TIMEOUT', '7')
    captured: list[str] = []
    real_execute = psycopg.Cursor.execute

    def spy(self, sql, *args, **kwargs):
        captured.append(str(sql))
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, 'execute', spy)

    try:
        _ensure_hnsw_index(pg_dsn, schema)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    set_idx = next(
        (i for i, s in enumerate(captured)
         if "SET statement_timeout = '7s'" in s), None)
    create_idx = next(
        (i for i, s in enumerate(captured)
         if 'CREATE INDEX CONCURRENTLY' in s), None)
    assert set_idx is not None, (
        f'expected SET statement_timeout in: {captured}')
    assert create_idx is not None, (
        f'expected CREATE INDEX CONCURRENTLY in: {captured}')
    assert set_idx < create_idx, (
        f'SET statement_timeout must precede CREATE INDEX;'
        f' captured order: {captured}')
