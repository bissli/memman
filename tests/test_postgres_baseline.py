"""Baseline contract tests against raw psycopg + pgvector.

Validates the primitives a future Postgres backend depends on:

1. `vector(512)` round-trip with a Voyage-shaped 512-dim list[float].
2. HNSW index correctness (top-5 against an exact seqscan).
3. `pg_try_advisory_lock` contention: only one connection wins.
4. `SET search_path` persists across cursor close in autocommit
   mode (pool-reuse hazard documentation).
5. Advisory lock released on connection close (no explicit unlock
   needed -- the crash-recovery mechanism the drain-lock contract
   relies on).

Gated behind `@pytest.mark.postgres` so SQLite-only `make test`
runs are unaffected.
"""

from __future__ import annotations

import random
import threading

import psycopg
import pytest
from pgvector.psycopg import register_vector
from memman.store.model import Insight
from memman.store.postgres import EMBEDDING_DIM, PostgresCluster
from memman.store.postgres import _ensure_baseline_schema, _ensure_hnsw_index
from memman.store.postgres import _store_schema
from memman.store.errors import BackendError
from tests.fixtures.postgres import SCHEMA, drain_connection_pair
from tests.fixtures.postgres import simulate_drain_connection_drop, wait_for

pytestmark = pytest.mark.postgres


def _voyage_shape_vector(seed: int = 0, dim: int = 512) -> list[float]:
    """Return a deterministic 512-dim float list approximating a Voyage embedding.

    Values land in [-1, 1] but are NOT unit-normalized -- pgvector's
    cosine distance handles normalization implicitly.
    """
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def test_vector_512_round_trip(pg_conn):
    """A 512-dim list[float] survives INSERT and SELECT through pgvector."""
    register_vector(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(f'SET search_path = {SCHEMA}, public')
        cur.execute(
            'CREATE TABLE vec_test ('
            ' id INTEGER PRIMARY KEY,'
            ' embedding vector(512))')
        original = _voyage_shape_vector(seed=42)
        cur.execute(
            'INSERT INTO vec_test (id, embedding) VALUES (%s, %s)',
            (1, original))
        cur.execute('SELECT embedding FROM vec_test WHERE id = 1')
        roundtripped = list(cur.fetchone()[0])
    assert len(roundtripped) == 512
    for a, b in zip(original, roundtripped):
        assert abs(a - b) < 1e-5, (
            'pgvector float32 truncation should be < 1e-5 per dim')


def test_hnsw_top5_correctness(pg_conn):
    """HNSW returns the same top-5 IDs as a sequential scan on 100 rows.

    Forces `enable_seqscan = on` so HNSW approximation noise can't
    explain a divergence; this validates that `<=>` semantics +
    cosine direction match the score-direction contract
    (`1 - distance` -> similarity in [-1, 1], higher better).
    """
    register_vector(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(f'SET search_path = {SCHEMA}, public')
        cur.execute(
            'CREATE TABLE corpus ('
            ' id INTEGER PRIMARY KEY,'
            ' embedding vector(512))')
        rows = [
            (i, _voyage_shape_vector(seed=i))
            for i in range(100)
            ]
        cur.executemany(
            'INSERT INTO corpus (id, embedding) VALUES (%s, %s)',
            rows)
        cur.execute(
            'CREATE INDEX hnsw_corpus ON corpus'
            ' USING hnsw (embedding vector_cosine_ops)')
        import numpy as np
        query_vec = np.asarray(_voyage_shape_vector(seed=7))
        cur.execute('SET enable_seqscan = on')
        cur.execute(
            'SELECT id, 1 - (embedding <=> %s) AS sim FROM corpus'
            ' ORDER BY embedding <=> %s LIMIT 5',
            (query_vec, query_vec))
        seqscan_top5 = [r[0] for r in cur.fetchall()]
        cur.execute('SET enable_seqscan = off')
        cur.execute(
            'SELECT id FROM corpus'
            ' ORDER BY embedding <=> %s LIMIT 5',
            (query_vec,))
        index_top5 = [r[0] for r in cur.fetchall()]
    assert len(seqscan_top5) == 5
    assert len(index_top5) == 5
    overlap = len(set(seqscan_top5) & set(index_top5))
    assert overlap >= 4, (
        f'HNSW top-5 should match seqscan top-5 in >=4 of 5'
        f' (got {overlap}); index={index_top5} seq={seqscan_top5}')


def test_pg_try_advisory_lock_contention(pg_dsn):
    """Holding pg_try_advisory_lock from one conn blocks a second."""
    lock_id = 9991
    with drain_connection_pair(pg_dsn) as (conn_a, conn_b):
        with conn_a.cursor() as cur_a:
            cur_a.execute(
                'SELECT pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_a.fetchone()[0] is True, (
                'first connection should win the advisory lock')
        with conn_b.cursor() as cur_b:
            cur_b.execute(
                'SELECT pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_b.fetchone()[0] is False, (
                'second connection should be denied while first holds')
        with conn_a.cursor() as cur_a:
            cur_a.execute(
                'SELECT pg_advisory_unlock(%s)', (lock_id,))
            assert cur_a.fetchone()[0] is True
        with conn_b.cursor() as cur_b:
            cur_b.execute(
                'SELECT pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_b.fetchone()[0] is True, (
                'second connection should now acquire after release')
            cur_b.execute(
                'SELECT pg_advisory_unlock(%s)', (lock_id,))


def test_search_path_persists_across_cursor_close_in_autocommit(pg_dsn):
    """`SET search_path` in autocommit mode persists to the next cursor.

    Documents the pool-reuse hazard: a pooled connection that ran
    `SET search_path = store_a, public` for one logical request will
    still report `search_path = store_a, public` when the next
    request acquires it from the pool. Implementations of
    `RecallSession.__exit__` must explicitly reset `search_path`
    before returning the connection to a pool.
    """
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}')
            cur.execute(f'SET search_path = {SCHEMA}, public')
            cur.execute('SHOW search_path')
            assert SCHEMA in cur.fetchone()[0]
        with conn.cursor() as cur2:
            cur2.execute('SHOW search_path')
            after = cur2.fetchone()[0]
            assert SCHEMA in after, (
                f'search_path should persist across cursor close in'
                f' autocommit mode; got {after!r}. Pool reuse without'
                f' explicit reset would leak schema selection.')


def test_advisory_lock_released_on_connection_close(pg_dsn):
    """Closing a connection releases its advisory locks without explicit unlock.

    This is the crash-recovery mechanism the drain-lock contract
    relies on: if a drain worker hangs or the host dies, the lock
    is released by Postgres detecting the dead TCP session, and
    another agent can claim the drain.
    """
    lock_id = 9992
    holder = psycopg.connect(pg_dsn, autocommit=True)
    try:
        with holder.cursor() as cur:
            cur.execute(
                'SELECT pg_try_advisory_lock(%s)', (lock_id,))
            assert cur.fetchone()[0] is True
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with observer.cursor() as cur:
                cur.execute(
                    'SELECT pg_try_advisory_lock(%s)', (lock_id,))
                assert cur.fetchone()[0] is False, (
                    'lock should still be held by holder')
    finally:
        simulate_drain_connection_drop(holder)
    with psycopg.connect(pg_dsn, autocommit=True) as later:

        def _can_acquire() -> bool:
            with later.cursor() as cur:
                cur.execute(
                    'SELECT pg_try_advisory_lock(%s)', (lock_id,))
                got = cur.fetchone()[0]
                if got:
                    cur.execute(
                        'SELECT pg_advisory_unlock(%s)', (lock_id,))
                return bool(got)

        assert wait_for(_can_acquire, timeout_sec=5.0), (
            'advisory lock should be released within 5s of'
            ' connection close (Postgres detects dead session)')


def _pg_vec(seed: int) -> list[float]:
    """Reproducible synthetic vector for postgres backend tests."""
    return [(seed + i) * 0.001 for i in range(EMBEDDING_DIM)]


@pytest.fixture
def _pg_store_backend(pg_dsn):
    """A PostgresBackend bound to a fresh `store_pg_salvage` schema."""
    store_name = 'pg_salvage'
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    backend = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    try:
        yield backend, pg_dsn, store_name
    finally:
        backend.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_hnsw_partial_index_built_concurrently(
        _pg_store_backend):
    """HNSW index uses vector_cosine_ops WHERE deleted_at IS NULL.

    Inspects pg_index for cosine ops, partial WHERE, and valid
    (`indisvalid = true`).
    """
    backend, pg_dsn, store_name = _pg_store_backend
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT i.indisvalid, am.amname,'
                ' pg_get_indexdef(i.indexrelid)'
                ' FROM pg_index i'
                ' JOIN pg_class c ON c.oid = i.indexrelid'
                ' JOIN pg_am am ON am.oid = ('
                '   SELECT relam FROM pg_class'
                '   WHERE oid = i.indexrelid)'
                ' WHERE c.relname = %s',
                (f'idx_insights_hnsw_{schema}',))
            row = cur.fetchone()
    assert row is not None, 'HNSW index missing'
    assert row[0] is True, 'HNSW index is invalid'
    assert row[1] == 'hnsw'
    indexdef = row[2]
    assert 'vector_cosine_ops' in indexdef
    assert 'deleted_at IS NULL' in indexdef


def test_bulk_update_embedding_chunks_at_1000(
        _pg_store_backend, monkeypatch):
    """bulk_update_embedding splits >1000 rows into <=1000 commits.

    Pure shape test: stubs `executemany` to record batch sizes without
    inserting 1001 rows.
    """
    backend, _pg_dsn, _store_name = _pg_store_backend
    sizes: list[int] = []
    real_executemany = psycopg.Cursor.executemany

    def spy(self, sql, args):
        rows = list(args)
        sizes.append(len(rows))
        return real_executemany(self, sql, rows)

    monkeypatch.setattr(psycopg.Cursor, 'executemany', spy)
    big_batch = [
        (f'b-{i}', _pg_vec(i), 'voyage-3-lite') for i in range(1500)
        ]
    backend.nodes.bulk_update_embedding(big_batch)
    assert sizes, 'executemany never called'
    assert all(n <= 1000 for n in sizes), (
        f'expected all batches <=1000, got {sizes}')
    assert sum(sizes) >= 1500


def test_reindex_drops_invalid_hnsw_remnant(pg_dsn):
    """An invalid HNSW remnant from an aborted CONCURRENTLY build is
    dropped before the next reindex retries.
    """
    store_name = 'pg_remnant'
    schema = _store_schema(store_name)
    _ensure_baseline_schema(pg_dsn, store_name)
    index_name = f'idx_insights_hnsw_{schema}'
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP INDEX IF EXISTS {schema}.{index_name}')
            cur.execute(
                f'CREATE INDEX {index_name}'
                f' ON {schema}.insights'
                f' USING hnsw (embedding vector_cosine_ops)'
                f' WHERE deleted_at IS NULL')
            cur.execute(
                'UPDATE pg_index SET indisvalid = false'
                ' WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            cur.execute(
                'SELECT indisvalid FROM pg_index WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            assert cur.fetchone()[0] is False
    _ensure_hnsw_index(pg_dsn, schema)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT indisvalid FROM pg_index WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            row = cur.fetchone()
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    assert row is not None
    assert row[0] is True


def test_reembed_lock_session_scoped_and_releases_on_close(
        pg_dsn):
    """Two PostgresBackends compete for `reembed_lock`. The second
    gets False; once the first connection closes, the second acquires.
    """
    store_name = 'pg_reembed'
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
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_reindex_concurrent_writers_no_edges_lost(pg_dsn):
    """Thread A inserts a manual edge inside `write_lock("reindex")`;
    Thread B runs `reindex_auto_edges` concurrently. The manual edge
    must survive.
    """
    from memman.graph.engine import reindex_auto_edges
    from memman.store.model import Edge

    store_name = 'pg_concurrent'
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
                    f'seed{i}', _pg_vec(i), 'voyage-3-lite')
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
                        'manual1', _pg_vec(99), 'voyage-3-lite')
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
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_open_with_stored_version_ahead_refuses(pg_dsn):
    """Seeding `meta.pg_schema_version = '999'` causes the next open
    to raise BackendError instead of silently writing.
    """
    from memman.store.postgres import _apply_pending_migrations

    store_name = 'pg_version_ahead'
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
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_memman_reindex_timeout_caps_hnsw_build(
        pg_dsn, monkeypatch):
    """`MEMMAN_REINDEX_TIMEOUT=7` puts statement_timeout on the
    autocommit connection used for HNSW build. Asserts the SET
    statement_timeout SQL is issued before CREATE INDEX by spying
    on cursor.execute.
    """
    store_name = 'pg_timeout'
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
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    set_idx = next(
        (i for i, s in enumerate(captured)
         if "statement_timeout = '7s'" in s.lower()), None)
    create_idx = next(
        (i for i, s in enumerate(captured)
         if 'create index concurrently' in s.lower()), None)
    assert set_idx is not None, (
        f'expected SET statement_timeout in: {captured}')
    assert create_idx is not None, (
        f'expected CREATE INDEX CONCURRENTLY in: {captured}')
    assert set_idx < create_idx, (
        f'SET statement_timeout must precede CREATE INDEX;'
        f' captured order: {captured}')
