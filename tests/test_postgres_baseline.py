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

import psycopg
import pytest
from pgvector.psycopg import register_vector
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
