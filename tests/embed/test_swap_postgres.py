"""Postgres online swap workflow.

Verifies the DDL workflow: PG version preflight, ADD COLUMN
embedding_pending, CREATE INDEX CONCURRENTLY, backfill predicate
WHERE embedding_pending IS NULL, atomic cutover (drop+rename),
and abort cleanup. Recall is expected to keep working throughout
because the reads continue to hit `embedding` until cutover commits.
"""

import psycopg
import pytest

from memman.embed.fingerprint import (
    Fingerprint,
    stored_fingerprint,
    write_fingerprint,
    )
from memman.embed.swap import (
    META_STATE,
    STATE_DONE,
    SwapPlan,
    abort_swap,
    run_swap,
    )
from memman.store.postgres import (
    EMBEDDING_DIM,
    PostgresCluster,
    _store_schema,
    )

pytestmark = pytest.mark.postgres


def _pg_vec(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    return [(seed + i) * 0.001 for i in range(dim)]


class _StubEmbedder:
    """Second embedder bound to a different (provider, model, dim)."""

    name = 'stub-target'

    def __init__(self, dim: int = 768) -> None:
        self.model = f'stub-target-d{dim}'
        self.dim = dim

    def available(self) -> bool:
        return True

    def prepare(self) -> None:
        return

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_pg_vec(hash(t) % 1000, dim=self.dim) for t in texts]

    def unavailable_message(self) -> str:
        return ''


def _drop_schema(pg_dsn: str, store_name: str) -> None:
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')


def _seed(backend, n: int) -> list[str]:
    """Insert n insights with current-dim embeddings; return ids."""
    from memman.store.model import Insight
    ids = []
    with backend.transaction():
        for i in range(n):
            rid = f'seed{i:04d}'
            ids.append(rid)
            backend.nodes.insert(
                Insight(id=rid, content=f'content {i}', importance=3))
            backend.nodes.update_embedding(
                rid, _pg_vec(i), 'voyage-3-lite')
    return ids


@pytest.fixture
def swap_backend(pg_dsn):
    store_name = 'pg_swap'
    _drop_schema(pg_dsn, store_name)
    backend = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    write_fingerprint(
        backend,
        Fingerprint(
            provider='voyage', model='voyage-3-lite',
            dim=EMBEDDING_DIM))
    try:
        yield backend, pg_dsn, store_name
    finally:
        backend.close()
        _drop_schema(pg_dsn, store_name)


def test_swap_completes_full_workflow(swap_backend):
    """run_swap walks all rows, cuts over, marks done."""
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 4)
    schema = _store_schema(store_name)
    ec = _StubEmbedder(dim=384)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d384',
        target_dim=384)

    progress = run_swap(backend, ec, plan, batch_size=2)

    assert progress.state == STATE_DONE
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select column_name from information_schema.columns'
                ' where table_schema = %s and table_name = %s',
                (schema, 'insights'))
            cols = {r[0] for r in cur.fetchall()}
            cur.execute(
                'select atttypmod from pg_attribute'
                ' where attrelid = (%s||\'.insights\')::regclass'
                '   and attname = %s and not attisdropped',
                (schema, 'embedding'))
            row = cur.fetchone()
    assert 'embedding' in cols
    assert 'embedding_pending' not in cols
    assert int(row[0]) == 384


def test_swap_writes_fingerprint(swap_backend):
    """meta.embed_fingerprint reflects the target after cutover."""
    backend, _pg_dsn, _store_name = swap_backend
    _seed(backend, 2)
    ec = _StubEmbedder(dim=256)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d256',
        target_dim=256)

    run_swap(backend, ec, plan, batch_size=10)

    fp = stored_fingerprint(backend)
    assert fp == Fingerprint(
        provider='stub-target',
        model='stub-target-d256',
        dim=256)


def test_swap_abort_drops_pending_column(swap_backend):
    """abort_swap drops embedding_pending and clears swap meta."""
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 3)
    schema = _store_schema(store_name)
    backend.swap_prepare(384)
    backend.meta.set('embed_swap_state', 'backfilling')
    backend.meta.set('embed_swap_target_dim', '384')

    abort_swap(backend)

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select column_name from information_schema.columns'
                ' where table_schema = %s and table_name = %s',
                (schema, 'insights'))
            cols = {r[0] for r in cur.fetchall()}
    assert 'embedding_pending' not in cols
    assert (backend.meta.get('embed_swap_state') or '') == ''


def test_assert_dim_accepts_pending_during_swap(swap_backend):
    """_assert_vector_dim_matches accepts pending dim during backfill.

    Without this, opening the store mid-swap with the new client would
    crash on the dim assertion.
    """
    from memman.store.postgres import _assert_vector_dim_matches
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 1)
    backend.swap_prepare(256)
    backend.meta.set('embed_swap_state', 'backfilling')

    _assert_vector_dim_matches(pg_dsn, store_name, 256)
    _assert_vector_dim_matches(pg_dsn, store_name, EMBEDDING_DIM)


def test_swap_lock_blocks_concurrent_swap(swap_backend):
    """The session-scoped embed_swap lock blocks a second swap."""
    backend, pg_dsn, store_name = swap_backend
    other = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    try:
        with backend.swap_lock() as held_a:
            assert held_a is True
            with other.swap_lock() as held_b:
                assert held_b is False
    finally:
        other.close()
