"""Postgres online swap workflow.

Verifies the DDL workflow: PG version preflight, ADD COLUMN
embedding_pending, CREATE INDEX CONCURRENTLY, backfill predicate
WHERE embedding_pending IS NULL, atomic cutover (drop+rename),
and abort cleanup. Recall is expected to keep working throughout
because the reads continue to hit `embedding` until cutover commits.
"""

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.embed.fingerprint import Fingerprint, stored_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.embed.swap import STATE_DONE, SwapPlan, abort_swap, run_swap
from memman.store.errors import BackendError
from memman.store.model import Insight
from memman.store.postgres import _assert_vector_dim_matches, _store_schema
from memman.store.postgres import open_postgres_backend
from tests.conftest import EMBEDDING_DIM

pytestmark = pytest.mark.postgres


def _pg_vec(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    return [(seed + i) * 0.001 for i in range(dim)]


class _StubEmbedder:
    """Second embedder bound to a different (model, dim).
    """

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
    """Insert n insights with current-dim embeddings; return ids.
    """
    ids = []
    with backend.transaction():
        for i in range(n):
            rid = f'seed{i:04d}'
            ids.append(rid)
            backend.nodes.insert(Insight(id=rid, content=f'content {i}'))
            backend.nodes.update_embedding(
                rid, _pg_vec(i), 'voyage-3-lite')
    return ids


@pytest.fixture
def swap_backend(pg_dsn):
    store_name = 'pg_swap'
    _drop_schema(pg_dsn, store_name)
    backend = open_postgres_backend(store_name, pg_dsn, create=True)
    write_fingerprint(
        backend,
        Fingerprint(
            model='voyage-3-lite',
            dim=EMBEDDING_DIM))
    try:
        yield backend, pg_dsn, store_name
    finally:
        backend.close()
        _drop_schema(pg_dsn, store_name)


def test_swap_completes_full_workflow(swap_backend, monkeypatch):
    """run_swap walks all rows, cuts over, marks done.

    Mutation: cutover leaving `embedding_pending` in place, or not
        resizing `embedding` to the target dimension.
    Oracle: `information_schema` column set and `atttypmod` of 384.
    """
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 4)
    schema = _store_schema(store_name)
    ec = _StubEmbedder(dim=384)
    plan = SwapPlan(
        target_model='stub-target-d384',
        target_dim=384)

    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '2')
    progress = run_swap(backend, ec, plan)

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
                " where attrelid = (%s||'.insights')::regclass"
                '   and attname = %s and not attisdropped',
                (schema, 'embedding'))
            row = cur.fetchone()
    assert 'embedding' in cols
    assert 'embedding_pending' not in cols
    assert int(row[0]) == 384


def test_swap_writes_fingerprint(swap_backend, monkeypatch):
    """meta.embed_fingerprint reflects the target after cutover.

    Mutation: `run_swap` skipping `write_fingerprint`, so the store
        keeps claiming the old model and dim.
    Oracle: the `Fingerprint` built from the plan's hand-set values.
    """
    backend, _pg_dsn, _store_name = swap_backend
    _seed(backend, 2)
    ec = _StubEmbedder(dim=256)
    plan = SwapPlan(
        target_model='stub-target-d256',
        target_dim=256)

    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '10')
    run_swap(backend, ec, plan)

    fp = stored_fingerprint(backend)
    assert fp == Fingerprint(
        model='stub-target-d256',
        dim=256)


def test_swap_abort_drops_pending_column(swap_backend):
    """abort_swap drops embedding_pending and clears swap meta.

    Mutation: `abort_swap` clearing the meta but leaving the pending
        column, or the reverse.
    Oracle: `information_schema` column set and the empty swap state.
    """
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

    Mutation: the check comparing against `embedding` only, so opening
        the store mid-swap with the new client raises.
    Oracle: no raise for both the pending dim 256 and the live dim.
    """
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 1)
    backend.swap_prepare(256)
    backend.meta.set('embed_swap_state', 'backfilling')

    _assert_vector_dim_matches(pg_dsn, store_name, 256)
    _assert_vector_dim_matches(pg_dsn, store_name, EMBEDDING_DIM)


def test_swap_lock_blocks_concurrent_swap(swap_backend):
    """The session-scoped embed_swap lock blocks a second swap.

    Mutation: `swap_lock` using a per-connection key or a
        non-exclusive lock, so both holders acquire it.
    Oracle: the first holder sees True, the second sees False.
    """
    backend, pg_dsn, store_name = swap_backend
    other = open_postgres_backend(store_name, pg_dsn)
    try:
        with backend.swap_lock() as held_a:
            assert held_a is True
            with other.swap_lock() as held_b:
                assert held_b is False
    finally:
        other.close()


def test_swap_resume_finishes_after_a_crash_past_cutover(
        swap_backend, monkeypatch):
    """--resume finishes a swap whose cutover committed before a crash.

    Mutation: the bug itself - the resumed cutover re-runs a check on
        `embedding_pending`, a column the committed cutover already
        renamed, so the swap can never finish or clear its keys.
    Oracle: the plan's hand-set target fingerprint and an empty swap
        state after the resume.
    """
    backend, _pg_dsn, _store_name = swap_backend
    _seed(backend, 3)
    ec = _StubEmbedder(dim=384)
    plan = SwapPlan(
        target_model='stub-target-d384',
        target_dim=384)

    def _crash(*args, **kwargs):
        raise RuntimeError('crash after cutover')

    monkeypatch.setattr('memman.embed.swap.write_fingerprint', _crash)
    with pytest.raises(RuntimeError, match='crash after cutover'):
        run_swap(backend, ec, plan)
    monkeypatch.setattr(
        'memman.embed.swap.write_fingerprint', write_fingerprint)

    progress = run_swap(backend, ec, plan)

    assert progress.state == STATE_DONE
    assert stored_fingerprint(backend) == Fingerprint(
        model='stub-target-d384', dim=384)
    assert backend.meta.get('embed_swap_state') is None


def test_swap_resume_recovers_after_the_cutover_check_refuses(
        swap_backend, monkeypatch):
    """A refused cutover returns the swap to backfill so --resume ends it.

    Mutation: the refusal leaving the state at `cutover`, so every
        resume reruns the same failing check, abort refuses, and the
        store is stuck until someone edits its meta table.
    Oracle: a row inserted below the backfill cursor after a crash -
        the case the check exists to catch - and the plan's target
        fingerprint once the second resume finishes.
    """
    backend, _pg_dsn, _store_name = swap_backend
    _seed(backend, 4)
    ec = _StubEmbedder(dim=384)
    plan = SwapPlan(
        target_model='stub-target-d384',
        target_dim=384)
    real_embed_batch = ec.embed_batch
    calls = {'n': 0}

    def _crash_on_second_batch(texts):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('embed outage')
        return real_embed_batch(texts)

    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '2')
    monkeypatch.setattr(ec, 'embed_batch', _crash_on_second_batch)
    with pytest.raises(RuntimeError, match='embed outage'):
        run_swap(backend, ec, plan)
    with backend.transaction():
        backend.nodes.insert(Insight(id='0000-new', content='late row'))
        backend.nodes.update_embedding(
            '0000-new', _pg_vec(99), 'voyage-3-lite')

    with pytest.raises(BackendError, match='backfill is incomplete'):
        run_swap(backend, ec, plan)
    progress = run_swap(backend, ec, plan)

    assert progress.state == STATE_DONE
    assert stored_fingerprint(backend) == Fingerprint(
        model='stub-target-d384', dim=384)


def test_swap_abort_refuses_while_another_session_holds_the_lock(swap_backend):
    """abort_swap refuses while another session holds the swap lock.

    Mutation: abort taking no lock, so it drops `embedding_pending`
        under a running swap whose cutover then finds no column, skips
        the switch, and writes the target fingerprint over old vectors.
    Oracle: the pending column the first session prepared, still
        present after the refused abort.
    """
    backend, pg_dsn, store_name = swap_backend
    _seed(backend, 2)
    backend.swap_prepare(384)
    other = open_postgres_backend(store_name, pg_dsn)
    try:
        with backend.swap_lock() as held:
            assert held is True
            with pytest.raises(RuntimeError, match='another swap'):
                abort_swap(other)
    finally:
        other.close()

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select 1 from information_schema.columns'
                ' where table_schema = %s and table_name = %s'
                " and column_name = 'embedding_pending'",
                (_store_schema(store_name), 'insights'))
            assert cur.fetchone() is not None


def test_swap_keeps_cutover_state_when_a_committed_cutover_errors(
        swap_backend, monkeypatch):
    """An error after the cutover commits leaves the state at cutover.

    Mutation: returning the swap to backfill on any `BackendError`
        from the cutover, so a commit that landed but failed to confirm
        lets abort clear the keys over the new vectors, and resume
        reads a column the cutover already renamed.
    Oracle: a real cutover followed by a raised error, then the plan's
        target fingerprint once resume finishes the swap.
    """
    backend, _pg_dsn, _store_name = swap_backend
    _seed(backend, 3)
    ec = _StubEmbedder(dim=384)
    plan = SwapPlan(
        target_model='stub-target-d384',
        target_dim=384)
    real_cutover = backend.swap_cutover

    def _cutover_then_lose_the_ack(target):
        real_cutover(target)
        raise BackendError('postgres query failed: connection lost')

    monkeypatch.setattr(backend, 'swap_cutover', _cutover_then_lose_the_ack)
    with pytest.raises(BackendError, match='connection lost'):
        run_swap(backend, ec, plan)
    monkeypatch.setattr(backend, 'swap_cutover', real_cutover)

    assert backend.meta.get('embed_swap_state') == 'cutover'
    assert run_swap(backend, ec, plan).state == STATE_DONE
    assert stored_fingerprint(backend) == Fingerprint(
        model='stub-target-d384', dim=384)
