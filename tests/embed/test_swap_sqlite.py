"""SQLite shadow-column swap workflow.

Verifies that `run_swap` walks all rows, populates `embedding_pending`,
cuts over to a new (provider, model, dim) fingerprint, and recall keeps
working throughout. The cutover transaction does not rebuild HNSW or
recall snapshots -- that's a follow-up concern. Recall correctness
across the new dim is covered by the broader test suite once a swapped
store is opened.
"""

from datetime import datetime, timezone

import pytest
from memman.embed.fingerprint import Fingerprint, stored_fingerprint
from memman.embed.swap import STATE_DONE, SwapPlan, abort_swap
from memman.embed.swap import read_progress, run_swap
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.store.db import open_db
from memman.store.sqlite import SqliteBackend


class _StubEmbedder:
    """Second embedder bound to a different (provider, model, dim).

    Mirrors the EmbeddingProvider Protocol surface used by `swap.py`.
    `embed_batch` returns deterministic dim-N vectors derived from
    text content via the conftest-shared `_mock_embed`.
    """

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
        from tests.conftest import _mock_embed
        return [_mock_embed(self, t) for t in texts]

    def unavailable_message(self) -> str:
        return ''


def _seed_insights(backend: SqliteBackend, n: int) -> list[str]:
    """Insert n rows with stub 512-dim embeddings; return ids."""
    now = datetime.now(timezone.utc).isoformat()
    ids = []
    with backend.transaction():
        for i in range(n):
            rid = f'id-{i:04d}'
            ids.append(rid)
            blob = serialize_vector([0.1 * i] * 512)
            backend._db._exec(
                'insert into insights'
                ' (id, content, embedding, embedding_model,'
                '  created_at, updated_at)'
                ' values (?, ?, ?, ?, ?, ?)',
                (rid, f'content {i}', blob, 'voyage-3-lite', now, now))
    return ids


@pytest.fixture
def swap_backend(tmp_path):
    """Open a fresh SQLite backend rooted at tmp_path."""
    db = open_db(str(tmp_path))
    backend = SqliteBackend(db)
    try:
        yield backend
    finally:
        db.close()


def test_swap_completes_full_workflow(swap_backend):
    """run_swap fills embedding_pending, cuts over, marks done."""
    _seed_insights(swap_backend, 5)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)

    progress = run_swap(swap_backend, ec, plan, batch_size=2)

    assert progress.state == STATE_DONE
    rows = swap_backend._db._query(
        'select id, embedding, embedding_model, embedding_pending'
        ' from insights order by id').fetchall()
    assert all(r[2] == 'stub-target-d768' for r in rows)
    assert all(r[3] is None for r in rows)
    vec = deserialize_vector(rows[0][1])
    assert len(vec) == 768


def test_swap_writes_fingerprint(swap_backend):
    """After cutover, meta.embed_fingerprint matches the target."""
    _seed_insights(swap_backend, 3)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)

    run_swap(swap_backend, ec, plan, batch_size=10)

    fp = stored_fingerprint(swap_backend)
    assert fp == Fingerprint(
        provider='stub-target',
        model='stub-target-d768',
        dim=768)


def test_swap_clears_meta_after_done(swap_backend):
    """All embed_swap_* meta keys are deleted after a successful cutover.

    Absence of the keys is the canonical "no swap in flight" signal;
    `read_progress` reports `state=''` and the doctor check
    `check_no_stale_swap_meta` passes.
    """
    _seed_insights(swap_backend, 2)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)

    run_swap(swap_backend, ec, plan)

    leftover = [
        k for k in swap_backend.meta
        if k.startswith('embed_swap_')]
    assert leftover == []


def test_swap_resume_skips_already_filled_rows(swap_backend):
    """A second run after a partial backfill resumes from the cursor.

    Simulated by manually filling embedding_pending for half the rows
    and seeding `meta.embed_swap_*` to mid-flight, then calling
    `run_swap` and observing that the second half completes.
    """
    ids = _seed_insights(swap_backend, 6)
    ec = _StubEmbedder(dim=768)
    fake_pending = serialize_vector([0.5] * 768)
    with swap_backend.transaction():
        for rid in ids[:3]:
            swap_backend._db._exec(
                'update insights set embedding_pending = ?'
                ' where id = ?', (fake_pending, rid))
        swap_backend.meta.set('embed_swap_state', 'backfilling')
        swap_backend.meta.set('embed_swap_cursor', ids[2])
        swap_backend.meta.set(
            'embed_swap_target_provider', 'stub-target')
        swap_backend.meta.set(
            'embed_swap_target_model', 'stub-target-d768')
        swap_backend.meta.set('embed_swap_target_dim', '768')

    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)
    progress = run_swap(swap_backend, ec, plan, batch_size=2)

    assert progress.state == STATE_DONE
    rows = swap_backend._db._query(
        'select id, embedding from insights'
        ' order by id').fetchall()
    first_three_blobs = [r[1] for r in rows[:3]]
    assert all(blob == fake_pending for blob in first_three_blobs)


def test_swap_abort_clears_pending_and_meta(swap_backend):
    """abort_swap nulls embedding_pending and clears all swap meta."""
    ids = _seed_insights(swap_backend, 4)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)
    swap_backend.swap_prepare(768)
    with swap_backend.transaction():
        for rid in ids[:2]:
            swap_backend._db._exec(
                'update insights set embedding_pending = ?'
                ' where id = ?',
                (serialize_vector([0.5] * 768), rid))
        swap_backend.meta.set('embed_swap_state', 'backfilling')
        swap_backend.meta.set('embed_swap_target_dim', '768')

    abort_swap(swap_backend)

    progress = read_progress(swap_backend)
    assert progress.state == ''
    rows = swap_backend._db._query(
        'select embedding_pending from insights').fetchall()
    assert all(r[0] is None for r in rows)


def test_swap_target_mismatch_in_flight_raises(swap_backend):
    """Resuming with a different target than the in-flight one errors.

    Forces operators to abort first instead of silently switching
    targets across a running backfill.
    """
    _seed_insights(swap_backend, 3)
    ec_first = _StubEmbedder(dim=768)
    plan_first = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d768',
        target_dim=768)
    swap_backend.swap_prepare(768)
    with swap_backend.transaction():
        swap_backend.meta.set('embed_swap_state', 'backfilling')
        swap_backend.meta.set(
            'embed_swap_target_provider', 'stub-target')
        swap_backend.meta.set(
            'embed_swap_target_model', 'stub-target-d768')
        swap_backend.meta.set('embed_swap_target_dim', '768')

    plan_diff = SwapPlan(
        target_provider='stub-target',
        target_model='stub-target-d1024',
        target_dim=1024)

    with pytest.raises(RuntimeError) as exc:
        run_swap(swap_backend, ec_first, plan_diff)
    assert 'in-flight' in str(exc.value).lower()
