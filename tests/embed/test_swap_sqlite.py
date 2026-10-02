"""SQLite shadow-column swap workflow.

`run_swap` walks all rows, populates `embedding_pending`, and cuts over
to a new (model, dim) fingerprint. The cutover transaction
does not rebuild HNSW or recall snapshots.
"""

from datetime import datetime, timezone

import pytest
from memman.embed.fingerprint import Fingerprint, stored_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.embed.swap import STATE_DONE, SwapPlan, abort_swap, read_progress
from memman.embed.swap import run_swap
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.store.db import open_db
from memman.store.sqlite import SqliteBackend
from tests.conftest import _mock_embed


class _StubEmbedder:
    """Second embedder bound to a different (model, dim).

    Mirrors the EmbeddingProvider Protocol surface used by `swap.py`.
    `embed_batch` returns deterministic dim-N vectors derived from
    text content via the conftest-shared `_mock_embed`.
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
        return [_mock_embed(self, t) for t in texts]

    def unavailable_message(self) -> str:
        return ''


def _seed_insights(backend: SqliteBackend, n: int) -> list[str]:
    """Insert n rows with stub 512-dim embeddings; return ids.
    """
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
    """Open a fresh SQLite backend rooted at tmp_path.
    """
    db = open_db(str(tmp_path))
    backend = SqliteBackend(db)
    try:
        yield backend
    finally:
        db.close()


def test_swap_completes_full_workflow(swap_backend, monkeypatch):
    """run_swap fills embedding_pending, cuts over, marks done.

    Mutation: cutover leaving old-dim blobs in `embedding`, leaving
        `embedding_pending` set, or stamping the old model name.
    Oracle: every row's model is the target, pending is null, and row 0
        decodes to 768 floats.
    """
    _seed_insights(swap_backend, 5)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_model='stub-target-d768',
        target_dim=768)

    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '2')
    progress = run_swap(swap_backend, ec, plan)

    assert progress.state == STATE_DONE
    rows = swap_backend._db._query(
        'select id, embedding, embedding_model, embedding_pending'
        ' from insights order by id').fetchall()
    assert all(r[2] == 'stub-target-d768' for r in rows)
    assert all(r[3] is None for r in rows)
    vec = deserialize_vector(rows[0][1])
    assert len(vec) == 768


def test_swap_writes_fingerprint(swap_backend, monkeypatch):
    """After cutover, meta.embed_fingerprint matches the target.

    Mutation: `run_swap` skipping `write_fingerprint`.
    Oracle: the `Fingerprint` built from the plan's hand-set values.
    """
    _seed_insights(swap_backend, 3)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_model='stub-target-d768',
        target_dim=768)

    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '10')
    run_swap(swap_backend, ec, plan)

    fp = stored_fingerprint(swap_backend)
    assert fp == Fingerprint(
        model='stub-target-d768',
        dim=768)


def test_swap_clears_meta_after_done(swap_backend):
    """All embed_swap_* meta keys are deleted after a successful cutover.

    Mutation: `run_swap` deleting only some of the `_META_KEYS`, so
        `read_progress` or the stale-swap doctor check sees a swap
        in flight.
    Oracle: no meta key with the `embed_swap_` prefix remains.
    """
    _seed_insights(swap_backend, 2)
    ec = _StubEmbedder(dim=768)
    plan = SwapPlan(
        target_model='stub-target-d768',
        target_dim=768)

    run_swap(swap_backend, ec, plan)

    leftover = [
        k for k in swap_backend.meta.keys()  # noqa: SIM118
        if k.startswith('embed_swap_')]
    assert leftover == []


def test_swap_resume_skips_already_filled_rows(swap_backend, monkeypatch):
    """A second run after a partial backfill resumes from the cursor.

    Mutation: `run_swap` ignoring the stored cursor and re-embedding
        from the first row, overwriting the pre-filled vectors.
    Oracle: the first three rows keep the hand-written `fake_pending`
        blob after cutover.
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
            'embed_swap_target_model', 'stub-target-d768')
        swap_backend.meta.set('embed_swap_target_dim', '768')

    plan = SwapPlan(
        target_model='stub-target-d768',
        target_dim=768)
    monkeypatch.setenv('MEMMAN_EMBED_SWAP_BATCH_SIZE', '2')
    progress = run_swap(swap_backend, ec, plan)

    assert progress.state == STATE_DONE
    rows = swap_backend._db._query(
        'select id, embedding from insights'
        ' order by id').fetchall()
    first_three_blobs = [r[1] for r in rows[:3]]
    assert all(blob == fake_pending for blob in first_three_blobs)


def test_swap_abort_clears_pending_and_meta(swap_backend):
    """abort_swap nulls embedding_pending and clears all swap meta.

    Mutation: `abort_swap` leaving filled `embedding_pending` values
        or the swap state meta behind.
    Oracle: `read_progress` state is empty and every pending value is
        null.
    """
    ids = _seed_insights(swap_backend, 4)
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

    Mutation: `run_swap` dropping the target comparison, silently
        switching targets across a running backfill.
    Oracle: `RuntimeError` whose message mentions `in-flight`.
    """
    _seed_insights(swap_backend, 3)
    ec_first = _StubEmbedder(dim=768)
    swap_backend.swap_prepare(768)
    with swap_backend.transaction():
        swap_backend.meta.set('embed_swap_state', 'backfilling')
        swap_backend.meta.set(
            'embed_swap_target_model', 'stub-target-d768')
        swap_backend.meta.set('embed_swap_target_dim', '768')

    plan_diff = SwapPlan(
        target_model='stub-target-d1024',
        target_dim=1024)

    with pytest.raises(RuntimeError) as exc:
        run_swap(swap_backend, ec_first, plan_diff)
    assert 'in-flight' in str(exc.value).lower()


def test_swap_abort_refuses_once_the_cutover_state_is_recorded(
        swap_backend, monkeypatch):
    """abort_swap refuses a swap whose cutover may already have landed.

    Mutation: the bug itself - an abort after a crash past the
        committed cutover clears the swap keys and leaves the old
        fingerprint over the new vectors, so recall embeds queries with
        a model the stored vectors no longer match.
    Oracle: the swap state and the old fingerprint both still in place
        after the refused abort, so --resume can finish the swap.
    """
    _seed_insights(swap_backend, 3)
    old_fp = Fingerprint(model='voyage-3-lite', dim=512)
    write_fingerprint(swap_backend, old_fp)
    plan = SwapPlan(
        target_model='stub-target-d768',
        target_dim=768)

    def _crash(*args, **kwargs):
        raise RuntimeError('crash after cutover')

    monkeypatch.setattr('memman.embed.swap.write_fingerprint', _crash)
    with pytest.raises(RuntimeError, match='crash after cutover'):
        run_swap(swap_backend, _StubEmbedder(dim=768), plan)

    with pytest.raises(RuntimeError, match='--resume'):
        abort_swap(swap_backend)

    assert read_progress(swap_backend).state == 'cutover'
    assert stored_fingerprint(swap_backend) == old_fp


def test_swap_abort_refuses_while_another_handle_holds_the_lock(
        swap_backend, tmp_path):
    """abort_swap refuses while another handle on the store holds the lock.

    Mutation: a SQLite `swap_lock` that always yields True, so an abort
        from a second shell clears `embedding_pending` under a running
        swap whose cutover then copies nothing and writes the target
        fingerprint over the old vectors.
    Oracle: the pending vector the first handle wrote, still present
        after the refused abort.
    """
    ids = _seed_insights(swap_backend, 2)
    with swap_backend.transaction():
        swap_backend._db._exec(
            'update insights set embedding_pending = ? where id = ?',
            (serialize_vector([0.5] * 768), ids[0]))
        swap_backend.meta.set('embed_swap_state', 'backfilling')
    other_db = open_db(str(tmp_path))
    try:
        with swap_backend.swap_lock() as held:
            assert held is True
            with pytest.raises(RuntimeError, match='another swap'):
                abort_swap(SqliteBackend(other_db))
    finally:
        other_db.close()

    pending = swap_backend._db._query(
        'select embedding_pending from insights where id = ?',
        (ids[0],)).fetchone()
    assert pending[0] is not None
