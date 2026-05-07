"""Cross-backend contract for `BaseNodeStore` defaulted verbs.

Locks the behavior of the four NodeStore verbs that compose from
other Protocol verbs and have a Python-side default in
`memman.store.base`. Both SQLite and Postgres implementations
must continue to return identical shapes after the override is
dropped (Postgres: `review_content_quality`) or the default is
inherited (SQLite: `iter_embeddings_as_vecs`).

`get_pending_link_ids`, `count_pending_links`, and `clear_linked_at`
are deliberately NOT defaulted in `BaseNodeStore` even though
`Insight` now carries `linked_at` and a Python-level filter is
formally possible. Reason: Postgres has indexed pushdown via the
partial index `idx_insights_pending_link_{schema}` (defined at
`src/memman/store/postgres.py:_PG_BASELINE_SCHEMA`), so a
default that calls `get_all_active()` and filters in Python would
fetch every row instead of the index-only scan. The SQLite verbs
are tiny single-column queries; collapsing them into a default
buys nothing and would only obscure the perf-critical Postgres
path. `boost_retention` is excluded because `increment_access_count`
skips `updated_at` and cannot stand in for it.
"""

from tests.conftest import _vec, make_insight


def _seed(backend, ids: list[tuple[str, str, str]],
          *, with_embedding: list[str] = ()) -> None:
    """Insert (id, content, source) rows; optionally attach embeddings.
    """
    embed_ids = set(with_embedding)
    with backend.transaction():
        for rid, content, source in ids:
            backend.nodes.insert(
                make_insight(
                    id=rid, content=content, source=source,
                    importance=3))
        for rid in embed_ids:
            backend.nodes.update_embedding(
                rid, _vec(0.1, 0.2, 0.3), 'voyage-3-lite')


class TestHasActiveWithSource:
    """`has_active_with_source` reports presence by source string."""

    def test_returns_true_when_source_present(self, backend):
        """Active row with the source returns True.
        """
        _seed(backend, [('h-1', 'a', 'cli')])
        assert backend.nodes.has_active_with_source('cli') is True

    def test_returns_false_when_source_absent(self, backend):
        """No row with the source returns False.
        """
        _seed(backend, [('h-2', 'a', 'cli')])
        assert backend.nodes.has_active_with_source('hook') is False

    def test_excludes_soft_deleted(self, backend):
        """Soft-deleted rows do not count as active.
        """
        _seed(backend, [('h-3', 'a', 'cli')])
        backend.nodes.soft_delete('h-3')
        assert backend.nodes.has_active_with_source('cli') is False


class TestGetWithoutEmbedding:
    """`get_without_embedding` returns active rows lacking embeddings."""

    def test_returns_only_unembedded(self, backend):
        """Rows with embedding are excluded.
        """
        _seed(
            backend,
            [('w-1', 'a', 'cli'), ('w-2', 'b', 'cli')],
            with_embedding=['w-1'])
        rows = backend.nodes.get_without_embedding(limit=10)
        ids = {r.id for r in rows}
        assert ids == {'w-2'}

    def test_respects_limit(self, backend):
        """Returns at most `limit` rows.
        """
        _seed(
            backend,
            [(f'w-{i:02d}', f'c{i}', 'cli') for i in range(5)])
        rows = backend.nodes.get_without_embedding(limit=2)
        assert len(rows) == 2

    def test_excludes_soft_deleted(self, backend):
        """Soft-deleted rows are excluded.
        """
        _seed(backend, [('w-d1', 'a', 'cli')])
        backend.nodes.soft_delete('w-d1')
        rows = backend.nodes.get_without_embedding(limit=10)
        assert all(r.id != 'w-d1' for r in rows)


class TestReviewContentQuality:
    """`review_content_quality` flags content with transient patterns."""

    def test_flags_transient_marker(self, backend):
        """Rows containing the 'currently' marker are flagged.
        """
        _seed(
            backend,
            [('rcq-1', 'this is currently broken', 'cli'),
             ('rcq-2', 'a stable observation', 'cli')])
        flagged = backend.nodes.review_content_quality(limit=10)
        ids = {f['insight'].id for f in flagged}
        assert 'rcq-1' in ids
        assert 'rcq-2' not in ids

    def test_returns_warnings_per_row(self, backend):
        """Each flagged row carries a non-empty warnings list.
        """
        _seed(backend, [('rcq-3', 'state is clean', 'cli')])
        flagged = backend.nodes.review_content_quality(limit=10)
        assert flagged
        assert all(f['quality_warnings'] for f in flagged)

    def test_respects_limit(self, backend):
        """Returns at most `limit` flagged rows.
        """
        _seed(
            backend,
            [(f'rcq-l{i}', 'currently broken', 'cli')
             for i in range(5)])
        flagged = backend.nodes.review_content_quality(limit=2)
        assert len(flagged) == 2


class TestIterEmbeddingsAsVecs:
    """`iter_embeddings_as_vecs` yields (id, vec) pairs."""

    def test_yields_only_embedded(self, backend):
        """Only rows with an embedding are yielded.
        """
        _seed(
            backend,
            [('iev-1', 'a', 'cli'), ('iev-2', 'b', 'cli')],
            with_embedding=['iev-1'])
        emitted = dict(backend.nodes.iter_embeddings_as_vecs())
        assert 'iev-1' in emitted
        assert 'iev-2' not in emitted

    def test_yields_list_of_floats(self, backend):
        """Each yielded vec is a list of numeric scalars.
        """
        _seed(
            backend, [('iev-3', 'a', 'cli')],
            with_embedding=['iev-3'])
        for _id, vec in backend.nodes.iter_embeddings_as_vecs():
            assert isinstance(vec, list)
            assert vec
            assert all(float(x) == float(x) for x in vec)

    def test_excludes_soft_deleted(self, backend):
        """Soft-deleted rows are not yielded.
        """
        _seed(
            backend, [('iev-4', 'a', 'cli')],
            with_embedding=['iev-4'])
        backend.nodes.soft_delete('iev-4')
        emitted = dict(backend.nodes.iter_embeddings_as_vecs())
        assert 'iev-4' not in emitted
