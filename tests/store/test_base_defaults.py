"""Cross-backend contract for `BaseNodeStore` defaulted verbs.

Locks the behavior of the NodeStore verb that composes from other
Protocol verbs and has a Python-side override in
`memman.store.base`: `review_content_quality`. Both SQLite and
Postgres implementations must continue to return identical shapes.

`has_active_with_queue_uuid` (which replaced the defaulted
`has_active_with_source`) deliberately has NO Python-side default:
a `==` scan would match legacy null `queue_uuid` rows against a
None argument, where SQL's `= ?` never matches NULL.

`get_pending_link_ids` and `count_pending_links` are deliberately
NOT defaulted in `BaseNodeStore` even though `Insight` now carries
`linked_at` and a Python-level filter is formally possible. Reason:
Postgres has indexed pushdown via the partial index
`idx_insights_pending_link_{schema}` (defined at
`src/memman/store/postgres.py:PG_BASELINE_SCHEMA`), so a
default that calls `get_all_active()` and filters in Python would
fetch every row instead of the index-only scan. The SQLite verbs
are tiny single-column queries; collapsing them into a default
buys nothing and would only obscure the perf-critical Postgres
path.
"""

from tests.conftest import make_insight


def _seed(backend, ids: list[tuple[str, str, str]]) -> None:
    """Insert (id, content, source) rows.
    """
    with backend.transaction():
        for rid, content, source in ids:
            backend.nodes.insert(
                make_insight(
                    id=rid, content=content, source=source,
                    importance=3))


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
