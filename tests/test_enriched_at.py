"""Tests for enriched_at column lifecycle in link_pending."""

from unittest.mock import MagicMock

from memman.graph.engine import link_pending
from memman.store.node import insert_insight
from tests.conftest import make_insight


def _insert_pending(db, insight_id, content='test content'):
    """Insert an insight with linked_at = NULL."""
    insert_insight(db, make_insight(id=insight_id, content=content))
    db._conn.execute(
        'UPDATE insights SET linked_at = NULL WHERE id = ?',
        (insight_id,))


class TestEnrichedAtColumn:
    """enriched_at column exists and is backfilled from migration."""

    def test_column_exists(self, tmp_db):
        """Fresh DB has enriched_at column."""
        cols = tmp_db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        col_names = {row[1] for row in cols}
        assert 'enriched_at' in col_names

    def test_backfill_from_linked_at(self, tmp_db):
        """Insights with linked_at get enriched_at backfilled."""
        insert_insight(tmp_db, make_insight(
            id='bf-1', content='backfill test'))
        row = tmp_db._conn.execute(
            'SELECT enriched_at, linked_at FROM insights'
            " WHERE id = 'bf-1'").fetchone()
        assert row[0] == row[1]


class TestEnrichedAtOnLinkPending:
    """link_pending sets enriched_at only when LLM enrichment succeeds."""

    def test_no_llm_sets_linked_at_only(self, tmp_db, tmp_backend):
        """Without LLM client, linked_at is set but enriched_at stays NULL."""
        _insert_pending(tmp_db, 'nl-1', 'test without llm')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'nl-1'")

        link_pending(tmp_backend, llm_client=None)

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'nl-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None

    def test_llm_success_sets_enriched_at(self, tmp_db, tmp_backend):
        """With successful LLM enrichment, enriched_at is set."""
        _insert_pending(tmp_db, 'ls-1', 'test with llm enrichment')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'ls-1'")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"keywords": ["test"], "summary": "test",'
            ' "semantic_facts": [], "entities": []}')

        link_pending(tmp_backend, llm_client=mock_llm)

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'ls-1'").fetchone()
        assert row[0] is not None
        assert row[1] is not None
