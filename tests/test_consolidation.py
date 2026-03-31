"""Tests for consolidate_pending and consolidated_at lifecycle."""

from datetime import datetime, timedelta, timezone

from mnemon.graph.engine import MAX_CONSOLIDATION_BATCH, consolidate_pending
from mnemon.graph.engine import rebuild_auto_edges
from mnemon.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _insert_pending(db, insight_id, content='test content', **kw):
    """Insert an insight with consolidated_at = NULL."""
    insert_insight(db, make_insight(id=insight_id, content=content, **kw))
    db._conn.execute(
        'UPDATE insights SET consolidated_at = NULL WHERE id = ?',
        (insight_id,))


class TestConsolidatePending:
    """consolidate_pending processes insights with consolidated_at IS NULL."""

    def test_processes_null_consolidated_at(self, tmp_db):
        """Insights with NULL consolidated_at get stamped after processing."""
        _insert_pending(tmp_db, 'cp-1', 'database migration completed')
        _insert_pending(tmp_db, 'cp-2', 'schema update for production')

        processed = consolidate_pending(tmp_db)
        assert processed == 2

        row = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE consolidated_at IS NULL AND deleted_at IS NULL'
            ).fetchone()
        assert row[0] == 0

    def test_skips_already_consolidated(self, tmp_db):
        """Insights with consolidated_at set are not re-processed."""
        insert_insight(tmp_db, make_insight(
            id='ac-1', content='already consolidated insight'))
        tmp_db._conn.execute(
            "UPDATE insights SET consolidated_at = created_at"
            " WHERE id = 'ac-1'")

        processed = consolidate_pending(tmp_db)
        assert processed == 0

    def test_batch_cap_respected(self, tmp_db):
        """Only MAX_CONSOLIDATION_BATCH insights processed per call."""
        for i in range(MAX_CONSOLIDATION_BATCH + 5):
            _insert_pending(
                tmp_db, f'batch-{i}',
                f'batch content number {i}',
                created_at=OLD + timedelta(seconds=i))

        processed = consolidate_pending(tmp_db)
        assert processed == MAX_CONSOLIDATION_BATCH

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE consolidated_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 5

    def test_recall_path_batch_cap(self, tmp_db):
        """max_batch=2 limits processing to 2 insights."""
        for i in range(5):
            _insert_pending(tmp_db, f'rb-{i}', f'recall batch {i}')

        processed = consolidate_pending(tmp_db, max_batch=2)
        assert processed == 2

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE consolidated_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 3

    def test_llm_none_skips_inference(self, tmp_db):
        """llm_client=None processes insights without LLM calls."""
        _insert_pending(tmp_db, 'ln-1', 'content for llm none test')

        processed = consolidate_pending(tmp_db, llm_client=None)
        assert processed == 1

        row = tmp_db._conn.execute(
            'SELECT consolidated_at FROM insights WHERE id = ?',
            ('ln-1',)).fetchone()
        assert row[0] is not None

    def test_zero_pending_noop(self, tmp_db):
        """No pending insights returns 0 without error."""
        insert_insight(tmp_db, make_insight(
            id='zp-1', content='all consolidated'))
        tmp_db._conn.execute(
            "UPDATE insights SET consolidated_at = created_at"
            " WHERE id = 'zp-1'")
        processed = consolidate_pending(tmp_db)
        assert processed == 0

    def test_rebuild_stamps_consolidated_at(self, tmp_db):
        """rebuild_auto_edges sets consolidated_at on all active insights."""
        insert_insight(tmp_db, make_insight(
            id='rb-1', content='rebuild test one'))
        insert_insight(tmp_db, make_insight(
            id='rb-2', content='rebuild test two'))
        tmp_db._conn.execute(
            'UPDATE insights SET consolidated_at = NULL')

        rebuild_auto_edges(tmp_db)

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE consolidated_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 0
