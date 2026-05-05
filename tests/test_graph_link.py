"""Tests for link_pending and linked_at lifecycle."""

from datetime import datetime, timezone

from memman.graph.engine import MAX_LINK_BATCH, link_pending
from memman.graph.engine import reindex_auto_edges
from memman.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _insert_pending(db, insight_id, content='test content', **kw):
    """Insert an insight with linked_at = NULL."""
    insert_insight(db, make_insight(id=insight_id, content=content, **kw))
    db._conn.execute(
        'UPDATE insights SET linked_at = NULL WHERE id = ?',
        (insight_id,))


class TestLinkPending:
    """link_pending processes insights with linked_at IS NULL."""

    def test_processes_null_linked_at(self, tmp_db, tmp_backend):
        """Insights with NULL linked_at get stamped after processing."""
        _insert_pending(tmp_db, 'cp-1', 'database migration completed')
        _insert_pending(tmp_db, 'cp-2', 'schema update for production')

        processed = link_pending(tmp_backend)
        assert processed == 2

        row = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL AND deleted_at IS NULL'
            ).fetchone()
        assert row[0] == 0

    def test_skips_already_linked(self, tmp_db, tmp_backend):
        """Insights with linked_at set are not re-processed."""
        insert_insight(tmp_db, make_insight(
            id='ac-1', content='already linked insight'))
        tmp_db._conn.execute(
            "UPDATE insights SET linked_at = created_at"
            " WHERE id = 'ac-1'")

        processed = link_pending(tmp_backend)
        assert processed == 0

    def test_batch_cap_respected(self, tmp_db, tmp_backend):
        """Only MAX_LINK_BATCH insights processed per call."""
        for i in range(MAX_LINK_BATCH + 5):
            _insert_pending(
                tmp_db, f'batch-{i}',
                f'batch content number {i}')

        processed = link_pending(tmp_backend)
        assert processed == MAX_LINK_BATCH

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 5

    def test_max_batch_parameter_respected(self, tmp_db, tmp_backend):
        """max_batch parameter caps processing to the given count."""
        for i in range(5):
            _insert_pending(tmp_db, f'rb-{i}', f'recall batch {i}')

        processed = link_pending(tmp_backend, max_batch=2)
        assert processed == 2

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 3

    def test_llm_none_skips_inference(self, tmp_db, tmp_backend):
        """llm_client=None processes insights without LLM calls."""
        _insert_pending(tmp_db, 'ln-1', 'content for llm none test')

        processed = link_pending(tmp_backend, llm_client=None)
        assert processed == 1

        row = tmp_db._conn.execute(
            'SELECT linked_at FROM insights WHERE id = ?',
            ('ln-1',)).fetchone()
        assert row[0] is not None

    def test_zero_pending_noop(self, tmp_db, tmp_backend):
        """No pending insights returns 0 without error."""
        insert_insight(tmp_db, make_insight(
            id='zp-1', content='all linked'))
        tmp_db._conn.execute(
            "UPDATE insights SET linked_at = created_at"
            " WHERE id = 'zp-1'")
        processed = link_pending(tmp_backend)
        assert processed == 0

    def test_llm_calls_outside_transaction(self, tmp_db, tmp_backend):
        """LLM HTTP calls must not occur inside BEGIN IMMEDIATE."""
        _insert_pending(tmp_db, 'tx-1', 'transaction test content')

        call_log = []

        class TxTrackingLLM:
            def complete(self, system, user):
                call_log.append({
                    'in_tx': tmp_db._in_tx,
                    'call': 'complete',
                    })
                return '[]'

        link_pending(tmp_backend, llm_client=TxTrackingLLM())

        llm_calls_in_tx = [c for c in call_log if c['in_tx']]
        assert llm_calls_in_tx == [], (
            f'LLM calls made inside transaction: {llm_calls_in_tx}')

    def test_progress_callback_called(self, tmp_db, tmp_backend):
        """on_progress receives enrich, causal, done stages per insight."""
        _insert_pending(tmp_db, 'pc-1', 'callback test content')

        calls = []

        def on_progress(stage, insight):
            calls.append((stage, insight.id))

        link_pending(tmp_backend, on_progress=on_progress)

        stages = [c[0] for c in calls]
        assert 'enrich' in stages
        assert 'causal' in stages
        assert 'done' in stages
        assert all(c[1] == 'pc-1' for c in calls)

    def test_relink_clears_linked_at(self, tmp_db, tmp_backend):
        """reindex_auto_edges clears linked_at for re-linking."""
        insert_insight(tmp_db, make_insight(
            id='rb-1', content='relink test one'))
        insert_insight(tmp_db, make_insight(
            id='rb-2', content='relink test two'))
        tmp_db._conn.execute(
            'UPDATE insights SET linked_at = created_at')

        reindex_auto_edges(tmp_backend)

        pending = tmp_db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL AND deleted_at IS NULL'
            ).fetchone()[0]
        assert pending == 2
