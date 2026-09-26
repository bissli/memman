"""Tests for link_pending."""

from datetime import datetime, timezone

from memman.graph.engine import MAX_LINK_BATCH, link_pending
from memman.store.node import insert_insight
from tests.conftest import insert_pending as _insert_pending
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


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

    def test_unreachable_llm_still_links(
            self, tmp_db, tmp_backend, monkeypatch):
        """An unreachable LLM still links the insight."""
        from memman.graph import engine as engine_mod

        def _unavailable(*args, **kwargs):
            raise RuntimeError('no LLM credential')

        monkeypatch.setattr(engine_mod, 'get_llm_client', _unavailable)
        _insert_pending(tmp_db, 'ln-1', 'content for llm none test')

        processed = link_pending(tmp_backend)
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
            def complete(self, system, user, **kwargs):
                call_log.append({
                    'in_tx': tmp_db._in_tx,
                    'call': 'complete',
                    })
                return '[]'

        link_pending(tmp_backend, metadata_llm_client=TxTrackingLLM())

        llm_calls_in_tx = [c for c in call_log if c['in_tx']]
        assert llm_calls_in_tx == [], (
            f'LLM calls made inside transaction: {llm_calls_in_tx}')

    def test_progress_callback_called(self, tmp_db, tmp_backend):
        """on_progress receives enrich and done stages per insight.

        Mutation: dropping an `on_progress` call, or emitting a stage
            name no consumer expects -- a caller rendering a progress
            bar then stalls on a stage that never arrives.
            `'causal'` is asserted absent because its emitter is gone.
        Oracle: the stage list collected by the callback, checked
            against the closed set the pass can emit.
        """
        _insert_pending(tmp_db, 'pc-1', 'callback test content')

        calls = []

        def on_progress(stage, insight):
            calls.append((stage, insight.id))

        link_pending(tmp_backend, on_progress=on_progress)

        stages = [c[0] for c in calls]
        assert 'enrich' in stages
        assert 'causal' not in stages
        assert 'done' in stages
        assert all(c[1] == 'pc-1' for c in calls)
