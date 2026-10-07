"""Tests for enrich_pending.
"""

from datetime import datetime, timezone

from memman.pipeline import enrich as enrich_mod
from memman.pipeline.enrich import MAX_ENRICH_BATCH, enrich_pending
from memman.store.node import insert_insight
from tests.conftest import insert_pending as _insert_pending
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


class TestEnrichPending:
    """enrich_pending processes insights with enrich_attempted_at is null.
    """

    def test_processes_null_enrich_attempted_at(self, tmp_db, tmp_backend):
        """Verify pending insights are stamped after processing.

        Mutation: enrich_pending skipping the stamp_enrich_attempted
            call, leaving the row pending forever.
        Oracle: a post-run SQL count of rows with enrich_attempted_at
            still NULL.
        """
        _insert_pending(tmp_db, 'cp-1', 'database migration completed')
        _insert_pending(tmp_db, 'cp-2', 'schema update for production')

        processed = enrich_pending(tmp_backend)
        assert processed == 2

        row = tmp_db._conn.execute(
            'select count(*) from insights'
            ' where enrich_attempted_at is null and deleted_at is null'
            ).fetchone()
        assert row[0] == 0

    def test_skips_already_attempted(self, tmp_db, tmp_backend):
        """Verify an already-attempted insight is not re-processed.

        Mutation: get_pending_enrich_ids dropping the
            `enrich_attempted_at is null` filter, sweeping in an
            already-attempted row.
        Oracle: processed count of 0 for a store holding one
            pre-stamped row.
        """
        insert_insight(tmp_db, make_insight(
            id='ac-1', content='already attempted insight'))
        tmp_db._conn.execute(
            "update insights set enrich_attempted_at = created_at"
            " where id = 'ac-1'")

        processed = enrich_pending(tmp_backend)
        assert processed == 0

    def test_batch_cap_respected(self, tmp_db, tmp_backend):
        """Verify one call processes at most MAX_ENRICH_BATCH insights.

        Mutation: get_pending_enrich_ids ignoring its limit argument,
            or enrich_pending defaulting max_batch to an unbounded
            value.
        Oracle: exact processed count of MAX_ENRICH_BATCH against a
            backlog of MAX_ENRICH_BATCH + 5 rows, with 5 left pending.
        """
        for i in range(MAX_ENRICH_BATCH + 5):
            _insert_pending(
                tmp_db, f'batch-{i}',
                f'batch content number {i}')

        processed = enrich_pending(tmp_backend)
        assert processed == MAX_ENRICH_BATCH

        pending = tmp_db._conn.execute(
            'select count(*) from insights'
            ' where enrich_attempted_at is null and deleted_at is null'
            ).fetchone()[0]
        assert pending == 5

    def test_max_batch_parameter_respected(self, tmp_db, tmp_backend):
        """Verify max_batch caps processing to the given count.

        Mutation: enrich_pending ignoring its max_batch argument and
            passing MAX_ENRICH_BATCH to get_pending_enrich_ids
            instead.
        Oracle: processed count of 2 against 5 pending rows, with 3
            left pending.
        """
        for i in range(5):
            _insert_pending(tmp_db, f'rb-{i}', f'recall batch {i}')

        processed = enrich_pending(tmp_backend, max_batch=2)
        assert processed == 2

        pending = tmp_db._conn.execute(
            'select count(*) from insights'
            ' where enrich_attempted_at is null and deleted_at is null'
            ).fetchone()[0]
        assert pending == 3

    def test_unreachable_llm_still_attempts(
            self, tmp_db, tmp_backend, monkeypatch):
        """Verify an unreachable LLM still stamps enrich_attempted_at.

        Mutation: letting get_llm_client's exception propagate out of
            enrich_pending, or skipping stamp_enrich_attempted when
            enrichment raises, so a failing row loops forever.
        Oracle: the row's enrich_attempted_at column read back
            not-None after the raised RuntimeError.
        """
        def _unavailable(*args, **kwargs):
            raise RuntimeError('no LLM credential')

        monkeypatch.setattr(enrich_mod, 'get_llm_client', _unavailable)
        _insert_pending(tmp_db, 'ln-1', 'content for llm none test')

        processed = enrich_pending(tmp_backend)
        assert processed == 1

        row = tmp_db._conn.execute(
            'select enrich_attempted_at from insights where id = ?',
            ('ln-1',)).fetchone()
        assert row[0] is not None

    def test_zero_pending_noop(self, tmp_db, tmp_backend):
        """Verify no pending insights returns 0 without error.

        Mutation: enrich_pending raising, or returning a nonzero
            count, when get_pending_enrich_ids finds no rows.
        Oracle: the returned processed count equals 0.
        """
        insert_insight(tmp_db, make_insight(
            id='zp-1', content='all attempted'))
        tmp_db._conn.execute(
            "update insights set enrich_attempted_at = created_at"
            " where id = 'zp-1'")
        processed = enrich_pending(tmp_backend)
        assert processed == 0

    def test_llm_calls_outside_transaction(self, tmp_db, tmp_backend):
        """Verify LLM calls run outside the write transaction.

        Mutation: moving the `enrich_with_llm` call inside the
            `with backend.transaction():` block, holding the write
            lock for the network round trip.
        Oracle: the store's `_in_tx` flag recorded at the moment the
            stub LLM's `complete` ran.
        """
        _insert_pending(tmp_db, 'tx-1', 'transaction test content')

        call_log = []

        class TxTrackingLLM:
            model = 'tx-model'

            def complete(self, system, user, **kwargs):
                call_log.append({
                    'in_tx': tmp_db._in_tx,
                    'call': 'complete',
                    })
                return '[]'

        enrich_pending(tmp_backend, llm_client=TxTrackingLLM())

        llm_calls_in_tx = [c for c in call_log if c['in_tx']]
        assert llm_calls_in_tx == [], (
            f'LLM calls made inside transaction: {llm_calls_in_tx}')

    def test_progress_callback_called(self, tmp_db, tmp_backend):
        """Verify on_progress receives enrich and done stages per insight.

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

        enrich_pending(tmp_backend, on_progress=on_progress)

        stages = [c[0] for c in calls]
        assert 'enrich' in stages
        assert 'causal' not in stages
        assert 'done' in stages
        assert all(c[1] == 'pc-1' for c in calls)


class _StubLLM:
    """An LLM client that answers one fixed body under a fixed model id.
    """

    def __init__(self, model, body):
        self.model = model
        self.body = body

    def complete(self, system, user, **kwargs):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class TestSummaryModel:
    """enrich_pending stores the client's model id beside the summary.
    """

    def test_stamps_the_clients_model(self, backend):
        """Verify a pass stores the model id of the client it ran.

        Mutation: `enrich_pending` passing the configured model or a
            constant for `summary_model` instead of `llm_client.model`,
            or dropping the argument so the column stays null.
        Oracle: the stub client's model id, a string no config holds.
        """
        backend.nodes.insert(make_insight(
            id='sm-1', content='Redis backs the session cache'))
        client = _StubLLM('stub-vendor/stub-model', '{"summary": "Cache."}')

        enrich_pending(backend, llm_client=client)

        row = backend.nodes.get_raw('sm-1')
        assert row.summary == 'Cache.'
        assert row.summary_model == 'stub-vendor/stub-model'

    def test_failed_enrichment_leaves_summary_model_unset(self, backend):
        """Verify a failed enrichment call records no model.

        Mutation: writing `summary_model` outside the `if enrichment`
            guard, so a row the model never answered names a model.
        Oracle: the stored row's `summary_model`, null beside its null
            summary, after a client that raises.
        """
        backend.nodes.insert(make_insight(
            id='sm-2', content='Redis backs the session cache'))
        client = _StubLLM(
            'stub-vendor/stub-model', ConnectionError('forced failure'))

        enrich_pending(backend, llm_client=client)

        row = backend.nodes.get_raw('sm-2')
        assert row.summary is None
        assert row.summary_model is None

    def test_empty_summary_still_records_the_model(self, backend):
        """Verify an empty summary is stored with the model that returned it.

        Mutation: writing `summary_model` only when the summary is
            non-empty, so a parse failure or a length-guard drop
            hides which model produced it.
        Oracle: a stub body whose summary is '', which `enrich_with_llm`
            returns as `{'summary': ''}`.
        """
        backend.nodes.insert(make_insight(
            id='sm-3', content='Redis backs the session cache'))
        client = _StubLLM('stub-vendor/stub-model', '{"summary": ""}')

        enrich_pending(backend, llm_client=client)

        row = backend.nodes.get_raw('sm-3')
        assert row.summary == ''
        assert row.summary_model == 'stub-vendor/stub-model'
