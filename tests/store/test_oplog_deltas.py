"""Oplog `before` / `after` content deltas.

Persists pre- and post-state on reconcile / forget / auto_prune so
forensic questions ("what did insight X say before reconciliation?")
can be answered from the oplog instead of requiring a backup.
"""

import json

import pytest
from memman.store.model import insight_to_delta_dict
from tests.conftest import make_insight


class TestOplogLogAcceptsDeltas:
    """`Oplog.log` accepts and persists `before` / `after` kwargs."""

    def test_round_trips_before_and_after(self, backend):
        """Both fields populated on log read back via `recent`.
        """
        with backend.transaction():
            backend.oplog.log(
                operation='reconcile-update',
                insight_id='x-1',
                detail='replaced',
                before={'content': 'old', 'importance': 3},
                after={'content': 'new', 'importance': 4})
        entries = backend.oplog.recent(limit=1)
        assert entries
        e = entries[0]
        assert e.before == {'content': 'old', 'importance': 3}
        assert e.after == {'content': 'new', 'importance': 4}

    def test_default_none_preserves_legacy_call_sites(self, backend):
        """Logging with no before/after kwargs leaves both null.
        """
        with backend.transaction():
            backend.oplog.log(
                operation='remember',
                insight_id='x-2',
                detail='legacy')
        entries = backend.oplog.recent(limit=1)
        assert entries
        assert entries[0].before is None
        assert entries[0].after is None

    def test_only_before_for_forget_shape(self, backend):
        """Forget records before only (no after).
        """
        with backend.transaction():
            backend.oplog.log(
                operation='forget', insight_id='x-3',
                detail='', before={'content': 'gone'})
        entries = backend.oplog.recent(limit=1)
        assert entries[0].before == {'content': 'gone'}
        assert entries[0].after is None


class TestInsightToDeltaDict:
    """`insight_to_delta_dict` shapes the dict for oplog deltas."""

    def test_includes_content_and_metadata(self):
        """The delta dict carries content/category/importance/source.
        """
        ins = make_insight(
            id='d-1', content='hello', category='fact',
            importance=4, source='cli')
        d = insight_to_delta_dict(ins)
        assert d['content'] == 'hello'
        assert d['category'] == 'fact'
        assert d['importance'] == 4
        assert d['source'] == 'cli'

    def test_round_trips_through_json(self):
        """The delta dict is JSON-serializable as written.
        """
        ins = make_insight(id='d-2', content='x', entities=['a', 'b'])
        d = insight_to_delta_dict(ins)
        json.dumps(d)


@pytest.fixture
def _sched_started(monkeypatch):
    """Force scheduler state to STARTED so write CLI verbs proceed.
    """
    from memman.setup import scheduler as sched_mod
    monkeypatch.setattr(
        sched_mod, 'read_state', lambda: sched_mod.STATE_STARTED)


class TestForgetWritesBefore:
    """`memman forget <id>` records the pre-deletion content."""

    def test_forget_logs_before(self, backend, _sched_started):
        """Forget oplog row carries the deleted insight's content.
        """
        with backend.transaction():
            backend.nodes.insert(
                make_insight(id='f-1', content='goodbye world'))
        from memman.cli import _forget_insight
        _forget_insight(backend, 'f-1')
        entries = [
            e for e in backend.oplog.recent(limit=10)
            if e.operation == 'forget' and e.insight_id == 'f-1']
        assert entries
        assert entries[0].before is not None
        assert entries[0].before.get('content') == 'goodbye world'


class TestDoctorOplogDeltaCoverage:
    """`memman doctor` surfaces `oplog_delta_coverage` percentage."""

    def test_coverage_reports_percentage(self, backend):
        """Mix of populated and bare oplog rows yields 50%.
        """
        with backend.transaction():
            backend.oplog.log(
                operation='reconcile-update', insight_id='m-1',
                detail='', before={'a': 1}, after={'a': 2})
            backend.oplog.log(
                operation='remember', insight_id='m-2', detail='')
        from memman.doctor import check_oplog_delta_coverage
        result = check_oplog_delta_coverage(backend)
        assert result['name'] == 'oplog_delta_coverage'
        assert result['detail']['total_oplog_rows'] == 2
        assert (
            result['detail']['rows_with_delta'] == 1)
        assert (
            result['detail']['coverage_pct'] == 50.0)
