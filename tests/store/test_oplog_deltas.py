"""Oplog `before` / `after` content deltas.

The oplog persists pre- and post-state on replace and forget, so a
question such as "what did insight X say before it was replaced?" is
answered from the oplog, without a backup.
"""

import json

import pytest
from memman.cli import _forget_insight
from memman.setup import scheduler as sched_mod
from memman.store.model import insight_to_delta_dict
from tests.conftest import make_insight


class TestOplogLogAcceptsDeltas:
    """`Oplog.log` accepts and persists `before` / `after` kwargs.
    """

    def test_round_trips_before_and_after(self, backend):
        """Both fields populated on log read back via `recent`.

        Mutation: `Oplog.log` persisting only one of the two deltas, or
            swapping `before` and `after`.
        Oracle: the hand-written dicts passed in.
        """
        with backend.transaction():
            backend.oplog.log(
                operation='replace',
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

        Mutation: `Oplog.log` storing `{}` or `'null'` text for an
            omitted delta instead of None.
        Oracle: `None` for both fields on a hand-logged plain entry.
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

        Mutation: `Oplog.log` defaulting `after` to a copy of `before`
            or an empty dict.
        Oracle: `before` equals the passed dict and `after` is None.
        """
        with backend.transaction():
            backend.oplog.log(
                operation='forget', insight_id='x-3',
                detail='', before={'content': 'gone'})
        entries = backend.oplog.recent(limit=1)
        assert entries[0].before == {'content': 'gone'}
        assert entries[0].after is None


class TestInsightToDeltaDict:
    """`insight_to_delta_dict` shapes the dict for oplog deltas.
    """

    def test_includes_content_and_metadata(self):
        """The delta dict carries content and category.

        Mutation: dropping the content or category key from the dict.
        Oracle: the two fields the insight was built with.
        """
        ins = make_insight(id='d-1', content='hello', category='fact')
        d = insight_to_delta_dict(ins)
        assert d['content'] == 'hello'
        assert d['category'] == 'fact'

    def test_round_trips_through_json(self):
        """The delta dict is JSON-serializable as written.

        Mutation: a non-serializable value (e.g. a datetime) left in
            the dict without conversion.
        Oracle: `json.dumps` raising `TypeError` on such a value.
        """
        ins = make_insight(id='d-2', content='x')
        d = insight_to_delta_dict(ins)
        json.dumps(d)


@pytest.fixture
def _sched_started(monkeypatch):
    """Force scheduler state to STARTED so write CLI verbs proceed.
    """
    monkeypatch.setattr(
        sched_mod, 'read_state', lambda: sched_mod.STATE_STARTED)


class TestForgetWritesBefore:
    """`memman forget <id>` records the pre-deletion content.
    """

    def test_forget_logs_before(self, backend, _sched_started):
        """Forget oplog row carries the deleted insight's content.

        Mutation: `_forget_insight` logging the forget without
            `before`, or with a `before` lacking the content.
        Oracle: the content string the test inserted.
        """
        with backend.transaction():
            backend.nodes.insert(
                make_insight(id='f-1', content='goodbye world'))
        _forget_insight(backend, 'f-1')
        entries = [
            e for e in backend.oplog.recent(limit=10)
            if e.operation == 'forget' and e.insight_id == 'f-1']
        assert entries
        assert entries[0].before is not None
        assert entries[0].before.get('content') == 'goodbye world'
