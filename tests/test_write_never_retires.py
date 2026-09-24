"""A `remember` write adds a row or skips an exact duplicate; it never retires.

Only `replace <id>` and `supersede` retire a stored row. A write whose
nearest stored row contradicts it lands beside that row, and both stay
in recall.
"""

import json
import uuid
from datetime import datetime, timezone

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from memman.store.model import Insight
from tests.conftest import _mock_llm_complete, make_insight


def _supersede_everything(self, system, user, **kwargs):
    """Answer every retiring prompt as a supersede, enrichment as usual.

    The screen, verdict and merge prompts each carry a marker of
    their own, so a write path that still sends any of them gets the
    answer that retires the stored row.
    """
    if 'CONTRADICTS|REFINES|RESTATES|UNRELATED' in system:
        return json.dumps({
            'relation': 'CONTRADICTS',
            'contradicted_clauses': ['The message broker is kombu'],
            'reason': 'the broker changed'})
    if 'ADD|UPDATE|SUPERSEDE|NONE' in system:
        return json.dumps({'actions': [{
            'action': 'SUPERSEDE', 'target_id': 0,
            'reason': 'the broker changed'}]})
    if 'SUCCESSOR TEXT' in system:
        return json.dumps({
            'merged_text': 'The message broker is redis, not kombu'})
    return _mock_llm_complete(self, system, user, **kwargs)


def test_a_contradicting_write_is_added_and_retires_nothing(
        tmp_backend, monkeypatch):
    """Verify a write that contradicts its nearest row lands beside it.

    Mutation: the verdict path still retiring - the stored row gets
        `superseded_by` and the write reports `supersede`.
    Oracle: the stored row read back current with `superseded_by`
        None, and the write's own row added with the agent's text.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-broker', content='The message broker is kombu'))
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        _supersede_everything)
    content = 'The message broker is redis, not kombu'
    now = datetime.now(timezone.utc)
    parent = Insight(
        id=str(uuid.uuid4()), content=content, category='fact',
        importance=3, entities=[], source='test', access_count=0,
        created_at=now, updated_at=now)

    res = run_remember(
        tmp_backend, parent, content,
        ec=bound_embedder(tmp_backend), store_name='test')

    fact = res['facts'][0]
    assert [f['action'] for f in res['facts']] == ['add']
    assert fact['content'] == content
    old = tmp_backend.nodes.get_include_deleted('old-broker')
    assert old.superseded_by is None
    assert tmp_backend.nodes.get('old-broker') is not None
