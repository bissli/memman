"""What a write consults a model for, and what it never does.

No model reads a write before it is stored: nothing judges it
non-durable, rewords it, or picks its category. And nothing a model
might say retires an existing row -- only `replace <id>` does that.
"""

import json
import uuid
from datetime import datetime, timezone

from memman.embed.fingerprint import bound_embedder
from memman.llm import usage as llm_usage
from memman.pipeline.remember import run_remember
from memman.store.model import Insight
from tests.conftest import _mock_llm_complete, make_insight


def test_a_write_makes_exactly_one_llm_call_on_enrichment(
        tmp_backend, monkeypatch):
    """Verify a write calls the model once, for enrichment only.

    Mutation: any model call restored on the write path, whatever its
        prompt text -- a screen, a verdict, a merge, or a second
        enrichment pass.
    Oracle: the stage each spied call names, against the stored row's
        content and category.
    """
    stages: list[str] = []

    def spy_complete(self, system, user, **kwargs):
        stages.append(kwargs.get('stage'))
        return _mock_llm_complete(self, system, user, **kwargs)

    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete', spy_complete)
    ec = bound_embedder(tmp_backend)
    content = 'Stored rows in goog and demo-v3 carry bissli.'
    parent = make_insight(id='one-call-1', content=content, category='fact')

    res = run_remember(tmp_backend, parent, ec=ec)

    assert stages == [llm_usage.STAGE_ENRICHMENT]
    stored = tmp_backend.nodes.get(res['id'])
    assert stored.content == content
    assert stored.category == 'fact'


def _retire_everything(self, system, user, **kwargs):
    """Answer every retiring prompt with a SUPERSEDE, enrichment as usual.

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
        `replaced_by`.
    Oracle: the stored row read back current with `replaced_by`
        None, and the write's own row added with the agent's text.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-broker', content='The message broker is kombu'))
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        _retire_everything)
    content = 'The message broker is redis, not kombu'
    now = datetime.now(timezone.utc)
    parent = Insight(
        id=str(uuid.uuid4()), content=content, category='fact',
        created_at=now, updated_at=now)

    res = run_remember(tmp_backend, parent, ec=bound_embedder(tmp_backend))

    assert res['action'] == 'add'
    assert res['content'] == content
    old = tmp_backend.nodes.get_include_deleted('old-broker')
    assert old.replaced_by is None
    assert tmp_backend.nodes.get('old-broker') is not None


def test_an_identical_write_adds_a_second_row(tmp_backend):
    """Verify a write identical to a current row lands as its own row.

    Mutation: the exact-duplicate lookup kept - the write reports
        `skipped` onto the stored row and adds nothing.
    Oracle: two current rows carrying the text, the stored one and
        the write's own.
    """
    content = 'Redis caches session tokens'
    tmp_backend.nodes.insert(make_insight(id='stored', content=content))
    now = datetime.now(timezone.utc)
    parent = Insight(
        id=str(uuid.uuid4()), content=content, category='fact',
        created_at=now, updated_at=now)

    res = run_remember(tmp_backend, parent, ec=bound_embedder(tmp_backend))

    assert res['action'] == 'add'
    current = [
        ins for ins in tmp_backend.nodes.get_all_active()
        if ins.content == content]
    assert len(current) == 2
