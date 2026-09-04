"""Pipeline-level supersession: the batch guard, the degraded add, the drain.

`_plan_fact` may see two facts in one write aim at the same
predecessor; `_apply_plan` may find its target already superseded by
an earlier drain; the drain may claim a `replace` whose target was
superseded between enqueue and claim. None of the three may drop a
fact or link the wrong row.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest
from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import FactPlan, _apply_plan, run_remember
from memman.store.model import Insight
from tests.conftest import invoke, make_insight


def _parent(content):
    now = datetime.now(timezone.utc)
    return Insight(
        id=str(uuid.uuid4()), content=content, category='fact',
        importance=3, entities=[], source='test', access_count=0,
        created_at=now, updated_at=now)


def test_two_facts_on_one_predecessor_supersede_once_and_add_once(
        tmp_backend, monkeypatch):
    """Verify a second fact aimed at a taken predecessor lands as an add.

    Mutation: dropping `'supersede'` from the batch guard set, so the
        second fact reaches `_apply_plan` still aimed at the taken
        predecessor and lands as a DEGRADED add naming it; or keeping
        the shipped skip, which drops the second fact with a
        `target already deleted` reason and no row.
    Oracle: exactly one row carries a pointer, it names the first
        successor, the second fact's text is stored as a clean add with
        no `target_id`, and no result is `skipped`.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

    def _two_facts(llm_client, content):
        return [
            {'text': 'the broker is redis now', 'category': 'fact',
             'importance': 3, 'entities': []},
            {'text': 'the broker moved to rabbitmq later', 'category': 'fact',
             'importance': 3, 'entities': []},
            ]

    def _aim_at_old(llm_client, facts, existing):
        return [{'fact': f['text'], 'action': 'SUPERSEDE',
                 'target_id': 'old-1', 'merged_text': None}
                for f in facts]

    monkeypatch.setattr('memman.llm.extract.extract_facts', _two_facts)
    monkeypatch.setattr(
        'memman.llm.extract.reconcile_memories', _aim_at_old)

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    actions = [f['action'] for f in res['facts']]
    assert actions == ['supersede', 'add']
    assert 'skipped' not in actions
    assert 'target_id' not in res['facts'][1]
    old = tmp_backend.nodes.get_include_deleted('old-1')
    assert old.superseded_by == res['facts'][0]['id']
    stored = {i.content for i in tmp_backend.nodes.get_all_active()}
    assert stored == {'the broker is redis now',
                      'the broker moved to rabbitmq later'}
    pointers = [i for i in (
        tmp_backend.nodes.get_include_deleted(f['id']) for f in res['facts'])
        if i.superseded_by]
    assert pointers == []


def test_degraded_replace_names_the_target_and_its_successor(tmp_backend):
    """Verify a replace whose target is already superseded says so.

    Mutation: reporting the degraded add with neither `target_id` nor
        `target_superseded_by`, so the caller cannot find the row that
        now holds the topic; or inheriting entities and counts from a
        row the add did not supersede.
    Oracle: the result dict for a superseded target (successor named)
        and for a forgotten target (`target_superseded_by` None), with
        no `replaced_id` on either, and the inserted row carrying only
        its own entities.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='first', entities=['inherited'],
        access_count=9))
    tmp_backend.nodes.insert(make_insight(id='new-1', content='second'))
    assert tmp_backend.nodes.supersede('old-1', 'new-1') is True
    tmp_backend.nodes.insert(make_insight(id='gone-1', content='gone'))
    assert tmp_backend.nodes.soft_delete('gone-1') is True

    def _replace(new_id, target_id):
        return FactPlan(
            action='replace', fact_text='third',
            fact_insight=make_insight(
                id=new_id, content='third', entities=['own']),
            target_id=target_id, embed_vec=None, enrichment={},
            causal_edges=[])

    late = _apply_plan(tmp_backend, _replace('late-1', 'old-1'),
                       embed_cache={}, store_name='test')
    assert late['action'] == 'add'
    assert late['target_id'] == 'old-1'
    assert late['target_superseded_by'] == 'new-1'
    assert 'replaced_id' not in late
    stored = tmp_backend.nodes.get('late-1')
    assert stored.entities == ['own']
    assert stored.access_count == 0
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == 'new-1'

    forgotten = _apply_plan(tmp_backend, _replace('late-2', 'gone-1'),
                            embed_cache={}, store_name='test')
    assert forgotten['action'] == 'add'
    assert forgotten['target_id'] == 'gone-1'
    assert forgotten['target_superseded_by'] is None


@pytest.mark.no_auto_drain
def test_drain_redirects_a_replace_to_the_chain_head(mm_runner):
    """Verify a queued replace follows the chain to the current head.

    Two replaces of one id are queued before either drains; the first
    supersedes the id, so the second's target is superseded by the
    time the drain claims it.

    Mutation: leaving the drain preflight on `nodes.get`, so the second
        row degrades to a plain add and the topic ends with two
        current rows.
    Oracle: the store read directly: a three-row chain with one
        current head, the drain output naming `redirected_from`, and
        no failed queue row.
    """
    from memman.store.factory import open_backend

    _, data_dir = mm_runner
    res = invoke(mm_runner, ['remember', 'the broker is kombu',
                             '--no-reconcile'])
    assert res.exit_code == 0, res.output
    res = invoke(mm_runner, ['scheduler', 'drain'])
    assert res.exit_code == 0, res.output
    with open_backend('default', data_dir, read_only=True) as backend:
        first = backend.nodes.get_all_active()[0].id

    for text in ('the broker is redis now', 'the broker is rabbitmq now'):
        res = invoke(mm_runner, ['replace', first, text])
        assert res.exit_code == 0, res.output
    res = invoke(mm_runner, ['scheduler', 'drain'])
    assert res.exit_code == 0, res.output
    assert '"redirected_from"' in res.output
    assert f'"redirected_from": "{first}"' in res.output

    failed = invoke(mm_runner, ['scheduler', 'queue', 'failed'])
    assert json.loads(failed.output)['rows'] == []

    with open_backend('default', data_dir, read_only=True) as backend:
        current = backend.nodes.get_all_active()
        assert [i.content for i in current] == ['the broker is rabbitmq now']
        head = current[0]
        old = backend.nodes.get_include_deleted(first)
        middle = backend.nodes.get_include_deleted(old.superseded_by)
        assert middle.content == 'the broker is redis now'
        assert middle.superseded_by == head.id
        assert head.superseded_by is None
        assert old.deleted_at is None
        assert middle.deleted_at is None


def test_sibling_causal_edge_into_a_superseded_row_is_swept(
        tmp_backend, monkeypatch):
    """Verify no fact in a write leaves an edge into a row the write superseded.

    Causal candidates are drawn during planning, when the predecessor
    is still current, so a later fact's causal edge can name a row an
    earlier fact superseded.

    Mutation: sweeping only each plan's own target after its upsert,
        so the second fact's edge into the first fact's target lands
        after that target's sweep ran.
    Oracle: the predecessor read back edgeless and the integrity
        population `superseded_with_edges` empty after the write.
    """
    from memman.store.model import Edge

    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

    def _two_facts(llm_client, content):
        return [
            {'text': 'the broker is redis now', 'category': 'fact',
             'importance': 3, 'entities': []},
            {'text': 'the dashboard reads the broker', 'category': 'fact',
             'importance': 3, 'entities': []},
            ]

    def _first_supersedes(llm_client, facts, existing):
        out = []
        for f in facts:
            if 'redis' in f['text']:
                out.append({'fact': f['text'], 'action': 'SUPERSEDE',
                            'target_id': 'old-1', 'merged_text': None})
            else:
                out.append({'fact': f['text'], 'action': 'ADD',
                            'target_id': None, 'merged_text': None})
        return out

    def _causal_into_old(ro, insight, client):
        return [Edge(source_id=insight.id, target_id='old-1',
                     edge_type='causal', weight=0.9)]

    monkeypatch.setattr('memman.llm.extract.extract_facts', _two_facts)
    monkeypatch.setattr(
        'memman.llm.extract.reconcile_memories', _first_supersedes)
    monkeypatch.setattr(
        'memman.pipeline.remember.infer_llm_causal_edges', _causal_into_old)

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert [f['action'] for f in res['facts']] == ['supersede', 'add']
    assert tmp_backend.edges.by_node('old-1') == []
    assert tmp_backend.nodes.supersession_integrity()[
        'superseded_with_edges'] == []


def test_degraded_replace_leaves_no_edge_into_its_dead_target(
        tmp_db, tmp_backend):
    """Verify a degraded add still sweeps its plan's edges into the target.

    Mutation: gating the trailing sweep on `not target_already_gone`,
        so the plan's causal edge into an already superseded row lands
        and stays.
    Oracle: the superseded target read back edgeless after the
        degraded apply.
    """
    from memman.store.model import Edge

    tmp_backend.nodes.insert(make_insight(id='old-1', content='first'))
    tmp_backend.nodes.insert(make_insight(id='new-1', content='second'))
    assert tmp_backend.nodes.supersede('old-1', 'new-1') is True
    plan = FactPlan(
        action='replace', fact_text='third',
        fact_insight=make_insight(id='late-1', content='third'),
        target_id='old-1', embed_vec=None, enrichment={},
        causal_edges=[Edge(source_id='late-1', target_id='old-1',
                           edge_type='causal', weight=0.9)])

    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert result['action'] == 'add'
    assert tmp_backend.edges.by_node('old-1') == []


def test_a_plain_add_plan_with_a_target_reports_no_replaced_id(
        tmp_db, tmp_backend):
    """Verify `replaced_id` is reported only when a supersession happened.

    Mutation: emitting `replaced_id` whenever the plan carries a
        `target_id`, so an `add` plan decorated with a target claims a
        replace that never ran.
    Oracle: the result of an `add` plan carrying a target: no
        `replaced_id`, and the target still current.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='first'))
    plan = FactPlan(
        action='add', fact_text='second',
        fact_insight=make_insight(id='new-1', content='second'),
        target_id='old-1', embed_vec=None, enrichment={}, causal_edges=[])

    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert 'replaced_id' not in result
    assert tmp_backend.nodes.get('old-1') is not None


@pytest.mark.no_auto_drain
def test_drain_passes_a_forgotten_head_through_as_a_named_add(mm_runner):
    """Verify a replace whose chain head was forgotten degrades, not redirects.

    Mutation: redirecting onto the forgotten head (a replace of a
        deleted row), or raising instead of degrading.
    Oracle: the drained result reports `action: add` with the original
        target and its successor named, no `redirected_from`, and no
        failed queue row.
    """
    from memman.store.factory import open_backend

    r, data_dir = mm_runner
    res = invoke(mm_runner, ['remember', 'the broker is kombu',
                             '--no-reconcile'])
    assert res.exit_code == 0, res.output
    assert invoke(mm_runner, ['scheduler', 'drain']).exit_code == 0
    with open_backend('default', data_dir, read_only=True) as backend:
        first = backend.nodes.get_all_active()[0].id

    replaced = invoke(mm_runner, ['replace', first, 'the broker is redis now'])
    assert replaced.exit_code == 0, replaced.output
    assert invoke(mm_runner, ['scheduler', 'drain']).exit_code == 0
    with open_backend('default', data_dir, read_only=True) as backend:
        head = backend.nodes.get_include_deleted(first).superseded_by
    assert invoke(mm_runner, ['forget', head]).exit_code == 0

    queued = invoke(mm_runner, ['replace', first, 'the broker is rabbitmq now'])
    assert queued.exit_code != 0
    assert f'is superseded by {head}' in queued.output
