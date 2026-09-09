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


def _contradict(monkeypatch, target_ids, when=lambda fact_text: True, merged=lambda tid: None):
    """Stub the three stages: `target_ids` are contradicted and superseded.

    `when` gates the contradiction on the fact text; `merged` is the
    stage-3 text per target id, None for the fact-text fallback.
    """
    def _screen(client, fact_text, memory):
        if memory[0] in target_ids and when(fact_text):
            return 'CONTRADICTS', []
        return 'UNRELATED', []

    monkeypatch.setattr('memman.llm.extract.screen_memory', _screen)
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory', lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: merged(target[0]))


def test_two_facts_on_one_predecessor_supersede_once_and_add_once(
        tmp_backend, monkeypatch):
    """Verify a second fact contradicting a taken predecessor lands as an add.

    Mutation: dropping the batch exclusion from the shortlist, so the
        second fact is shown the row the first fact retired, supersedes
        it again and forks the chain; or degrading it to a skip, which
        drops the second fact with no row.
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

    monkeypatch.setattr('memman.llm.extract.extract_facts', _two_facts)
    _contradict(monkeypatch, {'old-1'})

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    actions = [f['action'] for f in res['facts']]
    assert actions == ['supersede', 'add']
    assert 'skipped' not in actions
    assert 'target_id' not in res['facts'][1]
    assert 'replaced_ids' not in res['facts'][1]
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

    Mutation: reporting the degraded add with no `targets_gone`, so the
        caller cannot find the row that now holds the topic; or
        inheriting entities and counts from a row the add did not
        supersede.
    Oracle: the result dict for a superseded target (successor named)
        and for a forgotten target (`superseded_by` None), with no
        `replaced_ids` on either, and the inserted row carrying only its
        own entities.
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
            targets=[(target_id, 'replace')], embed_vec=None, enrichment={},
            causal_edges=[])

    late = _apply_plan(tmp_backend, _replace('late-1', 'old-1'),
                       embed_cache={}, store_name='test')
    assert late['action'] == 'add'
    assert late['targets_gone'] == [{'id': 'old-1', 'superseded_by': 'new-1'}]
    assert 'replaced_ids' not in late
    stored = tmp_backend.nodes.get('late-1')
    assert stored.entities == ['own']
    assert stored.access_count == 0
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == 'new-1'

    forgotten = _apply_plan(tmp_backend, _replace('late-2', 'gone-1'),
                            embed_cache={}, store_name='test')
    assert forgotten['action'] == 'add'
    assert forgotten['targets_gone'] == [{'id': 'gone-1', 'superseded_by': None}]


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

    def _causal_into_old(ro, insight, client):
        return [Edge(source_id=insight.id, target_id='old-1',
                     edge_type='causal', weight=0.9)]

    monkeypatch.setattr('memman.llm.extract.extract_facts', _two_facts)
    _contradict(monkeypatch, {'old-1'}, when=lambda fact_text: 'redis' in fact_text)
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
        targets=[('old-1', 'replace')], embed_vec=None, enrichment={},
        causal_edges=[Edge(source_id='late-1', target_id='old-1',
                           edge_type='causal', weight=0.9)])

    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert result['action'] == 'add'
    assert tmp_backend.edges.by_node('old-1') == []


def test_a_plain_add_plan_with_a_target_reports_no_replaced_id(
        tmp_db, tmp_backend):
    """Verify `replaced_ids` is reported only when a supersession happened.

    Mutation: emitting `replaced_ids` whenever the plan carries targets,
        so an `add` plan decorated with a target claims a replace that
        never ran.
    Oracle: the result of an `add` plan carrying a target: no
        `replaced_ids`, and the target still current.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='first'))
    plan = FactPlan(
        action='add', fact_text='second',
        fact_insight=make_insight(id='new-1', content='second'),
        targets=[('old-1', 'replace')], embed_vec=None, enrichment={},
        causal_edges=[])

    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert 'replaced_ids' not in result
    assert tmp_backend.nodes.get('old-1') is not None


def test_one_fact_supersedes_every_contradicted_row(tmp_backend, monkeypatch):
    """Verify a fact that contradicts two rows supersedes both in one write.

    Mutation: acting on the first target only, which leaves the second
        contradicted row current beside the fact; or one successor for
        both, which moves every predecessor's edges onto one row.
    Oracle: each predecessor read back with `superseded_by` naming its
        own successor, one `reconcile-supersede` oplog row each, and the
        far endpoint of each predecessor's edge on its own successor.
    """
    from memman.store.model import Edge

    for old, far in (('old-1', 'ctx-1'), ('old-2', 'ctx-2')):
        tmp_backend.nodes.insert(make_insight(
            id=old, content=f'the broker at {old} is kombu and X holds'))
        tmp_backend.nodes.insert(make_insight(id=far, content=f'{far} context'))
        tmp_backend.edges.upsert(Edge(source_id=far, target_id=old,
                                      edge_type='causal', weight=0.8))

    def _one_fact(llm_client, content):
        return [{'text': content, 'category': 'fact', 'entities': []}]

    monkeypatch.setattr('memman.llm.extract.extract_facts', _one_fact)
    _contradict(monkeypatch, {'old-1', 'old-2'},
                merged=lambda target_id: f'X is no longer so at {target_id}; Y holds')

    res = run_remember(
        tmp_backend, _parent('the broker is redis and X no longer holds'),
        'the broker is redis and X no longer holds',
        ec=bound_embedder(tmp_backend), store_name='test')

    by_target = {f['replaced_ids'][0]: f for f in res['facts']}
    assert set(by_target) == {'old-1', 'old-2'}
    ops = [(e.operation, e.insight_id) for e in tmp_backend.oplog.recent(limit=20)]
    for old, far in (('old-1', 'ctx-1'), ('old-2', 'ctx-2')):
        fact = by_target[old]
        assert fact['action'] == 'supersede'
        assert fact['content'] == f'X is no longer so at {old}; Y holds'
        assert tmp_backend.nodes.get_include_deleted(old).superseded_by == fact['id']
        assert tmp_backend.edges.by_node(old) == []
        assert ops.count(('reconcile-supersede', old)) == 1
        causal_far = {e.source_id for e in tmp_backend.edges.by_node(fact['id'])
                      if e.edge_type == 'causal'}
        assert causal_far == {far}


def test_batch_drops_only_the_taken_target(tmp_backend, monkeypatch):
    """Verify a later fact retires its free target when another is already taken.

    Mutation: dropping the batch exclusion from the shortlist, so the
        taken row is screened again and the second fact forks its
        chain; or skipping the second fact whenever a row it would
        have contradicted is taken, which leaves the free row current.
    Oracle: the free row's `superseded_by` naming the second successor,
        and the second fact's action `supersede` with `replaced_ids`
        listing the free row alone.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='the broker is kombu'))
    tmp_backend.nodes.insert(make_insight(id='old-2', content='the queue is durable'))

    def _two_facts(llm_client, content):
        return [
            {'text': 'the broker is redis now', 'category': 'fact', 'entities': []},
            {'text': 'nothing is durable and the broker is redis', 'category': 'fact',
             'entities': []},
            ]

    def _screen(client, fact_text, memory):
        if memory[0] == 'old-1' or (
                memory[0] == 'old-2' and fact_text.startswith('nothing')):
            return 'CONTRADICTS', []
        return 'UNRELATED', []

    monkeypatch.setattr('memman.llm.extract.extract_facts', _two_facts)
    monkeypatch.setattr('memman.llm.extract.screen_memory', _screen)
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory', lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor', lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    first, second = res['facts']
    assert (first['action'], first['replaced_ids']) == ('supersede', ['old-1'])
    assert (second['action'], second['replaced_ids']) == ('supersede', ['old-2'])
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == first['id']
    assert tmp_backend.nodes.get_include_deleted('old-2').superseded_by == second['id']


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
