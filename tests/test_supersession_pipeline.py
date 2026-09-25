"""Pipeline-level supersession: the batch guard, the degraded add, the drain.

`_plan_fact` may see two facts in one write aim at the same
predecessor; `_apply_plan` may find its target already superseded by
an earlier drain; the drain may claim a `replace` whose target was
superseded between enqueue and claim. None of the three may drop a
fact or link the wrong row.
"""

import json

import pytest
from memman.pipeline.remember import FactPlan, _apply_plan
from tests.conftest import invoke, make_insight, mint_edge_into


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
        id='old-1', content='first', entities=['inherited']))
    tmp_backend.nodes.insert(make_insight(id='new-1', content='second'))
    assert tmp_backend.nodes.supersede('old-1', 'new-1') is True
    tmp_backend.nodes.insert(make_insight(id='gone-1', content='gone'))
    assert tmp_backend.nodes.soft_delete('gone-1') is True

    def _replace(new_id, target_id):
        return FactPlan(
            action='replace',
            fact_insight=make_insight(
                id=new_id, content='third', entities=['own']),
            targets=[(target_id, 'replace')], embed_vec=None,
            enrichment={})

    late = _apply_plan(tmp_backend, _replace('late-1', 'old-1'),
                       embed_cache={}, store_name='test')
    assert late['action'] == 'add'
    assert late['targets_gone'] == [{'id': 'old-1', 'superseded_by': 'new-1'}]
    assert 'replaced_ids' not in late
    stored = tmp_backend.nodes.get('late-1')
    assert stored.entities == ['own']
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
    res = invoke(mm_runner, ['remember', 'the broker is kombu'])
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


def test_degraded_replace_leaves_no_edge_into_its_dead_target(
        tmp_db, tmp_backend, monkeypatch):
    """Verify a degraded add still sweeps its own edges into the target.

    Mutation: gating the trailing sweep on `not target_already_gone`,
        so an edge into an already superseded row lands and stays.
    Oracle: the superseded target read back edgeless after the
        degraded apply.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='first'))
    tmp_backend.nodes.insert(make_insight(id='new-1', content='second'))
    assert tmp_backend.nodes.supersede('old-1', 'new-1') is True

    mint_edge_into(monkeypatch, 'old-1')
    plan = FactPlan(
        action='replace',
        fact_insight=make_insight(id='late-1', content='third'),
        targets=[('old-1', 'replace')], embed_vec=None, enrichment={})

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
        action='add',
        fact_insight=make_insight(id='new-1', content='second'),
        targets=[('old-1', 'replace')], embed_vec=None, enrichment={})

    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert 'replaced_ids' not in result
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
    res = invoke(mm_runner, ['remember', 'the broker is kombu'])
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
