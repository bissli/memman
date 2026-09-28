"""Pipeline-level replacement: the degraded add and the drain.

`_apply_plan` may find its target already replaced by an earlier
drain; the drain may claim a `replace` whose target was replaced
between enqueue and claim. Neither may drop a write or link the wrong
row.
"""

import json

import pytest
from memman.pipeline.remember import _apply_plan
from tests.conftest import invoke, make_insight


def test_degraded_replace_names_the_target_and_its_successor(tmp_backend):
    """Verify a replace whose target is already replaced says so.

    Mutation: reporting the degraded add with no `target_gone`, so the
        caller cannot find the row that now holds the topic.
    Oracle: the result dict for a replaced target (successor named)
        and for a forgotten target (`replaced_by` None), with no
        `replaced_id` on either.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='first'))
    tmp_backend.nodes.insert(make_insight(id='new-1', content='second'))
    assert tmp_backend.nodes.mark_replaced('old-1', 'new-1') is True
    tmp_backend.nodes.insert(make_insight(id='gone-1', content='gone'))
    assert tmp_backend.nodes.soft_delete('gone-1') is True

    def _replace(new_id, target_id):
        return _apply_plan(
            tmp_backend, make_insight(id=new_id, content='third'),
            target_id, None, {})

    late = _replace('late-1', 'old-1')
    assert late['action'] == 'add'
    assert late['target_gone'] == {'id': 'old-1', 'replaced_by': 'new-1'}
    assert 'replaced_id' not in late
    assert tmp_backend.nodes.get_include_deleted('old-1').replaced_by == 'new-1'

    forgotten = _replace('late-2', 'gone-1')
    assert forgotten['action'] == 'add'
    assert forgotten['target_gone'] == {'id': 'gone-1', 'replaced_by': None}


@pytest.mark.no_auto_drain
def test_drain_redirects_a_replace_to_the_chain_head(mm_runner):
    """Verify a queued replace follows the chain to the current head.

    Two replaces of one id are queued before either drains, as two
    concurrent `replace` calls do when both pass the pending check
    before either inserts; the first replaces the id, so the second's
    target was replaced by the time the drain claims it.

    Mutation: leaving the drain preflight on `nodes.get`, so the second
        row degrades to a plain add and the topic ends with two
        current rows.
    Oracle: the store read directly: a three-row chain with one
        current head, the drain output naming `redirected_from`, and
        no failed queue row.
    """
    from memman.queue import enqueue, queue_db
    from memman.store.factory import open_backend

    _, data_dir = mm_runner
    res = invoke(mm_runner, ['remember', 'the broker is kombu'])
    assert res.exit_code == 0, res.output
    res = invoke(mm_runner, ['scheduler', 'drain'])
    assert res.exit_code == 0, res.output
    with open_backend('default', data_dir, read_only=True) as backend:
        first = backend.nodes.get_all_active()[0].id

    with queue_db(data_dir) as conn:
        for text in ('the broker is redis now', 'the broker is rabbitmq now'):
            enqueue(
                conn, store='default', content=text,
                category='fact', replaced_id=first)
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
        middle = backend.nodes.get_include_deleted(old.replaced_by)
        assert middle.content == 'the broker is redis now'
        assert middle.replaced_by == head.id
        assert head.replaced_by is None
        assert old.deleted_at is None
        assert middle.deleted_at is None
