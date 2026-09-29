"""CLI surfaces for replacement: the history walk and the status bucket.

`insights show <id> --history` walks a chain in both directions, and
`memman status` counts replaced rows as their own bucket.
"""

import json
import sqlite3

import pytest
from memman.store.db import store_dir
from tests.conftest import invoke, parse_remember, queued_contents


def _remember(runner, text, *flags):
    """Store `text` verbatim and return its id.
    """
    res = invoke(runner, ['remember', text, *flags])
    assert res.exit_code == 0, res.output
    return parse_remember(res, runner)['id']


def _replace(runner, target, text):
    """Replace `target` with `text` and return the successor id.
    """
    res = invoke(runner, ['replace', target, text])
    assert res.exit_code == 0, res.output
    return parse_remember(res, runner)['id']


def test_forget_refuses_a_replacement_and_names_replace(mm_runner):
    """`forget` refuses a current row that replaced another.

    Mutation: `forget` soft-deleting a replacement, which leaves the
        topic with no current row and the replaced row hidden for good.
    Oracle: a remember then a replace; the refusal names `replace` on
        the replacement's id.
    """
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _replace(mm_runner, p1, 'the broker is redis now')

    r = invoke(mm_runner, ['forget', p2])

    assert r.exit_code != 0
    assert f'replace {p2}' in r.output


def test_forget_accepts_a_replaced_row_that_replaced_another(mm_runner):
    """`forget` takes a replaced row, even one that replaced another.

    Mutation: the guard refusing every row with a predecessor, so the
        middle row of a chain, already out of recall, cannot be
        forgotten.
    Oracle: P1 -> P2 -> P3; forgetting P2 succeeds.
    """
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _replace(mm_runner, p1, 'the broker is redis now')
    _replace(mm_runner, p2, 'the broker is rabbitmq now')

    assert invoke(mm_runner, ['forget', p2]).exit_code == 0


def test_forget_accepts_a_replacement_whose_predecessor_is_forgotten(
        mm_runner):
    """`forget` takes a replacement once the row it replaced is forgotten.

    Mutation: the guard counting forgotten predecessors, so after the
        replaced row is forgotten the replacement can never leave
        recall.
    Oracle: P1 -> P2; forgetting P1, then P2, both succeed.
    """
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _replace(mm_runner, p1, 'the broker is redis now')
    assert invoke(mm_runner, ['forget', p1]).exit_code == 0

    assert invoke(mm_runner, ['forget', p2]).exit_code == 0


def test_history_walks_a_three_row_chain_oldest_first(mm_runner):
    """Verify `--history` lists the whole chain from any member.

    Mutation: walking one direction only (a middle id then shows half
        the chain), emitting content for a forgotten row, or refusing
        a forgotten id.
    Oracle: a hand-built P1 -> P2 -> P3 with P1 forgotten: the same
        ordered ids and states from the middle id and from the
        forgotten one, no content on P1, content on the other two.
    """
    _, data_dir = mm_runner
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _replace(mm_runner, p1, 'the broker is redis now')
    p3 = _replace(mm_runner, p2, 'the broker is rabbitmq now')
    assert invoke(mm_runner, ['forget', p1]).exit_code == 0

    from_middle = invoke(mm_runner, ['insights', 'show', p2, '--history'])
    assert from_middle.exit_code == 0, from_middle.output
    data = json.loads(from_middle.output)
    assert data['requested'] == p2
    chain = data['chain']
    assert [c['id'] for c in chain] == [p1, p2, p3]
    assert [c['state'] for c in chain] == ['forgotten', 'replaced', 'current']
    assert [c['replaced_by'] for c in chain] == [p2, p3, None]
    assert 'content' not in chain[0]
    assert chain[1]['content'] == 'the broker is redis now'
    assert chain[2]['content'] == 'the broker is rabbitmq now'

    from_forgotten = invoke(mm_runner, ['insights', 'show', p1, '--history'])
    assert from_forgotten.exit_code == 0, from_forgotten.output
    assert json.loads(from_forgotten.output)['chain'] == chain

    missing = invoke(mm_runner, ['insights', 'show', 'no-such', '--history'])
    assert missing.exit_code != 0
    assert 'not found' in missing.output


def test_history_lists_both_predecessors_of_a_hand_made_fork(mm_runner):
    """Verify `--history` from one predecessor also finds the other.

    Mutation: draining the backward walk before the forward walk and
        never feeding forward-discovered successors back, so
        `show P1 --history` on P1 -> S <- P2 omits P2.
    Oracle: a fork set by raw SQL; the chain from P1 names all three
        rows, the successor last.
    """
    _, data_dir = mm_runner
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _remember(mm_runner, 'the broker was celery before kombu')
    s = _remember(mm_runner, 'the broker is redis now')
    with sqlite3.connect(f'{store_dir(data_dir, "default")}/memman.db') as conn:
        conn.execute('update insights set replaced_by = ? where id in (?, ?)',
                     (s, p1, p2))
        conn.commit()

    res = invoke(mm_runner, ['insights', 'show', p1, '--history'])
    assert res.exit_code == 0, res.output
    chain = json.loads(res.output)['chain']
    assert {c['id'] for c in chain} == {p1, p2, s}
    assert chain[-1]['id'] == s


def test_status_reports_the_replaced_bucket(mm_runner):
    """Verify `memman status` shows replaced rows as their own bucket.

    Mutation: leaving `replaced_insights` off the status dict, so a
        replaced row is in no bucket and the three counts no longer
        sum to the table.
    Oracle: one replacement on a store of three rows -> current 2,
        replaced 1, deleted 0.
    """
    old = _remember(mm_runner, 'the broker is kombu')
    _replace(mm_runner, old, 'the broker is redis now')
    _remember(mm_runner, 'the dashboard reads the broker')

    res = invoke(mm_runner, ['status'])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert (out['total_insights'], out['replaced_insights'],
            out['deleted_insights']) == (2, 1, 0)


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('drained', [False, True],
                         ids=['queued-target', 'stored-target'])
def test_replace_refuses_a_target_with_a_replace_pending(mm_runner, drained):
    """Verify a second replace of one target is refused with the first's text.

    Mutation: no pending-replace check, the check in only the queued or
        only the stored branch, or a refusal without the first
        replace's text. Each lets the drain retire the first correction
        under a second one written from the original row.
    Oracle: the first replace's printed id and text in the refusal,
        and a queue that never holds the second text.
    """
    _, data_dir = mm_runner
    target = json.loads(invoke(mm_runner, [
        'remember', 'the broker is kombu and the retry cap is three',
        ]).output)['id']
    if drained:
        assert invoke(mm_runner, ['scheduler', 'drain']).exit_code == 0
    first = invoke(mm_runner, [
        'replace', target, 'the broker is kombu and the retry cap is five'])
    assert first.exit_code == 0, first.output

    second = invoke(mm_runner, [
        'replace', target, 'the broker is redis and the retry cap is three'])

    assert second.exit_code == 1
    assert json.loads(first.output)['id'] in second.output
    assert 'the broker is kombu and the retry cap is five' in second.output
    assert ('the broker is redis and the retry cap is three'
            not in queued_contents(data_dir))
