"""CLI surfaces for supersession: history, supersede, unsupersede.

`insights show <id> --history` walks a chain in both directions;
`memman supersede` links two rows that both already exist; `memman
unsupersede` reverses a link whose successor is no longer current and
brings the predecessor back into the graph.
"""

import json

from memman.store.factory import open_backend
from tests.conftest import invoke, parse_remember


def _remember(runner, text, *flags):
    """Store `text` verbatim and return its id."""
    res = invoke(runner, ['remember', text, '--no-reconcile', *flags])
    assert res.exit_code == 0, res.output
    return parse_remember(res, runner)['id']


def _replace(runner, target, text):
    """Replace `target` with `text` and return the successor id."""
    res = invoke(runner, ['replace', target, text])
    assert res.exit_code == 0, res.output
    return parse_remember(res, runner)['id']


def _read(data_dir):
    return open_backend('default', data_dir, read_only=True)


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
    assert [c['state'] for c in chain] == ['forgotten', 'superseded', 'current']
    assert [c['superseded_by'] for c in chain] == [p2, p3, None]
    assert 'content' not in chain[0]
    assert chain[1]['content'] == 'the broker is redis now'
    assert chain[2]['content'] == 'the broker is rabbitmq now'

    from_forgotten = invoke(mm_runner, ['insights', 'show', p1, '--history'])
    assert from_forgotten.exit_code == 0, from_forgotten.output
    assert json.loads(from_forgotten.output)['chain'] == chain

    missing = invoke(mm_runner, ['insights', 'show', 'no-such', '--history'])
    assert missing.exit_code != 0
    assert 'not found' in missing.output


def test_supersede_command_links_two_current_rows(mm_runner):
    """Verify `memman supersede` links existing rows and moves the edges.

    Mutation: not moving the predecessor's edges (its neighborhood
        vanishes), accepting a non-current predecessor (a fork), or
        accepting the same id twice (a self-pointer).
    Oracle: the store read directly after the command, and the
        refusal text for each non-current shape.
    """
    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    ctx = _remember(mm_runner, 'the broker feeds the dashboard')
    assert invoke(mm_runner, ['graph', 'link', old, ctx,
                              '--type', 'causal']).exit_code == 0

    res = invoke(mm_runner, ['supersede', old, new])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert (out['predecessor'], out['successor']) == (old, new)
    assert out['edges_moved'] >= 2

    with _read(data_dir) as backend:
        assert backend.nodes.get_include_deleted(old).superseded_by == new
        assert backend.nodes.get(old) is None
        assert backend.edges.by_node(old) == []
        moved = {(e.source_id, e.target_id) for e in backend.edges.by_node(new)
                 if e.edge_type == 'causal'}
        assert moved == {(new, ctx), (ctx, new)}
        ops = [e for e in backend.oplog.recent(limit=20)
               if e.operation == 'supersede']
        assert [(e.insight_id, e.detail) for e in ops] == [
            (old, f'replaced by {new}')]

    again = invoke(mm_runner, ['supersede', old, ctx])
    assert again.exit_code != 0
    assert f'is superseded by {new}' in again.output

    same = invoke(mm_runner, ['supersede', ctx, ctx])
    assert same.exit_code != 0
    assert 'same' in same.output

    gone = _remember(mm_runner, 'a row to forget')
    assert invoke(mm_runner, ['forget', gone]).exit_code == 0
    forgotten = invoke(mm_runner, ['supersede', gone, ctx])
    assert forgotten.exit_code != 0
    assert 'was forgotten' in forgotten.output

    missing = invoke(mm_runner, ['supersede', 'no-such', ctx])
    assert missing.exit_code != 0
    assert 'not found' in missing.output


def test_unsupersede_refuses_while_the_successor_is_current(mm_runner):
    """Verify `unsupersede` will not create two current rows for one fact.

    Mutation: dropping the successor-is-current guard.
    Oracle: the refusal names the successor and the predecessor stays
        superseded.
    """
    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0

    res = invoke(mm_runner, ['unsupersede', old])
    assert res.exit_code != 0
    assert new in res.output
    assert 'current' in res.output
    with _read(data_dir) as backend:
        assert backend.nodes.get_include_deleted(old).superseded_by == new

    not_superseded = invoke(mm_runner, ['unsupersede', new])
    assert not_superseded.exit_code != 0
    assert 'not superseded' in not_superseded.output


def test_unsupersede_relinks_reembeds_and_writes_its_oplog_row(mm_runner):
    """Verify `unsupersede` brings the predecessor back into the graph.

    Mutation: clearing the pointer without rebuilding edges (a current
        row with zero degree), without re-embedding (no vector after
        an embed swap), or without the oplog row.
    Oracle: after `forget succ` then `unsupersede pred`: the row is
        current, its entity edge to a peer exists, no temporal edge was
        minted, its embedding is present, and the oplog names the
        successor it was superseded by.
    """
    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu', '--entities', 'kombu')
    peer = _remember(mm_runner, 'kombu retries are exponential',
                     '--entities', 'kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0
    assert invoke(mm_runner, ['forget', new]).exit_code == 0

    res = invoke(mm_runner, ['unsupersede', old])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out['id'] == old
    assert out['was_superseded_by'] == new
    assert out['embedded'] is True

    with _read(data_dir) as backend:
        row = backend.nodes.get(old)
        assert row is not None
        assert row.superseded_by is None
        assert backend.nodes.get_embedding(old) is not None
        edges = backend.edges.by_node(old)
        assert {e.edge_type for e in edges} <= {'entity', 'semantic'}
        assert any(e.edge_type == 'entity'
                   and peer in (e.source_id, e.target_id) for e in edges)
        ops = [(e.insight_id, e.detail) for e in backend.oplog.recent(limit=20)
               if e.operation == 'unsupersede']
        assert ops == [(old, f'was superseded by {new}')]
