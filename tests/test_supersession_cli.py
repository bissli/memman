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
    with open_backend('default', data_dir) as backend:
        # An embed swap leaves a superseded row without a vector; the
        # nulled column is what the re-embed must fill.
        backend._db._exec(
            'update insights set embedding = null where id = ?', (old,))

    res = invoke(mm_runner, ['unsupersede', old])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out['id'] == old
    assert out['was_superseded_by'] == new

    with _read(data_dir) as backend:
        row = backend.nodes.get(old)
        assert row is not None
        assert row.superseded_by is None
        assert backend.nodes.get_embedding(old) is not None
        edges = backend.edges.by_node(old)
        assert {e.edge_type for e in edges} <= {'entity', 'semantic'}
        assert any(e.edge_type == 'entity'
                   and peer in {e.source_id, e.target_id} for e in edges)
        ops = [(e.insight_id, e.detail) for e in backend.oplog.recent(limit=20)
               if e.operation == 'unsupersede']
        assert ops == [(old, f'was superseded by {new}')]


def test_supersede_command_joins_a_second_predecessor(mm_runner):
    """Verify a successor may take a second predecessor: a join, not a fork.

    A correction the reconciler wrote as a merge already has one
    predecessor; curating a sibling claim onto it is the common case
    on the live fleet (measured 2026-09-04 on both verified pairs).

    Mutation: refusing a successor that already has a predecessor, so
        the sibling claim stays current beside its correction.
    Oracle: both commands succeed, both predecessors leave the active
        set, `--history` from either predecessor lists all three rows
        with the successor last, and the doctor passes.
    """
    _, data_dir = mm_runner
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _remember(mm_runner, 'the broker was celery before kombu')
    s = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', p1, s]).exit_code == 0

    res = invoke(mm_runner, ['supersede', p2, s])
    assert res.exit_code == 0, res.output
    with _read(data_dir) as backend:
        assert backend.nodes.get(p1) is None
        assert backend.nodes.get(p2) is None
        assert {i.id for i in backend.nodes.predecessors(s)} == {p1, p2}
        assert all(not ids for ids in
                   backend.nodes.supersession_integrity().values())
    for start in (p1, p2):
        chain = json.loads(invoke(
            mm_runner, ['insights', 'show', start, '--history']).output)['chain']
        assert {c['id'] for c in chain} == {p1, p2, s}
        assert chain[-1]['id'] == s


def test_history_lists_both_predecessors_of_a_hand_made_fork(mm_runner):
    """Verify `--history` from one predecessor also finds the other.

    Mutation: draining the backward walk before the forward walk and
        never feeding forward-discovered successors back, so
        `show P1 --history` on P1 -> S <- P2 omits P2.
    Oracle: a fork set by raw SQL; the chain from P1 names all three
        rows, the successor last.
    """
    import sqlite3

    from memman.store.db import store_dir

    _, data_dir = mm_runner
    p1 = _remember(mm_runner, 'the broker is kombu')
    p2 = _remember(mm_runner, 'the broker was celery before kombu')
    s = _remember(mm_runner, 'the broker is redis now')
    with sqlite3.connect(f'{store_dir(data_dir, "default")}/memman.db') as conn:
        conn.execute('update insights set superseded_by = ? where id in (?, ?)',
                     (s, p1, p2))
        conn.commit()

    res = invoke(mm_runner, ['insights', 'show', p1, '--history'])
    assert res.exit_code == 0, res.output
    chain = json.loads(res.output)['chain']
    assert {c['id'] for c in chain} == {p1, p2, s}
    assert chain[-1]['id'] == s


def test_unsupersede_embeds_before_taking_the_write_lock(mm_runner, monkeypatch):
    """Verify the embed call runs outside the store's write transaction.

    Mutation: calling `ec.embed` inside `backend.transaction()`, so a
        30-second embed holds SQLite's writer lock and a concurrent
        drain fails with `database is locked`.
    Oracle: an embedder stub that, while embedding, writes through a
        second connection with a short busy timeout; it succeeds only
        when no write lock is held.
    """
    import sqlite3

    from memman.embed import fingerprint as fp_mod
    from memman.store.db import store_dir

    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0
    assert invoke(mm_runner, ['forget', new]).exit_code == 0

    real_bound = fp_mod.bound_embedder
    db_path = f'{store_dir(data_dir, "default")}/memman.db'
    probe = {'locked': None}

    class _ProbingEmbedder:
        def __init__(self, inner):
            self._inner = inner
            self.model = inner.model

        def embed(self, text):
            conn = sqlite3.connect(db_path, timeout=0.2)
            try:
                conn.execute("update meta set value = value where key = 'probe'")
                conn.commit()
                probe['locked'] = False
            except sqlite3.OperationalError:
                probe['locked'] = True
            finally:
                conn.close()
            return self._inner.embed(text)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        fp_mod, 'bound_embedder',
        lambda backend: _ProbingEmbedder(real_bound(backend)))

    res = invoke(mm_runner, ['unsupersede', old])
    assert res.exit_code == 0, res.output
    assert probe['locked'] is False


def test_status_reports_the_superseded_bucket(mm_runner):
    """Verify `memman status` shows superseded rows as their own bucket.

    Mutation: leaving `superseded_insights` off the status dict, so a
        superseded row is in no bucket and the three counts no longer
        sum to the table.
    Oracle: one supersession on a store of three rows -> current 2,
        superseded 1, deleted 0.
    """
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    _remember(mm_runner, 'the dashboard reads the broker')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0

    res = invoke(mm_runner, ['status'])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert (out['total_insights'], out['superseded_insights'],
            out['deleted_insights']) == (2, 1, 0)


def test_unsupersede_refuses_a_row_whose_successor_was_superseded(mm_runner):
    """Verify a chain unwinds from its head, never from the middle.

    Mutation: guarding only a CURRENT successor, so `unsupersede a` on
        a -> b -> c brings `a` back beside the current head `c`.
    Oracle: the refusal names `b`, and `a` stays superseded.
    """
    _, data_dir = mm_runner
    a = _remember(mm_runner, 'the broker is kombu')
    b = _remember(mm_runner, 'the broker is redis now')
    c = _remember(mm_runner, 'the broker is rabbitmq now')
    assert invoke(mm_runner, ['supersede', a, b]).exit_code == 0
    assert invoke(mm_runner, ['supersede', b, c]).exit_code == 0

    res = invoke(mm_runner, ['unsupersede', a])
    assert res.exit_code != 0
    assert b in res.output
    assert 'head' in res.output
    with _read(data_dir) as backend:
        assert backend.nodes.get_include_deleted(a).superseded_by == b


def test_unsupersede_refuses_when_the_embed_fails(mm_runner, monkeypatch):
    """Verify an embed failure leaves the row superseded, not half restored.

    Mutation: restoring the row anyway, so it re-enters recall with no
        vector (Postgres) or the stale-width blob an embed swap left
        behind (SQLite), which `check_embedding_consistency` then
        fails on.
    Oracle: a non-zero exit naming the embed, the pointer still set,
        and no edges rebuilt.
    """
    import httpx
    from memman.embed import fingerprint as fp_mod

    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu', '--entities', 'kombu')
    _remember(mm_runner, 'kombu retries are exponential', '--entities', 'kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0
    assert invoke(mm_runner, ['forget', new]).exit_code == 0
    with open_backend('default', data_dir) as backend:
        backend._db._exec(
            'update insights set embedding = null where id = ?', (old,))

    real_bound = fp_mod.bound_embedder

    class _FailingEmbedder:
        def __init__(self, inner):
            self._inner = inner
            self.model = inner.model

        def embed(self, text):
            raise httpx.ReadTimeout('provider timed out')

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        fp_mod, 'bound_embedder',
        lambda backend: _FailingEmbedder(real_bound(backend)))

    res = invoke(mm_runner, ['unsupersede', old])
    assert res.exit_code != 0
    assert 'embedding failed' in res.output
    with _read(data_dir) as backend:
        assert backend.nodes.get(old) is None
        assert backend.nodes.get_include_deleted(old).superseded_by == new
        assert backend.edges.by_node(old) == []


def test_supersede_command_drops_a_self_edge_instead_of_moving_it(mm_runner):
    """Verify a self-edge on the predecessor does not become one on the successor.

    Mutation: re-pointing both endpoints with no far-endpoint check in
        `move_edges`, which turns old -> old into new -> new.
    Oracle: the successor's edge list holds no edge with both
        endpoints equal to it, and `edges_moved` excludes the self-edge.
    """
    _, data_dir = mm_runner
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    with open_backend('default', data_dir) as backend:
        from memman.store.model import Edge
        backend.edges.upsert(Edge(
            source_id=old, target_id=old, edge_type='causal', weight=0.7))

    res = invoke(mm_runner, ['supersede', old, new])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)['edges_moved'] == 0
    with _read(data_dir) as backend:
        assert not [e for e in backend.edges.by_node(new)
                    if e.source_id == new and e.target_id == new]
