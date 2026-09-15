"""Defensive edge sweep in `_apply_plan` for the replace/update flow.

A replace mints the successor's edges after the target is already
superseded, so an edge generator that names the replaced predecessor
would leave a dangling edge pointing at it. The sweep at the end of the
apply phase removes any such edges.

The sweep must not touch edges to retained (current) history nodes.
"""

from memman.pipeline.remember import FactPlan, _apply_plan
from memman.store.edge import get_edges_by_node, insert_edge
from memman.store.node import get_insight_by_id_include_deleted, insert_insight
from tests.conftest import make_edge, make_insight, mint_edge_into


def _make_plan(new_id, target_id):
    """Build a minimal replace FactPlan that points at `target_id`."""
    new_insight = make_insight(
        id=new_id, content='replacement content', importance=3)
    return FactPlan(
        action='replace',
        fact_text='replacement content',
        fact_insight=new_insight,
        targets=[(target_id, 'replace')],
        embed_vec=None,
        enrichment={},
        )


def test_apply_plan_sweeps_edges_pointing_at_replaced_target(
        tmp_db, tmp_backend, monkeypatch):
    """Verify a replace supersedes the target and leaves it edgeless.

    Mutation: writing the pointer after `nodes.insert` (the successor
        then chains its temporal backbone to the row it replaced), or
        dropping the trailing `delete_by_node` sweep, which leaves the
        apply's own freshly minted edge dangling into the predecessor.
    Oracle: the predecessor read back current-but-superseded
        (`deleted_at` null, `superseded_by` = the successor) with no
        edges, and the far endpoint present on the successor.
    """
    insert_insight(tmp_db, make_insight(id='old-1', content='original'))
    insert_insight(tmp_db, make_insight(id='ctx-1', content='context'))
    insert_edge(tmp_db, make_edge(
        source_id='ctx-1', target_id='old-1',
        edge_type='entity', weight=0.6))

    mint_edge_into(monkeypatch, 'old-1')

    plan = _make_plan(new_id='new-1', target_id='old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    old = get_insight_by_id_include_deleted(tmp_db, 'old-1')
    assert old is not None
    assert old.deleted_at is None
    assert old.superseded_by == 'new-1'
    assert get_edges_by_node(tmp_db, 'old-1') == []
    moved = {(e.source_id, e.target_id, e.edge_type)
             for e in get_edges_by_node(tmp_db, 'new-1')}
    assert ('ctx-1', 'new-1', 'entity') in moved


def test_apply_plan_preserves_edges_to_retained_history_nodes(
        tmp_db, tmp_backend, monkeypatch):
    """Sweep targets only the replaced id, not other retained nodes.

    Mutation: sweeping every node the apply touched rather than the
        replaced target alone, which drops a live edge into a row the
        write never retired.
    Oracle: the retained node read back holding both its pre-existing
        edge and the one this apply minted at it.
    """
    insert_insight(tmp_db, make_insight(id='old-2', content='original'))
    insert_insight(
        tmp_db, make_insight(id='history-1', content='retained history'))

    pre_existing = make_edge(
        source_id='history-1', target_id='history-1',
        edge_type='temporal', weight=1.0)
    insert_edge(tmp_db, pre_existing)

    mint_edge_into(monkeypatch, 'history-1')

    plan = _make_plan(new_id='new-2', target_id='old-2')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    edges = get_edges_by_node(tmp_db, 'history-1')
    edge_keys = {(e.source_id, e.target_id, e.edge_type) for e in edges}
    assert ('history-1', 'history-1', 'temporal') in edge_keys
    assert ('new-2', 'history-1', 'semantic') in edge_keys

    deleted_edges = get_edges_by_node(tmp_db, 'old-2')
    assert len(deleted_edges) == 0
