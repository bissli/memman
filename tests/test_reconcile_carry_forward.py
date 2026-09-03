"""Metadata and edges the reconcile merge must carry from its target.

A reconcile UPDATE is not an in-place edit: `_apply_plan` soft-deletes
the target and inserts a successor built from the incoming write. Every
field the successor does not explicitly copy is therefore destroyed,
along with the target's whole edge neighborhood.

These tests pin what survives a merge.
"""

from memman.pipeline.remember import FactPlan, _apply_plan
from memman.store.edge import get_edges_by_node, insert_edge
from memman.store.node import get_insight_by_id, insert_insight
from tests.conftest import make_edge, make_insight


def _merge_plan(new_id, target_id, **insight_overrides):
    """Build a reconcile-UPDATE FactPlan targeting `target_id`."""
    overrides = {
        'id': new_id,
        'content': 'merged content',
        'importance': 3,
        }
    overrides.update(insight_overrides)
    return FactPlan(
        action='update',
        fact_text='merged content',
        fact_insight=make_insight(**overrides),
        target_id=target_id,
        embed_vec=None,
        enrichment={},
        causal_edges=[],
        )


def test_merge_unions_target_entities_into_successor(tmp_db, tmp_backend):
    """Verify a merge keeps the target's entities, not only the incoming ones.

    Mutation: dropping the union so the successor carries only the
        incoming write's entity list.
    Oracle: hand-computed union of the two disjoint lists.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='original',
        entities=['KeePassXC', 'transcrypt', 'chezmoi']))

    plan = _merge_plan('new-1', 'old-1', entities=['systemd'])
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    successor = get_insight_by_id(tmp_db, 'new-1')
    assert successor is not None
    assert set(successor.entities) == {
        'KeePassXC', 'transcrypt', 'chezmoi', 'systemd'}


def test_merge_repoints_target_edges_to_successor(tmp_db, tmp_backend):
    """Verify the target's edges move to the successor rather than vanish.

    Mutation: leaving the bare `delete_by_node` with no re-point, which
        drops the target's whole neighborhood.
    Oracle: the causal edge's own type and weight read back off the
        successor. `fast_edges` mints temporal-proximity edges between
        any two nodes created moments apart, so matching on the
        neighbor id alone passes without the re-point.
    """
    insert_insight(tmp_db, make_insight(id='old-1', content='original'))
    insert_insight(tmp_db, make_insight(id='ctx-1', content='context'))
    insert_edge(tmp_db, make_edge(
        source_id='ctx-1', target_id='old-1',
        edge_type='causal', weight=0.83))

    plan = _merge_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert get_edges_by_node(tmp_db, 'old-1') == []
    carried = [
        e for e in get_edges_by_node(tmp_db, 'new-1')
        if e.edge_type == 'causal']
    assert len(carried) == 1
    assert carried[0].source_id == 'ctx-1'
    assert carried[0].target_id == 'new-1'
    assert carried[0].weight == 0.83


def test_merge_repoint_drops_target_self_edge(tmp_db, tmp_backend):
    """Verify a self-edge on the target does not become one on the successor.

    Mutation: re-pointing both endpoints with no far-endpoint check,
        which turns old-1 -> old-1 into new-1 -> new-1.
    Oracle: absence of any edge whose two endpoints are both 'new-1'.
    """
    insert_insight(tmp_db, make_insight(id='old-1', content='original'))
    insert_edge(tmp_db, make_edge(
        source_id='old-1', target_id='old-1',
        edge_type='causal', weight=0.7))

    plan = _merge_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert not [
        e for e in get_edges_by_node(tmp_db, 'new-1')
        if e.source_id == 'new-1' and e.target_id == 'new-1']


def test_merge_carries_target_corroboration_count(tmp_db, tmp_backend):
    """Verify corroboration earned by the target survives the merge.

    Mutation: dropping the carry so the successor resets the count to
        the incoming write's zero.
    Oracle: hand-computed 4, the target's stored count.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='original', corroboration_count=4))

    plan = _merge_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    successor = get_insight_by_id(tmp_db, 'new-1')
    assert successor is not None
    assert successor.corroboration_count == 4


def test_merge_carries_target_access_count(tmp_db, tmp_backend):
    """Verify recall history on the target survives the merge.

    Mutation: leaving access_count at the incoming write's zero, which
        erases every recall the target had served.
    Oracle: hand-computed 7, the target's stored count.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='original', access_count=7))

    plan = _merge_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    successor = get_insight_by_id(tmp_db, 'new-1')
    assert successor is not None
    assert successor.access_count == 7
