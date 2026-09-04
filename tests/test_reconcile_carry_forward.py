"""Metadata and edges a reconcile merge must carry from its predecessor.

A reconcile UPDATE or SUPERSEDE is not an in-place edit: `_apply_plan`
supersedes the target and inserts a successor built from the incoming
write. The predecessor keeps its content behind `superseded_by`, but
every field the successor does not explicitly copy is missing from the
current view, along with the predecessor's whole edge neighborhood.

These tests pin what the successor carries.
"""

from memman.pipeline.remember import FactPlan, _apply_plan
from memman.store.edge import get_edges_by_node, insert_edge
from memman.store.node import get_insight_by_id, insert_insight
from tests.conftest import make_edge, make_insight


def _merge_plan(new_id, target_id, *, action='update', fact_text=None,
                **insight_overrides):
    """Build a reconcile FactPlan of `action` targeting `target_id`."""
    overrides = {
        'id': new_id,
        'content': 'merged content',
        'importance': 3,
        }
    overrides.update(insight_overrides)
    return FactPlan(
        action=action,
        fact_text=fact_text or 'merged content',
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


def test_supersede_plan_links_the_predecessor_and_keeps_it(tmp_db, tmp_backend):
    """Verify a SUPERSEDE plan supersedes the target instead of deleting it.

    Mutation: routing `supersede` through `soft_delete` (the shipped
        merge shape), or reading `merged_text` for UPDATE only so the
        successor stores the bare fact.
    Oracle: the predecessor read back with `deleted_at` null and
        `superseded_by` naming the successor, the successor's merged
        content, and a `reconcile-supersede` oplog row naming both.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='the broker is kombu'))

    plan = _merge_plan(
        'new-1', 'old-1', action='supersede',
        fact_text='the broker is redis now',
        content='the broker is redis now (was kombu)')
    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    old = tmp_backend.nodes.get_include_deleted('old-1')
    assert old.deleted_at is None
    assert old.superseded_by == 'new-1'
    assert get_insight_by_id(tmp_db, 'old-1') is None
    assert get_insight_by_id(tmp_db, 'new-1').content == (
        'the broker is redis now (was kombu)')
    assert result['action'] == 'supersede'
    assert result['replaced_id'] == 'old-1'
    ops = {(e.operation, e.insight_id, e.detail)
           for e in tmp_backend.oplog.recent(limit=10)}
    assert ('reconcile-supersede', 'old-1', 'replaced by new-1') in ops


def test_supersede_does_not_carry_corroboration_but_update_does(
        tmp_db, tmp_backend):
    """Verify corroboration carries on UPDATE and resets on SUPERSEDE.

    Mutation: carrying the count on a contradiction, which credits the
        new fact with every restatement of the claim it just falsified;
        or dropping the carry on UPDATE.
    Oracle: hand-computed 4 on the UPDATE successor and 0 on the
        SUPERSEDE successor, from predecessors both stored at 4.
    """
    insert_insight(tmp_db, make_insight(
        id='old-u', content='refined later', corroboration_count=4))
    insert_insight(tmp_db, make_insight(
        id='old-s', content='contradicted later', corroboration_count=4))

    _apply_plan(tmp_backend, _merge_plan('new-u', 'old-u'),
                embed_cache={}, store_name='test')
    _apply_plan(tmp_backend, _merge_plan('new-s', 'old-s', action='supersede'),
                embed_cache={}, store_name='test')

    assert get_insight_by_id(tmp_db, 'new-u').corroboration_count == 4
    assert get_insight_by_id(tmp_db, 'new-s').corroboration_count == 0


def test_unmerged_supersede_is_marked_in_the_oplog(tmp_db, tmp_backend):
    """Verify a SUPERSEDE that stored the bare fact is marked as unmerged.

    Mutation: no marker, so the rate of successors that dropped the
        predecessor's still-true clauses cannot be measured.
    Oracle: the oplog detail ends with `(unmerged)` exactly when the
        successor's content equals the fact text, and carries no
        marker when the model supplied merged text.
    """
    insert_insight(tmp_db, make_insight(id='old-a', content='a'))
    insert_insight(tmp_db, make_insight(id='old-b', content='b'))

    _apply_plan(tmp_backend, _merge_plan(
        'new-a', 'old-a', action='supersede',
        fact_text='bare fact', content='bare fact'),
        embed_cache={}, store_name='test')
    _apply_plan(tmp_backend, _merge_plan(
        'new-b', 'old-b', action='supersede',
        fact_text='bare fact', content='bare fact merged with b'),
        embed_cache={}, store_name='test')

    details = {e.insight_id: e.detail
               for e in tmp_backend.oplog.recent(limit=10)
               if e.operation == 'reconcile-supersede'}
    assert details['old-a'] == 'replaced by new-a (unmerged)'
    assert details['old-b'] == 'replaced by new-b'
