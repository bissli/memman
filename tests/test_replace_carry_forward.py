"""Metadata and edges a replace must carry from its predecessor.

A replace is not an in-place edit: `_apply_plan` supersedes the target
and inserts a successor built from the incoming write. The predecessor
keeps its content behind `superseded_by`, but every field the
successor does not explicitly copy is missing from the current view,
along with the predecessor's whole edge neighborhood.

These tests pin what the successor carries.
"""

from memman.pipeline.remember import FactPlan, _apply_plan
from memman.store.edge import get_edges_by_node, insert_edge
from memman.store.node import get_insight_by_id, insert_insight
from tests.conftest import make_edge, make_insight


def _replace_plan(new_id, target_id, *, fact_text=None, **insight_overrides):
    """Build a replace FactPlan targeting `target_id`."""
    overrides = {
        'id': new_id,
        'content': 'merged content',
        'importance': 3,
        }
    overrides.update(insight_overrides)
    return FactPlan(
        action='replace',
        fact_text=fact_text or 'merged content',
        fact_insight=make_insight(**overrides),
        targets=[(target_id, 'replace')],
        embed_vec=None,
        enrichment={},
        )


def test_replace_repoints_target_edges_to_successor(tmp_db, tmp_backend):
    """Verify the target's edges move to the successor rather than vanish.

    Mutation: leaving the bare `delete_by_node` with no re-point, which
        drops the target's whole neighborhood.
    Oracle: the entity edge's own type and weight read back off the
        successor. `fast_edges` mints temporal-proximity edges between
        any two nodes created moments apart, so matching on the
        neighbor id alone passes without the re-point.
    """
    insert_insight(tmp_db, make_insight(id='old-1', content='original'))
    insert_insight(tmp_db, make_insight(id='ctx-1', content='context'))
    insert_edge(tmp_db, make_edge(
        source_id='ctx-1', target_id='old-1',
        edge_type='entity', weight=0.83))

    plan = _replace_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert get_edges_by_node(tmp_db, 'old-1') == []
    carried = [
        e for e in get_edges_by_node(tmp_db, 'new-1')
        if e.edge_type == 'entity']
    assert len(carried) == 1
    assert carried[0].source_id == 'ctx-1'
    assert carried[0].target_id == 'new-1'
    assert carried[0].weight == 0.83


def test_replace_repoint_drops_target_self_edge(tmp_db, tmp_backend):
    """Verify a self-edge on the target does not become one on the successor.

    Mutation: re-pointing both endpoints with no far-endpoint check,
        which turns old-1 -> old-1 into new-1 -> new-1.
    Oracle: absence of any edge whose two endpoints are both 'new-1'.
    """
    insert_insight(tmp_db, make_insight(id='old-1', content='original'))
    insert_edge(tmp_db, make_edge(
        source_id='old-1', target_id='old-1',
        edge_type='entity', weight=0.7))

    plan = _replace_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert not [
        e for e in get_edges_by_node(tmp_db, 'new-1')
        if e.source_id == 'new-1' and e.target_id == 'new-1']


def test_replace_carries_target_access_count(tmp_db, tmp_backend):
    """Verify recall history on the target survives the replace.

    Mutation: leaving access_count at the incoming write's zero, which
        erases every recall the target had served.
    Oracle: hand-computed 7, the target's stored count.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='original', access_count=7))

    plan = _replace_plan('new-1', 'old-1')
    _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    successor = get_insight_by_id(tmp_db, 'new-1')
    assert successor is not None
    assert successor.access_count == 7


def test_replace_plan_links_the_predecessor_and_keeps_it(tmp_db, tmp_backend):
    """Verify a replace plan supersedes the target instead of deleting it.

    Mutation: routing `replace` through `soft_delete`, which drops the
        predecessor's content instead of keeping it behind
        `superseded_by`.
    Oracle: the predecessor read back with `deleted_at` null and
        `superseded_by` naming the successor, the successor's stored
        content, and a `replace` oplog row naming both.
    """
    insert_insight(tmp_db, make_insight(
        id='old-1', content='the broker is kombu'))

    plan = _replace_plan(
        'new-1', 'old-1', fact_text='the broker is redis now',
        content='the broker is redis now')
    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    old = tmp_backend.nodes.get_include_deleted('old-1')
    assert old.deleted_at is None
    assert old.superseded_by == 'new-1'
    assert get_insight_by_id(tmp_db, 'old-1') is None
    assert get_insight_by_id(tmp_db, 'new-1').content == 'the broker is redis now'
    assert result['action'] == 'replace'
    assert result['replaced_ids'] == ['old-1']
    ops = {(e.operation, e.insight_id, e.detail)
           for e in tmp_backend.oplog.recent(limit=10)}
    assert ('replace', 'old-1', 'replaced by new-1') in ops


def test_a_gone_target_is_recorded_in_the_oplog(tmp_db, tmp_backend):
    """Verify a dropped target leaves an operator-readable record.

    A target can vanish between the plan and the apply, and the write
    then degrades to a plain add: no pointer, no supersession, and the
    caller's correction silently does not attach. The row IS stored,
    so nothing is lost, but without a record the operator who ran
    `replace` has no way to learn the correction did not land.

    Mutation: dropping the gone target with the warning log alone,
        which reaches the drain's own stdout and no per-store surface,
        so `memman log list` shows the degraded write as an ordinary
        add.
    Oracle: the oplog read back for the successor, holding a
        `target-gone` row naming the requested target, and no
        `replace` row at all since the single target never linked.
    """
    insert_insight(tmp_db, make_insight(id='gone-1', content='forgotten claim'))
    assert tmp_backend.nodes.soft_delete('gone-1') is True

    plan = FactPlan(
        action='replace', fact_text='the correction',
        fact_insight=make_insight(id='new-1', content='the correction'),
        targets=[('gone-1', 'replace')],
        embed_vec=None, enrichment={})
    result = _apply_plan(tmp_backend, plan, embed_cache={}, store_name='test')

    assert result['action'] == 'add'
    ops = {(e.operation, e.insight_id): e.detail
           for e in tmp_backend.oplog.recent(limit=50)}
    assert ('target-gone', 'new-1') in ops
    assert 'gone-1' in ops[('target-gone', 'new-1')]
    assert ('replace', 'gone-1') not in ops


def test_superseded_target_leaves_the_drain_cache(tmp_db, tmp_backend):
    """Verify a superseded target stops being a semantic candidate.

    Mutation: dropping `embed_cache.pop(target_id, None)` from the
        apply phase's sweep, so the row it just superseded stays a
        live neighbor and the next row of the same drain mints a
        semantic edge onto a row that must be edgeless.
    Oracle: the store's own dangling-edge count, whose definition is
        that a superseded row owns no edges.
    """
    vec = [1.0, 0.0, 0.0]
    insert_insight(tmp_db, make_insight(id='old-1', content='the target'))
    # The drain snapshots every current row's vector once, before any
    # plan runs, so the target stays a candidate until something evicts
    # it from this dict.
    embed_cache = {'old-1': list(vec)}

    _apply_plan(
        tmp_backend,
        _replace_plan('new-1', 'old-1'),
        embed_cache=embed_cache, store_name='test')

    later = FactPlan(
        action='add', fact_text='a later row',
        fact_insight=make_insight(id='later-1', content='a later row'),
        targets=[], embed_vec=list(vec), enrichment={})
    _apply_plan(
        tmp_backend, later, embed_cache=embed_cache, store_name='test')

    assert tmp_backend.edges.count_dangling_by_type() == {}
