"""Metadata a replace must carry from its predecessor.

A replace is not an in-place edit: `_apply_plan` supersedes the target
and inserts a successor built from the incoming write. The predecessor
keeps its content behind `superseded_by`, but every field the
successor does not explicitly copy is missing from the current view.

These tests pin what the successor carries.
"""

from memman.pipeline.remember import FactPlan, _apply_plan
from memman.store.node import get_insight_by_id, insert_insight
from tests.conftest import make_insight


def _replace_plan(new_id, target_id, **insight_overrides):
    """Build a replace FactPlan targeting `target_id`."""
    overrides = {
        'id': new_id,
        'content': 'merged content',
        'importance': 3,
        }
    overrides.update(insight_overrides)
    return FactPlan(
        action='replace',
        fact_insight=make_insight(**overrides),
        targets=[(target_id, 'replace')],
        embed_vec=None,
        enrichment={},
        )


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
        'new-1', 'old-1', content='the broker is redis now')
    result = _apply_plan(tmp_backend, plan)

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
        action='replace',
        fact_insight=make_insight(id='new-1', content='the correction'),
        targets=[('gone-1', 'replace')],
        embed_vec=None, enrichment={})
    result = _apply_plan(tmp_backend, plan)

    assert result['action'] == 'add'
    ops = {(e.operation, e.insight_id): e.detail
           for e in tmp_backend.oplog.recent(limit=50)}
    assert ('target-gone', 'new-1') in ops
    assert 'gone-1' in ops[('target-gone', 'new-1')]
    assert ('replace', 'gone-1') not in ops
