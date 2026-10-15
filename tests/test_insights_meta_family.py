"""Queue and family behavior of `memman insights meta`.

The tests run on SQLite and on a Postgres parent with SQLite branches.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from memman import branch as branch_mod
from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.fingerprint import stored_fingerprint, write_fingerprint
from memman.queue import queue_db
from memman.store import factory
from memman.store.db import store_dir, store_exists
from memman.store.model import Insight
from memman.store.overlay import OverlayNodeStore, token_refusal
from memman.store.sqlite import open_sqlite_backend
from tests.conftest import force_drain, invoke
from tests.test_store_branch import _branch, _merge, _retire, _set_meta
from tests.test_store_branch import _snapshot


def _row_id(tag):
    """A full row id whose first eight characters are tag.
    """
    return f'{tag}-0000-4000-8000-000000000000'


def _make_branch(data_dir, parent, label='rearch'):
    """Branch parent, seeding its embed fingerprint when it has none.
    """
    with factory.open_backend(parent, data_dir) as backend:
        if stored_fingerprint(backend) is None:
            write_fingerprint(backend, seed_default_fingerprint())
    return _branch(data_dir, parent, label)


def _put(data_dir, store, tag, meta=None):
    """Insert a current row with id `_row_id(tag)` into store.
    """
    row_id = _row_id(tag)
    with factory.open_backend(store, data_dir) as backend:
        backend.nodes.insert(Insight(
            id=row_id, content=f'The {tag} service restarts nightly.',
            meta=meta))
    return row_id


def _store_name():
    """The parent store name the cross-backend fixture selected.
    """
    return os.environ.get('MEMMAN_STORE', 'default')


def _meta_of(data_dir, store, row_id):
    """Meta pairs of row_id as store holds it.
    """
    return _snapshot(data_dir, store)[0][row_id].meta


def _run_meta(runner, store, *tokens, where=None):
    """Run `insights meta` on store, with an optional --where pair.
    """
    args = ['insights', 'meta', '--store', store]
    if where is not None:
        args += ['--where', where]
    return invoke(runner, args + list(tokens))


def _changed(result):
    """The sorted `changed` list of a successful reply.
    """
    assert result.exit_code == 0, result.output
    reply = json.loads(result.output)
    assert reply['action'] == 'meta'
    return sorted(reply['changed'])


def _remember_meta(runner, store, text, pair):
    """Queue text on store with one --meta pair and return its queue uuid.
    """
    result = invoke(runner, [
        'remember', '--store', store, '--meta', pair, text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def _branch_state(data_dir, branch):
    """Rows and oplog of a branch read straight from its database.
    """
    with open_sqlite_backend(branch, data_dir, read_only=True) as backend:
        rows = {
            row_id: backend.nodes.get_raw(row_id)
            for row_id in backend.nodes.get_all_ids()
            }
        oplog = backend.oplog.recent(limit=10_000)
    return rows, oplog


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('queued_in', ['parent', 'branch'])
def test_id_still_queued_in_any_family_store_is_refused(
        cross_backend_runner, queued_in):
    """Verify a queued id is refused and the text names its store.

    Mutation: queue lookup limited to the invoking store.
    Oracle: the queue_uuid the `remember` call printed.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    queue_store = parent if queued_in == 'parent' else branch
    queued = _remember_meta(
        runner, queue_store, 'The retry cap for batch jobs is three.',
        'doc=a')

    result = _run_meta(runner, parent, queued, 'doc=b')

    assert result.exit_code != 0
    assert (
        f'is still queued in store {queue_store}; re-run once memman'
        f' insights show {queued} --store {queue_store} prints it'
        ) in result.output


@pytest.mark.no_auto_drain
@pytest.mark.parametrize(
    ('queued_in', 'run_on'), [('branch', 'parent'), ('parent', 'branch')])
def test_pending_remember_holding_the_where_pair_refuses(
        cross_backend_runner, queued_in, run_on):
    """Verify a pending remember holding the --where pair blocks the edit.

    Mutation: no queue check, a view-only store filter, or raw text
        matching.
    Oracle: the stored row and the queue_uuid, before and after.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    stored = {'parent': parent, 'branch': branch}
    _put(data_dir, parent, 'aaaa0001', {'doc': 'x/y'})
    _put(data_dir, branch, 'bbbb0001', {'doc': 'x/y'})
    queued = _remember_meta(
        runner, stored[queued_in], 'The retry cap for batch jobs is three.',
        'doc=x/y')
    parent_before = _snapshot(data_dir, parent)
    branch_before = _snapshot(data_dir, branch)

    result = _run_meta(
        runner, stored[run_on], 'doc=z', where='doc=x/y')

    assert result.exit_code != 0
    assert (
        f'queued write {queued} in store {stored[queued_in]} holds'
        f' doc=x/y; re-run once memman insights show {queued} --store'
        f' {stored[queued_in]} prints it') in result.output
    assert _snapshot(data_dir, parent) == parent_before
    assert _snapshot(data_dir, branch) == branch_before


@pytest.mark.no_auto_drain
def test_pending_remember_of_a_longer_path_does_not_block_where(
        cross_backend_runner):
    """Verify a pending remember holding only `doc=x/y/` lets the edit run.

    Mutation: raw text matching of the queued meta against the pair.
    Oracle: the stored row's meta after the call.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_id = _put(data_dir, parent, 'aaaa0001', {'doc': 'x/y'})
    _remember_meta(
        runner, branch, 'The retry cap for batch jobs is three.',
        'doc=x/y/')

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert _changed(result) == [row_id[:8]]
    assert _meta_of(data_dir, parent, row_id) == {'doc': 'z'}


@pytest.mark.no_auto_drain
def test_queue_refusal_wins_over_zero_match(cross_backend_runner):
    """Verify a pending remember holding the pair refuses with no stored match.

    Mutation: the zero-match check ordered before the queue check.
    Oracle: the refusal text names the queued write.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    queued = _remember_meta(
        runner, parent, 'The retry cap for batch jobs is three.', 'doc=x/y')

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert result.exit_code != 0
    assert f'queued write {queued} in store {parent} holds doc=x/y' in (
        result.output)
    assert 'no current row holds' not in result.output


@pytest.mark.no_auto_drain
def test_pending_replace_with_the_old_pair_refuses_where(
        cross_backend_runner):
    """Verify a pending replace carrying the --where pair blocks the edit.

    Mutation: checking `replaced_id is null` queue rows only.
    Oracle: the queued `replace --meta doc=x/y` row's queue_uuid.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    _put(data_dir, parent, 'aaaa0001', {'doc': 'x/y'})
    target = _put(data_dir, parent, 'aaaa0002')
    result = invoke(runner, [
        'replace', '--meta', 'doc=x/y', target,
        'The retry cap for batch jobs is five.'])
    assert result.exit_code == 0, result.output
    queued = json.loads(result.output)['id']

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert result.exit_code != 0
    assert f'queued write {queued} in store {parent} holds doc=x/y' in (
        result.output)


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('meta_arg', ['doc=a', 'doc='])
def test_pending_replace_with_explicit_meta_of_a_selected_id_refuses(
        cross_backend_runner, meta_arg):
    """Verify a queued replace with its own meta blocks an id-form edit.

    Mutation: explicit meta ignored, or a truthy test in place of
        `is not None`, so `--meta doc=` (queued as {}) passes.
    Oracle: the queued replace row's meta and queue_uuid.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    target = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    result = invoke(runner, [
        'replace', '--meta', meta_arg, target,
        'The retry cap for batch jobs is five.'])
    assert result.exit_code == 0, result.output
    queued = json.loads(result.output)['id']
    before = _snapshot(data_dir, parent)

    result = _run_meta(runner, parent, target, 'doc=b')

    assert result.exit_code != 0
    assert f'queued replace {queued} in store {parent} of ' in result.output
    assert (
        f'sets its own meta; re-run on {queued} once memman insights show'
        f' {queued} --store {parent} prints it') in result.output
    assert _snapshot(data_dir, parent) == before


@pytest.mark.no_auto_drain
def test_pending_replace_without_meta_lets_the_successor_inherit_the_edit(
        cross_backend_runner):
    """Verify a meta-less queued replace is allowed and inherits the new value.

    Mutation: refusing None-meta replaces, or the drain caching the
        target's meta at enqueue.
    Oracle: the successor's meta equals the pair the edit set.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    target = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    result = invoke(runner, [
        'replace', target, 'The retry cap for batch jobs is five.'])
    assert result.exit_code == 0, result.output

    result = _run_meta(runner, parent, target, 'doc=new')
    force_drain(data_dir)

    assert _changed(result) == [target[:8]]
    successor = _snapshot(data_dir, parent)[0][target].replaced_by
    assert successor is not None
    assert _meta_of(data_dir, parent, successor) == {'doc': 'new'}


@pytest.mark.no_auto_drain
def test_failed_and_done_queue_rows_holding_the_pair_are_ignored(
        cross_backend_runner):
    """Verify only pending queue rows block a --where edit.

    Mutation: the queue read taking `status in ('pending','failed')`.
    Oracle: statuses set by direct SQL on queue.db; exit 0.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_id = _put(data_dir, parent, 'aaaa0001', {'doc': 'x/y'})
    failed = _remember_meta(
        runner, parent, 'The retry cap for batch jobs is three.', 'doc=x/y')
    done = _remember_meta(
        runner, parent, 'The backup job runs at midnight.', 'doc=x/y')
    with queue_db(data_dir) as conn:
        conn.execute(
            "update queue set status = 'failed' where queue_uuid = ?",
            (failed,))
        conn.execute(
            "update queue set status = 'done' where queue_uuid = ?",
            (done,))
        conn.commit()

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert _changed(result) == [row_id[:8]]


def test_where_from_a_branch_edits_parent_rows_but_not_retired_copies(
        cross_backend_runner):
    """Verify --where on a branch edits P, Q and B and leaves Q's copy alone.

    Mutation: the family loop skips the parent, or writes copies.
    Oracle: hand-built P, Q and B with the old pair, Q forgotten on the
        branch through `_retire`; the branch's copy of Q keeps its pair.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    row_q = _put(data_dir, parent, 'aaaa0002', {'doc': 'old'})
    branch = _make_branch(data_dir, parent)
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})
    _retire(data_dir, branch, row_q)

    result = _run_meta(runner, branch, 'doc=new', where='doc=old')

    assert _changed(result) == sorted(r[:8] for r in (row_p, row_q, row_b))
    assert _meta_of(data_dir, parent, row_p) == {'doc': 'new'}
    assert _meta_of(data_dir, parent, row_q) == {'doc': 'new'}
    assert _meta_of(data_dir, branch, row_b) == {'doc': 'new'}
    assert _meta_of(data_dir, branch, row_q) == {'doc': 'old'}


def test_merge_after_an_edit_from_the_parent_carries_the_new_value(
        cross_backend_runner):
    """Verify a branch row edited from the parent merges with its new meta.

    Mutation: members limited to the invoking store.
    Oracle: B's meta in the parent after the merge.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')
    _merge(runner, branch)

    assert _changed(result) == [row_b[:8]]
    assert _meta_of(data_dir, parent, row_b) == {'doc': 'new'}


def test_where_from_the_parent_reaches_two_branches(cross_backend_runner):
    """Verify --where on the parent edits one row in each of two branches.

    Mutation: the member loop stops after one branch.
    Oracle: one hand-made row per branch.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    first = _make_branch(data_dir, parent, 'one')
    second = _make_branch(data_dir, parent, 'two')
    row_1 = _put(data_dir, first, 'bbbb0001', {'doc': 'old'})
    row_2 = _put(data_dir, second, 'cccc0001', {'doc': 'old'})

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    assert _changed(result) == sorted([row_1[:8], row_2[:8]])
    assert _meta_of(data_dir, first, row_1) == {'doc': 'new'}
    assert _meta_of(data_dir, second, row_2) == {'doc': 'new'}


def test_a_merging_branch_with_its_token_refuses_and_writes_nothing(
        cross_backend_runner):
    """Verify a mid-merge branch whose token the parent holds blocks the run.

    Mutation: the merging check made after the parent write.
    Oracle: `_snapshot` of the parent and the branch, unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    branch = _make_branch(data_dir, parent)
    _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})
    _set_meta(data_dir, branch, 'branch_merging', '2026-10-06T00:00:00Z')
    parent_before = _snapshot(data_dir, parent)
    branch_before = _snapshot(data_dir, branch)

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    assert result.exit_code != 0
    assert (
        f'store {branch} is merging into its parent; re-run memman store'
        f' merge {branch} to finish it') in result.output
    assert _snapshot(data_dir, parent) == parent_before
    assert _snapshot(data_dir, branch) == branch_before


def test_a_branch_merged_past_the_parent_commit_is_skipped(
        cross_backend_runner, monkeypatch):
    """Verify a branch whose merge passed the parent commit is not written.

    Mutation: refusing the run, or writing the branch.
    Oracle: the parent rows edited; the branch's rows and oplog
        unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    branch = _make_branch(data_dir, parent)
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})

    def crash(*args, **kwargs):
        raise RuntimeError('killed before the removal')

    with monkeypatch.context() as patch:
        patch.setattr(branch_mod, '_remove_branch', crash)
        assert invoke(runner, ['store', 'merge', branch]).exit_code != 0
    branch_before = _branch_state(data_dir, branch)

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    assert _changed(result) == sorted([row_p[:8], row_b[:8]])
    assert _meta_of(data_dir, parent, row_p) == {'doc': 'new'}
    assert _meta_of(data_dir, parent, row_b) == {'doc': 'new'}
    assert _branch_state(data_dir, branch) == branch_before


def test_a_missing_or_unreadable_branch_directory_is_skipped(
        cross_backend_runner):
    """Verify a gone branch directory and a db-less one are skipped.

    Mutation: members taken from the parent's tokens, or a raise on the
        unreadable directory.
    Oracle: `store_exists` stays False for the gone branch, the db-less
        directory gains no database, and the parent row has the new
        meta.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    gone = _make_branch(data_dir, parent, 'gone')
    shutil.rmtree(store_dir(data_dir, gone))
    bare = f'{parent}__bare'
    Path(store_dir(data_dir, bare)).mkdir(parents=True)

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    assert _changed(result) == [row_p[:8]]
    assert _meta_of(data_dir, parent, row_p) == {'doc': 'new'}
    assert not store_exists(data_dir, gone)
    assert not (Path(store_dir(data_dir, bare)) / 'memman.db').exists()


def test_foreign_token_skips_a_sibling_and_refuses_the_invoking_branch(
        cross_backend_runner):
    """Verify a sibling with a foreign token is skipped, the invoker refused.

    Mutation: no token check.
    Oracle: the branch token rewritten by `_set_meta`; `token_refusal`
        wording; the sibling's row unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    stale = _make_branch(data_dir, parent, 'stale')
    live = _make_branch(data_dir, parent, 'live')
    row_stale = _put(data_dir, stale, 'bbbb0001', {'doc': 'old'})
    row_live = _put(data_dir, live, 'cccc0001', {'doc': 'old'})
    _set_meta(data_dir, stale, 'branch_token', 'foreign:token')

    sibling_run = _run_meta(runner, live, 'doc=new', where='doc=old')
    refused = _run_meta(runner, stale, 'doc=newer', where='doc=new')

    assert _changed(sibling_run) == sorted([row_p[:8], row_live[:8]])
    assert _meta_of(data_dir, stale, row_stale) == {'doc': 'old'}
    assert refused.exit_code != 0
    assert token_refusal(stale, parent) in refused.output
    assert _meta_of(data_dir, parent, row_p) == {'doc': 'new'}


def test_id_form_from_the_parent_edits_a_branch_only_row(
        cross_backend_runner):
    """Verify the id form from the parent reaches a branch-only row.

    Mutation: ids resolved in the invoking store only.
    Oracle: the id the branch's `remember` returned.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_b = _remember_meta(
        runner, branch, 'The grackle colony nests by the river.', 'doc=old')

    result = _run_meta(runner, parent, row_b, 'doc=new')

    assert _changed(result) == [row_b[:8]]
    assert _meta_of(data_dir, branch, row_b) == {'doc': 'new'}


def test_prefix_matching_rows_in_two_members_refuses(cross_backend_runner):
    """Verify a prefix held by rows in two members is refused.

    Mutation: the first member's match taken.
    Oracle: hand-set ids sharing the prefix `abcd`; both rows unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    _put(data_dir, parent, 'abcd0001', {'doc': 'old'})
    _put(data_dir, branch, 'abcd0002', {'doc': 'old'})
    parent_before = _snapshot(data_dir, parent)
    branch_before = _snapshot(data_dir, branch)

    result = _run_meta(runner, parent, 'abcd', 'doc=new')

    assert result.exit_code != 0
    assert (
        'prefix abcd matches several rows or queued writes;'
        ' give more of the id') in result.output
    assert _snapshot(data_dir, parent) == parent_before
    assert _snapshot(data_dir, branch) == branch_before


def test_id_form_gives_the_same_result_from_every_member(
        cross_backend_runner):
    """Verify the id form edits both rows whichever member --store names.

    Mutation: refusing on one member and editing on another.
    Oracle: each run's `changed` and the rows' meta.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})
    expected = sorted([row_p[:8], row_b[:8]])

    from_parent = _run_meta(runner, parent, row_p, row_b, 'doc=a')
    after_parent = (
        _meta_of(data_dir, parent, row_p), _meta_of(data_dir, branch, row_b))
    from_branch = _run_meta(runner, branch, row_p, row_b, 'doc=b')
    after_branch = (
        _meta_of(data_dir, parent, row_p), _meta_of(data_dir, branch, row_b))

    assert _changed(from_parent) == expected
    assert after_parent == ({'doc': 'a'}, {'doc': 'a'})
    assert _changed(from_branch) == expected
    assert after_branch == ({'doc': 'b'}, {'doc': 'b'})


def test_fault_between_members_leaves_the_parent_committed(
        cross_backend_runner, monkeypatch):
    """Verify a branch write fault keeps the parent edit and a rerun is minimal.

    Mutation: a rerun rewriting changed rows, or an error naming no
        committed member.
    Oracle: the error text, and `changed` of the second run.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_p = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})

    def fail(*args, **kwargs):
        raise RuntimeError('disk full')

    with monkeypatch.context() as patch:
        patch.setattr(OverlayNodeStore, 'update_meta', fail, raising=False)
        faulted = _run_meta(runner, parent, row_p, row_b, 'doc=new')
    rerun = _run_meta(runner, parent, row_p, row_b, 'doc=new')

    assert faulted.exit_code != 0
    assert (
        f'meta edit failed in store {branch}: disk full; stores already'
        f' written: {parent}; re-run to finish') in faulted.output
    assert _changed(rerun) == [row_b[:8]]
    assert _meta_of(data_dir, parent, row_p) == {'doc': 'new'}
    assert _meta_of(data_dir, branch, row_b) == {'doc': 'new'}


@pytest.mark.no_auto_drain
def test_pending_replace_of_a_replaced_row_blocks_an_edit_of_its_head(
        cross_backend_runner):
    """Verify the queued replace target is followed along its chain to Y.

    Mutation: the queued `replaced_id` compared literally instead of
        through `_queued_replace_target`.
    Oracle: Y, the head the drain would retire.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_x = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    result = invoke(runner, [
        'replace', '--meta', 'doc=a', row_x,
        'The retry cap for batch jobs is five.'])
    assert result.exit_code == 0, result.output
    queued = json.loads(result.output)['id']
    row_y = _put(data_dir, parent, 'aaaa0002', {'doc': 'old'})
    with factory.open_backend(parent, data_dir) as backend:
        assert backend.nodes.mark_replaced(row_x, row_y)
    before = _snapshot(data_dir, parent)

    result = _run_meta(runner, parent, row_y, 'doc=b')

    assert result.exit_code != 0
    assert (
        f'queued replace {queued} in store {parent} of ') in result.output
    assert (
        f'sets its own meta; re-run on {queued} once memman insights show'
        f' {queued} --store {parent} prints it') in result.output
    assert _snapshot(data_dir, parent) == before


def test_a_branch_whose_parent_cannot_open_refuses_before_any_write(
        mm_runner):
    """Verify an unreadable parent refuses the run and the branch is intact.

    Mutation: branch members written before the parent opens.
    Oracle: the branch rows and oplog, read from its database before and
        after.
    """
    _, data_dir = mm_runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_b = _remember_meta(
        mm_runner, branch, 'The grackle colony nests by the river.',
        'doc=old')
    before = _branch_state(data_dir, branch)
    shutil.rmtree(store_dir(data_dir, parent))

    result = _run_meta(mm_runner, branch, 'doc=new', where='doc=old')

    assert result.exit_code != 0
    assert f'cannot read store {parent!r}' in result.output
    assert row_b in before[0]
    assert _branch_state(data_dir, branch) == before


def test_cap_refusal_on_a_branch_row_writes_no_parent_row(
        cross_backend_runner):
    """Verify an overflow on a branch row stops the parent row's write too.

    Mutation: the cap checked per member inside its transaction, or each
        row written as it passes its check.
    Oracle: the parent row, first by id and by member order, keeps no
        meta after the refusal.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_p = _put(data_dir, parent, 'aaaa0001')
    row_b = _put(data_dir, branch, 'bbbb0001', {'ticket': 'y' * 100})

    result = _run_meta(runner, parent, row_p, row_b, f'doc={"x" * 100}')

    assert result.exit_code == 1
    assert f'meta of {row_b[:8]} would be 214 bytes' in result.output
    assert _meta_of(data_dir, parent, row_p) is None
    assert _meta_of(data_dir, branch, row_b) == {'ticket': 'y' * 100}


def test_a_branch_row_edit_is_logged_in_the_branch_oplog(
        cross_backend_runner):
    """Verify the oplog entry for a branch row goes to the branch's oplog.

    Mutation: every member's entries written to the parent's oplog.
    Oracle: the branch oplog read from its database, and the parent's
        oplog, after an edit run from the parent.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    row_b = _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})

    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    _, branch_oplog = _branch_state(data_dir, branch)
    with factory.open_backend(parent, data_dir, read_only=True) as backend:
        parent_oplog = backend.oplog.recent(limit=10_000)
    assert _changed(result) == [row_b[:8]]
    assert [(entry.operation, entry.insight_id) for entry in branch_oplog
            if entry.operation == 'meta'] == [('meta', row_b)]
    assert [entry for entry in parent_oplog if entry.operation == 'meta'] == []


def test_a_branch_whose_parent_has_no_database_creates_none(mm_runner):
    """Verify a parent directory with no database is refused, not created.

    Mutation: the parent opened writable without a read check, which
        creates an empty parent database before the token refusal.
    Oracle: no `memman.db*` file in the parent directory afterward; the
        branch unchanged.
    """
    _, data_dir = mm_runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})
    before = _branch_state(data_dir, branch)
    parent_dir = Path(store_dir(data_dir, parent))
    for path in parent_dir.glob('memman.db*'):
        path.unlink()

    result = _run_meta(mm_runner, branch, 'doc=new', where='doc=old')

    assert result.exit_code != 0
    assert f'cannot read store {parent!r}' in result.output
    assert list(parent_dir.glob('memman.db*')) == []
    assert _branch_state(data_dir, branch) == before


def test_a_branch_whose_parent_database_is_empty_refuses_cleanly(mm_runner):
    """Verify a zero-byte parent database gives the cannot-read refusal.

    Mutation: `sqlite3.Error` left out of the parent open's except, so
        the run ends on a raw "no such table" error.
    Oracle: the `cannot read store <parent>` text; the branch unchanged.
    """
    _, data_dir = mm_runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)
    _put(data_dir, branch, 'bbbb0001', {'doc': 'old'})
    before = _branch_state(data_dir, branch)
    parent_dir = Path(store_dir(data_dir, parent))
    for path in parent_dir.glob('memman.db*'):
        path.unlink()
    (parent_dir / 'memman.db').touch()

    result = _run_meta(mm_runner, branch, 'doc=new', where='doc=old')

    assert result.exit_code != 0
    assert f'cannot read store {parent!r}' in result.output
    assert _branch_state(data_dir, branch) == before


@pytest.mark.no_auto_drain
def test_pending_replace_chained_through_a_queued_replace_refuses(
        cross_backend_runner):
    """Verify a --meta replace of a queued replace of X blocks an edit of X.

    Mutation: the queued target compared without following an earlier
        queued replace, so the edit X passes to U0 is dropped by U1.
    Oracle: U1, the queued `replace U0 --meta doc=a`, named in the
        refusal; the store unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_x = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    first = invoke(runner, [
        'replace', row_x, 'The retry cap for batch jobs is five.'])
    assert first.exit_code == 0, first.output
    second = invoke(runner, [
        'replace', '--meta', 'doc=a', json.loads(first.output)['id'],
        'The retry cap for batch jobs is six.'])
    assert second.exit_code == 0, second.output
    queued = json.loads(second.output)['id']
    before = _snapshot(data_dir, parent)

    result = _run_meta(runner, parent, row_x, 'ticket=t1')

    assert result.exit_code != 0
    assert (
        f'queued replace {queued} in store {parent} of {row_x[:8]} sets its'
        ' own meta') in result.output
    assert _snapshot(data_dir, parent) == before


@pytest.mark.no_auto_drain
def test_a_pending_row_already_stored_does_not_block(cross_backend_runner):
    """Verify a queue row stored but still pending counts as stored.

    Mutation: every pending row checked, so a row the drain stored before
        a crash blocks until the next drain marks it done.
    Oracle: the stored row's id in `changed`, after its queue row is set
        back to pending by direct SQL.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_id = _remember_meta(
        runner, parent, 'The retry cap for batch jobs is three.', 'doc=x/y')
    force_drain(data_dir)
    with queue_db(data_dir) as conn:
        conn.execute(
            "update queue set status = 'pending' where queue_uuid = ?",
            (row_id,))
        conn.commit()

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert _changed(result) == [row_id[:8]]


def test_prefix_matching_two_rows_in_one_store_refuses(cross_backend_runner):
    """Verify a prefix held by two rows of one store is refused by name.

    Mutation: the `ValueError` from `resolve_id` left uncaught, so the run
        ends on a traceback.
    Oracle: hand-set ids sharing the prefix `abcd`; the store unchanged.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    _put(data_dir, parent, 'abcd0001', {'doc': 'old'})
    _put(data_dir, parent, 'abcd0002', {'doc': 'old'})
    before = _snapshot(data_dir, parent)

    result = _run_meta(runner, parent, 'abcd', 'doc=new')

    assert result.exit_code == 1
    assert (
        'prefix abcd matches several rows or queued writes;'
        ' give more of the id') in result.output
    assert _snapshot(data_dir, parent) == before


def test_a_row_retired_before_its_write_is_left_out_of_changed(
        cross_backend_runner, monkeypatch):
    """Verify a row whose update_meta returns False is not reported changed.

    Mutation: every selected row added to `changed` or given an oplog
        entry, written or not.
    Oracle: a stub returning False for one row, standing in for a forget
        between the read and the write; the parent's `meta` oplog rows.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    kept = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    gone = _put(data_dir, parent, 'aaaa0002', {'doc': 'old'})
    with factory.open_backend(parent, data_dir) as backend:
        node_class = type(backend.nodes)
    real_update = node_class.update_meta

    def update_meta(self, id, meta):
        return False if id == gone else real_update(self, id, meta)

    monkeypatch.setattr(node_class, 'update_meta', update_meta)
    result = _run_meta(runner, parent, 'doc=new', where='doc=old')

    with factory.open_backend(parent, data_dir, read_only=True) as backend:
        logged = [entry.insight_id for entry in backend.oplog.recent(limit=100)
                  if entry.operation == 'meta']
    assert _changed(result) == [kept[:8]]
    assert logged == [kept]


@pytest.mark.no_auto_drain
def test_several_queue_blockers_name_the_first_and_count_the_rest(
        cross_backend_runner):
    """Verify two blocking queued writes give the first and `(+1 more)`.

    Mutation: the count suffix dropped or off by one.
    Oracle: two queued remembers holding the pair, oldest first.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    _put(data_dir, parent, 'aaaa0001', {'doc': 'x/y'})
    first = _remember_meta(
        runner, parent, 'The retry cap for batch jobs is three.', 'doc=x/y')
    _remember_meta(
        runner, parent, 'The backup job runs at midnight.', 'doc=x/y')

    result = _run_meta(runner, parent, 'doc=z', where='doc=x/y')

    assert result.exit_code == 1
    assert f'queued write {first} in store {parent} holds' in result.output
    assert result.output.rstrip().endswith('prints it (+1 more)')


def test_zero_match_names_every_member_parent_first(cross_backend_runner):
    """Verify the zero-match text lists the parent, then each branch.

    Mutation: only the invoking store named, or branches listed first.
    Oracle: the parent and one branch, run from the branch.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    branch = _make_branch(data_dir, parent)

    result = _run_meta(runner, branch, 'doc=new', where='doc=absent')

    assert result.exit_code == 1
    assert (
        f'no current row holds doc=absent in {parent}, {branch};'
        ) in result.output


def test_a_replaced_id_whose_chain_ends_forgotten_says_so(
        cross_backend_runner):
    """Verify a replaced id with a forgotten head names no row to edit.

    Mutation: the forgotten head offered as the row to set meta on.
    Oracle: X replaced by Y, then Y forgotten, by hand.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = _store_name()
    row_x = _put(data_dir, parent, 'aaaa0001', {'doc': 'old'})
    row_y = _put(data_dir, parent, 'aaaa0002', {'doc': 'old'})
    with factory.open_backend(parent, data_dir) as backend:
        assert backend.nodes.mark_replaced(row_x, row_y)
        assert backend.nodes.soft_delete(row_y)

    result = _run_meta(runner, parent, row_x, 'doc=new')

    assert result.exit_code == 1
    assert (
        f'insight {row_x} was replaced by {row_y}; its chain ends in a'
        f' forgotten row, or run insights show {row_x} --history'
        ) in result.output
