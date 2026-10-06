"""Store branches and the missing-store refusal they rely on.
"""

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from memman import branch as branch_mod
from memman import config
from memman.cli import cli, list_agent_commands, list_claude_permissions
from memman.embed import registry
from memman.embed.fingerprint import META_KEY, Fingerprint, swap_command
from memman.embed.fingerprint import write_fingerprint
from memman.migrate import MigrateInsight
from memman.pipeline.enrich import enrich_pending
from memman.queue import queue_db
from memman.setup import claude as claude_setup
from memman.setup import scheduler as sched_mod
from memman.setup.codex import install_codex
from memman.store import factory
from memman.store.db import list_local_store_dirs, read_active, store_dir
from memman.store.db import write_active
from memman.store.errors import StoreMissingError
from memman.store.model import Insight
from memman.store.sqlite import SqliteRecallSession, open_sqlite_backend
from tests.conftest import EMBEDDING_DIM, _set_env_file_value, _vec
from tests.conftest import force_drain, invoke, queued_contents
from tests.embed.test_swap_cli import _FakeTargetProvider

CREATED = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def _remember(runner, text, store=None):
    """Store text, drained, and return its id.
    """
    args = ['remember', text] if store is None else [
        'remember', '--store', store, text]
    result = invoke(runner, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def _replace(runner, target, text, store=None):
    """Replace target with text, drained, and return the successor id.
    """
    args = ['replace', target, text] if store is None else [
        'replace', '--store', store, target, text]
    result = invoke(runner, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def _branch(data_dir, parent='default', label='rearch'):
    """Create a branch of parent and return its store name.
    """
    return branch_mod.create_branch(data_dir, parent, label)['store']


def _row(row_id, content, embedding, **kwargs):
    """A full row for `insert_raw`, with fixed timestamps.
    """
    fields = {
        'id': row_id, 'content': content, 'summary': None,
        'embedding': embedding, 'enrich_attempted_at': None,
        'enriched_at': None, 'created_at': CREATED, 'updated_at': CREATED,
        'deleted_at': None, 'prompt_version': None, 'embedding_model': None,
        'queue_uuid': None, 'replaced_by': None, 'author': 'alice',
        }
    fields.update(kwargs)
    return MigrateInsight(**fields)


def _snapshot(data_dir, store):
    """Every row, meta key and oplog row of store.
    """
    with factory.open_backend(store, data_dir, read_only=True) as backend:
        rows = {
            row_id: backend.nodes.get_raw(row_id)
            for row_id in backend.nodes.get_all_ids()
            }
        meta = {
            key: backend.meta.get(key)
            for key in backend.meta.keys()  # noqa: SIM118
            }
        oplog = backend.oplog.recent(limit=10_000)
    return rows, meta, oplog


def _active_contents(data_dir, store):
    """Content of every row recall can see in store.
    """
    with factory.open_backend(store, data_dir) as backend:
        return {ins.content for ins in backend.nodes.get_all_active()}


def test_branch_work_writes_nothing_to_the_parent(
        cross_backend_runner, monkeypatch):
    """Verify add, replace, forget and the drain leave the parent unchanged.

    Mutation: a parent write method called from the overlay, enrich or
        embed reading the parent's pending rows, or a writable parent
        open.
    Oracle: the parent's rows, meta and oplog before and after, and a
        recording stub of `factory.open_backend`.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = os.environ.get('MEMMAN_STORE', 'default')
    target = _remember(runner, 'The retry cap for batch jobs is three.')
    gone = _remember(runner, 'The backup job runs at midnight.')
    with factory.open_backend(parent, data_dir) as backend:
        backend.nodes.insert(Insight(
            id='pending-parent', content='The scheduler polls every minute.'))
    branch = _branch(data_dir, parent)
    before = _snapshot(data_dir, parent)
    opens = []
    real_open = factory.open_backend

    def recording_open(store, data_dir_arg, **kwargs):
        opens.append((store, kwargs))
        return real_open(store, data_dir_arg, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(factory, 'open_backend', recording_open)
        _remember(runner, 'The grackle colony nests by the river.', branch)
        _replace(
            runner, target, 'The retry cap for batch jobs is five.', branch)
        assert invoke(
            runner, ['forget', '--store', branch, gone]).exit_code == 0
        force_drain(data_dir)
    parent_opens = [kwargs for store, kwargs in opens if store == parent]

    assert parent_opens
    assert all(
        kwargs == {'read_only': True, 'create': False}
        for kwargs in parent_opens)
    assert _snapshot(data_dir, parent) == before


def test_recall_on_a_branch_leaves_out_a_row_the_branch_retired(mm_runner):
    """Verify a parent row the branch forgot reaches no recall channel.

    Mutation: "branch first, then parent" lookups that never hide a
        parent row the branch holds.
    Oracle: the forgotten row's own text as the query, which matches it
        on keyword, vector and recency.
    """
    _, data_dir = mm_runner
    text = 'The grackle colony nests by the river.'
    target = _remember(mm_runner, text)
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    assert invoke(mm_runner, ['forget', '--store', branch, target]).exit_code == 0

    full = invoke(mm_runner, ['recall', '--store', branch, text])
    basic = invoke(mm_runner, ['recall', '--store', branch, '--basic', text])

    assert full.exit_code == 0, full.output
    assert basic.exit_code == 0, basic.output
    assert 'grackle' not in full.output
    assert 'grackle' not in basic.output
    assert 'retry cap' in full.output


def test_recall_on_a_branch_sees_old_and_new_parent_rows(mm_runner):
    """Verify recall reads parent rows from before and after branch creation.

    Mutation: the overlay reading only the branch, or a copy of the
        parent taken at creation.
    Oracle: one parent row written before the branch, one after.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    branch = _branch(data_dir)
    _remember(mm_runner, 'The scheduler polls the queue every minute.')

    result = invoke(mm_runner, [
        'recall', '--store', branch, 'backup job scheduler queue'])

    assert result.exit_code == 0, result.output
    assert 'backup job runs at midnight' in result.output
    assert 'polls the queue every minute' in result.output


def test_vector_anchors_fill_k_when_hidden_rows_lead_the_parent(mm_runner):
    """Verify the anchor list holds k visible ids past hidden parent rows.

    Mutation: hidden ids filtered after the cut to k, or the parent
        asked for only k.
    Oracle: four parent rows at hand-set cosines to the query; the
        branch hides the two nearest, so the next two are the answer.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    rows = [
        _row(f'p{n}', f'row {n}', _vec(1.0, 0.1 * n)) for n in range(1, 5)]
    with open_sqlite_backend('default', data_dir) as parent:
        for row in rows:
            parent.nodes.insert_raw(row)
    with open_sqlite_backend(branch, data_dir) as backend:
        for row in rows[:2]:
            backend.nodes.insert_raw(
                dataclasses.replace(row, deleted_at=CREATED))

    with factory.open_backend(branch, data_dir) as overlay, \
            overlay.recall_session() as session:
        anchors = session.vector_anchors(_vec(1.0), k=2)

    assert [row_id for row_id, _ in anchors] == ['p3', 'p4']


def test_vector_anchors_over_ask_the_parent_by_copies_only(
        mm_runner, monkeypatch):
    """Verify the parent is asked for k plus the copies, not every branch id.

    Mutation: the over-ask counting every branch id, which past a few
        hundred branch-only rows pushes Postgres past its ef_search cap.
    Oracle: a stub that records the k of each call; the branch holds
        three branch-only rows and one copy.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    copied = _row('p1', 'parent row', _vec(1.0))
    with open_sqlite_backend('default', data_dir) as parent:
        parent.nodes.insert_raw(copied)
    with open_sqlite_backend(branch, data_dir) as backend:
        backend.nodes.insert_raw(dataclasses.replace(copied, deleted_at=CREATED))
        for n in range(3):
            backend.nodes.insert_raw(_row(f'b{n}', f'branch row {n}', _vec(0.5)))
    asked = []
    real_anchors = SqliteRecallSession.vector_anchors

    def recording_anchors(self, query_vec, *, k=10):
        asked.append(k)
        return real_anchors(self, query_vec, k=k)

    monkeypatch.setattr(
        SqliteRecallSession, 'vector_anchors', recording_anchors)

    with factory.open_backend(branch, data_dir) as overlay, \
            overlay.recall_session() as session:
        session.vector_anchors(_vec(1.0), k=5)

    assert sorted(asked) == [5, 6]


def test_basic_query_with_a_negative_limit_returns_every_visible_row(
        mm_runner):
    """Verify a negative limit, unlimited on SQLite, cuts no row on a branch.

    Mutation: the overlay adding the hidden count to a negative limit
        and slicing by it, which drops rows.
    Oracle: three parent rows, one forgotten in the branch, and one
        branch-only row: three visible rows.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    with open_sqlite_backend('default', data_dir) as parent:
        for n in range(1, 4):
            parent.nodes.insert_raw(_row(f'p{n}', f'parent row {n}', _vec(1.0)))
    with open_sqlite_backend(branch, data_dir) as backend:
        backend.nodes.insert_raw(_row(
            'p1', 'parent row 1', _vec(1.0), deleted_at=CREATED))
        backend.nodes.insert_raw(_row('b1', 'branch row', _vec(1.0)))

    with factory.open_backend(branch, data_dir) as overlay:
        ids = {ins.id for ins in overlay.nodes.query(keyword='row', limit=-1)}

    assert ids == {'p2', 'p3', 'b1'}


def test_enrich_on_a_branch_makes_no_llm_call_for_a_parent_row(
        mm_runner, monkeypatch):
    """Verify enrich on a branch never reaches a pending parent row.

    Mutation: the pending-enrich read taken from the parent, which bills
        an LLM call per parent row on every drain tick.
    Oracle: a stub LLM that records the text of each call.
    """
    _, data_dir = mm_runner
    with open_sqlite_backend('default', data_dir) as parent:
        parent.nodes.insert(Insight(
            id='pending-parent', content='The grackle colony nests early.'))
    branch = _branch(data_dir)
    calls = []

    def recording_complete(self, system, user, *, stage):
        calls.append(user)
        return '{"summary": "a summary"}'

    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete', recording_complete)

    with factory.open_backend(branch, data_dir) as overlay:
        enrich_pending(overlay)

    assert not any('grackle' in call for call in calls)


def test_recall_on_a_branch_refuses_a_parent_on_another_embed_model(
        mm_runner):
    """Verify recall refuses when the parent's fingerprint moved.

    Mutation: no fingerprint check in `OverlayRecallSession`, which
        compares parent cosines against a query embedded by another
        model.
    Oracle: the branch's swap command in the refusal.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    branch = _branch(data_dir)
    with open_sqlite_backend('default', data_dir) as parent:
        write_fingerprint(
            parent, Fingerprint(model='other/model', dim=EMBEDDING_DIM))

    result = invoke(mm_runner, ['recall', '--store', branch, 'backup'])

    assert result.exit_code != 0
    assert swap_command(branch) in result.output


@pytest.mark.no_auto_drain
def test_replace_after_the_parent_replaced_the_target_keeps_both_claims(
        mm_runner):
    """Verify a branch replace stays a plain add when the parent won.

    Mutation: the drain's chain-follow following a successor the parent
        wrote, which copies it into the branch and retires it.
    Oracle: both replacement texts, each written by hand.
    """
    _, data_dir = mm_runner
    target = json.loads(invoke(mm_runner, [
        'remember', 'The retry cap for batch jobs is three.']).output)['id']
    force_drain(data_dir)
    branch = _branch(data_dir)
    invoke(mm_runner, ['replace', target, 'The retry cap is five.'])
    invoke(mm_runner, [
        'replace', '--store', branch, target, 'The retry cap is seven.'])

    force_drain(data_dir)

    contents = _active_contents(data_dir, branch)
    assert 'The retry cap is five.' in contents
    assert 'The retry cap is seven.' in contents


def test_a_prefix_matching_rows_in_both_stores_raises(mm_runner):
    """Verify a prefix matching a branch row and a parent row is refused.

    Mutation: resolving in the branch first and taking its match.
    Oracle: two hand-set ids sharing the prefix `abc`.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    with open_sqlite_backend('default', data_dir) as parent:
        parent.nodes.insert_raw(_row('abc-parent', 'parent row', _vec(1.0)))
    with open_sqlite_backend(branch, data_dir) as backend:
        backend.nodes.insert_raw(_row('abc-branch', 'branch row', _vec(1.0)))

    with factory.open_backend(branch, data_dir) as overlay, \
            pytest.raises(ValueError, match='matches 2 rows'):
        overlay.nodes.resolve_id('abc')


def test_count_active_on_a_branch_counts_what_recall_sees(mm_runner):
    """Verify the branch count follows forgets in the branch and parent.

    Mutation: both stores summed with no hidden-id subtraction, a
        hidden id subtracted again after the parent retires it too, or
        the branch's own current rows left out.
    Oracle: the parent's own count_active, and hand counts.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The backup job runs at midnight.')
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)

    def counts():
        with factory.open_backend(branch, data_dir) as overlay, \
                open_sqlite_backend('default', data_dir) as parent:
            return overlay.nodes.count_active(), parent.nodes.count_active()

    fresh = counts()
    _remember(mm_runner, 'The grackle colony nests by the river.', branch)
    assert invoke(mm_runner, ['forget', '--store', branch, target]).exit_code == 0
    after_branch_forget = counts()
    assert invoke(mm_runner, ['forget', target]).exit_code == 0
    after_parent_forget = counts()

    assert fresh == (2, 2)
    assert after_branch_forget == (2, 2)
    assert after_parent_forget == (2, 1)


def test_prime_on_a_fresh_branch_prints_the_parent_count(
        mm_runner, monkeypatch):
    """Verify prime on a fresh branch reports the rows recall would see.

    Mutation: prime reading the branch's own SQLite stats, which print
        0 insights and tell the agent the store is empty.
    Oracle: the two rows written to the parent.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.setenv('MEMMAN_STORE', branch)

    result = invoke(mm_runner, ['prime'])

    assert '[memman] Memory active (2 insights).' in result.output


def test_prime_during_a_parent_outage_prints_no_count(
        mm_runner, monkeypatch):
    """Verify prime prints no count when a branch cannot read its parent.

    Mutation: the status line set from the branch's own SQLite stats
        before the overlay read raises, which prints 0 insights.
    Oracle: the count-free status line prime prints on any failure.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    branch = _branch(data_dir)
    shutil.rmtree(store_dir(data_dir, 'default'))
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.setenv('MEMMAN_STORE', branch)

    result = invoke(mm_runner, ['prime'])

    assert result.output.splitlines()[0] == '[memman] Memory active.'


def test_forget_on_a_branch_refuses_a_parent_correction(mm_runner):
    """Verify forget on a branch keeps a parent row that replaced another.

    Mutation: `predecessors` read from the branch only, so the branch
        forgets a correction and the topic leaves recall.
    Oracle: the single-store refusal text for the same case.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The backup job runs at midnight.')
    successor = _replace(
        mm_runner, stale, 'The backup job runs at two in the morning.')
    branch = _branch(data_dir)

    result = invoke(mm_runner, ['forget', '--store', branch, successor])

    assert result.exit_code != 0
    assert 'replaced an earlier row' in result.output


def test_forget_on_a_branch_of_a_row_the_parent_replaced_names_the_head(
        mm_runner):
    """Verify forget of a row the parent already replaced names its head.

    Mutation: the overlay answering "not found" for a retired parent row.
    Oracle: the successor id the parent's replace printed.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The backup job runs at midnight.')
    successor = _replace(
        mm_runner, stale, 'The backup job runs at two in the morning.')
    branch = _branch(data_dir)

    result = invoke(mm_runner, ['forget', '--store', branch, stale])

    assert result.exit_code != 0
    assert 'already retired in default' in result.output
    assert f'forget {successor}' in result.output


def test_replace_on_a_branch_of_a_row_the_parent_replaced_names_the_head(
        mm_runner):
    """Verify replace of a row the parent already replaced names its head.

    Mutation: the single-store refusal, which names the next successor
        and never says the row is retired in the parent.
    Oracle: the head id the parent's second replace printed.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The backup job runs at midnight.')
    middle = _replace(
        mm_runner, stale, 'The backup job runs at one in the morning.')
    head = _replace(
        mm_runner, middle, 'The backup job runs at two in the morning.')
    branch = _branch(data_dir)

    result = invoke(mm_runner, [
        'replace', '--store', branch, stale, 'The backup job runs at three.'])

    assert result.exit_code != 0
    assert 'already retired in default' in result.output
    assert f'replace {head}' in result.output


def test_branch_starts_empty_and_names_its_parent(mm_runner):
    """Verify a new branch holds no row and carries the branch keys.

    Mutation: rows copied at creation, the fingerprint taken from the
        env model, a token missing on either side, a wrong name shape,
        or the backend or rerank env key left unwritten.
    Oracle: a parent fingerprint whose model differs from the env model,
        a parent rerank key set by hand, and the name pattern.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    parent_fp = Fingerprint(model='other/model', dim=EMBEDDING_DIM)
    with open_sqlite_backend('default', data_dir) as parent:
        write_fingerprint(parent, parent_fp)
    _set_env_file_value(config.RERANK_ENABLED_FOR('default'), 'false')

    reply = branch_mod.create_branch(data_dir, 'default', 'rearch')

    branch = reply['store']
    with open_sqlite_backend(branch, data_dir) as backend:
        ids = backend.nodes.get_all_ids()
        meta = {
            key: backend.meta.get(key)
            for key in backend.meta.keys()  # noqa: SIM118
            }
    with open_sqlite_backend('default', data_dir) as parent:
        parent_token = parent.meta.get(f'branch_token:{branch}')
    env = config.parse_env_file(config.env_file_path(data_dir))
    assert env[config.BACKEND_FOR(branch)] == 'sqlite'
    assert env[config.RERANK_ENABLED_FOR(branch)] == 'false'
    assert branch.startswith('default__rearch_')
    assert len(branch) == 20
    assert ids == set()
    assert meta['branch_parent'] == 'default'
    assert meta['branch_created_at'] == reply['created_at']
    assert Fingerprint.from_json(meta['embed_fingerprint']) == parent_fp
    assert meta['branch_token'] == parent_token
    assert len(parent_token) == 32
    assert reply['instruction'] == (
        f'Use memman store {branch} for this thread: pass --store {branch}'
        ' to every memman recall, remember, replace, forget and insights'
        ' show call.')


def test_branch_creation_changes_only_the_parent_token(cross_backend_runner):
    """Verify creating a branch writes nothing to the parent but its token.

    Mutation: creation writing a parent row, an oplog row, or a meta key
        other than `branch_token:<branch>`, such as a branch key set on
        the parent's handle.
    Oracle: the parent's rows, meta and oplog read before the call.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = os.environ.get('MEMMAN_STORE', 'default')
    _remember(runner, 'The backup job runs at midnight.')
    rows_before, meta_before, oplog_before = _snapshot(data_dir, parent)

    branch = _branch(data_dir, parent)

    rows_after, meta_after, oplog_after = _snapshot(data_dir, parent)
    token_key = f'branch_token:{branch}'
    assert (rows_after, oplog_after) == (rows_before, oplog_before)
    assert token_key in meta_after
    assert {
        key: value for key, value in meta_after.items() if key != token_key
        } == meta_before


@pytest.mark.parametrize(('case', 'refusal'), [
    ('missing parent', 'store "ghost" does not exist'),
    ('branch parent', 'is a branch of'),
    ('double underscore label', 'invalid branch label'),
    ('no fingerprint', 'has no embed_fingerprint meta key'),
    ('swap in progress', 'embed swap in progress'),
    ('reembed in progress', 're-embed in progress'),
    ('corrupt fingerprint', 'corrupt embed_fingerprint'),
    ])
def test_branch_refuses_and_creates_nothing(mm_runner, case, refusal):
    """Verify each creation refusal names its cause and creates no store.

    Mutation: one refusal missing, so creation branches a store it must
        not, or stops on a later error.
    Oracle: each refusal's own text, and the store directories, env
        file and parent meta before and after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    parent, label = 'default', 'rearch'
    if case == 'missing parent':
        parent = 'ghost'
    elif case == 'branch parent':
        parent = _branch(data_dir)
    elif case == 'double underscore label':
        label = 're__arch'
    else:
        with open_sqlite_backend('default', data_dir) as backend:
            if case == 'no fingerprint':
                backend.meta.delete('embed_fingerprint')
            elif case == 'swap in progress':
                backend.meta.set('embed_swap_state', 'backfilling')
            elif case == 'corrupt fingerprint':
                backend.meta.set('embed_fingerprint', '{not json')
            else:
                backend.meta.set('embed_reembed_state', 'in_progress')
    stores_before = set(list_local_store_dirs(data_dir))
    env_path = config.env_file_path(data_dir)
    env_before = env_path.read_text() if env_path.exists() else ''
    meta_before = _snapshot(data_dir, 'default')[1]

    with pytest.raises(click.ClickException, match=refusal):
        branch_mod.create_branch(data_dir, parent, label)

    assert set(list_local_store_dirs(data_dir)) == stores_before
    assert (env_path.read_text() if env_path.exists() else '') == env_before
    assert _snapshot(data_dir, 'default')[1] == meta_before


def test_failed_branch_leaves_no_directory_and_no_env_keys(
        mm_runner, monkeypatch):
    """Verify a failure after the first write removes what it wrote.

    Mutation: the cleanup arm dropped, so a half-made branch directory
        and its backend key outlive the failure, or the parent token
        written before the branch steps.
    Oracle: the store directories, env keys and parent meta before the
        call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    stores_before = set(list_local_store_dirs(data_dir))
    env_path = config.env_file_path(data_dir)
    env_before = config.parse_env_file(env_path) if env_path.exists() else {}

    def failing_open(*args, **kwargs):
        raise RuntimeError('disk full')

    monkeypatch.setattr(branch_mod, 'open_sqlite_backend', failing_open)

    meta_before = _snapshot(data_dir, 'default')[1]

    with pytest.raises(RuntimeError, match='disk full'):
        branch_mod.create_branch(data_dir, 'default', 'rearch')

    assert set(list_local_store_dirs(data_dir)) == stores_before
    assert config.parse_env_file(env_path) == env_before
    assert _snapshot(data_dir, 'default')[1] == meta_before


def _merge(runner, branch):
    """Run store merge and return its parsed JSON output.
    """
    result = invoke(runner, ['store', 'merge', branch])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _drop(runner, branch):
    """Run store drop and return its parsed JSON output.
    """
    result = invoke(runner, ['store', 'drop', branch])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _set_meta(data_dir, store, key, value):
    """Write one meta key into a SQLite store, or delete it for None.
    """
    with open_sqlite_backend(store, data_dir) as backend:
        if value is None:
            backend.meta.delete(key)
        else:
            backend.meta.set(key, value)


def _retire(data_dir, store, row_id, *, successor=None):
    """Replace row_id by successor, or forget it when successor is None.

    A branch copies a parent row on the way, as the CLI verbs do.
    """
    with factory.open_backend(store, data_dir) as backend:
        if successor is None:
            assert backend.nodes.soft_delete(row_id)
        else:
            assert backend.nodes.mark_replaced(row_id, successor)


def test_merge_fault_after_the_inserts_leaves_the_parent_unchanged(
        cross_backend_runner, monkeypatch):
    """Verify merge writes the parent in one transaction.

    Mutation: the inserts committed apart from the retirements and the
        token removal, as `Migrator.apply` on its own connection does,
        a database error after the inserts reported with no re-run
        hint, or the branch removed or its merge flag cleared after the
        failure.
    Oracle: the parent's rows, meta and oplog before the merge, a stub
        retirement raising sqlite3's lock error, and the branch read
        after the call.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    parent = os.environ.get('MEMMAN_STORE', 'default')
    target = _remember(runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir, parent)
    _remember(runner, 'The grackle colony nests by the river.', branch)
    _retire(data_dir, branch, target)
    before = _snapshot(data_dir, parent)

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(branch_mod, '_retire_copied', locked)

    result = invoke(runner, ['store', 'merge', branch])

    assert isinstance(result.exception, SystemExit), result.exception
    assert f'memman store merge {branch}' in result.output
    assert _snapshot(data_dir, parent) == before
    assert 'branch_merging' in _snapshot(data_dir, branch)[1]


@pytest.mark.no_auto_drain
def test_merge_failing_before_the_parent_write_clears_its_flag(
        mm_runner, monkeypatch):
    """Verify a parent read failing in merge's checks leaves the branch writable.

    Mutation: only the refusal path clears `branch_merging`, so a parent
        outage during the checks freezes every branch write.
    Oracle: the branch meta after the call, with a stub that raises on
        the parent's read-only open behind a pending parent replace.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    target = json.loads(invoke(mm_runner, [
        'remember', '--store', 'work',
        'The retry cap for batch jobs is three.',
        ]).output)['id']
    force_drain(data_dir)
    branch = _branch(data_dir, parent='work')
    invoke(mm_runner, [
        'replace', '--store', 'work', target,
        'The retry cap for batch jobs is five.'])
    real_open = factory.open_backend

    def parent_down(store, data_dir_arg, *args, read_only=False, **kwargs):
        if store == 'work' and read_only:
            raise StoreMissingError('work')
        return real_open(store, data_dir_arg, *args, read_only=read_only,
                         **kwargs)

    monkeypatch.setattr(factory, 'open_backend', parent_down)

    result = invoke(mm_runner, ['store', 'merge', branch])

    assert result.exit_code != 0
    assert 'branch_merging' not in _snapshot(data_dir, branch)[1]


def test_merge_failing_after_the_parent_commit_claims_no_empty_parent(
        mm_runner, monkeypatch):
    """Verify a failure after the parent commit never says nothing was written.

    Mutation: the error for any failure in the parent step says merge
        wrote nothing, though the commit already holds the copied rows.
    Oracle: the branch row in the parent after the call, with a stub
        whose close raises once the parent token is gone.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    own = _remember(mm_runner, 'The grackle colony nests by the river.', branch)
    real_open = factory.open_backend

    def lost_ack(store, data_dir_arg, *args, **kwargs):
        backend = real_open(store, data_dir_arg, *args, **kwargs)
        real_close = backend.close

        def close():
            committed = (
                store == 'default'
                and backend.meta.get(f'branch_token:{branch}') is None)
            real_close()
            if committed:
                raise OSError('connection reset')

        backend.close = close
        return backend

    with monkeypatch.context() as patch:
        patch.setattr(factory, 'open_backend', lost_ack)
        result = invoke(mm_runner, ['store', 'merge', branch])

    assert result.exit_code != 0
    assert f'memman store merge {branch}' in result.output
    assert 'wrote nothing' not in result.output
    assert own in _snapshot(data_dir, 'default')[0]


@pytest.mark.parametrize('own_row', [False, True])
def test_forget_on_a_merging_branch_refuses(mm_runner, own_row):
    """Verify the overlay refuses a forget once a merge set its flag.

    Mutation: no `branch_merging` check on the overlay's retire path,
        so a forget between merge's read and its removal is lost.
    Oracle: the branch's rows, meta and oplog before the call, for a
        parent row and for a row the branch wrote.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    own = _remember(mm_runner, 'The grackle colony nests by the river.', branch)
    _set_meta(data_dir, branch, 'branch_merging', '2026-10-06T00:00:00Z')
    before = _snapshot(data_dir, branch)

    result = invoke(mm_runner, [
        'forget', '--store', branch, own if own_row else target])

    assert result.exit_code != 0
    assert f'memman store merge {branch}' in result.output
    assert _snapshot(data_dir, branch) == before


@pytest.mark.parametrize('token', [None, 'f' * 32])
def test_merge_refuses_a_parent_whose_token_is_missing_or_differs(
        mm_runner, token):
    """Verify merge refuses a parent that does not hold the branch's token.

    Mutation: no token check, so merge replays into a parent that was
        removed and recreated, or restored from an older backup.
    Oracle: both stores' rows, meta and oplog before the call.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    _remember(mm_runner, 'The grackle colony nests by the river.', branch)
    _retire(data_dir, branch, target)
    _set_meta(data_dir, 'default', f'branch_token:{branch}', token)
    parent_before = _snapshot(data_dir, 'default')
    branch_before = _snapshot(data_dir, branch)

    result = invoke(mm_runner, ['store', 'merge', branch])

    assert result.exit_code != 0
    assert 'token' in result.output
    assert _snapshot(data_dir, 'default') == parent_before
    assert _snapshot(data_dir, branch) == branch_before


def test_merge_conflict_names_the_head_of_the_parent_chain(mm_runner):
    """Verify a conflict entry names the parent's current head and texts.

    Mutation: the next successor reported in place of the head, which
        names a retired row the agent cannot act on.
    Oracle: the ids and texts each replace printed or was given.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The backup job runs at midnight.')
    branch = _branch(data_dir)
    mine = _replace(mm_runner, stale, 'The backup job runs at three.', branch)
    middle = _replace(mm_runner, stale, 'The backup job runs at one.')
    head = _replace(mm_runner, middle, 'The backup job runs at two.')

    reply = _merge(mm_runner, branch)

    assert reply['conflicts'] == [{
        'id': stale,
        'branch_state': 'replaced',
        'parent_state': 'replaced',
        'branch_successor': mine,
        'parent_head': head,
        'branch_content': 'The backup job runs at three.',
        'parent_content': 'The backup job runs at two.',
        }]


def test_a_parent_embed_swap_blocks_only_recall_and_merge(
        mm_runner, monkeypatch):
    """Verify a branch over a swapped parent still opens for every other verb.

    Mutation: the fingerprint check run at overlay open, which blocks
        the branch's own swap, the drain, doctor and drop.
    Oracle: a parent fingerprint moved by hand to the swap target, and
        each verb's exit status.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The backup job runs at midnight.')
    branch = _branch(data_dir)
    target = _FakeTargetProvider()
    target.prepare()
    monkeypatch.setitem(registry._GET_FOR_CACHE, target.model, target)
    with open_sqlite_backend('default', data_dir) as parent:
        write_fingerprint(parent, Fingerprint(model=target.model, dim=384))

    recall = invoke(mm_runner, ['recall', '--store', branch, 'backup'])
    merge = invoke(mm_runner, ['store', 'merge', branch])
    own = _remember(mm_runner, 'The grackle colony nests by the river.', branch)
    doctor = json.loads(
        invoke(mm_runner, ['--store', branch, 'doctor']).output)
    with monkeypatch.context() as patch:
        patch.setattr(
            sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)
        swap = invoke(mm_runner, [
            '--store', branch, 'embed', 'swap', '--to', target.model])
    recall_after_swap = invoke(mm_runner, [
        'recall', '--store', branch, 'backup'])
    drop = _drop(mm_runner, branch)

    branches = next(c for c in doctor['checks'] if c['name'] == 'branches')
    assert recall.exit_code != 0
    assert merge.exit_code != 0
    assert swap_command(branch) in merge.output
    assert [b['store'] for b in branches['detail']['branches']] == [branch]
    assert swap.exit_code == 0, swap.output
    assert recall_after_swap.exit_code == 0, recall_after_swap.output
    assert [row['id'] for row in drop['dropped']] == [own]


def test_merge_rerun_after_the_parent_commit_finishes_the_removal(
        mm_runner, monkeypatch):
    """Verify a crash after the parent commit leaves a merge a re-run ends.

    Mutation: the parent token left in place after the parent commit,
        or a re-run that refuses the missing token in place of resuming.
    Oracle: the parent's meta after the crash, and the parent's rows,
        meta and oplog, unchanged by the re-run.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    successor = _replace(
        mm_runner, target, 'The retry cap for batch jobs is seven.', branch)

    def crash(*args, **kwargs):
        raise RuntimeError('killed before the removal')

    with monkeypatch.context() as patch:
        patch.setattr(branch_mod, '_remove_branch', crash)
        assert invoke(mm_runner, ['store', 'merge', branch]).exit_code != 0
    after_crash = _snapshot(data_dir, 'default')

    reply = _merge(mm_runner, branch)

    assert f'branch_token:{branch}' not in after_crash[1]
    assert after_crash[0][target].replaced_by == successor
    assert (reply['copied'], reply['retired'], reply['conflicts']) == (
        0, 0, [])
    assert _snapshot(data_dir, 'default') == after_crash
    assert not Path(store_dir(data_dir, branch)).exists()


def test_merge_rerun_reports_the_conflicts_of_the_crashed_run(
        mm_runner, monkeypatch):
    """Verify a resumed merge reports the conflicts and writes nothing.

    Mutation: a resume that replies with no conflicts, so the agent
        never rules on a conflict the crashed run found, or a resume
        that writes the parent, which forgets a row restored after the
        crash.
    Oracle: the conflict entries of a merge that did not crash, and the
        parent's rows, meta and oplog after the crash and a hand restore
        of a row the merge forgot.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The backup job runs at midnight.')
    restored = _remember(mm_runner, 'The grackle colony nests by the river.')
    branch = _branch(data_dir)
    mine = _replace(mm_runner, stale, 'The backup job runs at three.', branch)
    head = _replace(mm_runner, stale, 'The backup job runs at two.')
    _retire(data_dir, branch, restored)

    def crash(*args, **kwargs):
        raise RuntimeError('killed before the removal')

    with monkeypatch.context() as patch:
        patch.setattr(branch_mod, '_remove_branch', crash)
        assert invoke(mm_runner, ['store', 'merge', branch]).exit_code != 0
    with open_sqlite_backend('default', data_dir) as parent, \
            parent.transaction():
        parent._db._conn.execute(
            'update insights set deleted_at = null where id = ?', (restored,))
    after_restore = _snapshot(data_dir, 'default')

    reply = _merge(mm_runner, branch)

    assert sorted(reply['conflicts'], key=lambda entry: entry['id']) == sorted([
        {
            'id': stale,
            'branch_state': 'replaced',
            'parent_state': 'replaced',
            'branch_successor': mine,
            'parent_head': head,
            'branch_content': 'The backup job runs at three.',
            'parent_content': 'The backup job runs at two.',
            },
        {
            'id': restored,
            'branch_state': 'deleted',
            'parent_state': 'current',
            'branch_successor': None,
            'parent_head': restored,
            'branch_content': None,
            'parent_content': 'The grackle colony nests by the river.',
            },
        ], key=lambda entry: entry['id'])
    assert _snapshot(data_dir, 'default') == after_restore


@pytest.mark.no_auto_drain
def test_remember_on_a_merging_branch_refuses(mm_runner):
    """Verify enqueue refuses a branch whose merge set its flag.

    Mutation: enqueue checking only that the store exists, so a write
        queues for a branch merge is about to remove.
    Oracle: the queue, which holds no row for the write.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    _set_meta(data_dir, branch, 'branch_merging', '2026-10-06T00:00:00Z')

    result = invoke(mm_runner, [
        'remember', '--store', branch, 'The worker splits in two.'])

    assert result.exit_code != 0
    assert 'The worker splits in two.' not in queued_contents(data_dir)


@pytest.mark.no_auto_drain
def test_merge_refuses_a_write_queued_just_before_its_flag(
        mm_runner, monkeypatch):
    """Verify merge re-checks the queue after it sets its flag.

    Mutation: the queue checked only before the flag, so merge writes
        the parent and then strands the late write in a merging branch.
    Oracle: the parent's rows before the call, and a flag stub that
        queues a write as the flag is set.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    invoke(mm_runner, [
        'remember', '--store', branch, 'The grackle colony nests early.'])
    force_drain(data_dir)
    parent_before = _snapshot(data_dir, 'default')
    real_set = branch_mod._set_merging

    def queue_then_flag(data_dir_arg, store, value):
        if value is not None:
            invoke(mm_runner, [
                'remember', '--store', branch, 'The worker splits in two.'])
        return real_set(data_dir_arg, store, value)

    monkeypatch.setattr(branch_mod, '_set_merging', queue_then_flag)

    result = invoke(mm_runner, ['store', 'merge', branch])
    force_drain(data_dir)

    assert result.exit_code != 0
    assert 'queued writes' in result.output
    assert _snapshot(data_dir, 'default') == parent_before
    assert 'The worker splits in two.' in _active_contents(data_dir, branch)


def test_merge_replays_a_retirement_after_the_branch_oplog_is_emptied(
        mm_runner):
    """Verify merge finds a copied row by id, never through the oplog.

    Mutation: copy classification read from the branch oplog, which
        the drain trims by age and count.
    Oracle: the parent row's state after the merge.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    assert invoke(mm_runner, ['forget', '--store', branch, target]).exit_code == 0
    with open_sqlite_backend(branch, data_dir) as backend, \
            backend.transaction():
        backend._db._conn.execute('delete from oplog')

    reply = _merge(mm_runner, branch)

    with open_sqlite_backend('default', data_dir) as parent:
        row = parent.nodes.get_include_deleted(target)
    assert reply['retired'] == 1
    assert row.deleted_at is not None


def test_drop_lists_each_branch_row_with_the_parent_row_it_replaced(
        mm_runner):
    """Verify drop lists current branch-only rows, then removes the branch.

    Mutation: a copy or a retired row listed, `replaces` naming the
        direct predecessor in place of the parent row at the chain's
        root, or the directory, env keys or parent token left behind.
    Oracle: the ids each remember and replace printed.
    """
    _, data_dir = mm_runner
    stale = _remember(mm_runner, 'The queue runs on SQLite.')
    branch = _branch(data_dir)
    kept = _remember(mm_runner, 'The worker splits in two.', branch)
    first = _replace(mm_runner, stale, 'The queue moves to Redis.', branch)
    second = _replace(mm_runner, first, 'The queue moves to Kafka.', branch)
    own = _remember(mm_runner, 'The cache sits on disk.', branch)
    own_next = _replace(mm_runner, own, 'The cache sits in memory.', branch)

    reply = _drop(mm_runner, branch)

    listed = {row['id']: row['replaces'] for row in reply['dropped']}
    assert listed == {kept: None, second: stale, own_next: None}
    assert reply['parent'] == 'default'
    assert not Path(store_dir(data_dir, branch)).exists()
    assert not [key for key in _env_keys(data_dir) if key.endswith(branch)]
    assert f'branch_token:{branch}' not in _snapshot(data_dir, 'default')[1]


_BRANCH_VERBS = [('store', 'branch'), ('store', 'drop'), ('store', 'merge')]


def test_branch_verbs_are_agent_callable_on_both_hosts(tmp_path):
    """Verify branch, merge and drop reach the Claude and Codex allow lists.

    Mutation: a branch verb missing @claude_callable, so one host
        prompts on every call.
    Oracle: the three verb paths, read in the Claude permission strings
        and in the Codex rules file install_codex writes.
    """
    codex_env = {
        'skills_dir': str(tmp_path / 'skills'),
        'config_dir': str(tmp_path / 'codex'),
        }

    install_codex(codex_env, no_wizard=True)

    rules = (tmp_path / 'codex' / 'rules' / 'memman.rules').read_text()
    permissions = list_claude_permissions()
    for path in _BRANCH_VERBS:
        assert path in list_agent_commands()
        assert f'Bash(memman {" ".join(path)}:*)' in permissions
        assert json.dumps(['memman', *path]) in rules


@pytest.mark.parametrize('verb', ['branch', 'merge', 'drop'])
def test_branch_verbs_take_no_per_verb_store(mm_runner, verb):
    """Verify the branch verbs name their stores as arguments only.

    Mutation: store_option left at its default on a branch verb, so a
        --store value could disagree with the named store.
    Oracle: click's no-such-option usage error.
    """
    result = invoke(mm_runner, ['store', verb, '--store', 'x', 'a', 'b'])

    assert result.exit_code == 2
    assert 'No such option' in result.output


def test_store_branch_prints_the_branch_and_its_instruction(mm_runner):
    """Verify the CLI verb creates a branch and replies with its fields.

    Mutation: `store branch` wired to another function, or its reply
        missing the instruction line a session pastes.
    Oracle: the branch meta read back from the store the reply names.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['store', 'branch', 'default', 'rearch'])

    reply = json.loads(result.output)
    with open_sqlite_backend(reply['store'], data_dir) as backend:
        parent = backend.meta.get('branch_parent')
    assert (reply['action'], parent) == ('branched', 'default')
    assert f'--store {reply["store"]}' in reply['instruction']


def test_branch_whose_directory_cannot_be_made_writes_no_env_keys(
        mm_runner, monkeypatch):
    """Verify a failed branch directory create leaves no env keys.

    Mutation: the env keys written before the directory, outside the
        cleanup, which leaves backend keys for a store that never
        existed.
    Oracle: the env file read after the call, and a refusal in place of
        a traceback.
    """
    _, data_dir = mm_runner

    class UnwritablePath(type(Path())):
        def mkdir(self, *args, **kwargs):
            raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(branch_mod, 'Path', UnwritablePath)

    result = invoke(mm_runner, ['store', 'branch', 'default', 'rearch'])

    assert isinstance(result.exception, SystemExit), result.exception
    assert not [k for k in _env_keys(data_dir) if '__rearch' in k]


def test_merge_copies_a_branch_row_whole_without_api_calls(
        mm_runner, monkeypatch):
    """Verify merge copies a branch-only row with its stored columns.

    Mutation: the copy sent through NodeStore.insert or the queue, which
        stamps a new created_at and re-embeds or re-enriches the row.
    Oracle: the branch's own row read before the merge, and API stubs
        that record every call.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    new_id = _remember(
        mm_runner, 'The rearchitecture splits the worker in two.', branch)
    with open_sqlite_backend(branch, data_dir) as backend:
        branch_row = backend.nodes.get_raw(new_id)
    calls = []
    monkeypatch.setattr(
        'memman.embed.client.Client.embed',
        lambda *a, **kw: calls.append('embed'))
    monkeypatch.setattr(
        'memman.embed.client.Client.embed_batch',
        lambda *a, **kw: calls.append('embed_batch'))
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        lambda *a, **kw: calls.append('complete'))

    reply = _merge(mm_runner, branch)

    with open_sqlite_backend('default', data_dir) as parent:
        parent_row = parent.nodes.get_raw(new_id)
    assert reply['copied'] == 1
    assert parent_row == branch_row
    assert calls == []


@pytest.mark.postgres
def test_merge_after_a_width_changing_swap_reaches_a_postgres_parent(
        mm_runner, env_file, pg_dsn, monkeypatch):
    """Verify merge copies a retired branch row the swap left at the old width.

    Mutation: merge copies a retired row's vector whatever its width, so
        the Postgres parent rejects the old-width vector, the merge
        fails on every re-run and the branch stays frozen.
    Oracle: a branch row replaced before both stores swap to a model of
        another width, and the parent's rows read after the merge.
    """
    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_mergepg', 'postgres')
    env_file('MEMMAN_POSTGRES_DSN_mergepg', pg_dsn)
    created = invoke(mm_runner, ['store', 'create', 'mergepg'])
    assert created.exit_code == 0, created.output
    try:
        branch = _branch(data_dir, parent='mergepg')
        old = _remember(
            mm_runner, 'The retry cap for batch jobs is three.', branch)
        new = _replace(
            mm_runner, old, 'The retry cap for batch jobs is seven.', branch)
        target = _FakeTargetProvider()
        target.prepare()
        monkeypatch.setitem(registry._GET_FOR_CACHE, target.model, target)
        monkeypatch.setattr(
            sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)
        for store in ('mergepg', branch):
            swap = invoke(mm_runner, [
                '--store', store, 'embed', 'swap', '--to', target.model])
            assert swap.exit_code == 0, swap.output

        result = invoke(mm_runner, ['store', 'merge', branch])

        assert result.exit_code == 0, result.output
        with factory.open_backend('mergepg', data_dir) as parent:
            assert parent.nodes.get_raw(old).replaced_by == new
            assert len(parent.nodes.get_raw(new).embedding) == target.dim
    finally:
        factory.drop_store('mergepg', data_dir)


@pytest.mark.parametrize(('branch_state', 'parent_state', 'expected'), [
    ('replaced', 'current', 'retired'),
    ('replaced', 'replaced by successor', 'done'),
    ('replaced', 'replaced by other', 'conflict'),
    ('replaced', 'deleted', 'conflict'),
    ('deleted', 'current', 'retired'),
    ('deleted', 'current with live predecessor', 'conflict'),
    ('deleted', 'deleted', 'done'),
    ('deleted', 'replaced by other', 'conflict'),
    ('replaced and deleted', 'replaced by successor', 'done'),
    ])
def test_merge_applies_the_retirement_table(
        mm_runner, branch_state, parent_state, expected):
    """Verify merge gives each branch and parent state pair its table result.

    Mutation: a table row swapped, the live-predecessor guard skipped,
        a branch row both replaced and deleted read as deleted, which
        turns a done row into a false conflict, or an oplog row written
        for a done or conflict row.
    Oracle: the retirement table of the spec, one row per case.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    if parent_state == 'current with live predecessor':
        target = _replace(
            mm_runner, target, 'The retry cap for batch jobs is five.')
    branch = _branch(data_dir)
    successor = _remember(
        mm_runner, 'The retry cap for batch jobs is seven.', branch)
    if branch_state == 'deleted':
        _retire(data_dir, branch, target)
    else:
        _retire(data_dir, branch, target, successor=successor)
    if branch_state == 'replaced and deleted':
        _retire(data_dir, branch, target)
    if parent_state == 'replaced by successor':
        _retire(data_dir, 'default', target, successor=successor)
    elif parent_state == 'replaced by other':
        _retire(data_dir, 'default', target, successor='other-row')
    elif parent_state == 'deleted':
        _retire(data_dir, 'default', target)
    with open_sqlite_backend('default', data_dir) as parent:
        parent_before = parent.nodes.get_raw(target)
    oplog_before = _snapshot(data_dir, 'default')[2]

    reply = _merge(mm_runner, branch)

    with open_sqlite_backend('default', data_dir) as parent:
        parent_after = parent.nodes.get_raw(target)
    if expected == 'retired':
        assert reply['retired'] == 1
        assert reply['conflicts'] == []
        assert (parent_after.replaced_by, parent_after.deleted_at is None) == (
            (successor, True) if branch_state == 'replaced' else (None, False))
    else:
        assert reply['retired'] == 0
        assert len(reply['conflicts']) == (expected == 'conflict')
        assert parent_after == parent_before
        assert _snapshot(data_dir, 'default')[2] == oplog_before


def test_merge_writes_an_oplog_row_per_retirement(mm_runner):
    """Verify each retirement merge applies leaves a parent oplog row.

    Mutation: the retirement written without its oplog row, so the
        parent's history loses the change.
    Oracle: the parent oplog's newest row, read after the merge.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    branch = _branch(data_dir)
    _retire(data_dir, branch, target)

    _merge(mm_runner, branch)

    newest = _snapshot(data_dir, 'default')[2][0]
    assert (newest.operation, newest.insight_id, newest.detail) == (
        'forget', target, f'merged from {branch}')


def _queue_status(data_dir, store, status):
    """Set every queue row of store to status.
    """
    with queue_db(data_dir) as conn:
        conn.execute(
            'update queue set status = ? where store = ?', (status, store))


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('case', [
    'pending branch row', 'failed branch row', 'pending parent replace',
    'failed parent replace', 'failed parent replace up the chain',
    'fingerprint mismatch', 'parent swap', 'branch reembed', 'active branch',
    'recreated parent',
    ])
def test_merge_refuses_and_changes_nothing(mm_runner, case):
    """Verify each merge refusal leaves the branch and its parent as they were.

    Mutation: one refusal missing, a failed parent replace let past, or
        one whose target the parent has since replaced into a row the
        branch retired, either of which a retry later follows onto the
        branch's successor, or the target read from the active store or
        a recreated store of the parent's name.
    Oracle: both stores' rows, meta and oplog before and after the call.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    target = json.loads(invoke(mm_runner, [
        'remember', '--store', 'work',
        'The retry cap for batch jobs is three.',
        ]).output)['id']
    force_drain(data_dir)
    branch = _branch(data_dir, parent='work')
    if case in {'pending branch row', 'failed branch row'}:
        invoke(mm_runner, [
            'remember', '--store', branch, 'The worker splits in two.'])
        if case == 'failed branch row':
            _queue_status(data_dir, branch, 'failed')
    elif case in {'pending parent replace', 'failed parent replace'}:
        _retire(data_dir, branch, target)
        invoke(mm_runner, [
            'replace', '--store', 'work', target,
            'The retry cap for batch jobs is five.'])
        if case == 'failed parent replace':
            _queue_status(data_dir, 'work', 'failed')
    elif case == 'failed parent replace up the chain':
        invoke(mm_runner, [
            'replace', '--store', 'work', target,
            'The retry cap for batch jobs is five.'])
        _queue_status(data_dir, 'work', 'failed')
        invoke(mm_runner, [
            'replace', '--store', 'work', target,
            'The retry cap for batch jobs is six.'])
        force_drain(data_dir)
        with open_sqlite_backend('work', data_dir) as backend:
            successor = backend.nodes.get_include_deleted(target).replaced_by
        _retire(data_dir, branch, successor)
    elif case == 'fingerprint mismatch':
        with open_sqlite_backend(branch, data_dir) as backend:
            write_fingerprint(
                backend, Fingerprint(model='other/model', dim=EMBEDDING_DIM))
    elif case == 'parent swap':
        _set_meta(data_dir, 'work', 'embed_swap_state', 'backfill')
    elif case == 'branch reembed':
        _set_meta(data_dir, branch, 'embed_reembed_state', 'in_progress')
    elif case == 'active branch':
        write_active(data_dir, branch)
    else:
        work_fp = _snapshot(data_dir, 'work')[1]['embed_fingerprint']
        factory.drop_store('work', data_dir)
        invoke(mm_runner, ['store', 'create', 'work'])
        _set_meta(data_dir, 'work', 'embed_fingerprint', work_fp)
    branch_before = _snapshot(data_dir, branch)
    parent_before = _snapshot(data_dir, 'work')

    result = invoke(mm_runner, ['store', 'merge', branch])

    assert result.exit_code != 0
    assert _snapshot(data_dir, branch) == branch_before
    assert _snapshot(data_dir, 'work') == parent_before


@pytest.mark.parametrize('verb', ['merge', 'drop'])
def test_merge_and_drop_refuse_an_ordinary_store(mm_runner, verb):
    """Verify merge and drop act only on a store with branch_parent meta.

    Mutation: the branch_parent check skipped, so an agent-callable verb
        deletes an ordinary store or fails on a later lookup.
    Oracle: the refusal text, and the store's rows, meta and oplog
        before and after.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    before = _snapshot(data_dir, 'work')

    result = invoke(mm_runner, ['store', verb, 'work'])

    assert 'is not a branch' in result.output
    assert _snapshot(data_dir, 'work') == before


@pytest.mark.parametrize(('parent_case', 'refuses'), [
    ('committed', True), ('rolled back', False), ('recreated', False),
    ('removed', False), ('unreachable', True),
    ])
def test_drop_refuses_only_a_branch_whose_merge_wrote_the_parent(
        mm_runner, monkeypatch, env_file, parent_case, refuses):
    """Verify drop refuses a merging branch only when merge would resume.

    Mutation: drop refusing on the flag alone, or ignoring the parent
        token, so a merge that failed before its parent commit leaves a
        branch neither merge nor drop removes; drop deleting a branch
        the parent already holds, which lists merged rows as dropped;
        or drop going ahead while the parent cannot answer whether it
        holds the branch.
    Oracle: a merge crashed after its parent commit, a flag set by hand
        beside the parent token, a parent recreated empty or removed,
        and a parent DSN on a closed port. The branch holds only a copy
        of a parent row, so its ids alone never tell the cases apart.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    target = _remember(
        mm_runner, 'The retry cap for batch jobs is three.', 'work')
    branch = _branch(data_dir, parent='work')
    _retire(data_dir, branch, target)
    if parent_case == 'committed':
        def crash(*args, **kwargs):
            raise RuntimeError('killed before the removal')

        with monkeypatch.context() as patch:
            patch.setattr(branch_mod, '_remove_branch', crash)
            assert invoke(mm_runner, ['store', 'merge', branch]).exit_code
    else:
        _set_meta(data_dir, branch, 'branch_merging', '2026-10-05T00:00:00Z')
    if parent_case in {'recreated', 'removed'}:
        factory.drop_store('work', data_dir)
    if parent_case == 'recreated':
        invoke(mm_runner, ['store', 'create', 'work'])
    if parent_case == 'unreachable':
        pytest.importorskip('psycopg')
        env_file('MEMMAN_BACKEND_work', 'postgres')
        env_file(
            'MEMMAN_POSTGRES_DSN_work',
            'host=127.0.0.1 port=1 dbname=x user=x password=x')

    result = invoke(mm_runner, ['store', 'drop', branch])

    assert (result.exit_code != 0) == refuses, result.output
    assert Path(store_dir(data_dir, branch)).exists() == refuses
    if refuses:
        assert f'memman store merge {branch}' in result.output


def test_drop_rereads_the_branch_meta_under_the_drain_lock(
        mm_runner, monkeypatch):
    """Verify drop refuses a merge that started before it took the lock.

    Mutation: the branch_merging check run before the drain lock, so a
        merge that commits the parent in that gap leaves a merged branch
        drop deletes.
    Oracle: a lock stub that writes branch_merging and removes the
        parent token as it is taken.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    acquire = branch_mod._acquire_drain_lock

    def merge_in_the_gap(data_dir_arg):
        _set_meta(data_dir, branch, 'branch_merging', '2026-10-05T00:00:00Z')
        _set_meta(data_dir, 'default', f'branch_token:{branch}', None)
        return acquire(data_dir_arg)

    monkeypatch.setattr(branch_mod, '_acquire_drain_lock', merge_in_the_gap)

    result = invoke(mm_runner, ['store', 'drop', branch])

    assert 'stopped part way' in result.output
    assert Path(store_dir(data_dir, branch)).exists()


def test_drop_caps_the_parent_check_connect_time(mm_runner, monkeypatch):
    """Verify drop sets the short connect timeout before the parent check.

    Mutation: the default libpq timeout kept, so a parent host that
        drops packets holds the drain lock for minutes.
    Oracle: a stub store_exists recording PGCONNECT_TIMEOUT at its call.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    monkeypatch.setenv('PGCONNECT_TIMEOUT', 'unset')
    monkeypatch.delenv('PGCONNECT_TIMEOUT')
    seen = []
    real_exists = branch_mod.factory.store_exists

    def recording_exists(store, data_dir_arg):
        seen.append((store, os.environ.get('PGCONNECT_TIMEOUT')))
        return real_exists(store, data_dir_arg)

    monkeypatch.setattr(branch_mod.factory, 'store_exists', recording_exists)

    _drop(mm_runner, branch)

    assert ('default', '3') in seen


@pytest.mark.parametrize('parent_case', ['removed', 'unreachable'])
def test_drop_lists_the_branch_rows_when_the_parent_cannot_answer(
        mm_runner, env_file, parent_case):
    """Verify drop deletes a branch whose parent is gone or unreachable.

    Mutation: drop reading the parent to choose the rows to list, which
        refuses or lists nothing when the parent cannot answer.
    Oracle: the id of the one row written in the branch.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    _remember(mm_runner, 'The retry cap for batch jobs is three.', 'work')
    branch = _branch(data_dir, parent='work')
    own = _remember(mm_runner, 'The worker splits in two.', branch)
    if parent_case == 'removed':
        factory.drop_store('work', data_dir)
    else:
        pytest.importorskip('psycopg')
        env_file('MEMMAN_BACKEND_work', 'postgres')
        env_file(
            'MEMMAN_POSTGRES_DSN_work',
            'host=127.0.0.1 port=1 dbname=x user=x password=x')

    reply = _drop(mm_runner, branch)

    assert reply['dropped'] == [{
        'id': own, 'content': 'The worker splits in two.', 'replaces': None}]
    assert not Path(store_dir(data_dir, branch)).exists()


@pytest.mark.parametrize('failing_open', ['read', 'token removal'])
def test_drop_lists_the_branch_rows_when_a_parent_call_raises(
        mm_runner, monkeypatch, failing_open):
    """Verify a parent database error never stops drop or loses its list.

    Mutation: a raw sqlite3 error from the parent read aborting drop, or
        one from the token removal after the branch is gone, which exits
        1 with the branch deleted and its rows never listed.
    Oracle: the id of the one row written in the branch, and a stub
        parent open that raises sqlite3's lock error on one of drop's
        two parent opens.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    own = _remember(mm_runner, 'The worker splits in two.', branch)
    real_open = branch_mod.factory.open_backend

    def locked_open(store, data_dir_arg, *args, read_only=False, **kwargs):
        if store == 'default' and read_only == (failing_open == 'read'):
            raise sqlite3.OperationalError('database is locked')
        return real_open(
            store, data_dir_arg, *args, read_only=read_only, **kwargs)

    monkeypatch.setattr(branch_mod.factory, 'open_backend', locked_open)

    reply = _drop(mm_runner, branch)

    assert [row['id'] for row in reply['dropped']] == [own]
    assert not Path(store_dir(data_dir, branch)).exists()


@pytest.mark.no_auto_drain
def test_a_write_queued_during_drop_makes_it_refuse(mm_runner, monkeypatch):
    """Verify drop re-checks the queue inside its removal transaction.

    Mutation: the queue checked only before the removal transaction, so
        a write queued in between is purged with the branch, or the
        parent token removed before that check, which leaves a kept
        branch that no merge accepts.
    Oracle: a remember run just before the removal, and the queue,
        branch directory and parent meta read after the call.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    real_remove = branch_mod._remove_branch

    def remember_then_remove(conn, store, data_dir_arg):
        invoke(mm_runner, [
            'remember', '--store', branch, 'The worker splits in two.'])
        return real_remove(conn, store, data_dir_arg)

    monkeypatch.setattr(branch_mod, '_remove_branch', remember_then_remove)

    result = invoke(mm_runner, ['store', 'drop', branch])

    assert result.exit_code != 0
    assert Path(store_dir(data_dir, branch)).exists()
    assert queued_contents(data_dir)[-1] == 'The worker splits in two.'
    assert f'branch_token:{branch}' in _snapshot(data_dir, 'default')[1]


@pytest.mark.no_auto_drain
def test_a_write_after_drop_meets_the_missing_store_refusal(mm_runner):
    """Verify a stale instruction line fails once the branch is dropped.

    Mutation: remember queuing for a store whose directory is gone,
        which the drain would recreate or fail on later.
    Oracle: the queue, which holds no row for the branch.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    _drop(mm_runner, branch)

    result = invoke(mm_runner, [
        'remember', '--store', branch, 'The worker splits in two.'])

    assert result.exit_code != 0
    assert 'The worker splits in two.' not in queued_contents(data_dir)


def test_store_use_refuses_a_branch(mm_runner):
    """Verify store use leaves the host-wide active file off a branch.

    Mutation: a branch written to the active file, which routes every
        session on the host into the thread.
    Oracle: the active file, read after the call.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)

    result = invoke(mm_runner, ['store', 'use', branch])

    assert result.exit_code != 0
    assert '--store' in result.output
    assert read_active(data_dir) == 'default'


@pytest.mark.parametrize('migrate_all', [False, True])
def test_migrate_skips_a_branch(mm_runner, monkeypatch, migrate_all):
    """Verify migrate skips a branch alone or under --all.

    Mutation: the branch_parent check missing, so migrate plans a branch
        for Postgres, where merge and drop cannot reach it.
    Oracle: the skip line naming merge and drop, and no branch in the
        planned stores.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    monkeypatch.setattr('memman.cli.shutil.which', lambda name: '/bin/true')
    selection = ['--all'] if migrate_all else ['--store', branch]

    result = invoke(mm_runner, [
        'migrate', *selection, '--to', 'postgres', '--dry-run'])

    skip_lines = [
        line for line in result.output.splitlines() if branch in line]
    assert skip_lines
    assert all('merge' in line and 'drop' in line for line in skip_lines)


def test_status_and_store_list_carry_the_branch_fields(mm_runner):
    """Verify status and store list name a branch's parent and date.

    Mutation: either output missing the branch fields.
    Oracle: the parent and created_at of the creation reply.
    """
    _, data_dir = mm_runner
    reply = branch_mod.create_branch(data_dir, 'default', 'rearch')
    branch = reply['store']

    status = json.loads(invoke(mm_runner, ['status', '--store', branch]).output)
    listing = json.loads(invoke(mm_runner, ['store', 'list']).output)

    assert (status['branch_parent'], status['branch_created_at']) == (
        'default', reply['created_at'])
    assert listing['branches'] == {
        branch: {'parent': 'default', 'created_at': reply['created_at']}}


def test_doctor_lists_open_branches_and_warns_on_a_missing_parent(mm_runner):
    """Verify doctor lists each branch and warns when its parent is gone.

    Mutation: the branch check missing, or a branch with no parent
        passed.
    Oracle: two branches, one of a parent removed after the branch.
    """
    _, data_dir = mm_runner
    kept = _branch(data_dir)
    invoke(mm_runner, ['store', 'create', 'work'])
    _remember(mm_runner, 'The worker splits in two.', 'work')
    orphan = _branch(data_dir, parent='work')
    factory.drop_store('work', data_dir)

    report = json.loads(invoke(mm_runner, ['doctor']).output)

    check = next(c for c in report['checks'] if c['name'] == 'branches')
    assert check['status'] == 'warn'
    assert {b['store'] for b in check['detail']['branches']} == {kept, orphan}
    assert check['detail']['missing_parent'] == [orphan]


@pytest.mark.parametrize('damage', ['corrupt database', 'missing database'])
def test_store_remove_removes_a_sqlite_store_it_cannot_read(mm_runner, damage):
    """Verify store remove still removes a SQLite store with a broken file.

    Mutation: the branch or token read raising out of store remove, so
        no verb can remove a corrupt store or a corrupt branch.
    Oracle: the store directory, gone after the call.
    """
    _, data_dir = mm_runner
    junk_dir = Path(store_dir(data_dir, 'junk'))
    junk_dir.mkdir(parents=True)
    if damage == 'corrupt database':
        (junk_dir / 'memman.db').write_bytes(b'not a database' * 100)

    result = invoke(mm_runner, ['store', 'remove', 'junk', '--yes'])

    assert result.exit_code == 0, result.output
    assert not junk_dir.exists()


@pytest.mark.parametrize('holder', ['local branch', 'branch on another host'])
def test_store_remove_refuses_a_parent_holding_a_branch_token(
        mm_runner, holder):
    """Verify store remove keeps a parent while any branch token is on it.

    Mutation: no token check in store remove, or a check of local
        branches only, which removes a Postgres parent shared with a
        branch on another host.
    Oracle: the parent's existence after the call, and the key named in
        the refusal.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    if holder == 'local branch':
        key = f'branch_token:{_branch(data_dir, parent="work")}'
    else:
        key = 'branch_token:work__elsewhere_1a2b'
        _set_meta(data_dir, 'work', key, 'f00d')

    result = invoke(mm_runner, ['store', 'remove', 'work', '--yes'])

    assert result.exit_code != 0
    assert key in result.output
    assert factory.store_exists('work', data_dir)


@pytest.mark.no_auto_drain
def test_store_remove_refuses_a_branch_and_names_store_drop(mm_runner):
    """Verify store remove keeps a branch and points at store drop.

    Mutation: no branch check in store remove, so drop_store purges the
        branch's pending writes with no warning.
    Oracle: the branch's existence and its queued write after the call.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    queued = invoke(mm_runner, [
        'remember', '--store', branch,
        'The grackle colony nests by the river.'])
    assert queued.exit_code == 0, queued.output

    result = invoke(mm_runner, ['store', 'remove', branch, '--yes'])

    assert result.exit_code != 0
    assert f'memman store drop {branch}' in result.output
    assert factory.store_exists(branch, data_dir)
    assert queued_contents(data_dir) == [
        'The grackle colony nests by the river.']


def test_reembed_follows_the_parent_backend_of_each_branch(
        mm_runner, env_file, pg_dsn, monkeypatch):
    """Verify reembed moves a branch of a SQLite parent, skips a Postgres one.

    Mutation: every branch skipped, so a branch keeps the old model
        while its SQLite parent moves; or none, so a branch moves while
        its Postgres parent stays.
    Oracle: the store names in the sweep's per-store report.
    """
    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_reembedpg', 'postgres')
    env_file('MEMMAN_POSTGRES_DSN_reembedpg', pg_dsn)
    created = invoke(mm_runner, ['store', 'create', 'reembedpg'])
    assert created.exit_code == 0, created.output
    try:
        sqlite_child = _branch(data_dir)
        pg_child = _branch(data_dir, parent='reembedpg')
        monkeypatch.setattr(
            sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)

        result = invoke(mm_runner, ['embed', 'reembed'])

        assert result.exit_code == 0, result.output
        swept = {s['store'] for s in json.loads(result.output)['stores']}
        assert sqlite_child in swept
        assert pg_child not in swept
    finally:
        factory.drop_store('reembedpg', data_dir)


def _env_keys(data_dir: str) -> dict[str, str]:
    """Keys of the env file under data_dir, empty when it is absent.
    """
    path = config.env_file_path(data_dir)
    return config.parse_env_file(path) if path.exists() else {}


@pytest.mark.no_auto_drain
def test_recall_on_a_missing_store_refuses_and_creates_nothing(mm_runner):
    """Verify recall on a missing store refuses and leaves no directory.

    Mutation: open_sqlite_backend creating the store directory on open.
    Oracle: the store directory and backend key absent after the call.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['--store', 'ghost', 'recall', 'anything'])

    assert result.exit_code != 0
    assert 'store "ghost" does not exist' in result.output
    assert not Path(store_dir(data_dir, 'ghost')).exists()
    assert 'MEMMAN_BACKEND_ghost' not in _env_keys(data_dir)


@pytest.mark.no_auto_drain
def test_remember_on_a_missing_store_refuses_and_queues_nothing(mm_runner):
    """Verify remember on a missing store refuses before it queues.

    Mutation: remember enqueuing before the existence check, or the
    backend key written before it.
    Oracle: an empty queue, no store directory and no backend key.
    """
    _, data_dir = mm_runner
    args = [
        '--store', 'ghost', 'remember',
        'The retry cap for batch jobs is three.',
        ]

    result = invoke(mm_runner, args)

    assert result.exit_code != 0
    assert 'store "ghost" does not exist' in result.output
    assert queued_contents(data_dir) == []
    assert not Path(store_dir(data_dir, 'ghost')).exists()
    assert 'MEMMAN_BACKEND_ghost' not in _env_keys(data_dir)


def _schema_exists(dsn: str, store: str) -> bool:
    """True when the Postgres database at dsn holds the store's schema.
    """
    import psycopg
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            'select 1 from pg_namespace where nspname = %s',
            (f'store_{store}',)).fetchone()
    return row is not None


@pytest.mark.postgres
@pytest.mark.no_auto_drain
def test_missing_postgres_store_refuses_and_creates_no_schema(
        mm_runner, env_file, pg_dsn):
    """Verify a missing Postgres store refuses recall and remember.

    Mutation: _ensure_baseline_schema run before the existence check,
    or remember enqueuing on a confirmed missing schema.
    Oracle: pg_namespace read straight from the container, and an empty
    queue.
    """
    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_pghost', 'postgres')
    env_file('MEMMAN_POSTGRES_DSN_pghost', pg_dsn)

    recalled = invoke(mm_runner, ['--store', 'pghost', 'recall', 'anything'])
    remembered = invoke(mm_runner, [
        '--store', 'pghost', 'remember',
        'The retry cap for batch jobs is three.'])

    assert 'store "pghost" does not exist' in recalled.output
    assert 'store "pghost" does not exist' in remembered.output
    assert queued_contents(data_dir) == []
    assert not _schema_exists(pg_dsn, 'pghost')


@pytest.mark.no_auto_drain
def test_remember_queues_when_the_postgres_check_cannot_connect(
        mm_runner, env_file):
    """Verify a connection error in remember's existence check queues.

    Mutation: a connection error read as a missing store, which refuses
    every write during a database outage.
    Oracle: a DSN on a closed local port, which refuses the connection.
    """
    pytest.importorskip('psycopg')
    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_pgdown', 'postgres')
    env_file(
        'MEMMAN_POSTGRES_DSN_pgdown',
        'host=127.0.0.1 port=1 dbname=x user=x password=x')
    content = 'The retry cap for batch jobs is three.'

    result = invoke(mm_runner, ['--store', 'pgdown', 'remember', content])

    assert result.exit_code == 0, result.output
    assert queued_contents(data_dir) == [content]


@pytest.mark.no_auto_drain
def test_remember_refuses_a_postgres_store_with_no_dsn(
        mm_runner, env_file, monkeypatch):
    """Verify a postgres store with no DSN refuses the write up front.

    Mutation: the missing-DSN config error read as a connection error,
    which queues a write the drain can only mark failed.
    Oracle: a store routed to postgres with no per-store or default DSN.
    """
    _, data_dir = mm_runner
    monkeypatch.delenv(config.DEFAULT_PG_DSN, raising=False)
    env_file(config.DEFAULT_PG_DSN, None)
    env_file('MEMMAN_BACKEND_nodsn', 'postgres')

    result = invoke(mm_runner, [
        'remember', '--store', 'nodsn', 'The retry cap is three.'])

    assert 'no DSN' in result.output
    assert queued_contents(data_dir) == []


@pytest.mark.no_auto_drain
def test_remember_refuses_a_postgres_store_without_the_extra(
        mm_runner, env_file, monkeypatch):
    """Verify a postgres store on a host without the extra refuses cleanly.

    Mutation: the ImportError from the existence check escaping as a
    traceback.
    Oracle: the postgres modules hidden from import.
    """
    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_pgx', 'postgres')
    env_file(
        'MEMMAN_POSTGRES_DSN_pgx',
        'host=127.0.0.1 port=1 dbname=x user=x password=x')
    monkeypatch.setitem(sys.modules, 'psycopg', None)
    monkeypatch.delitem(sys.modules, 'memman.store.postgres', raising=False)

    result = invoke(mm_runner, [
        'remember', '--store', 'pgx', 'The retry cap is three.'])

    assert isinstance(result.exception, SystemExit), result.exception
    assert 'memman[postgres]' in result.output
    assert queued_contents(data_dir) == []


def test_existence_check_keeps_an_import_bug_visible(monkeypatch):
    """Verify only a missing extra package reads as a missing extra.

    Mutation: every ImportError reworded as a missing extra, which hides
    a broken import inside memman.
    Oracle: a stub existence function raising an ImportError for a
    memman module.
    """
    from memman.store import factory

    def broken(store, data_dir):
        raise ImportError('cannot import name x', name='memman.oops')

    monkeypatch.setitem(factory.BACKENDS, 'sqlite', dataclasses.replace(
        factory.descriptor('sqlite'), store_exists_fn=broken))

    with pytest.raises(ImportError):
        factory.store_exists('default', '/nonexistent')


@pytest.mark.postgres
def test_postgres_existence_check_needs_no_vector_extension(pg_dsn):
    """Verify a database without pgvector reads as missing, not as an outage.

    Mutation: the existence probe registering pgvector types, which
    raises on a database that lacks the extension.
    Oracle: a fresh database made from template1, which has no vector
    extension.
    """
    import psycopg
    from memman.store.postgres import postgres_store_exists
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute('drop database if exists novector with (force)')
        conn.execute('create database novector')
    dsn = re.sub(r'dbname=\S+', 'dbname=novector', pg_dsn)

    try:
        exists = postgres_store_exists('ghost', dsn)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute('drop database novector with (force)')

    assert exists is False


def test_prime_prints_the_refusal_for_a_missing_store(mm_runner, monkeypatch):
    """Verify prime's status line names a missing resolved store.

    Mutation: prime falling back to the bare "Memory active." line.
    Oracle: the refusal text fixed by the spec.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_STORE', 'ghost')

    result = CliRunner().invoke(cli, ['prime'], input='{}')

    status_line = result.output.splitlines()[0]
    assert 'store "ghost" does not exist' in status_line


def test_doctor_reports_a_missing_store_and_runs_other_checks(mm_runner):
    """Verify doctor fails one store check and still runs the rest.

    Mutation: doctor aborting on the missing store, or passing it.
    Oracle: the check list, which holds a failed store_exists check and
    the queue_backlog check.
    """
    result = invoke(mm_runner, ['--store', 'ghost', 'doctor'])

    report = json.loads(result.output)
    status_by_name = {c['name']: c['status'] for c in report['checks']}
    assert result.exit_code == 1
    assert status_by_name['store_exists'] == 'fail'
    assert 'queue_backlog' in status_by_name


@pytest.mark.parametrize('agent_installed', [False, True])
def test_install_creates_the_store_the_active_file_names(
        mm_runner, monkeypatch, agent_installed):
    """Verify install creates and seeds the active store on every path.

    Mutation: install leaving the store to the first write, skipping the
    scheduler-only path, or creating 'default' in place of the active
    store.
    Oracle: the store's memman.db and its embed fingerprint, read after
    a flow with an active file naming 'work'.
    """
    _, data_dir = mm_runner
    write_active(data_dir, 'work')
    monkeypatch.setattr(
        claude_setup, 'install_scheduler',
        lambda data_dir, knobs: {'platform': 'systemd', 'actions': []})
    monkeypatch.setattr(
        claude_setup, '_install_claude_code', lambda *a, **kw: None)
    monkeypatch.setattr(
        claude_setup.openrouter_models, 'refresh_model_state',
        lambda *a, **kw: None)
    env = {
        'detected': False,
        'display': 'Claude Code',
        'version': '',
        'config_dir': '',
        }

    claude_setup._run_install_flow(
        env, claude_code=agent_installed, data_dir=data_dir, knobs={})

    with open_sqlite_backend('work', data_dir, read_only=True) as backend:
        stored = backend.meta.get(META_KEY)
    assert stored is not None


@pytest.mark.no_auto_drain
def test_per_verb_store_routes_like_the_group_flag(mm_runner):
    """Verify `remember --store X` queues the row for store X.

    Mutation: the per-verb option parsed but never copied into the
    context, so the write lands in the active store.
    Oracle: the store named in the reply, against the active 'default'.
    """
    invoke(mm_runner, ['store', 'create', 'work'])

    result = invoke(mm_runner, [
        'remember', '--store', 'work',
        'The retry cap for batch jobs is three.'])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['store'] == 'work'


def test_per_verb_and_group_store_that_differ_raise(mm_runner):
    """Verify two different --store values refuse the call.

    Mutation: one flag silently winning over the other.
    Oracle: click's usage-error exit code 2, with both names in the text.
    """
    invoke(mm_runner, ['store', 'create', 'work'])

    result = invoke(mm_runner, [
        '--store', 'default', 'recall', '--store', 'work', 'anything'])

    assert result.exit_code == 2
    assert 'default' in result.output
    assert 'work' in result.output


def test_store_create_refuses_a_double_underscore_name(mm_runner):
    """Verify store create refuses a name holding the branch separator.

    Mutation: the `__` reserve check missing from store create.
    Oracle: no store directory for the refused name.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['store', 'create', 'memman__rearch'])

    assert result.exit_code != 0
    assert not Path(store_dir(data_dir, 'memman__rearch')).exists()


def test_store_create_pins_the_backend_it_creates_on(
        mm_runner, env_file):
    """Verify a created store keeps its backend when the default changes.

    Mutation: store create leaving MEMMAN_BACKEND_<store> to the first
    drain, so a default switched before any write routes the store to a
    backend that lacks it, and every verb refuses while create says it
    exists.
    Oracle: a recall after the default backend moves to postgres.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    env_file(config.DEFAULT_BACKEND, 'postgres')

    result = invoke(mm_runner, ['recall', '--store', 'work', 'anything'])

    assert result.exit_code == 0, result.output


def test_failed_store_create_leaves_no_backend_pin(mm_runner, env_file):
    """Verify a create that fails pins nothing, so a retry can succeed.

    Mutation: the backend key written before the open, so a create that
    fails on a misconfigured default keeps routing the retry there.
    Oracle: a retry after the default moves back to sqlite.
    """
    _, data_dir = mm_runner
    env_file(config.DEFAULT_PG_DSN, None)
    env_file(config.DEFAULT_BACKEND, 'postgres')
    assert invoke(mm_runner, ['store', 'create', 'work']).exit_code != 0
    env_file(config.DEFAULT_BACKEND, 'sqlite')

    result = invoke(mm_runner, ['store', 'create', 'work'])

    assert result.exit_code == 0, result.output
    assert _env_keys(data_dir).get('MEMMAN_BACKEND_work') == 'sqlite'


def test_install_pins_the_backend_of_the_store_it_creates(mm_runner):
    """Verify install's store creation writes the store's backend key.

    Mutation: _init_default_store creating the store without pinning its
    backend, the same dead end as an unpinned store create.
    Oracle: the env file read after the call.
    """
    _, data_dir = mm_runner
    write_active(data_dir, 'work')

    claude_setup._init_default_store(data_dir)

    assert _env_keys(data_dir).get('MEMMAN_BACKEND_work') == 'sqlite'


@pytest.mark.parametrize('verb', ['use', 'remove'])
def test_store_use_and_remove_name_a_missing_store_alike(mm_runner, verb):
    """Verify store use and remove give the shared missing-store refusal.

    Mutation: a verb keeping its own wording for a missing store.
    Oracle: the StoreMissingError text every memory verb prints.
    """
    result = invoke(mm_runner, ['store', verb, 'ghost'])

    assert str(StoreMissingError('ghost')) in result.output
