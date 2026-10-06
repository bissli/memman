"""Store branches: an empty SQLite store layered over its live parent.
"""

import dataclasses
import json
import os
import shutil
from datetime import datetime, timezone

import click
import pytest
from memman import branch as branch_mod
from memman import config
from memman.embed.fingerprint import Fingerprint, swap_command
from memman.embed.fingerprint import write_fingerprint
from memman.migrate import MigrateInsight
from memman.pipeline.enrich import enrich_pending
from memman.store import factory
from memman.store.db import list_local_store_dirs, store_dir
from memman.store.model import Insight
from memman.store.sqlite import SqliteRecallSession, open_sqlite_backend
from tests.conftest import EMBEDDING_DIM, _set_env_file_value, _vec
from tests.conftest import force_drain, invoke

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
