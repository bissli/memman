"""Experiment forks and the missing-store refusal they rely on.
"""

import dataclasses
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman import fork as fork_mod
from memman.cli import cli, list_agent_commands, list_claude_permissions
from memman.embed.fingerprint import META_KEY, Fingerprint
from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.migrate import MigrateError, MigrateInsight, MigrationPayload
from memman.queue import queue_db
from memman.setup import claude as claude_setup
from memman.setup.codex import install_codex
from memman.store.db import list_local_store_dirs, read_active, store_dir
from memman.store.db import write_active
from memman.store.errors import StoreMissingError
from memman.store.sqlite import SqliteMigrator, open_sqlite_backend
from tests.conftest import EMBEDDING_DIM, force_drain, invoke, queued_contents


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
    """Verify store create refuses a name holding the fork separator.

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


_FORK_VERBS = [('store', 'drop'), ('store', 'fork'), ('store', 'merge')]


def test_fork_verbs_are_agent_callable_on_both_hosts(tmp_path):
    """Verify fork, merge and drop reach the Claude and Codex allow lists.

    Mutation: a fork verb missing @claude_callable, so one host prompts
    on every call.
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
    for path in _FORK_VERBS:
        assert path in list_agent_commands()
        assert f'Bash(memman {" ".join(path)}:*)' in permissions
        assert json.dumps(['memman', *path]) in rules


@pytest.mark.parametrize('verb', ['fork', 'merge', 'drop'])
def test_fork_verbs_take_no_per_verb_store(mm_runner, verb):
    """Verify the fork verbs name their stores as arguments only.

    Mutation: store_option left at its default on a fork verb, so a
    --store value could disagree with the named store.
    Oracle: click's no-such-option usage error.
    """
    result = invoke(mm_runner, ['store', verb, '--store', 'x', 'a', 'b'])

    assert result.exit_code == 2
    assert 'No such option' in result.output


def _remember(runner, text):
    """Store text in the active store, drained, and return its id.
    """
    result = invoke(runner, ['remember', text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def _seed_parent(runner):
    """Seed the default store with two current rows and two retired ones.

    Returns
    -------
    tuple[set[str], set[str]]
        Ids of the current rows, then ids of the retired rows.
    """
    kept = _remember(runner, 'The retry cap for batch jobs is three.')
    stale = _remember(runner, 'The backup job runs at midnight.')
    gone = _remember(runner, 'The scheduler runs every sixty seconds.')
    replaced = invoke(runner, [
        'replace', stale, 'The backup job runs at two in the morning.'])
    successor = json.loads(replaced.output)['id']
    assert invoke(runner, ['forget', gone]).exit_code == 0
    return {kept, successor}, {stale, gone}


def _fork(runner, parent='default', label='rearch'):
    """Run store fork and return its parsed JSON output.
    """
    result = invoke(runner, ['store', 'fork', parent, label])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_fork_leaves_the_parent_unchanged(mm_runner):
    """Verify the fork verb writes nothing into its parent.

    Mutation: the verb writing fork meta, an oplog row or a row change
    into the parent.
    Oracle: the parent's full gather before and after the fork.
    """
    _, data_dir = mm_runner
    _seed_parent(mm_runner)
    before = SqliteMigrator(data_dir).gather('default')

    _fork(mm_runner)

    assert SqliteMigrator(data_dir).gather('default') == before


def test_fork_copies_current_rows_and_names_the_fork(mm_runner):
    """Verify the fork holds only the parent's current rows and fork keys.

    Mutation: retired rows copied, the fingerprint seeded from the env
    model in place of the parent's, or a wrong name shape.
    Oracle: the ids _seed_parent reports current, a parent fingerprint
    whose model differs from the env model, and the name pattern.
    """
    _, data_dir = mm_runner
    current_ids, _ = _seed_parent(mm_runner)
    parent_fp = Fingerprint(model='other/model', dim=EMBEDDING_DIM)
    with open_sqlite_backend('default', data_dir) as parent:
        write_fingerprint(parent, parent_fp)

    reply = _fork(mm_runner)

    fork = SqliteMigrator(data_dir).gather(reply['store'])
    assert re.fullmatch(r'default__rearch_[0-9a-f]{4}', reply['store'])
    assert {ins.id for ins in fork.insights} == current_ids
    assert fork.meta['fork_parent'] == 'default'
    assert fork.meta['fork_rows'] == '2'
    assert fork.meta['fork_created_at'] == reply['created_at']
    assert fork.fingerprint == parent_fp
    assert fork.oplog == []


def _set_meta(data_dir, store, key, value):
    """Write one meta key into a SQLite store.
    """
    with open_sqlite_backend(store, data_dir) as backend:
        backend.meta.set(key, value)


def _break_fingerprint(data_dir):
    """Delete the default store's embed fingerprint row.
    """
    with open_sqlite_backend('default', data_dir) as backend:
        backend.meta.delete(META_KEY)


@pytest.mark.parametrize(('case', 'refusal'), [
    ('missing parent', 'store "ghost" does not exist'),
    ('fork parent', 'is a fork of'),
    ('double underscore label', 'invalid fork label'),
    ('no fingerprint', 'embed_fingerprint meta key'),
    ('corrupt fingerprint', 'cannot fork'),
    ('swap in progress', 'embed swap in progress'),
    ('reembed in progress', 're-embed in progress'),
    ])
def test_fork_refuses_and_creates_nothing(mm_runner, case, refusal):
    """Verify each fork refusal names its cause and creates no store.

    Mutation: one refusal missing, so the verb forks a store it must
    not copy or stops on a later error, or a corrupt fingerprint
    escaping as a traceback.
    Oracle: each refusal's own text, and the store directory list
    before and after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    parent, label = 'default', 'rearch'
    if case == 'missing parent':
        parent = 'ghost'
    elif case == 'fork parent':
        parent = _fork(mm_runner)['store']
    elif case == 'double underscore label':
        label = 're__arch'
    elif case == 'no fingerprint':
        _break_fingerprint(data_dir)
    elif case == 'corrupt fingerprint':
        _set_meta(data_dir, 'default', META_KEY, '{not json')
    elif case == 'swap in progress':
        _set_meta(data_dir, 'default', 'embed_swap_state', 'backfill')
    else:
        _set_meta(data_dir, 'default', 'embed_reembed_state', 'in_progress')
    stores_before = list_local_store_dirs(data_dir)

    result = invoke(mm_runner, ['store', 'fork', parent, label])

    assert isinstance(result.exception, SystemExit), result.exception
    assert refusal in result.output
    assert list_local_store_dirs(data_dir) == stores_before


def test_failed_fork_leaves_no_directory_and_no_env_keys(
        mm_runner, monkeypatch):
    """Verify a fork whose copy fails removes what it wrote.

    Mutation: the cleanup after a failed apply skipped, which leaves a
    `__` store with no fork_parent meta.
    Oracle: the store directories and the env file read after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    stores_before = list_local_store_dirs(data_dir)

    def failing_apply(self, store, payload):
        raise MigrateError('disk full')

    monkeypatch.setattr(SqliteMigrator, 'apply', failing_apply)

    result = invoke(mm_runner, ['store', 'fork', 'default', 'rearch'])

    assert result.exit_code != 0
    assert list_local_store_dirs(data_dir) == stores_before
    assert not [k for k in _env_keys(data_dir) if '__rearch' in k]


def test_fork_whose_directory_cannot_be_made_writes_no_env_keys(
        mm_runner, monkeypatch):
    """Verify a failed fork directory create leaves no env keys.

    Mutation: the env keys written before the directory, outside the
    cleanup, which leaves backend keys for a store that never existed.
    Oracle: the env file read after the call, and a refusal in place of
    a traceback.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')

    class UnwritablePath(type(Path())):
        def mkdir(self, *args, **kwargs):
            raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(fork_mod, 'Path', UnwritablePath)

    result = invoke(mm_runner, ['store', 'fork', 'default', 'rearch'])

    assert isinstance(result.exception, SystemExit), result.exception
    assert not [k for k in _env_keys(data_dir) if '__rearch' in k]


def test_fork_instruction_names_the_fork_parent_and_time(mm_runner):
    """Verify the instruction line carries what a session needs to route.

    Mutation: the instruction missing the fork name, the parent, or the
    fork date a parent recall filters on, or carrying the time of day a
    recall page never shows.
    Oracle: the store, parent and created_at fields of the same reply.
    """
    _remember(mm_runner, 'The retry cap for batch jobs is three.')

    reply = _fork(mm_runner)

    assert f'--store {reply["store"]}' in reply['instruction']
    assert '--store default' in reply['instruction']
    assert reply['created_at'][:10] in reply['instruction']
    assert reply['created_at'] not in reply['instruction']


def _merge(runner, fork):
    """Run store merge and return its parsed JSON output.
    """
    result = invoke(runner, ['store', 'merge', fork])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _rows_by_id(data_dir, store):
    """Every row of a SQLite store, retired rows included, keyed by id.
    """
    return {
        ins.id: ins for ins in SqliteMigrator(data_dir).gather(store).insights}


def test_merge_copies_a_fork_row_whole_without_api_calls(
        mm_runner, monkeypatch):
    """Verify merge copies a fork-only row with its stored columns.

    Mutation: the copy sent through NodeStore.insert or the queue, which
    stamps a new created_at and re-embeds or re-enriches the row.
    Oracle: the fork's own row read before the merge, and API stubs that
    record every call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    new_id = json.loads(invoke(mm_runner, [
        'remember', '--store', fork,
        'The rearchitecture splits the worker into two processes.',
        ]).output)['id']
    fork_row = _rows_by_id(data_dir, fork)[new_id]
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

    reply = _merge(mm_runner, fork)

    parent_row = _rows_by_id(data_dir, 'default')[new_id]
    assert reply['copied'] == 1
    assert parent_row.created_at == fork_row.created_at
    assert parent_row.summary == fork_row.summary
    assert parent_row.embedding == fork_row.embedding
    assert calls == []


def _retire(data_dir, store, row_id, *, successor=None):
    """Replace row_id by successor, or soft-delete it when successor is None.
    """
    with open_sqlite_backend(store, data_dir) as backend:
        if successor is None:
            assert backend.nodes.soft_delete(row_id)
        else:
            assert backend.nodes.mark_replaced(row_id, successor)


@pytest.mark.parametrize(('fork_state', 'parent_state', 'expected'), [
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
        mm_runner, fork_state, parent_state, expected):
    """Verify merge gives each fork and parent state pair its table result.

    Mutation: a table row swapped, the live-predecessor guard skipped, or
    a fork row both replaced and deleted read as deleted, which turns a
    done row into a false conflict.
    Oracle: the retirement table of the spec, one row per case.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    if parent_state == 'current with live predecessor':
        target = json.loads(invoke(mm_runner, [
            'replace', target, 'The retry cap for batch jobs is five.',
            ]).output)['id']
    fork = _fork(mm_runner)['store']
    successor = json.loads(invoke(mm_runner, [
        'remember', '--store', fork,
        'The retry cap for batch jobs is seven.',
        ]).output)['id']
    if fork_state == 'deleted':
        _retire(data_dir, fork, target)
    else:
        _retire(data_dir, fork, target, successor=successor)
    if fork_state == 'replaced and deleted':
        _retire(data_dir, fork, target)
    if parent_state == 'replaced by successor':
        _retire(data_dir, 'default', target, successor=successor)
    elif parent_state == 'replaced by other':
        _retire(data_dir, 'default', target, successor='other-row')
    elif parent_state == 'deleted':
        _retire(data_dir, 'default', target)
    parent_before = _rows_by_id(data_dir, 'default')[target]

    reply = _merge(mm_runner, fork)

    parent_after = _rows_by_id(data_dir, 'default')[target]
    if expected == 'retired':
        assert reply['retired'] == 1
        assert reply['conflicts'] == []
        assert (parent_after.replaced_by, parent_after.deleted_at is None) == (
            (successor, True) if fork_state == 'replaced' else (None, False))
    else:
        assert reply['retired'] == 0
        assert len(reply['conflicts']) == (expected == 'conflict')
        assert parent_after == parent_before


def test_merge_writes_an_oplog_row_per_retirement(mm_runner):
    """Verify each retirement merge applies leaves a parent oplog row.

    Mutation: the retirement written without its oplog row, so the
    parent's history loses the change.
    Oracle: the parent oplog's newest row, read after the merge.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    _retire(data_dir, fork, target)

    _merge(mm_runner, fork)

    newest = SqliteMigrator(data_dir).gather('default').oplog[-1]
    assert (newest.operation, newest.insight_id, newest.detail) == (
        'forget', target, f'merged from {fork}')


def _queue_status(data_dir, store, status):
    """Set every queue row of store to status.
    """
    with queue_db(data_dir) as conn:
        conn.execute(
            'update queue set status = ? where store = ?', (status, store))


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('case', [
    'pending fork row', 'failed fork row', 'pending parent replace',
    'fingerprint mismatch', 'parent swap', 'fork reembed', 'active fork',
    'recreated parent',
    ])
def test_merge_refuses_and_changes_nothing(mm_runner, case):
    """Verify each merge refusal leaves the fork and its parent as they were.

    Mutation: one refusal missing, or the target read from the active
    store or a recreated store of the parent's name.
    Oracle: both stores' full gathers before and after the call.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    target = json.loads(invoke(mm_runner, [
        'remember', '--store', 'work',
        'The retry cap for batch jobs is three.',
        ]).output)['id']
    force_drain(data_dir)
    fork = _fork(mm_runner, parent='work')['store']
    if case in {'pending fork row', 'failed fork row'}:
        invoke(mm_runner, [
            'remember', '--store', fork, 'The worker splits in two.'])
        if case == 'failed fork row':
            _queue_status(data_dir, fork, 'failed')
    elif case == 'pending parent replace':
        _retire(data_dir, fork, target)
        invoke(mm_runner, [
            'replace', '--store', 'work', target,
            'The retry cap for batch jobs is five.'])
    elif case == 'fingerprint mismatch':
        with open_sqlite_backend(fork, data_dir) as backend:
            write_fingerprint(
                backend, Fingerprint(model='other/model', dim=EMBEDDING_DIM))
    elif case == 'parent swap':
        _set_meta(data_dir, 'work', 'embed_swap_state', 'backfill')
    elif case == 'fork reembed':
        _set_meta(data_dir, fork, 'embed_reembed_state', 'in_progress')
    elif case == 'active fork':
        write_active(data_dir, fork)
    else:
        work_fp = SqliteMigrator(data_dir).gather('work').meta[META_KEY]
        invoke(mm_runner, ['store', 'remove', 'work', '--yes'])
        invoke(mm_runner, ['store', 'create', 'work'])
        _set_meta(data_dir, 'work', META_KEY, work_fp)
    fork_before = SqliteMigrator(data_dir).gather(fork)
    parent_before = SqliteMigrator(data_dir).gather('work')

    result = invoke(mm_runner, ['store', 'merge', fork])

    assert result.exit_code != 0
    assert SqliteMigrator(data_dir).gather(fork) == fork_before
    assert SqliteMigrator(data_dir).gather('work') == parent_before


def test_merge_rerun_after_a_crash_changes_nothing_more(
        mm_runner, monkeypatch):
    """Verify a merge re-run over its own earlier copy is a no-op on the parent.

    Mutation: a copy that inserts duplicates on the second pass, or a
    retirement applied twice, which writes a second oplog row.
    Oracle: the parent's gather after the crashed first run.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    successor = json.loads(invoke(mm_runner, [
        'remember', '--store', fork,
        'The retry cap for batch jobs is seven.',
        ]).output)['id']
    _retire(data_dir, fork, target, successor=successor)

    def crash(*args, **kwargs):
        raise RuntimeError('killed before the removal')

    with monkeypatch.context() as patch:
        patch.setattr(fork_mod, '_remove_fork', crash)
        assert invoke(mm_runner, ['store', 'merge', fork]).exit_code != 0
    after_first = SqliteMigrator(data_dir).gather('default')

    reply = _merge(mm_runner, fork)

    assert (reply['copied'], reply['retired'], reply['conflicts']) == (
        0, 0, [])
    assert SqliteMigrator(data_dir).gather('default') == after_first
    assert not Path(store_dir(data_dir, fork)).exists()


def test_merge_that_fails_after_the_copy_says_to_rerun(
        mm_runner, monkeypatch):
    """Verify a database error during the retirements names the re-run.

    Mutation: only memman's own error types caught after the copy, so a
    raw sqlite3 or psycopg error leaves a half-merged parent with no
    re-run hint.
    Oracle: a stub retirement raising sqlite3's lock error.
    """
    _, data_dir = mm_runner
    target = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    _retire(data_dir, fork, target)

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(fork_mod, '_retire_inherited', locked)

    result = invoke(mm_runner, ['store', 'merge', fork])

    assert isinstance(result.exception, SystemExit), result.exception
    assert f'memman store merge {fork}' in result.output
    assert Path(store_dir(data_dir, fork)).exists()


def _row(row_id, content, created_at):
    """A MigrateInsight with only the columns the apply test reads.
    """
    return MigrateInsight(
        id=row_id, content=content, summary=None, embedding=None,
        enrich_attempted_at=None, enriched_at=None, created_at=created_at,
        updated_at=created_at, deleted_at=None, prompt_version=None,
        embedding_model=None, queue_uuid=None, replaced_by=None,
        author=None)


def _apply_payload(existing_id):
    """A payload holding a clashing id, a new id, and empty meta.
    """
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fp = seed_default_fingerprint()
    return MigrationPayload(
        fingerprint=fp, embedding_dim=fp.dim,
        insights=[
            _row(existing_id, 'clashing text', when),
            _row('new-row', 'new text', when),
            ],
        oplog=[], meta={})


def test_sqlite_apply_into_a_populated_store_inserts_only_new_ids(
        mm_runner):
    """Verify SqliteMigrator.apply skips ids the store already holds.

    Mutation: the plain insert kept, which fails the whole apply on the
    first clashing id.
    Oracle: the existing row's text and the meta, read before the apply.
    """
    _, data_dir = mm_runner
    existing = _remember(mm_runner, 'The retry cap for batch jobs is three.')
    before = SqliteMigrator(data_dir).gather('default')

    SqliteMigrator(data_dir).apply('default', _apply_payload(existing))

    after = SqliteMigrator(data_dir).gather('default')
    rows = {ins.id: ins.content for ins in after.insights}
    assert rows[existing] == 'The retry cap for batch jobs is three.'
    assert rows['new-row'] == 'new text'
    assert after.meta == before.meta


@pytest.mark.postgres
def test_postgres_apply_into_a_populated_store_inserts_only_new_ids(
        pg_dsn):
    """Verify PostgresMigrator.apply skips held ids and keeps the meta.

    Mutation: an upsert of clashing rows, or a meta write that replaces
    the store's own keys.
    Oracle: the existing row's text and the meta, read before the apply.
    """
    from memman.store.postgres import PostgresMigrator, drop_postgres_store
    from memman.store.postgres import open_postgres_backend
    store = 'apply_populated'
    drop_postgres_store(store, pg_dsn)
    with open_postgres_backend(store, pg_dsn, create=True) as backend:
        backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
    migrator = PostgresMigrator(dsn=pg_dsn)
    before = migrator.gather(store)
    migrator.apply(store, _apply_payload('held-row'))
    clash = _apply_payload('held-row')
    clash.insights[0].content = 'second text'

    try:
        migrator.apply(store, clash)
        after = migrator.gather(store)
    finally:
        drop_postgres_store(store, pg_dsn)

    rows = {ins.id: ins.content for ins in after.insights}
    assert rows['held-row'] == 'clashing text'
    assert after.meta == before.meta


@pytest.mark.parametrize('verb', ['merge', 'drop'])
def test_merge_and_drop_refuse_an_ordinary_store(mm_runner, verb):
    """Verify merge and drop act only on a store with fork_parent meta.

    Mutation: the fork_parent check skipped, so an agent-callable verb
    deletes an ordinary store or fails on a later lookup.
    Oracle: the refusal text, and the store's gather before and after.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    before = SqliteMigrator(data_dir).gather('work')

    result = invoke(mm_runner, ['store', verb, 'work'])

    assert 'is not a fork' in result.output
    assert SqliteMigrator(data_dir).gather('work') == before


def _drop(runner, fork):
    """Run store drop and return its parsed JSON output.
    """
    result = invoke(runner, ['store', 'drop', fork])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _remember_in(runner, store, text):
    """Store text in store, drained, and return its id.
    """
    result = invoke(runner, ['remember', '--store', store, text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def test_drop_lists_the_fork_rows_the_parent_lacks_and_deletes_the_fork(
        mm_runner):
    """Verify drop prints the fork's own current rows, then removes it.

    Mutation: inherited or retired rows listed, or the fork's directory
    or env keys left behind.
    Oracle: the ids of the two current rows written in the fork.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    kept = _remember_in(mm_runner, fork, 'The worker splits in two.')
    stale = _remember_in(mm_runner, fork, 'The queue moves to Redis.')
    successor = json.loads(invoke(mm_runner, [
        'replace', '--store', fork, stale,
        'The queue stays in SQLite.']).output)['id']

    reply = _drop(mm_runner, fork)

    assert {row['id'] for row in reply['dropped']} == {kept, successor}
    assert reply['parent'] == 'default'
    assert not Path(store_dir(data_dir, fork)).exists()
    assert not [key for key in _env_keys(data_dir) if key.endswith(fork)]


def test_drop_refuses_a_fork_a_merge_started(mm_runner):
    """Verify drop refuses a fork whose merge stopped part way.

    Mutation: drop deleting a fork part of which is already in the
    parent, which strands the retirements the merge had not applied.
    Oracle: the fork directory, still present after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    _set_meta(data_dir, fork, 'fork_merging', '2026-10-05T00:00:00Z')

    result = invoke(mm_runner, ['store', 'drop', fork])

    assert result.exit_code != 0
    assert Path(store_dir(data_dir, fork)).exists()


def test_drop_rereads_the_fork_meta_under_the_drain_lock(
        mm_runner, monkeypatch):
    """Verify drop refuses a merge that started before it took the lock.

    Mutation: the fork_merging check run before the drain lock, so a
    merge that fails in that gap leaves a half-merged fork drop deletes.
    Oracle: a lock stub that writes fork_merging as it is taken.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    acquire = fork_mod._acquire_drain_lock

    def merge_in_the_gap(data_dir_arg):
        _set_meta(data_dir, fork, 'fork_merging', '2026-10-05T00:00:00Z')
        return acquire(data_dir_arg)

    monkeypatch.setattr(fork_mod, '_acquire_drain_lock', merge_in_the_gap)

    result = invoke(mm_runner, ['store', 'drop', fork])

    assert 'stopped part way' in result.output
    assert Path(store_dir(data_dir, fork)).exists()


def test_drop_caps_the_parent_check_connect_time(mm_runner, monkeypatch):
    """Verify drop sets the short connect timeout before the parent check.

    Mutation: the default libpq timeout kept, so a parent host that drops
    packets holds the drain lock for minutes.
    Oracle: a stub store_exists recording PGCONNECT_TIMEOUT at its call.
    """
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    monkeypatch.setenv('PGCONNECT_TIMEOUT', 'unset')
    monkeypatch.delenv('PGCONNECT_TIMEOUT')
    seen = []
    real_exists = fork_mod.factory.store_exists

    def recording_exists(store, data_dir):
        seen.append((store, os.environ.get('PGCONNECT_TIMEOUT')))
        return real_exists(store, data_dir)

    monkeypatch.setattr(fork_mod.factory, 'store_exists', recording_exists)

    _drop(mm_runner, fork)

    assert ('default', '3') in seen


def test_drop_refuses_when_the_parent_check_cannot_connect(
        mm_runner, env_file):
    """Verify an unreachable parent makes drop refuse, never list all rows.

    Mutation: a connection error read as a missing parent, which lists
    every inherited row as the fork's own.
    Oracle: a parent rerouted to a DSN on a closed local port.
    """
    pytest.importorskip('psycopg')
    _, data_dir = mm_runner
    invoke(mm_runner, ['store', 'create', 'work'])
    _remember_in(mm_runner, 'work', 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner, parent='work')['store']
    env_file('MEMMAN_BACKEND_work', 'postgres')
    env_file(
        'MEMMAN_POSTGRES_DSN_work',
        'host=127.0.0.1 port=1 dbname=x user=x password=x')

    result = invoke(mm_runner, ['store', 'drop', fork])

    assert result.exit_code != 0
    assert Path(store_dir(data_dir, fork)).exists()


@pytest.mark.no_auto_drain
def test_a_write_queued_during_drop_makes_it_refuse(mm_runner, monkeypatch):
    """Verify drop re-checks the queue inside its removal transaction.

    Mutation: the queue checked only before the removal transaction, so
    a write queued in between is purged with the fork.
    Oracle: a remember run just before the removal, and the queue and
    fork directory read after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    force_drain(data_dir)
    fork = _fork(mm_runner)['store']
    real_remove = fork_mod._remove_fork

    def remember_then_remove(conn, store, data_dir_arg):
        invoke(mm_runner, [
            'remember', '--store', fork, 'The worker splits in two.'])
        return real_remove(conn, store, data_dir_arg)

    monkeypatch.setattr(fork_mod, '_remove_fork', remember_then_remove)

    result = invoke(mm_runner, ['store', 'drop', fork])

    assert result.exit_code != 0
    assert Path(store_dir(data_dir, fork)).exists()
    assert queued_contents(data_dir)[-1] == 'The worker splits in two.'


@pytest.mark.no_auto_drain
def test_a_write_after_drop_meets_the_missing_store_refusal(mm_runner):
    """Verify a stale instruction line fails once the fork is dropped.

    Mutation: remember queuing for a store whose directory is gone,
    which the drain would recreate or fail on later.
    Oracle: the queue, which holds no row for the fork.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    force_drain(data_dir)
    fork = _fork(mm_runner)['store']
    _drop(mm_runner, fork)

    result = invoke(mm_runner, [
        'remember', '--store', fork, 'The worker splits in two.'])

    assert result.exit_code != 0
    assert 'The worker splits in two.' not in queued_contents(data_dir)


def test_store_use_refuses_a_fork(mm_runner):
    """Verify store use leaves the host-wide active file off a fork.

    Mutation: a fork written to the active file, which routes every
    session on the host into the experiment.
    Oracle: the active file, read after the call.
    """
    _, data_dir = mm_runner
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']

    result = invoke(mm_runner, ['store', 'use', fork])

    assert result.exit_code != 0
    assert '--store' in result.output
    assert read_active(data_dir) == 'default'


@pytest.mark.parametrize('migrate_all', [False, True])
def test_migrate_skips_a_fork(mm_runner, monkeypatch, migrate_all):
    """Verify migrate skips a fork alone or under --all.

    Mutation: the fork_parent check missing, so migrate plans a fork
    for Postgres, where merge and drop cannot reach it.
    Oracle: the skip line naming merge and drop, and no fork in the
    planned stores.
    """
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    fork = _fork(mm_runner)['store']
    monkeypatch.setattr('memman.cli.shutil.which', lambda name: '/bin/true')
    selection = ['--all'] if migrate_all else ['--store', fork]

    result = invoke(mm_runner, [
        'migrate', *selection, '--to', 'postgres', '--dry-run'])

    skip_lines = [
        line for line in result.output.splitlines() if fork in line]
    assert skip_lines
    assert all('merge' in line and 'drop' in line for line in skip_lines)


def test_status_and_store_list_carry_the_fork_fields(mm_runner):
    """Verify status and store list name a fork's parent and date.

    Mutation: either output missing the fork fields.
    Oracle: the parent and created_at of the fork verb's own reply.
    """
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    reply = _fork(mm_runner)
    fork = reply['store']

    status = json.loads(invoke(mm_runner, ['status', '--store', fork]).output)
    listing = json.loads(invoke(mm_runner, ['store', 'list']).output)

    assert (status['fork_parent'], status['fork_created_at']) == (
        'default', reply['created_at'])
    assert listing['forks'] == {
        fork: {'parent': 'default', 'created_at': reply['created_at']}}


def test_doctor_lists_open_forks_and_warns_on_a_missing_parent(mm_runner):
    """Verify doctor lists each fork and warns when its parent is gone.

    Mutation: the fork check missing, or a fork with no parent passed.
    Oracle: two forks, one of a parent removed after the fork.
    """
    _remember(mm_runner, 'The retry cap for batch jobs is three.')
    kept = _fork(mm_runner)['store']
    invoke(mm_runner, ['store', 'create', 'work'])
    _remember_in(mm_runner, 'work', 'The worker splits in two.')
    orphan = _fork(mm_runner, parent='work')['store']
    invoke(mm_runner, ['store', 'remove', 'work', '--yes'])

    report = json.loads(invoke(mm_runner, ['doctor']).output)

    check = next(c for c in report['checks'] if c['name'] == 'forks')
    assert check['status'] == 'warn'
    assert {f['store'] for f in check['detail']['forks']} == {kept, orphan}
    assert check['detail']['missing_parent'] == [orphan]
