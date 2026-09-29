"""Backend failures must present the same way on both backends.

The Postgres backend translates every statement failure into
`BackendError` at its connection scope; the SQLite backend translates
only the OPEN. The root group closes the gap, so `database is locked`
mid-recall exits as one clean line on either backend.

The seam is the root group. Many callers branch on a driver type (the
queue's stale-claim reclaim, recall's bookkeeping skip, the
`sqlite3.IntegrityError` arm in `sqlite.py`) and their handlers sit
deeper, so translating at `DB._query` would make each one dead code.

Alongside it: the worker keeps a stack the stream must never print,
`migrate` keeps the store name on a bare `BackendError`, and a failure
after the one-way embed cutover says so instead of reading as a
retryable connect error.
"""

import logging
import sqlite3
from pathlib import Path

import memman.migrate as mig
import pytest
from memman import cli as cli_mod
from memman.embed import fingerprint as fp_mod
from memman.embed import registry as ec_registry
from memman.embed import swap as swap_mod
from memman.setup import scheduler as sched_mod
from memman.store import node
from memman.store.db import _BASELINE_SCHEMA, open_db
from memman.store.errors import BackendError
from memman.store.sqlite import SqliteMigrator
from tests.conftest import _set_env_file_value, invoke

FAKE_DSN = 'postgresql://u:p@h:5432/d'


@pytest.fixture
def runner(mm_runner):
    """The `mm_runner` fixture under a shorter name.
    """
    return mm_runner


def _seed_store(data_dir, name, dim=512):
    """Create a migratable SQLite store carrying one insight.
    """
    sdir = Path(data_dir) / 'data' / name
    sdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             ('{"provider":"voyage","model":"voyage-3-lite","dim":'
              f'{dim}}}')))
        conn.execute(
            'insert into insights (id, content, category,'
            ' created_at, updated_at)'
            ' values (?, ?, ?, ?, ?)',
            ('11111111-1111-4111-8111-111111111111', 'seed text',
             'fact', '2026-01-01T00:00:00Z',
             '2026-01-01T00:00:00Z'))
        conn.commit()
    finally:
        conn.close()


def _stub_postgres(monkeypatch):
    """Seed a DSN and stub the two Postgres round-trips migrate makes.

    Without the DSN the command exits at its own missing-DSN gate,
    which would pass while the path under test never runs.
    """
    _set_env_file_value('MEMMAN_DEFAULT_POSTGRES_DSN', FAKE_DSN)
    monkeypatch.setattr(mig, 'preflight', lambda dsn: {'select_1': True})
    monkeypatch.setattr(
        mig, 'inspect_target_schemas',
        lambda dsn, stores: dict.fromkeys(stores, mig.SchemaState.ABSENT))


def test_sqlite_statement_error_exits_as_one_clean_line(runner, monkeypatch):
    """A mid-command sqlite failure reports like its Postgres twin.

    Mutation: dropping the root group's `except sqlite3.Error` arm, so
        the driver error escapes untranslated. `exit_code` cannot catch
        that on its own -- Click reports an in-command raise as exit 1
        exactly like a clean `ClickException`.
    Oracle: `result.exception` must not be the `sqlite3.Error` itself,
        and the output must carry `database is locked` behind an
        `Error:` prefix.
    """
    def _locked(*a, **kw):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(node, 'query_insights', _locked)
    result = invoke(runner, ['recall', 'anything', '--basic'])

    assert result.exit_code != 0
    assert not isinstance(result.exception, sqlite3.Error), result.output
    assert 'Error: sqlite query failed: database is locked' in result.output


def test_the_db_layer_leaves_driver_errors_driver_typed(tmp_path):
    """A failing statement reaches its caller as a raw `sqlite3` error.

    This pins where the seam sits, at the DB layer only. Callers branch
    on a driver type (the queue's stale-claim reclaim, recall's
    bookkeeping skip, the `sqlite3.IntegrityError` arm in `sqlite.py`),
    so the driver type must survive `DB._query` and `DB._exec`. A
    translation added higher, in the node layer or in `queue.py`, would
    leave this test green and is not pinned here.

    Mutation: translating inside `DB._query` / `DB._exec`, the
        symmetry with `postgres._connection`. Each driver-type
        handler silently becomes dead code.
    Oracle: the exception type out of a failing statement, which is
        `BackendError` under the mutation and `sqlite3.OperationalError`
        as shipped.
    """
    with open_db(str(tmp_path / 'store')) as db:
        with pytest.raises(sqlite3.OperationalError):
            db._query('select 1 from no_such_table')
        with pytest.raises(sqlite3.OperationalError):
            db._exec('insert into no_such_table (x) values (1)')


def test_worker_keeps_the_stack_the_stream_must_not_print(
        runner, monkeypatch, logger_state):
    """Worker file logging runs at DEBUG while stderr stays quiet.

    Mutation: setting the stream handler to DEBUG alongside the file
        handler, which prints every
        seam traceback to interactive users and undoes the clean
        one-line exit. Equally caught: leaving the logger at the
        configured level, which filters the record before any handler
        sees it.
        Also caught: moving the handler's own path, which decouples
        the file the worker writes from the file
        `memman log worker --stack` reads and leaves that command
        tailing a path nothing ever creates.
    Oracle: the three levels must disagree in one direction -- logger
        DEBUG, file handler DEBUG, stream handler at the configured
        WARNING -- and the handler's path is compared against
        `<data_dir>/logs/memman.log` spelled out independently.
    """
    _, data_dir = runner
    monkeypatch.setenv('MEMMAN_WORKER', '1')
    cli_mod._configure_logging(data_dir, verbose=False, debug=False)

    log = logger_state
    streams = [
        h for h in log.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, '_memman', False)]
    files = [
        h for h in log.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, '_memman', False)]

    assert streams
    assert files
    assert log.level == logging.DEBUG
    assert files[0].level == logging.DEBUG
    assert streams[0].level == logging.WARNING
    assert files[0].baseFilename == str(
        Path(data_dir) / 'logs' / 'memman.log')


def test_interactive_logging_attaches_no_worker_file(
        runner, monkeypatch, logger_state):
    """Off the worker, nothing is opened to DEBUG.

    Mutation: raising the logger to DEBUG unconditionally rather than
        only under `is_worker()`, which pays formatting cost on the
        synchronous recall path for records no handler will emit.
    Oracle: the logger sits at the configured WARNING and no rotating
        file handler exists at all.
    """
    _, data_dir = runner
    monkeypatch.delenv('MEMMAN_WORKER', raising=False)
    cli_mod._configure_logging(data_dir, verbose=False, debug=False)

    log = logger_state
    files = [
        h for h in log.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, '_memman', False)]

    assert not files
    assert log.level == logging.WARNING


def test_worker_says_where_the_stack_went(
        runner, monkeypatch, logger_state):
    """The terse worker line names the file holding the traceback.

    The stack goes to the rotated `logs/memman.log` because the
    worker's stderr is an unrotated systemd `append:` redirect, and
    that file sits under `--data-dir` rather than beside
    `enrich.err`, so a stack preserved and unnamed is a stack the
    operator cannot reach.

    Mutation: emitting the pointer as its own log record instead of on
        the message. `MEMMAN_LOG_LEVEL=ERROR` is an installable value,
        so the stream handler would drop a WARNING and restore the
        reported symptom -- one line, no route to the stack. Setting
        that level here is what catches it. Equally caught: trimming
        the command out of the message, which leaves an operator a
        path and no way to know what reads it; and emitting
        `--data-dir` after the subcommand, where Click rejects a group
        option, so the suggested command cannot run.
    Oracle: `memman.log` and the command that reads it, both named in
        the output Click itself writes, which no log level can
        suppress.
    """
    _, data_dir = runner
    monkeypatch.setenv('MEMMAN_WORKER', '1')
    _set_env_file_value('MEMMAN_LOG_LEVEL', 'ERROR')
    (Path(data_dir) / 'queue.db').write_bytes(b'not a sqlite database' * 8)

    result = invoke(runner, ['remember', 'a fact'])

    assert result.exit_code != 0
    assert 'memman.log' in result.output, result.output
    assert f'--data-dir {data_dir} log worker --stack' in result.output, \
        result.output


def test_the_stack_hint_drops_the_flag_on_a_default_data_dir(
        tmp_path, monkeypatch, logger_state):
    """The suggested command omits `--data-dir` when it is the default.

    Mutation: emitting `--data-dir` unconditionally, which pastes a
        redundant flag into every default install's error message; or
        emitting it never, which the sibling test catches from the other
        side.
    Oracle: `default_data_dir()` recomputed from the patched home and
        compared against the message, which must name the path and
        omit the flag.
    """
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    data_dir = str(tmp_path / '.memman')
    assert data_dir == cli_mod.default_data_dir()
    monkeypatch.setenv('MEMMAN_WORKER', '1')
    cli_mod._configure_logging(data_dir, verbose=False, debug=False)

    message = cli_mod.MemmanGroup._name_the_stack('it broke')

    assert 'log worker --stack' in message
    assert '--data-dir' not in message
    assert str(Path(data_dir) / 'logs' / 'memman.log') in message


def test_migrate_keeps_the_store_name_on_a_bare_backend_error(
        runner, monkeypatch):
    """A backend failure in the per-store loop still names the store.

    `gather` reaches the backend, so it raises a bare `BackendError`
    outside `except MigrateError`, and under `--all` the operator needs
    the store name to tell which store failed.

    Both types are reachable and the handler must catch each.
    `apply` and `_verify_destination_counts` reach the Postgres
    connection scope and raise `BackendError`; `SqliteMigrator.gather`
    runs its selects unwrapped, so a store that opens and then fails
    mid-read raises a bare `sqlite3.Error`. The sqlite leg is the one
    every `--to postgres` migration takes.

    Mutation: narrowing the handler to `except MigrateError`, or to
        `except (MigrateError, BackendError)` -- which looks complete
        and still drops the store name on the sqlite-source leg every
        `--to postgres` run takes.
    Oracle: the store name in the output, per raised type. The raw
        driver message alone does not carry it.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'alpha')
    _stub_postgres(monkeypatch)

    for raised in (
            BackendError('postgres query failed: connection reset'),
            sqlite3.OperationalError('no such table: oplog')):
        def _boom(self, store, exc=raised):
            raise exc

        monkeypatch.setattr(SqliteMigrator, 'gather', _boom)
        result = invoke(
            runner, ['migrate', '--store', 'alpha', '--to', 'postgres',
                     '--dry-run'])

        assert result.exit_code != 0, result.output
        assert not isinstance(
            result.exception, (BackendError, sqlite3.Error)), result.output
        assert 'alpha:' in result.output, (type(raised), result.output)


def test_swap_reports_a_post_cutover_failure_as_completed(
        runner, monkeypatch):
    """A failure after the one-way cutover must not read as retryable.

    `run_swap` writes the fingerprint in one transaction with its own
    state cleanup, so a `BackendError` from the CLI's confirming
    re-read means the data cutover already landed.

    Both types are reachable here, and `sqlite3.Error` is the one the
    DEFAULT backend raises: `stored_fingerprint` reaches `DB._query`,
    which deliberately does not translate. Catching `BackendError`
    alone covers Postgres only.

    Mutation: dropping the guard, or narrowing it to `except
        BackendError` -- which passes a Postgres-shaped test while the
        default backend still prints the generic pre-cutover line, the
        reading that makes an operator retry a one-way migration.
    Oracle: COMPLETED and the re-run guarantee in the output, per
        raised type; the generic seam message carries neither.
    """
    class _Stub:
        model = 'stub-target'
        dim = 768

        def available(self):
            return True

        def prepare(self):
            return

        def unavailable_message(self):
            return ''

    monkeypatch.setattr(
        sched_mod, 'read_state', lambda *a, **kw: sched_mod.STATE_STOPPED)
    monkeypatch.setattr(ec_registry, 'get_for', lambda *a, **kw: _Stub())
    monkeypatch.setattr(
        swap_mod, 'run_swap',
        lambda *a, **kw: swap_mod.SwapProgress(
            state=swap_mod.STATE_DONE, cursor='',
            target_provider='stub', target_model='stub-target',
            target_dim=768))
    for raised in (
            BackendError('postgres query failed: connection reset'),
            sqlite3.OperationalError('database is locked')):
        def _boom(*a, exc=raised, **kw):
            raise exc

        monkeypatch.setattr(fp_mod, 'stored_fingerprint', _boom)
        result = invoke(runner, ['embed', 'swap', '--to', 'stub-target'])

        assert result.exit_code != 0, result.output
        assert not isinstance(
            result.exception, (BackendError, sqlite3.Error)), result.output
        assert 'COMPLETED' in result.output, (type(raised), result.output)
        assert 're-embeds nothing' in result.output
