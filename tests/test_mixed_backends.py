"""Mixed-backend dispatch tests for per-store routing.

Each test creates two stores in one process: one resolves to sqlite,
the other to postgres (per-store env keys). `open_backend(store,
data_dir)` is the dispatch entry point; these tests pin the contract.
"""

import json
import os
import pathlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import cli as memman_cli
from memman import config
from memman.cli import _ensure_store_backend_key, cli
from memman.store.factory import resolve_store_backend


def _seed_sqlite_dir(data_dir: str, store: str) -> None:
    """Materialize a SQLite store dir so `list_stores` finds it.
    """
    sdir = Path(data_dir) / 'data' / store
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')


def test_resolve_store_backend_dispatches_per_store(tmp_path, env_file):
    """resolve_store_backend returns the per-store value, not the default.

    Mutation: resolve_store_backend ignoring the per-store key and returning
        the default for every store.
    Oracle: a sqlite default with one store keyed to postgres, each read back
        by name.
    """
    data_dir = os.environ[config.DATA_DIR]
    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.POSTGRES_DSN_FOR('shared'), 'postgresql://x@y/z')

    assert resolve_store_backend('default', data_dir) == 'sqlite'
    assert resolve_store_backend('shared', data_dir) == 'postgres'


def test_store_list_returns_mixed_stores(tmp_path, env_file):
    """Memman store list includes a sqlite-keyed store.

    Mutation: store list enumerating only the default backend, or dropping a
        keyed local store.
    Oracle: the JSON stores field holds the seeded store name.
    """
    data_dir = os.environ[config.DATA_DIR]
    _seed_sqlite_dir(data_dir, 'local')
    env_file(config.BACKEND_FOR('local'), 'sqlite')

    r = CliRunner()
    out = r.invoke(cli, ['--data-dir', data_dir, 'store', 'list'])
    assert out.exit_code == 0, out.output
    data = json.loads(out.output)
    assert 'local' in data['stores']


def test_hot_path_writes_per_store_key_on_first_open(tmp_path, env_file):
    """The first open of a store writes its per-store backend key.

    Mutation: _ensure_store_backend_key not persisting the default, so a later
        default change reroutes the store.
    Oracle: env file read back before and after the call.
    """
    data_dir = os.environ[config.DATA_DIR]
    env_file(config.DEFAULT_BACKEND, 'sqlite')
    assert config.BACKEND_FOR('autostore') not in (
        config.parse_env_file(config.env_file_path(data_dir)))

    _ensure_store_backend_key('autostore', data_dir)

    written = config.parse_env_file(config.env_file_path(data_dir))
    assert written.get(config.BACKEND_FOR('autostore')) == 'sqlite'


def test_status_reports_per_store_backend_and_summary(tmp_path, env_file):
    """Memman status reports the store backend and backends_in_use.

    Mutation: status reporting the default backend instead of the per-store
        one, or a backends_in_use that omits postgres.
    Oracle: a sqlite store beside a postgres-keyed store; JSON payload fields.
    """
    data_dir = os.environ[config.DATA_DIR]
    _seed_sqlite_dir(data_dir, 'local')
    env_file(config.BACKEND_FOR('local'), 'sqlite')
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.POSTGRES_DSN_FOR('shared'), 'postgresql://x@y/z')

    r = CliRunner()
    out = r.invoke(
        cli,
        ['--data-dir', data_dir, '--store', 'local', 'status'])
    assert out.exit_code == 0, out.output
    payload_start = out.output.find('{')
    payload_end = out.output.rfind('}')
    data = json.loads(out.output[payload_start:payload_end + 1])
    assert data['backend'] == 'sqlite'
    assert set(data['backends_in_use']) >= {'sqlite', 'postgres'}


def test_config_show_redacts_per_store_dsn(tmp_path, env_file):
    """Memman config show redacts every Postgres DSN value.

    Mutation: the redaction covering only the default DSN key and leaking the
        per-store one.
    Oracle: the raw secret substrings absent from the output, redaction marker
        present.
    """
    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.POSTGRES_DSN_FOR('shared'), 'postgresql://secret@host/db')
    env_file(config.DEFAULT_PG_DSN, 'postgresql://default-secret@host/db')

    r = CliRunner()
    out = r.invoke(cli, ['--data-dir', data_dir, 'config', 'show'])
    assert out.exit_code == 0, out.output
    assert 'secret' not in out.output
    data = json.loads(out.output)
    per_store = data.get('per_store') or data['env']
    pg_key = config.POSTGRES_DSN_FOR('shared')
    assert per_store.get(pg_key) == '***REDACTED***'
    assert data['env'].get(config.DEFAULT_PG_DSN) == '***REDACTED***'


@pytest.mark.postgres
@pytest.mark.no_auto_drain
def test_drain_processes_mixed_backends_in_one_batch(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """One drain stores a sqlite row and a postgres row from one queue.

    Mutation: drain resolving one backend for the whole batch, so the other
        store's queued row is never written.
    Oracle: one live insight row in the sqlite file and one in the postgres
        schema after a single drain over two queued remembers.
    """
    import psycopg
    from memman.store.postgres import _store_schema

    data_dir = os.environ[config.DATA_DIR]
    sqlite_store = 'mixed_sqlite'
    pg_store = 'mixed_pg'

    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR(sqlite_store), 'sqlite')
    env_file(config.BACKEND_FOR(pg_store), 'postgres')
    env_file(config.POSTGRES_DSN_FOR(pg_store), pg_dsn)

    schema = _store_schema(pg_store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    try:
        r = CliRunner()
        for store_name in (sqlite_store, pg_store):
            out = r.invoke(
                cli,
                ['--data-dir', data_dir, '--store', store_name,
                 'remember', f'note for {store_name}'])
            assert out.exit_code == 0, out.output

        out = r.invoke(
            cli, ['--data-dir', data_dir,
                  'scheduler', 'drain', '--limit', '10'])
        assert out.exit_code == 0, out.output

        sqlite_db = pathlib.Path(data_dir) / 'data' / sqlite_store / 'memman.db'
        with closing(sqlite3.connect(sqlite_db)) as sconn:
            sqlite_count = sconn.execute(
                'select count(*) from insights'
                ' where deleted_at is null').fetchone()[0]
        with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f'select count(*) from {schema}.insights'
                ' where deleted_at is null')
            pg_count = cur.fetchone()[0]
        assert (sqlite_count, pg_count) == (1, 1)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


@pytest.mark.postgres
@pytest.mark.no_auto_drain
def test_cross_backend_retry(tmp_path, env_file, pg_dsn, monkeypatch):
    """A retried row routes to the backend keyed at retry time.

    Mutation: a queue row or context caching the backend from its first
        attempt, so the retry lands on sqlite.
    Oracle: the row count in the postgres schema after the retry, and two
        process calls.
    """
    import psycopg
    from memman.store.postgres import _store_schema

    data_dir = os.environ[config.DATA_DIR]
    store = 'xfer'

    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR(store), 'sqlite')

    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    try:
        clock = [1_000_000.0]

        def _fake_time():
            return clock[0]

        monkeypatch.setattr('memman.queue.time.time', _fake_time)

        attempts_seen = [0]
        real_process = memman_cli._process_queue_row

        def flaky_process(row, ctx):
            attempts_seen[0] += 1
            if attempts_seen[0] == 1:
                raise RuntimeError('transient failure on sqlite leg')
            return real_process(row, ctx)

        monkeypatch.setattr(
            'memman.cli._process_queue_row', flaky_process)

        r = CliRunner()
        out = r.invoke(
            cli, ['--data-dir', data_dir, '--store', store,
                  'remember', 'cross-backend retry note'])
        assert out.exit_code == 0, out.output

        out = r.invoke(
            cli, ['--data-dir', data_dir,
                  'scheduler', 'drain', '--limit', '1'])
        assert out.exit_code == 0, out.output
        assert attempts_seen[0] == 1, (
            f'expected one process call on first drain,'
            f' saw {attempts_seen[0]}')

        env_file(config.BACKEND_FOR(store), 'postgres')
        env_file(config.POSTGRES_DSN_FOR(store), pg_dsn)
        config.reset_file_cache()

        clock[0] += 65

        out = r.invoke(
            cli, ['--data-dir', data_dir,
                  'scheduler', 'drain', '--limit', '1'])
        assert out.exit_code == 0, out.output
        assert attempts_seen[0] == 2, (
            f'expected a second process call after backend swap,'
            f' saw {attempts_seen[0]}')

        with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f'select count(*) from {schema}.insights'
                ' where deleted_at is null')
            count = cur.fetchone()[0]
        assert count > 0, (
            f'expected the retried row to land in {schema}.insights;'
            f' the per-drain _StoreContext should have routed to the'
            f' new postgres backend')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


@pytest.mark.postgres
def test_default_sqlite_with_work_postgres(
        tmp_path, env_file, pg_dsn):
    """A sqlite default beside a postgres-routed store works end to end.

    Mutation: a remember to the postgres-keyed store writing to sqlite, or to
        the default store writing nowhere.
    Oracle: a sqlite db file for default and a nonzero row count in the work
        schema.
    """
    import psycopg
    from memman.store.postgres import _store_schema

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR('work'), 'postgres')
    env_file(config.POSTGRES_DSN_FOR('work'), pg_dsn)

    schema = _store_schema('work')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    try:
        r = CliRunner()
        r.invoke(
            cli, ['--data-dir', data_dir, '--store', 'default',
                  'remember', 'sqlite-side fact'])
        r.invoke(
            cli, ['--data-dir', data_dir, '--store', 'work',
                  'remember', 'postgres-side fact'])

        sqlite_db = pathlib.Path(data_dir) / 'data' / 'default' / 'memman.db'
        assert sqlite_db.exists(), (
            f'expected sqlite store at {sqlite_db}')

        with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f'select count(*) from {schema}.insights'
                ' where deleted_at is null')
            count = cur.fetchone()[0]
        assert count > 0, (
            f'expected at least one row in {schema}.insights')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
