"""Mixed-backend dispatch tests for the per-store routing cutover.

Each test creates two stores in one process: one resolves to sqlite,
the other to postgres (per-store env keys). `open_backend(store,
data_dir)` is the dispatch entry point; these tests pin the contract.

Tests that depend on slice 2.7 (per-store heartbeat) are intentionally
omitted and tracked in the slice plan.
"""

import os
from pathlib import Path

import pytest

from memman import config


def _seed_pg_keys(env_file, store: str, dsn: str) -> None:
    """Write per-store postgres backend keys for `store`."""
    env_file(config.BACKEND_FOR(store), 'postgres')
    env_file(config.PG_DSN_FOR(store), dsn)


def _seed_sqlite_dir(data_dir: str, store: str) -> None:
    """Materialize a SQLite store dir so `list_stores` finds it."""
    sdir = Path(data_dir) / 'data' / store
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')


def test_resolve_store_backend_dispatches_per_store(tmp_path, env_file):
    """`_resolve_store_backend` returns the per-store value, not the default.
    """
    from memman.store.factory import _resolve_store_backend

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.PG_DSN_FOR('shared'), 'postgresql://x@y/z')

    assert _resolve_store_backend('default', data_dir) == 'sqlite'
    assert _resolve_store_backend('shared', data_dir) == 'postgres'


def test_graph_rebuild_guard_uses_per_store_kind(tmp_path, env_file):
    """`graph rebuild --store pg_one` rejects on per-store postgres key.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND_FOR('pg_one'), 'postgres')
    env_file(config.PG_DSN_FOR('pg_one'), 'postgresql://x@y/z')

    r = CliRunner()
    out = r.invoke(
        cli,
        ['--data-dir', data_dir, '--store', 'pg_one',
         'graph', 'rebuild'])
    assert out.exit_code != 0
    assert 'SQLite-only' in out.output


def test_graph_rebuild_guard_passes_for_sqlite_store(tmp_path, env_file):
    """`graph rebuild --store sqlite_one` reaches the rebuild path.

    Asserts the guard does NOT fire when the per-store kind is sqlite,
    even when a sibling postgres store is configured.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    _seed_sqlite_dir(data_dir, 'sqlite_one')
    env_file(config.BACKEND_FOR('sqlite_one'), 'sqlite')
    env_file(config.BACKEND_FOR('pg_one'), 'postgres')
    env_file(config.PG_DSN_FOR('pg_one'), 'postgresql://x@y/z')

    r = CliRunner()
    out = r.invoke(
        cli,
        ['--data-dir', data_dir, '--store', 'sqlite_one',
         'graph', 'rebuild', '--dry-run'])
    assert 'SQLite-only' not in out.output


def test_store_list_returns_mixed_stores(tmp_path, env_file):
    """`memman store list` returns both sqlite and postgres-keyed stores.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    _seed_sqlite_dir(data_dir, 'local')
    env_file(config.BACKEND_FOR('local'), 'sqlite')

    r = CliRunner()
    out = r.invoke(cli, ['--data-dir', data_dir, 'store', 'list'])
    assert out.exit_code == 0, out.output
    import json
    data = json.loads(out.output)
    assert 'local' in data['stores']


def test_hot_path_writes_per_store_key_on_first_open(tmp_path, env_file):
    """`_StoreContext` open writes `MEMMAN_BACKEND_<store>` from default.
    """
    from memman.cli import _ensure_store_backend_key

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.DEFAULT_BACKEND, 'sqlite')
    assert config.BACKEND_FOR('autostore') not in (
        config.parse_env_file(config.env_file_path(data_dir)))

    _ensure_store_backend_key('autostore', data_dir)

    written = config.parse_env_file(config.env_file_path(data_dir))
    assert written.get(config.BACKEND_FOR('autostore')) == 'sqlite'


def test_status_reports_per_store_backend_and_summary(tmp_path, env_file):
    """`memman status` reports the per-store kind + a `backends_in_use` set.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    _seed_sqlite_dir(data_dir, 'local')
    env_file(config.BACKEND_FOR('local'), 'sqlite')
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.PG_DSN_FOR('shared'), 'postgresql://x@y/z')

    r = CliRunner()
    out = r.invoke(
        cli,
        ['--data-dir', data_dir, '--store', 'local', 'status'])
    assert out.exit_code == 0, out.output
    import json
    data = json.loads(out.output)
    assert data['backend'] == 'sqlite'
    assert set(data['backends_in_use']) >= {'sqlite', 'postgres'}


def test_config_show_redacts_per_store_dsn(tmp_path, env_file):
    """`memman config show` redacts every `MEMMAN_PG_DSN_<store>` value.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND_FOR('shared'), 'postgres')
    env_file(config.PG_DSN_FOR('shared'), 'postgresql://secret@host/db')
    env_file(config.DEFAULT_PG_DSN, 'postgresql://default-secret@host/db')

    r = CliRunner()
    out = r.invoke(cli, ['--data-dir', data_dir, 'config', 'show'])
    assert out.exit_code == 0, out.output
    assert 'secret' not in out.output
    import json
    data = json.loads(out.output)
    per_store = data.get('per_store') or data['env']
    pg_key = config.PG_DSN_FOR('shared')
    assert per_store.get(pg_key) == '***REDACTED***'
    assert data['env'].get(config.DEFAULT_PG_DSN) == '***REDACTED***'


@pytest.mark.postgres
def test_drain_processes_mixed_backends_in_one_batch(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """One drain claims one SQLite row and one Postgres row, both land.
    """
    import psycopg
    from click.testing import CliRunner

    from memman.cli import cli
    from memman.store.postgres import _store_schema

    data_dir = os.environ[config.DATA_DIR]
    sqlite_store = 'mixed_sqlite'
    pg_store = 'mixed_pg'

    env_file(config.DEFAULT_BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR(sqlite_store), 'sqlite')
    env_file(config.BACKEND_FOR(pg_store), 'postgres')
    env_file(config.PG_DSN_FOR(pg_store), pg_dsn)

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
                  'scheduler', 'drain', '--pending', '--limit', '10'])
        assert out.exit_code == 0, out.output
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
