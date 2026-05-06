"""Tests for `bootstrap_per_store_keys(data_dir)`.

Slice 2.3: convert a legacy install (global `MEMMAN_BACKEND` /
`MEMMAN_PG_DSN`) to the per-store key shape. Idempotent and
operator-edit-preserving.
"""

import os
from pathlib import Path

import pytest

from memman import config


def _read_env(data_dir: str) -> dict[str, str]:
    return config.parse_env_file(config.env_file_path(data_dir))


def test_bootstrap_no_globals_returns_empty(tmp_path, env_file):
    """No global `MEMMAN_BACKEND`/`MEMMAN_PG_DSN` set -> no-op.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND, None)
    env_file(config.PG_DSN, None)
    actions = bootstrap_per_store_keys(data_dir)
    assert actions == []


def test_bootstrap_legacy_sqlite_writes_per_store_keys(tmp_path, env_file):
    """Two local SQLite dirs -> writes per-store backend keys + strips global.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    Path(data_dir, 'data', 'one').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'one', 'memman.db').write_bytes(b'')
    Path(data_dir, 'data', 'two').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'two', 'memman.db').write_bytes(b'')

    env_file(config.BACKEND, 'sqlite')

    actions = bootstrap_per_store_keys(data_dir)

    written = _read_env(data_dir)
    assert written.get(config.BACKEND_FOR('one')) == 'sqlite'
    assert written.get(config.BACKEND_FOR('two')) == 'sqlite'
    assert written.get(config.DEFAULT_BACKEND) == 'sqlite'
    assert config.BACKEND not in written
    assert any('MEMMAN_BACKEND_one' in a for a in actions)


def test_bootstrap_is_idempotent(tmp_path, env_file):
    """Re-run is a no-op once globals are stripped.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    Path(data_dir, 'data', 'one').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'one', 'memman.db').write_bytes(b'')

    env_file(config.BACKEND, 'sqlite')

    bootstrap_per_store_keys(data_dir)
    actions = bootstrap_per_store_keys(data_dir)
    assert actions == []


def test_bootstrap_does_not_overwrite_operator_edits(tmp_path, env_file):
    """If operator already wrote `MEMMAN_BACKEND_<store>`, leave it alone.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    Path(data_dir, 'data', 'special').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'special', 'memman.db').write_bytes(b'')

    env_file(config.BACKEND, 'sqlite')
    env_file(config.BACKEND_FOR('special'), 'postgres')

    bootstrap_per_store_keys(data_dir)
    written = _read_env(data_dir)
    assert written.get(config.BACKEND_FOR('special')) == 'postgres'


def test_bootstrap_rejects_invalid_store_dirs(tmp_path, env_file):
    """Non-`valid_store_name` directories under data/ are skipped.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    Path(data_dir, 'data', 'good').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'good', 'memman.db').write_bytes(b'')
    Path(data_dir, 'data', '.cache').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', '.cache', 'memman.db').write_bytes(b'')

    env_file(config.BACKEND, 'sqlite')
    bootstrap_per_store_keys(data_dir)

    written = _read_env(data_dir)
    assert config.BACKEND_FOR('good') in written
    assert config.BACKEND_FOR('.cache') not in written


def test_bootstrap_strips_global_pg_dsn(tmp_path, env_file):
    """Global `MEMMAN_PG_DSN` is removed even when no postgres stores enumerate.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND, 'sqlite')
    env_file(config.PG_DSN, 'postgresql://stale@localhost/db')

    bootstrap_per_store_keys(data_dir)

    written = _read_env(data_dir)
    assert config.PG_DSN not in written


def test_bootstrap_postgres_missing_psycopg_warns(
        tmp_path, env_file, monkeypatch):
    """Postgres legacy install with `psycopg` ImportError: warn, no writes.
    """
    import builtins

    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys

    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND, 'postgres')
    env_file(config.PG_DSN, 'postgresql://nope@localhost/db')

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith('psycopg') or 'postgres' in name:
            raise ImportError(f'blocked: {name}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)

    actions = bootstrap_per_store_keys(data_dir)
    written = _read_env(data_dir)
    assert any('psycopg' in a.lower() or 'skip' in a.lower() for a in actions)
    assert config.BACKEND not in written
    assert config.PG_DSN not in written


@pytest.mark.postgres
def test_bootstrap_legacy_postgres_writes_per_store_dsn(
        tmp_path, env_file, pg_dsn):
    """Legacy postgres install: enumerates schemas, writes per-store keys + DSN.
    """
    import psycopg

    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys
    from memman.store.postgres import _store_schema

    schema_a = _store_schema('boot_a')
    schema_b = _store_schema('boot_b')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema_a} cascade')
            cur.execute(f'drop schema if exists {schema_b} cascade')
            cur.execute(f'create schema {schema_a}')
            cur.execute(f'create schema {schema_b}')
    try:
        data_dir = os.environ[config.DATA_DIR]
        env_file(config.BACKEND, 'postgres')
        env_file(config.PG_DSN, pg_dsn)

        bootstrap_per_store_keys(data_dir)

        written = _read_env(data_dir)
        assert written.get(config.BACKEND_FOR('boot_a')) == 'postgres'
        assert written.get(config.BACKEND_FOR('boot_b')) == 'postgres'
        assert written.get(config.PG_DSN_FOR('boot_a')) == pg_dsn
        assert written.get(config.PG_DSN_FOR('boot_b')) == pg_dsn
        assert written.get(config.DEFAULT_BACKEND) == 'postgres'
        assert written.get(config.DEFAULT_PG_DSN) == pg_dsn
        assert config.BACKEND not in written
        assert config.PG_DSN not in written
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema_a} cascade')
                cur.execute(f'drop schema if exists {schema_b} cascade')


def test_migrate_keys_cli_command_exists(tmp_path, env_file):
    """`memman config migrate-keys` is wired and runs the bootstrap.
    """
    from click.testing import CliRunner

    from memman.cli import cli

    data_dir = os.environ[config.DATA_DIR]
    Path(data_dir, 'data', 'cliboot').mkdir(parents=True, exist_ok=True)
    Path(data_dir, 'data', 'cliboot', 'memman.db').write_bytes(b'')

    env_file(config.BACKEND, 'sqlite')

    runner = CliRunner()
    result = runner.invoke(cli, ['config', 'migrate-keys'])
    assert result.exit_code == 0, result.output

    written = _read_env(data_dir)
    assert written.get(config.BACKEND_FOR('cliboot')) == 'sqlite'
    assert config.BACKEND not in written
