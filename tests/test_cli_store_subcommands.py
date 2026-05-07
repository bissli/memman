"""Cross-backend `memman store {list,create,use,remove}` tests.

The 0.14.1 implementation used a sqlite-only `store_exists`
filesystem gate, so a postgres-only store was invisible to
`store list`, `store use`, and -- worst -- `store remove`, which
returned "store does not exist" while leaving the postgres schema
intact. F.1 swaps every gate to `factory.list_stores` and every
removal to `factory.drop_store`. These tests lock that contract.

Postgres-marked tests use the `pg_dsn` fixture from
`tests/fixtures/postgres.py`; they're skipped when psycopg /
testcontainers are unavailable. The `_runner_for_postgres` helper
seeds a postgres-routed store via `MEMMAN_BACKEND_<name>=postgres`
and `MEMMAN_PG_DSN_<name>=<dsn>`.
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import cli
from tests.conftest import invoke


def _make_runner_with_pg_store(tmp_path, pg_dsn, store_name):
    """Build a CliRunner whose env has `<store_name>` routed to postgres.

    Returns the standard `(runner, data_dir)` tuple.
    """
    from memman import config

    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = config.env_file_path(str(data_dir))
    env_path.parent.mkdir(parents=True, exist_ok=True)
    rows = config.parse_env_file(env_path) if env_path.exists() else {}
    rows.update({
        'OPENROUTER_API_KEY': 'mock-key-for-testing',
        'VOYAGE_API_KEY': 'mock-voyage-key-for-testing',
        f'MEMMAN_BACKEND_{store_name}': 'postgres',
        f'MEMMAN_PG_DSN_{store_name}': pg_dsn,
        'MEMMAN_DEFAULT_PG_DSN': pg_dsn,
        })
    env_path.write_text(
        '\n'.join(f'{k}={v}' for k, v in rows.items()) + '\n')
    config.reset_file_cache()
    return CliRunner(), str(data_dir)


def test_store_remove_uses_factory(monkeypatch, tmp_path, mm_runner):
    """`store remove` calls `factory.drop_store`, not raw rmtree.

    This test does not need postgres -- the contract is observable
    by patching `factory.drop_store` and asserting the call.
    """
    invoke(mm_runner, ['store', 'create', 'doomed'])

    calls: list[tuple[str, str]] = []

    def _fake_drop(name, dd):
        calls.append((name, dd))

    monkeypatch.setattr(
        'memman.store.factory.drop_store', _fake_drop)
    result = invoke(mm_runner, ['store', 'remove', '--yes', 'doomed'])
    assert result.exit_code == 0, result.output
    assert calls
    assert calls[0][0] == 'doomed'


@pytest.mark.postgres
def test_store_list_includes_postgres_only_store(tmp_path, pg_dsn):
    """A store routed to postgres appears in `store list` output.

    Pre-F.1: `cli.list_stores` was the sqlite-only filesystem
    scanner, so a postgres-only store never appeared. Now the import
    is `factory.list_stores`, which unions sqlite dirs with
    `pg_namespace`.
    """
    runner, data_dir = _make_runner_with_pg_store(tmp_path, pg_dsn, 'work')
    create = runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'create', 'work'])
    assert create.exit_code == 0, create.output
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'list'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert 'work' in payload['stores']


@pytest.mark.postgres
def test_store_use_accepts_postgres_only_store(tmp_path, pg_dsn):
    """`store use` accepts a store visible only via postgres.

    Pre-F.1: gated on `store_exists` (sqlite filesystem only) so the
    command rejected the name with "does not exist" even though the
    schema was present in the DB.
    """
    runner, data_dir = _make_runner_with_pg_store(tmp_path, pg_dsn, 'work')
    runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'create', 'work'])
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'use', 'work'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {'action': 'set', 'store': 'work'}


@pytest.mark.postgres
def test_store_remove_drops_postgres_schema(tmp_path, pg_dsn):
    """`store remove` actually drops the postgres schema.

    Pre-F.1: rejected as "does not exist", silently leaving the
    remote schema. This is the headline data-loss bug being fixed.
    """
    import psycopg
    runner, data_dir = _make_runner_with_pg_store(tmp_path, pg_dsn, 'work')
    runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'create', 'work'])
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select 1 from pg_namespace where nspname = 'store_work'")
            assert cur.fetchone() is not None
    result = runner.invoke(
        cli, ['--data-dir', data_dir,
              'store', 'remove', '--yes', 'work'])
    assert result.exit_code == 0, result.output
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select 1 from pg_namespace where nspname = 'store_work'")
            assert cur.fetchone() is None


@pytest.mark.postgres
def test_store_create_rejects_duplicate_postgres_store(tmp_path, pg_dsn):
    """A postgres-routed store cannot be re-created.

    Pre-F.1: the duplicate gate also went through `store_exists`, so
    repeat-create silently passed and re-bootstrapped the schema. The
    new gate routes through `factory.list_stores`.
    """
    runner, data_dir = _make_runner_with_pg_store(tmp_path, pg_dsn, 'work')
    first = runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'create', 'work'])
    assert first.exit_code == 0, first.output
    second = runner.invoke(
        cli, ['--data-dir', data_dir, 'store', 'create', 'work'])
    assert second.exit_code != 0
    assert 'already exists' in (second.output or '').lower()
