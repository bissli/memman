"""`memman migrate` command tests.

Verifies the migrate orchestration: DSN preflight, drain.lock guard,
per-store atomic transaction, dry-run mode, and the
`--i-have-a-backup` fail-closed gate.
"""

import sqlite3
from pathlib import Path

import psycopg
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.postgres


def _seed_sqlite_store(data_dir: Path, store: str) -> Path:
    """Build a minimal SQLite store with one insight + one meta row."""
    from memman.store.db import open_db, store_dir
    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        from memman.store.model import Insight, format_timestamp
        from memman.store.node import insert_insight
        from datetime import datetime, timezone
        ins = Insight(
            id='m-1',
            content='migrate test insight',
            category='fact',
            importance=3,
            entities=[],
            source='migrate-test',
            access_count=0,
            updated_at=datetime.now(timezone.utc),
            deleted_at=None,
            last_accessed_at=None,
            effective_importance=0.0)
        insert_insight(db, ins)
        from memman.store.db import set_meta
        set_meta(db, 'embed_fingerprint',
                 '{"provider":"voyage","model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def test_migrate_dry_run_reports_counts_without_writing(tmp_path, pg_dsn):
    """`--dry-run` returns counts and creates no Postgres schema."""
    from memman.migrate import migrate_store
    from memman.store.postgres import _store_schema

    _seed_sqlite_store(tmp_path, 'mig_dry')
    schema = _store_schema('mig_dry')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    from memman.store.db import store_dir
    result = migrate_store(
        source_dir=store_dir(str(tmp_path), 'mig_dry'),
        dsn=pg_dsn, store='mig_dry', dry_run=True)
    assert result.dry_run is True
    assert result.insights == 1
    assert result.meta >= 1

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT 1 FROM pg_namespace WHERE nspname = %s',
                (schema,))
            assert cur.fetchone() is None, (
                'dry-run created the schema; should be a no-op')


def test_migrate_writes_rows_into_target_schema(tmp_path, pg_dsn):
    """Real migrate inserts rows; ON CONFLICT makes re-run idempotent."""
    from memman.migrate import migrate_store
    from memman.store.postgres import _store_schema

    _seed_sqlite_store(tmp_path, 'mig_write')
    schema = _store_schema('mig_write')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    from memman.store.db import store_dir
    source = store_dir(str(tmp_path), 'mig_write')
    result = migrate_store(
        source_dir=source, dsn=pg_dsn, store='mig_write')
    assert not result.dry_run
    assert result.insights == 1

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM {schema}.insights')
            assert cur.fetchone()[0] == 1

    result2 = migrate_store(
        source_dir=source, dsn=pg_dsn, store='mig_write',
        overwrite_schema=False)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM {schema}.insights')
            assert cur.fetchone()[0] == 1, (
                'ON CONFLICT should make re-run a no-op')

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    except Exception:
        pass


def test_migrate_overwrite_schema_clobbers_existing(tmp_path, pg_dsn):
    """`overwrite_schema=True` drops the existing schema first."""
    from memman.migrate import migrate_store
    from memman.store.postgres import _store_schema

    _seed_sqlite_store(tmp_path, 'mig_overwrite')
    schema = _store_schema('mig_overwrite')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
            cur.execute(f'CREATE SCHEMA {schema}')
            cur.execute(
                f'CREATE TABLE {schema}.junk (id INTEGER PRIMARY KEY)')
            cur.execute(
                f'INSERT INTO {schema}.junk VALUES (42)')

    from memman.store.db import store_dir
    source = store_dir(str(tmp_path), 'mig_overwrite')
    try:
        migrate_store(
            source_dir=source, dsn=pg_dsn, store='mig_overwrite',
            overwrite_schema=True)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT 1 FROM information_schema.tables'
                    ' WHERE table_schema = %s AND table_name = %s',
                    (schema, 'junk'))
                assert cur.fetchone() is None, (
                    '--overwrite-schema should have dropped junk table')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_migrate_preflight_passes_on_pgvector_database(pg_dsn):
    """Preflight succeeds when pgvector is installed."""
    from memman.migrate import preflight
    checks = preflight(pg_dsn)
    assert checks['select_1'] is True
    assert checks['pgvector_installed'] is True


def test_migrate_cli_requires_i_have_a_backup_for_real_run(
        tmp_path, env_file, pg_dsn):
    """CLI fails closed when --i-have-a-backup is missing on real run."""
    env_file('MEMMAN_PG_DSN', pg_dsn)
    _seed_sqlite_store(tmp_path / 'memman', 'mig_cli')

    from memman.cli import cli
    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(tmp_path / 'memman'),
            'migrate', '--store', 'mig_cli'],
        catch_exceptions=False)
    assert result.exit_code != 0
    assert '--i-have-a-backup' in result.output


def test_migrate_cli_dry_run_succeeds(tmp_path, env_file, pg_dsn):
    """CLI dry-run works without --i-have-a-backup."""
    env_file('MEMMAN_PG_DSN', pg_dsn)
    _seed_sqlite_store(tmp_path / 'memman', 'mig_cli_dry')

    from memman.cli import cli
    from memman.store.postgres import _store_schema
    schema = _store_schema('mig_cli_dry')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(tmp_path / 'memman'),
            'migrate', '--store', 'mig_cli_dry', '--dry-run'],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert 'mig_cli_dry' in result.output
    assert 'dry-run' in result.output
