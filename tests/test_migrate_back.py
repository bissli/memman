"""`memman migrate --to sqlite` (postgres -> sqlite) tests.

Verifies the postgres-to-sqlite migration orchestration: pg_dump
pre-flight, target-dir-absent guard, copy-to-tmp + atomic rename,
postgres schema archive, env-key flip, snapshot regeneration, and
warn-only schema drop. Round-trip preservation of insight ids,
edge keys, and oplog ids (via `coalesce(legacy_id, id)`) is the
cornerstone test.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.postgres


def _seed_sqlite_store(data_dir: Path, store: str) -> Path:
    """Build a minimal SQLite store with one insight + one meta row."""
    from memman.store.db import open_db, set_meta, store_dir
    from memman.store.model import Insight
    from memman.store.node import insert_insight
    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        ins = Insight(
            id=f'rb-{store}-1',
            content='reverse migrate test insight',
            category='fact',
            importance=3,
            entities=['alpha', 'beta'],
            source='migrate-back-test',
            access_count=0,
            updated_at=datetime.now(timezone.utc),
            deleted_at=None,
            last_accessed_at=None,
            effective_importance=0.0)
        insert_insight(db, ins)
        set_meta(db, 'embed_fingerprint',
                 '{"provider":"voyage","model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def _drop_schema(pg_dsn: str, store: str) -> None:
    from memman.store.postgres import _store_schema
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_migrate_to_sqlite_round_trip(tmp_path, pg_dsn):
    """SQLite -> Postgres -> SQLite round-trip preserves row counts."""
    from memman.migrate import (
        SchemaState, migrate_store_to_postgres,
        migrate_store_to_sqlite,
        )
    from memman.store.db import open_db, store_dir

    store = 'rb_round'
    _seed_sqlite_store(tmp_path, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(tmp_path), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        result = migrate_store_to_sqlite(
            dsn=pg_dsn, target_dir=target, store=store)
        assert result.insights == 1
        assert result.meta >= 1

        db = open_db(target)
        try:
            actual = db.conn.execute(
                'select count(*) from insights').fetchone()[0]
            assert actual == 1
        finally:
            db.close()
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_to_sqlite_preserves_insight_ids(tmp_path, pg_dsn):
    """Insight ids survive the round-trip bit-exact."""
    from memman.migrate import (
        SchemaState, migrate_store_to_postgres,
        migrate_store_to_sqlite,
        )
    from memman.store.db import open_db, store_dir

    store = 'rb_ids'
    _seed_sqlite_store(tmp_path, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(tmp_path), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        migrate_store_to_sqlite(
            dsn=pg_dsn, target_dir=target, store=store)

        db = open_db(target)
        try:
            row = db.conn.execute(
                'select id from insights').fetchone()
            assert row[0] == f'rb-{store}-1'
        finally:
            db.close()
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_to_sqlite_preserves_oplog_legacy_ids(tmp_path, pg_dsn):
    """Round-trip oplog ids match the original sqlite ids via legacy_id."""
    from memman.migrate import (
        SchemaState, migrate_store_to_postgres,
        migrate_store_to_sqlite,
        )
    from memman.store.db import open_db, store_dir

    store = 'rb_oplog'
    _seed_sqlite_store(tmp_path, store)
    source = store_dir(str(tmp_path), store)
    db = open_db(source)
    try:
        db.conn.execute(
            "insert into oplog (operation, insight_id, detail, created_at)"
            " values ('insert', 'rb-X-1', 'seed', '2026-05-01T00:00:00Z')")
    finally:
        db.close()

    _drop_schema(pg_dsn, store)
    try:
        original_oplog_ids = []
        db = open_db(source)
        try:
            original_oplog_ids = [
                r[0] for r in db.conn.execute(
                    'select id from oplog order by id').fetchall()
                ]
        finally:
            db.close()

        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        migrate_store_to_sqlite(
            dsn=pg_dsn, target_dir=target, store=store)

        db = open_db(target)
        try:
            restored_ids = [
                r[0] for r in db.conn.execute(
                    'select id from oplog order by id').fetchall()
                ]
        finally:
            db.close()
        assert restored_ids == original_oplog_ids
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_to_sqlite_errors_when_schema_missing(tmp_path, pg_dsn):
    """Reverse migrate of a non-existent schema raises MigrateError."""
    from memman.migrate import MigrateError, migrate_store_to_sqlite

    store = 'rb_missing'
    _drop_schema(pg_dsn, store)
    target = tmp_path / 'data' / store
    with pytest.raises(MigrateError, match='does not exist'):
        migrate_store_to_sqlite(
            dsn=pg_dsn, target_dir=str(target), store=store)


def test_migrate_to_sqlite_errors_when_fingerprint_missing(
        tmp_path, pg_dsn):
    """Schema without meta.embed_fingerprint raises MigrateError."""
    from memman.migrate import MigrateError, migrate_store_to_sqlite
    from memman.store.postgres import _store_schema

    store = 'rb_nofp'
    schema = _store_schema(store)
    _drop_schema(pg_dsn, store)
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'create schema {schema}')
                cur.execute(
                    f'create table {schema}.meta'
                    ' (key text primary key, value text not null)')
                cur.execute(
                    f'create table {schema}.insights'
                    ' (id text primary key)')
                cur.execute(
                    f'create table {schema}.edges'
                    ' (source_id text, target_id text, edge_type text,'
                    '  primary key (source_id, target_id, edge_type))')
                cur.execute(
                    f'create table {schema}.oplog (id bigserial primary key)')
        target = tmp_path / 'data' / store
        with pytest.raises(MigrateError, match='embed_fingerprint'):
            migrate_store_to_sqlite(
                dsn=pg_dsn, target_dir=str(target), store=store)
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_archives_dump_and_drops_schema(
        tmp_path, env_file, pg_dsn):
    """CLI reverse migrate writes dump.pgdump, drops schema, flips env."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir
    from memman.store.postgres import _store_schema

    store = 'rb_cli_full'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert '(verified)' in result.output
        assert 'Archived postgres schema' in result.output
        assert f'MEMMAN_BACKEND_{store}=sqlite' in result.output

        archive_root = data_dir / 'archive' / store
        slots = sorted(archive_root.iterdir())
        assert len(slots) == 1
        dump = slots[0] / 'dump.pgdump'
        assert dump.exists()
        assert dump.stat().st_size > 0

        env_text = (data_dir / 'env').read_text()
        assert f'MEMMAN_BACKEND_{store}=sqlite' in env_text
        assert f'MEMMAN_PG_DSN_{store}=' not in env_text
        assert (data_dir / 'data' / store / 'memman.db').exists()

        schema = _store_schema(store)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'select 1 from pg_namespace where nspname = %s',
                    (schema,))
                assert cur.fetchone() is None
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_regenerates_snapshot(
        tmp_path, env_file, pg_dsn):
    """CLI reverse migrate writes recall_snapshot.v1.bin to target dir."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir
    from memman.store.snapshot import SNAPSHOT_FILENAME

    store = 'rb_snap'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        snap = data_dir / 'data' / store / SNAPSHOT_FILENAME
        assert snap.exists(), result.output
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_errors_when_pg_dump_missing(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """`shutil.which('pg_dump') is None` -> ClickException with install hint."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir

    store = 'rb_nopgdump'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        real_which = shutil.which

        def fake_which(name, *args, **kwargs):
            if name == 'pg_dump':
                return None
            return real_which(name, *args, **kwargs)

        monkeypatch.setattr('shutil.which', fake_which)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code != 0
        assert 'pg_dump' in result.output
        assert 'postgresql-client' in result.output
        assert (data_dir / 'data' / store).exists() is False
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_refuses_when_target_dir_exists(
        tmp_path, env_file, pg_dsn):
    """Pre-existing data/<store>/ guards against accidental overwrite."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir

    store = 'rb_target_exists'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code != 0
        assert 'already exists' in result.output
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_warns_when_already_sqlite(
        tmp_path, env_file, pg_dsn):
    """`--to sqlite` against a sqlite-routed store warns and exits 0."""
    from memman.cli import cli

    store = 'rb_already_sqlite'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)

    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(data_dir),
            'migrate', '--to', 'sqlite',
            '--store', store, '--yes'],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert 'already on sqlite' in result.output


def test_migrate_cli_to_postgres_warns_when_already_postgres(
        tmp_path, env_file, pg_dsn):
    """`--to postgres` against a postgres-routed store warns and exits 0."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir

    store = 'rb_already_pg'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'postgres',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert 'already on postgres' in result.output
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_cleans_leftover_tmp_dir(
        tmp_path, env_file, pg_dsn):
    """A pre-existing `<store>.tmp/` from a crashed run is cleaned at start."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir

    store = 'rb_tmpleft'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        leftover = data_dir / 'data' / f'{store}.tmp'
        leftover.mkdir(parents=True)
        (leftover / 'stale-file').write_text('garbage')

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert 'Cleaning leftover' in result.output
        assert not leftover.exists()
        assert (data_dir / 'data' / store / 'memman.db').exists()
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_drop_failure_is_warn_only(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """Drop-schema failure logs a warning but completes successfully."""
    from memman.cli import cli
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.db import store_dir

    store = 'rb_dropfail'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        migrate_store_to_postgres(
            source_dir=source, dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_PG_DSN_' + store, pg_dsn)

        def fake_drop(*args, **kwargs):
            raise RuntimeError('simulated drop failure')

        monkeypatch.setattr(
            'memman.store.postgres.drop_postgres_store', fake_drop)

        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'sqlite',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert 'failed to drop postgres schema' in result.output
        assert (data_dir / 'data' / store / 'memman.db').exists()

        env_text = (data_dir / 'env').read_text()
        assert f'MEMMAN_BACKEND_{store}=sqlite' in env_text
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_to_postgres_explicit_flag_matches_default(
        tmp_path, env_file, pg_dsn):
    """`--to postgres` is equivalent to default (no flag)."""
    from memman.cli import cli

    store = 'rb_explicit_pg'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    env_file('MEMMAN_DEFAULT_PG_DSN', pg_dsn)
    _drop_schema(pg_dsn, store)
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--to', 'postgres',
                '--store', store, '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert '(verified)' in result.output
        assert f'MEMMAN_BACKEND_{store}=postgres' in result.output
    finally:
        _drop_schema(pg_dsn, store)
