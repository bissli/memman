"""`memman migrate` command tests.

Verifies the migrate orchestration: DSN preflight, drain.lock guard,
per-store atomic transaction, dry-run mode, the interactive
confirmation flow, and the per-store env-key write
(`MEMMAN_BACKEND_<store>=postgres`) after a successful migrate.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.migrate import MigrateInsight, MigrationPayload, SchemaState
from memman.migrate import inspect_target_schemas, preflight
from memman.store.db import open_db, set_meta, store_dir
from memman.store.model import Insight
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteMigrator
from tests.conftest import invoke

psycopg = pytest.importorskip('psycopg')

from memman.store.postgres import PostgresMigrator, _store_schema
from memman.store.postgres import drop_postgres_store

pytestmark = pytest.mark.postgres


def _seed_sqlite_store(data_dir: Path, store: str) -> Path:
    """Build a minimal SQLite store with one insight + one meta row.
    """
    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        ins = Insight(
            id='m-1',
            content='migrate test insight',
            updated_at=datetime.now(timezone.utc),
            deleted_at=None)
        insert_insight(db, ins)
        set_meta(db, 'embed_fingerprint',
                 '{"model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def test_migrate_dry_run_reports_counts_without_writing(
        tmp_path, env_file, pg_dsn):
    """`migrate --dry-run` reports the source counts and changes nothing.

    Mutation: the dry-run branch falling through to the apply path, which
        creates the target schema, flips MEMMAN_BACKEND_<store> and archives
        the source; or the plan dropping the insight or meta counts.
    Oracle: the hand-written count line for one seeded insight, then the
        target schema absent in pg_namespace, the source db still in place
        and the store's backend key unset.
    """
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
    data_dir = tmp_path / 'memman'
    sdir = _seed_sqlite_store(data_dir, 'mig_dry')
    schema = _store_schema('mig_dry')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    result = CliRunner().invoke(
        cli, [
            '--data-dir', str(data_dir),
            'migrate', '--store', 'mig_dry', '--dry-run'],
        catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert 'mig_dry: insights=1 oplog=' in result.output
    assert '(dry-run)' in result.output
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select 1 from pg_namespace where nspname = %s',
                (schema,))
            assert cur.fetchone() is None
    assert (sdir / 'memman.db').exists()
    assert config.get(config.BACKEND_FOR('mig_dry')) != 'postgres'


def test_migrate_writes_rows_into_target_schema(tmp_path, pg_dsn):
    """Real migrate inserts the source rows into the target schema.

    Mutation: PostgresMigrator.apply skipping the insights insert, or writing
        to the wrong schema.
    Oracle: a direct count(*) on the target schema after apply, against one
        seeded insight.
    """
    _seed_sqlite_store(tmp_path, 'mig_write')
    schema = _store_schema('mig_write')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    src = SqliteMigrator(str(tmp_path))
    src.preflight_source('mig_write')
    payload = src.gather('mig_write')
    tgt = PostgresMigrator(dsn=pg_dsn)
    tgt.preflight_target('mig_write')
    tgt.apply('mig_write', payload)
    assert len(payload.insights) == 1

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'select count(*) from {schema}.insights')
            assert cur.fetchone()[0] == 1

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
    except Exception:
        pass


def test_migrate_populated_state_drops_and_recreates(tmp_path, pg_dsn):
    """A populated target schema is dropped and rebuilt.

    Mutation: drop_postgres_store leaving existing tables behind, so stale rows
        survive a migrate.
    Oracle: information_schema shows the pre-created junk table gone after
        apply.
    """
    _seed_sqlite_store(tmp_path, 'mig_overwrite')
    schema = _store_schema('mig_overwrite')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
            cur.execute(f'create schema {schema}')
            cur.execute(
                f'create table {schema}.junk (id integer primary key)')
            cur.execute(
                f'insert into {schema}.junk values (42)')

    try:
        src = SqliteMigrator(str(tmp_path))
        src.preflight_source('mig_overwrite')
        payload = src.gather('mig_overwrite')
        drop_postgres_store('mig_overwrite', pg_dsn)
        tgt = PostgresMigrator(dsn=pg_dsn)
        tgt.preflight_target('mig_overwrite')
        tgt.apply('mig_overwrite', payload)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'select 1 from information_schema.tables'
                    ' where table_schema = %s and table_name = %s',
                    (schema, 'junk'))
                assert cur.fetchone() is None, (
                    'POPULATED state should have dropped junk table')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_inspect_target_schemas_classifies_states(tmp_path, pg_dsn):
    """Each store maps to ABSENT, EMPTY or POPULATED.

    Mutation: inspect_target_schemas treating an empty schema as populated, or
        a missing one as empty.
    Oracle: hand-built schemas: one with an insights table, one bare, one never
        created.
    """
    pop_schema = _store_schema('mig_inspect_pop')
    empty_schema = _store_schema('mig_inspect_empty')
    absent = 'mig_inspect_absent'
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {pop_schema} cascade')
            cur.execute(f'drop schema if exists {empty_schema} cascade')
            cur.execute(
                f'drop schema if exists {_store_schema(absent)} cascade')
            cur.execute(f'create schema {pop_schema}')
            cur.execute(
                f'create table {pop_schema}.insights (id text)')
            cur.execute(f'create schema {empty_schema}')
    try:
        states = inspect_target_schemas(
            pg_dsn, ['mig_inspect_pop', 'mig_inspect_empty', absent])
        assert states['mig_inspect_pop'] is SchemaState.POPULATED
        assert states['mig_inspect_empty'] is SchemaState.EMPTY
        assert states[absent] is SchemaState.ABSENT
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {pop_schema} cascade')
                cur.execute(
                    f'drop schema if exists {empty_schema} cascade')


def test_migrate_preflight_passes_on_pgvector_database(pg_dsn):
    """Preflight passes on a database with pgvector installed.

    Mutation: preflight reporting pgvector missing, or skipping the select 1
        probe.
    Oracle: both checks true against the live test database.
    """
    checks = preflight(pg_dsn)
    assert checks['select_1'] is True
    assert checks['pgvector_installed'] is True


def test_migrate_cli_requires_confirmation_for_real_run(
        tmp_path, env_file, pg_dsn):
    """The CLI aborts when the confirmation prompt is declined.

    Mutation: the migrate command running without --yes and without asking.
    Oracle: answer n to the prompt; click prints Aborted and exits nonzero.
    """
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
    _seed_sqlite_store(tmp_path / 'memman', 'mig_cli')
    schema = _store_schema('mig_cli')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(tmp_path / 'memman'),
            'migrate', '--store', 'mig_cli'],
        input='n\n', catch_exceptions=False)
    assert result.exit_code != 0
    assert 'Aborted' in result.output


def test_migrate_cli_yes_flag_skips_prompt(
        tmp_path, env_file, pg_dsn):
    """--yes migrates without a prompt and writes the per-store backend keys.

    Mutation: migrate not writing the backend or DSN key, or leaving the sqlite
        store in place.
    Oracle: env file text, a single archive slot holding memman.db, and the
        data dir gone.
    """
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, 'mig_cli_yes')
    schema = _store_schema('mig_cli_yes')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    runner = CliRunner()
    try:
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--store', 'mig_cli_yes', '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert 'MEMMAN_BACKEND_mig_cli_yes' in result.output
        assert '(verified)' in result.output
        assert 'Archived source to' in result.output
        env_text = (data_dir / 'env').read_text()
        assert 'MEMMAN_BACKEND_mig_cli_yes=postgres' in env_text
        assert f'MEMMAN_POSTGRES_DSN_mig_cli_yes={pg_dsn}' in env_text
        archive_root = data_dir / 'archive' / 'mig_cli_yes'
        slots = sorted(archive_root.iterdir())
        assert len(slots) == 1, f'expected 1 archive slot, got {slots}'
        assert (slots[0] / 'memman.db').exists()
        assert not (data_dir / 'data' / 'mig_cli_yes').exists()
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_migrate_cli_per_store_dsn_without_default(
        tmp_path, env_file, pg_dsn):
    """--store resolves MEMMAN_POSTGRES_DSN_<store> when no default DSN is set.

    Mutation: DSN resolution reading only the default key.
    Oracle: only the per-store key is set; the target schema holds one row
        afterward.
    """
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, 'mig_per_store')
    schema = _store_schema('mig_per_store')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    env_file('MEMMAN_POSTGRES_DSN_mig_per_store', pg_dsn)

    runner = CliRunner()
    try:
        result = runner.invoke(
            cli, [
                '--data-dir', str(data_dir),
                'migrate', '--store', 'mig_per_store', '--yes'],
            catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert '(verified)' in result.output
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'select count(*) from {schema}.insights')
                assert cur.fetchone()[0] == 1
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_migrate_cli_per_store_missing_dsn_lists_both_keys(tmp_path):
    """With no DSN set, the error names both DSN keys.

    Mutation: the error naming only one key, or omitting the set-pg-dsn hint.
    Oracle: hand-listed key names in the CLI output.
    """
    _seed_sqlite_store(tmp_path / 'memman', 'mig_no_dsn')
    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(tmp_path / 'memman'),
            'migrate', '--store', 'mig_no_dsn', '--yes'],
        catch_exceptions=False)
    assert result.exit_code != 0
    assert 'MEMMAN_POSTGRES_DSN_mig_no_dsn' in result.output
    assert 'MEMMAN_DEFAULT_POSTGRES_DSN' in result.output
    assert 'set-pg-dsn' in result.output


def test_migrate_cli_dry_run_succeeds(tmp_path, env_file, pg_dsn):
    """The CLI dry run prints the plan and exits zero with no prompt.

    Mutation: --dry-run prompting for confirmation, or exiting nonzero.
    Oracle: exit code zero, plus the plan text and the store name in the
        output.
    """
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
    _seed_sqlite_store(tmp_path / 'memman', 'mig_cli_dry')

    schema = _store_schema('mig_cli_dry')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(tmp_path / 'memman'),
            'migrate', '--store', 'mig_cli_dry', '--dry-run'],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert 'Migration plan' in result.output
    assert 'mig_cli_dry' in result.output
    assert 'dry-run' in result.output


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
    existing = json.loads(invoke(mm_runner, [
        'remember', 'The retry cap for batch jobs is three.']).output)['id']
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
