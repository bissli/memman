"""`memman migrate --to sqlite` (postgres -> sqlite) tests.

Verifies the postgres-to-sqlite migration orchestration: pg_dump
pre-flight, target-dir-absent guard, copy-to-tmp + atomic rename,
postgres schema archive, env-key flip, and
warn-only schema drop. Round-trip preservation of insight ids,
edge keys, and oplog ids (via `coalesce(legacy_id, id)`) is the
cornerstone test.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip('psycopg')

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
            last_accessed_at=None)
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
    from memman.store.db import open_db, store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_round'
    _seed_sqlite_store(tmp_path, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(tmp_path), store)
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        rev_src.preflight_source(store)
        rev_payload = rev_src.gather(store)
        rev_tgt = SqliteMigrator(str(tmp_path))
        rev_tgt.preflight_target(store)
        rev_tgt.apply(store, rev_payload)
        assert len(rev_payload.insights) == 1
        assert len(rev_payload.meta) >= 1

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
    from memman.store.db import open_db, store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_ids'
    _seed_sqlite_store(tmp_path, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(tmp_path), store)
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        rev_src.preflight_source(store)
        rev_payload = rev_src.gather(store)
        rev_tgt = SqliteMigrator(str(tmp_path))
        rev_tgt.preflight_target(store)
        rev_tgt.apply(store, rev_payload)

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
    from memman.store.db import open_db, store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

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

        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        rev_src.preflight_source(store)
        rev_payload = rev_src.gather(store)
        rev_tgt = SqliteMigrator(str(tmp_path))
        rev_tgt.preflight_target(store)
        rev_tgt.apply(store, rev_payload)

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


_FIDELITY_ROW = {
    'id': 'rb-fid-1',
    'content': 'field fidelity round-trip subject',
    'category': 'decision',
    'importance': 4,
    'entities': ['alpha', 'beta'],
    'source': 'fidelity-test',
    'access_count': 11,
    'keywords': ['kw-one', 'kw-two'],
    'summary': 'the summary text',
    'semantic_facts': ['fact-one'],
    'last_accessed_at': datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    'linked_at': datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
    'enriched_at': datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc),
    'created_at': datetime(2026, 4, 5, 6, 7, 8, tzinfo=timezone.utc),
    'updated_at': datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc),
    'deleted_at': None,
    'prompt_version': 'pv-aaaa',
    'model_id': 'mid-bbbb',
    'embedding_model': 'em-cccc',
    'session_id': 'sess-dddd',
    'queue_uuid': 'quuid-eeee',
    'corroboration_count': 7,
    'superseded_by': 'rb-fid-successor',
    }


def _seed_fidelity_store(data_dir: Path, store: str) -> Path:
    """Write one insight whose every column carries a distinct value."""
    import json

    from memman.store.db import open_db, set_meta, store_dir
    from memman.store.model import format_timestamp
    r = _FIDELITY_ROW
    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        db.conn.execute(
            'insert into insights ('
            ' id, content, category, importance, entities, source,'
            ' access_count, keywords, summary, semantic_facts,'
            ' last_accessed_at, linked_at, enriched_at, created_at,'
            ' updated_at, deleted_at, prompt_version, model_id,'
            ' embedding_model, session_id, queue_uuid,'
            ' corroboration_count, superseded_by)'
            ' values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
            ' ?, ?, ?, ?, ?, ?, ?, ?)',
            (r['id'], r['content'], r['category'], r['importance'],
             json.dumps(r['entities']), r['source'], r['access_count'],
             json.dumps(r['keywords']), r['summary'],
             json.dumps(r['semantic_facts']),
             format_timestamp(r['last_accessed_at']),
             format_timestamp(r['linked_at']),
             format_timestamp(r['enriched_at']),
             format_timestamp(r['created_at']),
             format_timestamp(r['updated_at']), None,
             r['prompt_version'], r['model_id'], r['embedding_model'],
             r['session_id'], r['queue_uuid'], r['corroboration_count'],
             r['superseded_by']))
        db.conn.commit()
        set_meta(db, 'embed_fingerprint',
                 '{"provider":"voyage","model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def test_round_trip_preserves_every_insight_field(tmp_path, pg_dsn):
    """Each insight column survives sqlite -> postgres -> sqlite in place.

    All four migrator halves read and write the `insights` column
    list POSITIONALLY -- two `select ... r[N]` scans whose trailing
    optional columns are addressed off a hardcoded base offset, and
    two `insert` column lists paired with a value tuple by position.
    Adding or removing one column shifts every later index, and the
    shifted read still type-checks, so the corruption is silent.
    Distinct values per column are what make it audible.

    Mutation: dropping or inserting a column in either migrator's
        `iter_for_swap` select or apply insert without moving the
        `r[N]` indices and the `idx` base offset with it -- e.g.
        `linked_at` landing in `enriched_at`, or a timestamp landing
        in `deleted_at` and soft-deleting the row.
    Oracle: `_FIDELITY_ROW`, hand-written with a different value in
        every column, compared field by field after the round-trip.
    """
    from memman.store.db import open_db, store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_fidelity'
    _seed_fidelity_store(tmp_path, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(tmp_path), store)
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)

        target = store_dir(str(tmp_path), store)
        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        rev_src.preflight_source(store)
        rev_payload = rev_src.gather(store)
        rev_tgt = SqliteMigrator(str(tmp_path))
        rev_tgt.preflight_target(store)
        rev_tgt.apply(store, rev_payload)

        assert len(rev_payload.insights) == 1
        got = rev_payload.insights[0]
        for name, want in _FIDELITY_ROW.items():
            assert getattr(got, name) == want, (
                f'postgres gather returned {name}='
                f'{getattr(got, name)!r}, want {want!r}')

        db = open_db(target)
        try:
            row = db.conn.execute(
                'select category, importance, access_count, summary,'
                ' last_accessed_at, linked_at, enriched_at, created_at,'
                ' updated_at, deleted_at, prompt_version, model_id,'
                ' embedding_model, session_id, queue_uuid,'
                ' corroboration_count, superseded_by'
                ' from insights where id = ?',
                (_FIDELITY_ROW['id'],)).fetchone()
        finally:
            db.close()
        r = _FIDELITY_ROW
        assert tuple(row) == (
            r['category'], r['importance'], r['access_count'],
            r['summary'], '2026-01-02T03:04:05Z',
            '2026-02-03T04:05:06Z', '2026-03-04T05:06:07Z',
            '2026-04-05T06:07:08Z', '2026-05-06T07:08:09Z', None,
            r['prompt_version'], r['model_id'], r['embedding_model'],
            r['session_id'], r['queue_uuid'], r['corroboration_count'],
            r['superseded_by'])
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_to_sqlite_errors_when_schema_missing(tmp_path, pg_dsn):
    """Reverse migrate of a non-existent schema raises MigrateError."""
    from memman.migrate import MigrateError
    from memman.store.postgres import PostgresMigrator

    store = 'rb_missing'
    _drop_schema(pg_dsn, store)
    rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
    with pytest.raises(MigrateError, match='does not exist'):
        rev_src.preflight_source(store)


def test_migrate_to_sqlite_errors_when_fingerprint_missing(
        tmp_path, pg_dsn):
    """Schema without meta.embed_fingerprint raises MigrateError."""
    from memman.migrate import MigrateError
    from memman.store.postgres import PostgresMigrator, _store_schema

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
        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        with pytest.raises(MigrateError, match='embed_fingerprint'):
            rev_src.preflight_source(store)
    finally:
        _drop_schema(pg_dsn, store)


def test_migrate_cli_to_sqlite_archives_dump_and_drops_schema(
        tmp_path, env_file, pg_dsn):
    """CLI reverse migrate writes dump.pgdump, drops schema, flips env."""
    from memman.cli import cli
    from memman.store.db import store_dir
    from memman.store.postgres import PostgresMigrator, _store_schema
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_cli_full'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        src_mig = SqliteMigrator(str(data_dir))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(data_dir), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_' + store, pg_dsn)

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
        assert f'MEMMAN_POSTGRES_DSN_{store}=' not in env_text
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


def test_migrate_cli_to_sqlite_errors_when_pg_dump_missing(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """`shutil.which('pg_dump') is None` -> ClickException with install hint."""
    from memman.cli import cli
    from memman.store.db import store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_nopgdump'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        src_mig = SqliteMigrator(str(data_dir))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(data_dir), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_' + store, pg_dsn)

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


def test_migrate_cli_to_postgres_errors_when_pg_dump_missing(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """Forward (sqlite -> postgres) migrate also requires pg_dump.

    Reverse migration is always a possibility after a forward run; the
    operator must have `pg_dump` available before any postgres-touching
    migration so a roll-back path exists. The gate fires at command
    entry, before any DB work or filesystem mutation.
    """
    import shutil as _shutil

    from memman.cli import cli

    store = 'fwd_nopgdump'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)

    real_which = _shutil.which

    def fake_which(name, *args, **kwargs):
        if name == 'pg_dump':
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr('shutil.which', fake_which)
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)

    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '--data-dir', str(data_dir),
            'migrate', '--to', 'postgres',
            '--store', store, '--yes'],
        catch_exceptions=False)
    assert result.exit_code != 0
    assert 'pg_dump' in result.output
    assert 'postgresql-client' in result.output


def test_migrate_cli_to_sqlite_refuses_when_target_dir_exists(
        tmp_path, env_file, pg_dsn):
    """Pre-existing data/<store>/ guards against accidental overwrite."""
    from memman.cli import cli
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_target_exists'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        src_mig = SqliteMigrator(str(data_dir))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(data_dir), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_' + store, pg_dsn)

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
    from memman.store.db import store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_already_pg'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        src_mig = SqliteMigrator(str(data_dir))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(data_dir), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_' + store, pg_dsn)

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


def test_migrate_cli_to_sqlite_drop_failure_is_warn_only(
        tmp_path, env_file, pg_dsn, monkeypatch):
    """Drop-schema failure logs a warning but completes successfully."""
    from memman.cli import cli
    from memman.store.db import store_dir
    from memman.store.postgres import PostgresMigrator
    from memman.store.sqlite import SqliteMigrator

    store = 'rb_dropfail'
    data_dir = tmp_path / 'memman'
    _seed_sqlite_store(data_dir, store)
    _drop_schema(pg_dsn, store)
    try:
        source = store_dir(str(data_dir), store)
        src_mig = SqliteMigrator(str(data_dir))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(data_dir), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(source)
        env_file('MEMMAN_BACKEND_' + store, 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_' + store, pg_dsn)

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
    env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
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
