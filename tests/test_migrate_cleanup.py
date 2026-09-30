"""Stale-post-migrate-source doctor tests.

The migrate flow leaves the source SQLite artifacts (`memman.db`,
`memman.db-wal`, `memman.db-shm`) in place until the archive step.
Doctor's `check_stale_post_migrate_source` warns for any store whose
resolved backend is `postgres` and that still has the SQLite source on
disk.
"""

import os
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from memman.doctor import check_stale_post_migrate_source, run_all_checks
from memman.store.db import _BASELINE_SCHEMA
from memman.store.sqlite import SqliteBackend, SqliteMigrator

try:
    import psycopg
except ImportError:
    psycopg = None


def _seed_with_artifacts(store_dir: Path) -> None:
    """Build a SQLite store with its WAL/SHM side files.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.execute('pragma journal_mode=WAL')
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        vec = [0.5] * 512
        conn.execute(
            'insert into insights (id, content,'
            ' embedding, created_at, updated_at)'
            ' values (?, ?, ?, ?, ?)',
            (str(uuid.uuid4()), 'cleanup test',
             struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.postgres
def test_migrate_preserves_source_artifacts(pg_dsn, tmp_path):
    """Verify gather and apply leave the SQLite source file in place.

    Mutation: PostgresMigrator.apply or SqliteMigrator.gather deleting
        `memman.db`, which removes the copy of the pre-migrate state
        before the archive step runs.
    Oracle: the source file's existence after gather and apply.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_preserve'
    sdir = tmp_path / 'data' / store
    _seed_with_artifacts(sdir)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        assert (sdir / 'memman.db').exists()
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_doctor_warns_on_stale_post_migrate_source(
        tmp_path, env_file):
    """Verify a `memman.db` survivor in a postgres-routed store warns.

    Mutation: reporting `fail` or `pass` for a preserved source, or
        omitting the store from the detail list.
    Oracle: status 'warn' and the store name in detail['stores'].
    """
    data_dir = os.environ['MEMMAN_DATA_DIR']
    sdir = Path(data_dir) / 'data' / 'stale_store'
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')

    env_file('MEMMAN_DEFAULT_BACKEND', 'postgres')
    result = check_stale_post_migrate_source(data_dir)
    assert result['status'] == 'warn'
    assert 'stale_store' in result['detail']['stores']


def test_doctor_passes_when_postgres_store_is_clean(
        tmp_path, env_file):
    """Verify a store dir with no SQLite artifacts under postgres passes.

    Mutation: warning on every postgres-routed store directory whether
        or not `memman.db` remains.
    Oracle: status 'pass' for a directory holding no `memman.db`.
    """
    data_dir = os.environ['MEMMAN_DATA_DIR']
    sdir = Path(data_dir) / 'data' / 'clean_store'
    sdir.mkdir(parents=True, exist_ok=True)

    env_file('MEMMAN_DEFAULT_BACKEND', 'postgres')
    result = check_stale_post_migrate_source(data_dir)
    assert result['status'] == 'pass'


def test_doctor_skips_sqlite_routed_stores(tmp_path, env_file):
    """Verify a sqlite-routed store holding `memman.db` is not flagged.

    Mutation: flagging every `memman.db` regardless of the store's
        resolved backend, which warns on live sqlite stores.
    Oracle: status 'pass' with the default backend set to sqlite.
    """
    data_dir = os.environ['MEMMAN_DATA_DIR']
    sdir = Path(data_dir) / 'data' / 'sqlite_store'
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')

    env_file('MEMMAN_DEFAULT_BACKEND', 'sqlite')
    result = check_stale_post_migrate_source(data_dir)
    assert result['status'] == 'pass'


def test_run_all_checks_includes_stale_post_migrate_source(
        tmp_db, tmp_path, env_file):
    """Verify run_all_checks includes the stale-source check.

    Mutation: leaving the check out of the run_all_checks registry.
    Oracle: the check name in the output's checks list.
    """
    backend = SqliteBackend(tmp_db)
    out = run_all_checks(backend, str(tmp_path / 'memman'))
    names = [c['name'] for c in out['checks']]
    assert 'stale_post_migrate_source' in names
