"""Stale-post-migrate-source doctor tests.

The migrate flow preserves the source SQLite artifacts
(`memman.db`, `memman.db-wal`, `memman.db-shm`,
`recall_snapshot.v1.bin`) so the operator has a forensic copy of
pre-migrate state. Doctor's `check_stale_post_migrate_source`
warns (not fails) for any store whose resolved backend is
`postgres` and that still has the SQLite source on disk.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from memman.store.db import _BASELINE_SCHEMA

try:
    import psycopg
except ImportError:
    psycopg = None


def _seed_with_artifacts(store_dir: Path) -> None:
    """Build a SQLite store with WAL/SHM/snapshot side files.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        vec = [0.5] * 512
        conn.execute(
            'INSERT INTO insights (id, content, category, importance,'
            ' entities, source, access_count, embedding, created_at,'
            ' updated_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (str(uuid.uuid4()), 'cleanup test', 'fact', 3, '[]',
             'user', 0, struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()
    (store_dir / 'recall_snapshot.v1.bin').write_bytes(b'\x00' * 16)


@pytest.mark.postgres
def test_migrate_preserves_source_artifacts(pg_dsn, tmp_path):
    """After successful verify, the four source files are preserved.

    Pre-0.14.2 the migrate flow auto-deleted the SQLite source. That
    cleanup was accidental drift introduced after the docs were
    written; preserving the source is the documented contract and
    the documented operator-driven cleanup is `rm <store>/memman.db*`
    once the postgres data is verified.
    """
    from memman.migrate import SchemaState, migrate_store_to_postgres
    from memman.store.postgres import _store_schema

    store = 'mig_preserve'
    sdir = tmp_path / store
    _seed_with_artifacts(sdir)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        migrate_store_to_postgres(
            source_dir=str(sdir), dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        assert (sdir / 'memman.db').exists()
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_doctor_warns_on_stale_post_migrate_source(
        tmp_path, env_file):
    """A `memman.db` survivor in a postgres-routed store raises `warn`.

    The artifact is intentional preservation, not corruption -- so
    the check is `warn` (operator-burden) rather than `fail`.
    """
    import os

    from memman.doctor import check_stale_post_migrate_source

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
    """A store dir with no SQLite artifacts under postgres routing passes.
    """
    import os

    from memman.doctor import check_stale_post_migrate_source

    data_dir = os.environ['MEMMAN_DATA_DIR']
    sdir = Path(data_dir) / 'data' / 'clean_store'
    sdir.mkdir(parents=True, exist_ok=True)

    env_file('MEMMAN_DEFAULT_BACKEND', 'postgres')
    result = check_stale_post_migrate_source(data_dir)
    assert result['status'] == 'pass'


def test_doctor_skips_sqlite_routed_stores(tmp_path, env_file):
    """A sqlite-routed store with `memman.db` is the source of truth,
    not a stale artifact -- do not flag.
    """
    import os

    from memman.doctor import check_stale_post_migrate_source

    data_dir = os.environ['MEMMAN_DATA_DIR']
    sdir = Path(data_dir) / 'data' / 'sqlite_store'
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')

    env_file('MEMMAN_DEFAULT_BACKEND', 'sqlite')
    result = check_stale_post_migrate_source(data_dir)
    assert result['status'] == 'pass'


def test_run_all_checks_includes_stale_post_migrate_source(
        tmp_db, tmp_path, env_file):
    """The check is registered in `run_all_checks` output (data_dir gated).
    """
    from memman.doctor import run_all_checks
    from memman.store.sqlite import SqliteBackend

    backend = SqliteBackend(tmp_db)
    out = run_all_checks(backend, str(tmp_path / 'memman'))
    names = [c['name'] for c in out['checks']]
    assert 'stale_post_migrate_source' in names
