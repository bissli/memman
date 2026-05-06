"""Source-cleanup and orphan-doctor tests for migrate.

Slice 1.5: a successful migrate deletes the source SQLite artifacts
(memman.db, memman.db-wal, memman.db-shm, recall_snapshot.v1.bin).
Doctor's `check_orphan_storage` flags any survivor for a store whose
backend env value resolves to `postgres`.
"""

import os
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from memman.store.db import _BASELINE_SCHEMA

pytestmark = pytest.mark.postgres


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


def test_migrate_deletes_source_artifacts(pg_dsn, tmp_path):
    """After successful verify, the four source files are removed.
    """
    from memman.migrate import SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_cleanup'
    sdir = tmp_path / store
    _seed_with_artifacts(sdir)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        migrate_store(
            source_dir=str(sdir), dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        assert not (sdir / 'memman.db').exists()
        assert not (sdir / 'memman.db-wal').exists()
        assert not (sdir / 'memman.db-shm').exists()
        assert not (sdir / 'recall_snapshot.v1.bin').exists()
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_doctor_flags_orphan_after_partial_cleanup(
        tmp_path, monkeypatch):
    """If `memman.db` survives a postgres-backed store, doctor flags it.

    Simulates the crash-between-commit-and-cleanup window by leaving
    a memman.db file in place while MEMMAN_BACKEND=postgres.
    """
    from memman.doctor import check_orphan_storage

    data_dir = tmp_path / 'data_root'
    sdir = data_dir / 'data' / 'orphan_store'
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')

    monkeypatch.setenv('MEMMAN_BACKEND', 'postgres')
    result = check_orphan_storage(str(data_dir))
    assert result['status'] == 'fail'
    assert 'orphan_store' in result.get('detail', '')


def test_doctor_passes_when_postgres_store_is_clean(
        tmp_path, monkeypatch):
    """A store dir with no SQLite artifacts under postgres backend passes.
    """
    from memman.doctor import check_orphan_storage

    data_dir = tmp_path / 'data_clean'
    sdir = data_dir / 'data' / 'clean_store'
    sdir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('MEMMAN_BACKEND', 'postgres')
    result = check_orphan_storage(str(data_dir))
    assert result['status'] == 'pass'


def test_doctor_skips_when_backend_is_sqlite(tmp_path, monkeypatch):
    """SQLite backend keeps memman.db legitimately; doctor must not flag.
    """
    from memman.doctor import check_orphan_storage

    data_dir = tmp_path / 'data_sqlite'
    sdir = data_dir / 'data' / 'sqlite_store'
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'memman.db').write_bytes(b'')

    monkeypatch.setenv('MEMMAN_BACKEND', 'sqlite')
    result = check_orphan_storage(str(data_dir))
    assert result['status'] == 'pass'
