"""Oplog idempotency tests for migrate.

Slice 1.4: oplog rows from SQLite have a stable `id` that we copy
into a `legacy_id BIGINT UNIQUE` column on the destination. Re-running
migrate after a partial failure must not duplicate oplog rows.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from memman.store.db import _BASELINE_SCHEMA

pytestmark = pytest.mark.postgres


def _seed_store_with_oplog(
        store_dir: Path, n_oplog: int = 5) -> list[int]:
    """Build a SQLite store with `n_oplog` oplog rows.

    Returns the source ids in insertion order.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    ids = []
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        vec = [0.5] * 512
        ins_id = str(uuid.uuid4())
        conn.execute(
            'INSERT INTO insights (id, content, category, importance,'
            ' entities, source, access_count, embedding, created_at,'
            ' updated_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (ins_id, 'oplog test', 'fact', 3, '[]', 'user', 0,
             struct.pack(f'<{len(vec)}d', *vec), now, now))
        for i in range(n_oplog):
            cur = conn.execute(
                'INSERT INTO oplog (operation, insight_id, detail,'
                ' created_at) VALUES (?, ?, ?, ?)',
                (f'op-{i}', ins_id, f'detail-{i}', now))
            ids.append(cur.lastrowid)
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()
    return ids


def test_oplog_table_has_legacy_id_column(pg_dsn, tmp_path):
    """After migrate, the destination oplog has a `legacy_id` column.
    """
    from memman.migrate import SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_oplog_legacy'
    sdir = tmp_path / store
    _seed_store_with_oplog(sdir, n_oplog=3)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        migrate_store(
            source_dir=str(sdir), dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'select column_name from'
                    ' information_schema.columns'
                    ' where table_schema = %s'
                    " and table_name = 'oplog'"
                    " and column_name = 'legacy_id'",
                    (schema,))
                assert cur.fetchone() is not None
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_oplog_legacy_id_matches_source_id(pg_dsn, tmp_path):
    """Migrated oplog rows have `legacy_id = source.id`.
    """
    from memman.migrate import SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_oplog_match'
    sdir = tmp_path / store
    src_ids = _seed_store_with_oplog(sdir, n_oplog=4)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        migrate_store(
            source_dir=str(sdir), dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'select legacy_id from {schema}.oplog'
                    f' order by legacy_id')
                got = [r[0] for r in cur.fetchall()]
        assert got == sorted(src_ids)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_import_oplog_twice_does_not_duplicate_rows(pg_dsn, tmp_path):
    """Calling `_import_oplog` twice on the same source yields N rows.

    Simulates the partial-failure-then-resume case: a first run wrote
    rows but committed before crashing; a second run re-imports and
    must not duplicate. The ON CONFLICT (legacy_id) DO NOTHING clause
    is what makes this idempotent.
    """
    from pgvector.psycopg import register_vector
    from scripts.import_sqlite_to_postgres import (
        _ensure_schema, _import_oplog,
        )

    store = 'mig_oplog_twice'
    sdir = tmp_path / store
    src_ids = _seed_store_with_oplog(sdir, n_oplog=4)
    schema = f'staging_{store}'
    sqlite_conn = sqlite3.connect(
        f'file:{sdir / "memman.db"}?mode=ro', uri=True)
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
        try:
            with psycopg.connect(pg_dsn, autocommit=False) as pg_conn:
                register_vector(pg_conn)
                _ensure_schema(pg_conn, schema, dim=512)
                first = _import_oplog(sqlite_conn, pg_conn, schema)
                second = _import_oplog(sqlite_conn, pg_conn, schema)
                pg_conn.commit()
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'select count(*) from {schema}.oplog')
                    got = int(cur.fetchone()[0])
            assert first == len(src_ids)
            assert second == len(src_ids)
            assert got == len(src_ids), (
                f'expected {len(src_ids)} oplog rows after rerun,'
                f' got {got}')
        finally:
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'drop schema if exists {schema} cascade')
    finally:
        sqlite_conn.close()
