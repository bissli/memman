"""Post-commit verification tests for `migrate_store`.

Slice 1.3: after the destination commit lands, row counts in each
destination table must match the captured source counts. A mismatch
raises `MigrateError` and `MigrateResult.verified` is True only on
strict equality.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest
from memman.store.db import _BASELINE_SCHEMA

pytestmark = pytest.mark.postgres


def _seed_store_with_rows(store_dir: Path, n_rows: int = 4) -> None:
    """Build a SQLite store with `n_rows` insights and a fingerprint.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        for i in range(n_rows):
            vec = [0.1 * (i + 1)] * 512
            conn.execute(
                'INSERT INTO insights (id, content, category, importance,'
                ' entities, source, access_count, embedding, created_at,'
                ' updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}', 'fact', 3,
                 '[]', 'user', 0,
                 struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()


def test_migrate_result_marks_verified_on_count_match(pg_dsn, tmp_path):
    """Happy-path migrate sets `MigrateResult.verified = True`.
    """
    from memman.migrate import SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_verify_ok'
    sdir = tmp_path / store
    _seed_store_with_rows(sdir, n_rows=3)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        result = migrate_store(
            source_dir=str(sdir), dsn=pg_dsn, store=store,
            state=SchemaState.ABSENT)
        assert result.verified is True
        assert result.insights == 3
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_migrate_raises_on_destination_count_mismatch(pg_dsn, tmp_path):
    """If the destination ends up short, MigrateError is raised.

    Patch `_import_insights` to skip one row; verify step catches it.
    """
    from memman.migrate import MigrateError, SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_verify_mismatch'
    sdir = tmp_path / store
    _seed_store_with_rows(sdir, n_rows=4)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    from scripts import import_sqlite_to_postgres as imp
    real = imp._import_insights

    def short(sqlite_conn, pg_conn, schema, dim):
        rows = sqlite_conn.execute(
            'SELECT id FROM insights LIMIT 1').fetchone()
        if rows:
            with pg_conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM {schema}.insights WHERE id = %s',
                    (rows[0],))
        n = real(sqlite_conn, pg_conn, schema, dim)
        with pg_conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM {schema}.insights'
                f' WHERE id = (SELECT id FROM {schema}.insights'
                f' ORDER BY id LIMIT 1)')
        return n

    try:
        with patch.object(imp, '_import_insights', side_effect=short):
            with pytest.raises(MigrateError, match='verif'):
                migrate_store(
                    source_dir=str(sdir), dsn=pg_dsn, store=store,
                    state=SchemaState.ABSENT)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
