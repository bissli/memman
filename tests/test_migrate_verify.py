"""Post-commit verification tests for `migrate_store_to_postgres`.

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
    from memman.migrate import SchemaState
    from memman.store.postgres import _store_schema
    from tests._migrate_helpers import migrate_store_to_postgres

    store = 'mig_verify_ok'
    sdir = tmp_path / store
    _seed_store_with_rows(sdir, n_rows=3)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        result = migrate_store_to_postgres(
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

    Wraps `PostgresMigrator.apply` to delete one row before commit;
    the verify-counts step in the migrate helper catches the
    discrepancy and raises.
    """
    from memman.migrate import MigrateError, SchemaState
    from memman.store.postgres import PostgresMigrator, _connection
    from memman.store.postgres import _store_schema
    from tests._migrate_helpers import migrate_store_to_postgres

    store = 'mig_verify_mismatch'
    sdir = tmp_path / store
    _seed_store_with_rows(sdir, n_rows=4)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    real_apply = PostgresMigrator.apply

    def short_apply(self, store_arg, payload):
        real_apply(self, store_arg, payload)
        with _connection(self.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'delete from {schema}.insights'
                    f' where id = (select id from {schema}.insights'
                    f' order by id limit 1)')

    try:
        with patch.object(
                PostgresMigrator, 'apply', new=short_apply):
            with pytest.raises(MigrateError, match='verif'):
                migrate_store_to_postgres(
                    source_dir=str(sdir), dsn=pg_dsn, store=store,
                    state=SchemaState.ABSENT)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
