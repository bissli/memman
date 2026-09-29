"""Post-commit verification tests for sqlite -> postgres migration.

After the destination commit lands, row counts in each destination
table must match the captured source counts. A mismatch raises
`MigrateError`.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.migrate import MigrateError, _verify_destination_counts
from memman.store.db import _BASELINE_SCHEMA
from memman.store.sqlite import SqliteMigrator

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
                'insert into insights (id, content, category,'
                ' embedding, created_at, updated_at)'
                ' values (?, ?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}', 'fact',
                 struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()


def test_migrate_result_marks_verified_on_count_match(pg_dsn, tmp_path):
    """Verify destination row counts match the source after a full apply.

    Mutation: apply dropping insights, oplog, or meta rows, or the
        verifier comparing against the wrong table.
    Oracle: the counts taken from the gathered payload.
    """
    from memman.store.postgres import PostgresMigrator, _connection
    from memman.store.postgres import _store_schema

    store = 'mig_verify_ok'
    sdir = tmp_path / 'data' / store
    _seed_store_with_rows(sdir, n_rows=3)
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
        with _connection(pg_dsn, autocommit=True) as conn:
            _verify_destination_counts(
                conn, schema, store,
                expected={
                    'insights': len(payload.insights),
                    'oplog': len(payload.oplog),
                    'meta': len(payload.meta),
                    })
        assert len(payload.insights) == 3
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_migrate_raises_on_destination_count_mismatch(pg_dsn, tmp_path):
    """Verify a destination one row short raises MigrateError.

    Mutation: the count check skipping insights, or passing on a
        smaller destination count.
    Oracle: a wrapped apply that deletes one insights row, against the
        4 seeded rows in the payload.
    """
    from memman.store.postgres import PostgresMigrator, _connection
    from memman.store.postgres import _store_schema

    store = 'mig_verify_mismatch'
    sdir = tmp_path / 'data' / store
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
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        with patch.object(
                PostgresMigrator, 'apply', new=short_apply):
            tgt_mig.apply(store, payload)
        with pytest.raises(MigrateError, match='verif'):
            with _connection(pg_dsn, autocommit=True) as conn:
                _verify_destination_counts(
                    conn, schema, store,
                    expected={
                        'insights': len(payload.insights),
                        'oplog': len(payload.oplog),
                        'meta': len(payload.meta),
                        })
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
