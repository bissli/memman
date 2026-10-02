"""Oplog idempotency tests for migrate.

Each SQLite oplog row has a stable `id`, copied into a
`legacy_id BIGINT UNIQUE` column on the destination. Re-running
migrate after a partial failure must not duplicate oplog rows.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.store.db import _BASELINE_SCHEMA
from memman.store.sqlite import SqliteMigrator

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
            'insert into insights (id, content,'
            ' embedding, created_at, updated_at)'
            ' values (?, ?, ?, ?, ?)',
            (ins_id, 'oplog test',
             struct.pack(f'<{len(vec)}d', *vec), now, now))
        for i in range(n_oplog):
            cur = conn.execute(
                'insert into oplog (operation, insight_id, detail,'
                ' created_at) values (?, ?, ?, ?)',
                (f'op-{i}', ins_id, f'detail-{i}', now))
            ids.append(cur.lastrowid)
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()
    return ids


def test_oplog_table_has_legacy_id_column(pg_dsn, tmp_path):
    """Verify the destination oplog has a `legacy_id` column.

    Mutation: omitting legacy_id from the destination oplog schema.
    Oracle: information_schema.columns for the store's schema.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_oplog_legacy'
    sdir = tmp_path / 'data' / store
    _seed_store_with_oplog(sdir, n_oplog=3)
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
    """Verify each migrated oplog row carries its source id as legacy_id.

    Mutation: numbering legacy_id afresh or leaving it NULL.
    Oracle: the source rowids collected while seeding.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_oplog_match'
    sdir = tmp_path / 'data' / store
    src_ids = _seed_store_with_oplog(sdir, n_oplog=4)
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
    """Verify applying the same payload twice leaves one row per source row.

    Mutation: dropping the `on conflict (legacy_id) do nothing` clause
        in PostgresMigrator.apply, so a resumed run doubles the rows.
    Oracle: the seeded oplog count of 4.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_oplog_twice'
    sdir = tmp_path / 'data' / store
    src_ids = _seed_store_with_oplog(sdir, n_oplog=4)
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
        tgt_mig.apply(store, payload)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'select count(*) from {schema}.oplog')
                got = int(cur.fetchone()[0])
        assert got == len(src_ids), (
            f'expected {len(src_ids)} oplog rows after rerun,'
            f' got {got}')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
