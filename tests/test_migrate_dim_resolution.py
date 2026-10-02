"""Tests for source-dim resolution in the SQLite -> Postgres migrate path.

The migrate path resolves the embedding dim from the source store's
`meta.embed_fingerprint`, with no fixed default of 512.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

psycopg = pytest.importorskip('psycopg')

from memman.migrate import MigrateError
from memman.store.db import _BASELINE_SCHEMA
from memman.store.sqlite import SqliteMigrator


def _seed_store(store_dir: Path, dim: int, n_rows: int = 3) -> None:
    """Build a SQLite store with `n_rows` insights at the given dim.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        rng = np.random.default_rng(7)
        for i in range(n_rows):
            vec = rng.uniform(-1.0, 1.0, dim).astype(np.float64).tolist()
            conn.execute(
                'insert into insights (id, content,'
                ' embedding, created_at, updated_at)'
                ' values (?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}',
                 struct.pack(f'<{dim}d', *vec), now, now))
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"model":"fixture","dim":' +
             str(dim) + '}'))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.postgres
def test_migrate_resolves_non_512_dim_from_source(pg_dsn, tmp_path):
    """Verify a 1024-dim source store yields a `vector(1024)` column.

    Mutation: the migrator creating the embedding column at a fixed 512.
    Oracle: pg_attribute atttypmod of 1024 and all 3 seeded rows copied.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_dim_1024'
    sdir = tmp_path / 'data' / store
    _seed_store(sdir, dim=1024, n_rows=3)

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
                    'select atttypmod from pg_attribute'
                    " where attrelid = (%s || '.insights')::regclass"
                    "  and attname = 'embedding'",
                    (schema,))
                row = cur.fetchone()
                assert row is not None
                assert row[0] == 1024
                cur.execute(f'select count(*) from {schema}.insights')
                assert cur.fetchone()[0] == 3
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


@pytest.mark.postgres
def test_migrate_raises_on_mixed_dim_rows(pg_dsn, tmp_path):
    """Verify a row whose blob size disagrees with the fingerprint dim fails.

    Mutation: truncating or padding the 256-dim vector to the 512-dim
        fingerprint, which loses the vector silently.
    Oracle: MigrateError matching '256|dim' from a hand-built store.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema

    store = 'mig_mixed_dim'
    sdir = tmp_path / 'data' / store
    sdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        good = [0.1] * 512
        bad = [0.2] * 256
        for i, vec in enumerate([good, bad]):
            conn.execute(
                'insert into insights (id, content,'
                ' embedding, created_at, updated_at)'
                ' values (?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}',
                 struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()

    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as cn:
        with cn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        with pytest.raises(MigrateError, match='256|dim'):
            src_mig = SqliteMigrator(str(tmp_path))
            src_mig.preflight_source(store)
            payload = src_mig.gather(store)
            tgt_mig = PostgresMigrator(dsn=pg_dsn)
            tgt_mig.preflight_target(store)
            tgt_mig.apply(store, payload)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as cn:
            with cn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
