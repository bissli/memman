"""Tests for source-dim resolution in the SQLite -> Postgres migrate path.

Slice 1.1 of the per-store backend routing plan: the migrate path must
resolve the embedding dim from the source store's
`meta.embed_fingerprint`, not a hardcoded 512.
"""

import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
import pytest
from memman.store.db import _BASELINE_SCHEMA


def test_decode_embedding_accepts_dim_parameter():
    """_decode_embedding decodes a float64 blob of the requested dim.
    """
    from scripts.import_sqlite_to_postgres import _decode_embedding
    vec = [0.1] * 1024
    blob = struct.pack(f'<{len(vec)}d', *vec)
    out = _decode_embedding(blob, dim=1024)
    assert out is not None
    assert len(out) == 1024


def test_decode_embedding_returns_none_for_null():
    """`None`/empty blob returns None regardless of expected dim.
    """
    from scripts.import_sqlite_to_postgres import _decode_embedding
    assert _decode_embedding(None, dim=512) is None
    assert _decode_embedding(b'', dim=1024) is None


def test_decode_embedding_raises_on_dim_mismatch():
    """Wrong-size blob raises ValueError, not a silent skip.
    """
    from scripts.import_sqlite_to_postgres import _decode_embedding
    vec = [0.1] * 256
    blob = struct.pack(f'<{len(vec)}d', *vec)
    with pytest.raises(ValueError, match='256.*1024'):
        _decode_embedding(blob, dim=1024)


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
                'INSERT INTO insights (id, content, category, importance,'
                ' entities, source, access_count, embedding, created_at,'
                ' updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}', 'fact', 3,
                 '[]', 'user', 0,
                 struct.pack(f'<{dim}d', *vec), now, now))
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":' +
             str(dim) + '}'))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.postgres
def test_migrate_resolves_non_512_dim_from_source(pg_dsn, tmp_path):
    """A 1024-dim source store yields a `vector(1024)` destination.
    """
    from memman.migrate import SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_dim_1024'
    sdir = tmp_path / store
    _seed_store(sdir, dim=1024, n_rows=3)

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
    """A source row whose blob size disagrees with the fingerprint dim
    surfaces as ValueError, not a silent vector loss.
    """
    from memman.migrate import MigrateError, SchemaState, migrate_store
    from memman.store.postgres import _store_schema

    store = 'mig_mixed_dim'
    sdir = tmp_path / store
    sdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        good = [0.1] * 512
        bad = [0.2] * 256
        for i, vec in enumerate([good, bad]):
            conn.execute(
                'INSERT INTO insights (id, content, category, importance,'
                ' entities, source, access_count, embedding, created_at,'
                ' updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), f'row-{i}', 'fact', 3, '[]',
                 'user', 0, struct.pack(f'<{len(vec)}d', *vec), now, now))
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()

    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as cn:
        with cn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    try:
        with pytest.raises(MigrateError, match='256|dim'):
            migrate_store(
                source_dir=str(sdir), dsn=pg_dsn, store=store,
                state=SchemaState.ABSENT)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as cn:
            with cn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
