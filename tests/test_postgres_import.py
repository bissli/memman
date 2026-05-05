"""End-to-end import-and-parity probe for scripts/import_sqlite_to_postgres.py.

Builds a synthetic SQLite store with 50+ insights and edges, runs the
import script via subprocess against the pgvector testcontainer, then
asserts the top-5 set intersection between SQLite numpy cosine and
Postgres `SET enable_seqscan=on; ORDER BY embedding <=> $q` is >= 4/5
on a curated query set.

This is the merge-gate parity check for the bundled
test-infrastructure-plus-SQL-extraction PR.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
import pytest
from memman.store.db import _BASELINE_SCHEMA
from pgvector.psycopg import register_vector

pytestmark = pytest.mark.postgres

EMBED_DIM = 512
N_ROWS = 60


def _make_vec(seed: int) -> list[float]:
    """Deterministic 512-dim float vector for a given seed."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, EMBED_DIM).astype(np.float64).tolist()


def _serialize_blob(vec: list[float]) -> bytes:
    """Encode as memman's float64 blob format (`<Nd`)."""
    return struct.pack(f'<{EMBED_DIM}d', *vec)


def _populate_sqlite_store(store_dir: Path) -> list[tuple[str, list[float]]]:
    """Create a SQLite store with synthetic insights + embeddings.

    Returns a list of (id, vec) tuples for downstream parity checks.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    db_path = store_dir / 'memman.db'
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows = []
        ids = []
        for i in range(N_ROWS):
            iid = str(uuid.uuid4())
            ids.append(iid)
            vec = _make_vec(seed=i)
            rows.append((iid, vec))
            conn.execute(
                'INSERT INTO insights (id, content, category, importance,'
                ' entities, source, access_count, embedding, created_at,'
                ' updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (iid, f'fixture content {i}', 'fact', 3,
                 '["alpha","beta"]', 'user', 0,
                 _serialize_blob(vec), now, now))
        conn.executemany(
            'INSERT INTO edges (source_id, target_id, edge_type, weight,'
            ' metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            [
                (ids[i], ids[(i + 1) % N_ROWS], 'semantic', 0.7,
                 '{"created_by":"auto"}', now)
                for i in range(N_ROWS)
                ])
        conn.execute(
            'INSERT INTO oplog (operation, insight_id, detail, created_at)'
            ' VALUES (?, ?, ?, ?)',
            ('add', ids[0], 'fixture seed', now))
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":512}'))
        conn.commit()
    finally:
        conn.close()
    return list(zip(ids, [r[1] for r in rows]))


def _sqlite_top5(rows: list[tuple[str, list[float]]],
                 query: list[float]) -> list[str]:
    """Compute top-5 cosine similarity over the synthetic corpus in numpy."""
    matrix = np.asarray([r[1] for r in rows], dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    q = np.asarray(query, dtype=np.float64)
    q /= np.linalg.norm(q)
    sims = matrix @ q
    top_idx = np.argsort(-sims)[:5]
    return [rows[i][0] for i in top_idx]


def _pg_top5(pg_dsn: str, schema: str, query: list[float]) -> list[str]:
    """Query Postgres for top-5 cosine via seqscan (no HNSW noise)."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f'SET search_path = {schema}, public')
            cur.execute('SET enable_seqscan = on')
            qv = np.asarray(query, dtype=np.float32)
            cur.execute(
                'SELECT id FROM insights'
                ' WHERE deleted_at IS NULL'
                ' ORDER BY embedding <=> %s LIMIT 5',
                (qv,))
            return [r[0] for r in cur.fetchall()]


def test_import_script_round_trips_with_top5_parity(pg_dsn, tmp_path):
    """End-to-end: import a 60-row store, top-5 set intersection >= 4/5."""
    store_dir = tmp_path / 'fixture_store'
    rows = _populate_sqlite_store(store_dir)
    schema = 'staging_import_test'
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent
            / 'scripts' / 'import_sqlite_to_postgres.py'),
        '--source', str(store_dir),
        '--target', pg_dsn,
        '--schema', schema,
        ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ})
    assert result.returncode == 0, (
        f'import script failed: stdout={result.stdout!r}'
        f' stderr={result.stderr!r}')
    assert f'insights: imported {N_ROWS} rows' in result.stderr
    assert f'edges: imported {N_ROWS} rows' in result.stderr
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SET search_path = {schema}, public')
            cur.execute('SELECT COUNT(*) FROM insights')
            assert cur.fetchone()[0] == N_ROWS
            cur.execute('SELECT COUNT(*) FROM edges')
            assert cur.fetchone()[0] == N_ROWS
            cur.execute('SELECT COUNT(*) FROM oplog')
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT value FROM meta WHERE key = 'embed_fingerprint'")
            assert 'fixture' in cur.fetchone()[0]
    intersections = []
    for query_seed in (101, 202, 303):
        query = _make_vec(seed=query_seed)
        sqlite_top = _sqlite_top5(rows, query)
        pg_top = _pg_top5(pg_dsn, schema, query)
        intersections.append(len(set(sqlite_top) & set(pg_top)))
    assert min(intersections) >= 4, (
        f'top-5 set intersection should be >= 4/5 across queries;'
        f' got {intersections}')
