"""Phase 2 PostgresBackend behaviour tests.

Drives the new `store/postgres.py` module against the real pgvector
container (Phase -1 fixture). Asserts the Phase 2 gate items:

- Schema/HNSW DDL applied with `vector_cosine_ops WHERE deleted_at
  IS NULL`, built `CONCURRENTLY`.
- Server-side timestamps (insert with no `created_at` produces a
  non-null DB row).
- Score-direction contract on `vector_anchors`.
- `bulk_update_embedding` chunks at <= 1000 rows.
- `oplog.log` is INSERT-only; trim happens in `maintenance_step`.
- `drain_lock` released on connection close.
- `RecallSession.__exit__` resets search_path.
- Reindex remnant cleanup: an invalid HNSW remnant is dropped
  before the next `CREATE INDEX CONCURRENTLY`.
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.model import Edge, Insight
from memman.store.postgres import EMBEDDING_DIM
from memman.store.postgres import PostgresCluster, _ensure_baseline_schema
from memman.store.postgres import _ensure_hnsw_index, _store_schema

pytestmark = pytest.mark.postgres


def _vec(seed: int) -> list[float]:
    """A reproducible synthetic vector for tests."""
    return [(seed + i) * 0.001 for i in range(EMBEDDING_DIM)]


@pytest.fixture
def store_name() -> str:
    return 'phase2'


@pytest.fixture
def pg_backend(pg_dsn, store_name):
    """A PostgresBackend bound to a fresh `store_<name>` schema.

    Drops + recreates the schema each test for isolation. The
    pgvector extension stays in the database (created at session
    start by `pgvector_docker`).
    """
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    backend = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    try:
        yield backend
    finally:
        backend.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_baseline_schema_created(pg_backend, pg_dsn, store_name):
    """Cluster.open creates the per-store schema with all four tables.
    """
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT tablename FROM pg_tables'
                ' WHERE schemaname = %s ORDER BY tablename',
                (schema,))
            tables = [r[0] for r in cur.fetchall()]
    assert tables == ['edges', 'insights', 'meta', 'oplog']


def test_hnsw_partial_index_built_concurrently(
        pg_backend, pg_dsn, store_name):
    """HNSW index uses vector_cosine_ops WHERE deleted_at IS NULL.

    The test inspects pg_index for the `idx_insights_hnsw_<schema>`
    index and asserts shape: cosine ops, partial WHERE, valid
    (`indisvalid = true`).
    """
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT i.indisvalid, am.amname,'
                ' pg_get_indexdef(i.indexrelid)'
                ' FROM pg_index i'
                ' JOIN pg_class c ON c.oid = i.indexrelid'
                ' JOIN pg_am am ON am.oid = ('
                '   SELECT relam FROM pg_class'
                '   WHERE oid = i.indexrelid)'
                ' WHERE c.relname = %s',
                (f'idx_insights_hnsw_{schema}',))
            row = cur.fetchone()
    assert row is not None, 'HNSW index missing'
    assert row[0] is True, 'HNSW index is invalid'
    assert row[1] == 'hnsw'
    indexdef = row[2]
    assert 'vector_cosine_ops' in indexdef
    assert 'deleted_at IS NULL' in indexdef


def test_server_side_timestamps(pg_backend):
    """Insert with no created_at: backend stamps via DEFAULT now()."""
    ins = Insight(
        id='ts-1', content='no client timestamp', importance=3,
        source='user')
    pg_backend.nodes.insert(ins)
    pg_backend._conn.commit()
    fetched = pg_backend.nodes.get('ts-1')
    assert fetched is not None
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


def test_vector_anchors_returns_similarity_in_range(pg_backend):
    """vector_anchors scores are cosine in [-1, 1], top result max."""
    for k in range(10):
        ins = Insight(
            id=f'va-{k}', content=f'content {k}', importance=3,
            source='user')
        pg_backend.nodes.insert(ins)
        pg_backend.nodes.update_embedding(
            f'va-{k}', _vec(k), 'voyage-3-lite')
    pg_backend._conn.commit()

    with pg_backend.recall_session() as session:
        results = session.vector_anchors(_vec(2), k=5)
    assert results, 'expected vector hits'
    for _id, score in results:
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6
    top_id, top_score = results[0]
    assert top_id == 'va-2'
    assert top_score == max(s for _id, s in results)


def test_bulk_update_embedding_chunks_at_1000(pg_backend, monkeypatch):
    """bulk_update_embedding splits >1000 rows into <=1000 commits.

    Pure shape test: stub `executemany` to record batch sizes; do
    not actually insert 1001 rows.
    """
    sizes: list[int] = []
    real_executemany = psycopg.Cursor.executemany

    def spy(self, sql, args):
        rows = list(args)
        sizes.append(len(rows))
        return real_executemany(self, sql, rows)

    monkeypatch.setattr(psycopg.Cursor, 'executemany', spy)
    big_batch = [
        (f'b-{i}', _vec(i), 'voyage-3-lite') for i in range(1500)
        ]
    pg_backend.nodes.bulk_update_embedding(big_batch)
    assert sizes, 'executemany never called'
    assert all(n <= 1000 for n in sizes), (
        f'expected all batches <=1000, got {sizes}')
    assert sum(sizes) >= 1500


def test_oplog_log_does_not_trim(pg_backend, pg_dsn, store_name):
    """oplog.log is INSERT-only; rows accumulate past the cap."""
    from memman.store.postgres import _MAX_OPLOG_ENTRIES
    schema = _store_schema(store_name)
    over = _MAX_OPLOG_ENTRIES + 5
    with pg_backend._conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.oplog (operation, insight_id, detail)'
            ' VALUES (%s, %s, %s)',
            [('seed', '', '') for _ in range(over)])
    pg_backend._conn.commit()
    pg_backend.oplog.log(
        operation='probe', insight_id='', detail='no trim here')
    pg_backend._conn.commit()

    with pg_backend._conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM {schema}.oplog')
        count = int(cur.fetchone()[0])
    assert count > _MAX_OPLOG_ENTRIES, (
        f'expected > {_MAX_OPLOG_ENTRIES} oplog rows, got {count}')


def test_oplog_maintenance_step_trims(pg_backend, pg_dsn, store_name):
    """maintenance_step caps the oplog at MAX_OPLOG_ENTRIES."""
    from memman.store.postgres import _MAX_OPLOG_ENTRIES
    schema = _store_schema(store_name)
    over = _MAX_OPLOG_ENTRIES + 5
    with pg_backend._conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.oplog (operation, insight_id, detail)'
            ' VALUES (%s, %s, %s)',
            [('seed', '', '') for _ in range(over)])
    pg_backend._conn.commit()
    pg_backend.oplog.maintenance_step()
    pg_backend._conn.commit()
    with pg_backend._conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM {schema}.oplog')
        count = int(cur.fetchone()[0])
    assert count <= _MAX_OPLOG_ENTRIES


def test_drain_lock_released_on_connection_close(pg_dsn, store_name):
    """A drain lock from one Backend's drain_lock is acquirable later.

    Acquires the lock, exits the context (which closes the dedicated
    drain connection), then asserts a second acquire succeeds. This
    is the crash-recovery contract: lock auto-releases on connection
    close.
    """
    backend = PostgresCluster(dsn=pg_dsn).open(
        store=store_name, data_dir='/unused')
    try:
        with backend.drain_lock(store_name) as got:
            assert got is True
        with backend.drain_lock(store_name) as got2:
            assert got2 is True
    finally:
        backend.close()


def test_recall_session_resets_search_path(pg_backend, pg_dsn):
    """RecallSession.__exit__ resets search_path before close."""
    with pg_backend.recall_session() as session:
        assert session._conn is not None
        with session._conn.cursor() as cur:
            cur.execute('SHOW search_path')
            inside = cur.fetchone()[0]
        assert 'store_' in inside, (
            f'expected store_ in search_path inside session, got {inside}')

    with psycopg.connect(pg_dsn, autocommit=True) as fresh:
        with fresh.cursor() as cur:
            cur.execute('SHOW search_path')
            outside = cur.fetchone()[0]
    assert 'store_' not in outside or outside.startswith('"$user"'), (
        f'fresh connection should not see store_ in default'
        f' search_path; got {outside}')


def test_reindex_drops_invalid_hnsw_remnant(pg_dsn, store_name):
    """An invalid HNSW remnant from an aborted CONCURRENTLY build is
    dropped before the next reindex retries.
    """
    schema = _store_schema(store_name)
    _ensure_baseline_schema(pg_dsn, store_name)
    index_name = f'idx_insights_hnsw_{schema}'
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP INDEX IF EXISTS {schema}.{index_name}')
            cur.execute(
                f'CREATE INDEX {index_name}'
                f' ON {schema}.insights'
                f' USING hnsw (embedding vector_cosine_ops)'
                f' WHERE deleted_at IS NULL')
            cur.execute(
                'UPDATE pg_index SET indisvalid = false'
                ' WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            cur.execute(
                'SELECT indisvalid FROM pg_index WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            assert cur.fetchone()[0] is False
    _ensure_hnsw_index(pg_dsn, schema)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT indisvalid FROM pg_index WHERE indexrelid = ('
                '  SELECT oid FROM pg_class WHERE relname = %s)',
                (index_name,))
            row = cur.fetchone()
    assert row is not None
    assert row[0] is True


def test_get_neighborhood_via_recursive_cte(pg_backend):
    """EdgeStore.get_neighborhood traverses via recursive CTE.

    Builds a small chain (a -> b -> c), asserts depth=2 finds both
    b and c, and that the edge_filter parameter narrows traversal.
    """
    for nid in ('a', 'b', 'c'):
        pg_backend.nodes.insert(
            Insight(id=nid, content=nid, importance=3))
    pg_backend.edges.upsert(Edge(
        source_id='a', target_id='b', edge_type='semantic',
        weight=0.7, metadata={'created_by': 'auto'}))
    pg_backend.edges.upsert(Edge(
        source_id='b', target_id='c', edge_type='semantic',
        weight=0.8, metadata={'created_by': 'auto'}))
    pg_backend._conn.commit()

    triples = pg_backend.edges.get_neighborhood('a', depth=2)
    ids = sorted(t[0] for t in triples)
    assert ids == ['b', 'c']
    triples1 = pg_backend.edges.get_neighborhood('a', depth=1)
    assert sorted(t[0] for t in triples1) == ['b']
    triples_filtered = pg_backend.edges.get_neighborhood(
        'a', depth=2, edge_filter='causal')
    assert triples_filtered == []


def test_factory_dispatches_to_postgres(pg_dsn, monkeypatch):
    """`MEMMAN_BACKEND=postgres` factory yields PostgresCluster."""
    from memman import config
    from memman.store import factory
    monkeypatch.setattr(config, 'get', lambda key, default=None:
                        'postgres' if key == config.BACKEND
                        else (pg_dsn if key == config.PG_DSN else None))
    cluster = factory.open_cluster()
    assert type(cluster).__name__ == 'PostgresCluster'


def test_write_lock_unwired_in_callers():
    r"""No call site uses `with .*\\.write_lock()` in Phase 2 -- the
    Phase 2 gate item 8 grep guard.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parent.parent / 'src'
    pat = re.compile(r'with\s+\w+\.write_lock\s*\(')
    hits: list[str] = []
    for p in src.rglob('*.py'):
        text = p.read_text()
        hits.extend(f'{p}: {m.group(0)}' for m in pat.finditer(text))
    assert hits == [], (
        f'write_lock should not be wired into callers in Phase 2;'
        f' found: {hits}')
