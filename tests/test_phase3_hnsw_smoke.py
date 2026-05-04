"""Phase 3.6 -- HNSW-uses-pgvector smoke test.

Asserts that a Postgres `intent_aware_recall` invocation issues at
least one SQL statement containing the pgvector distance operator
`<=>`. If true, the production recall path is using the HNSW index
backed by pgvector. If false, the recall path is falling through to
`backend.nodes.iter_embeddings_as_vecs()` + Python-side cosine
(`vector_search_from_cache`) and the pgvector index is dead code.

The Phase 1b RecallSession verbs (`vector_anchors` on both Sqlite
and Postgres sessions) were finally wired in the cleanup-carryovers
slice; `intent_aware_recall` calls `session.vector_anchors(qvec)`
inside the `recall_session` context, which fires `embedding <=>` on
Postgres via HNSW. This test guards against regressions back to the
dead-strand state.
"""

import random

import psycopg
import pytest
from memman.embed.fingerprint import META_KEY, active_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight

EMBEDDING_DIM = 512
N_INSIGHTS = 10

pytestmark = pytest.mark.postgres


def _gaussian_unit(seed: int) -> list[float]:
    """Deterministic 512-dim unit Gaussian vector."""
    rng = random.Random(seed)
    floats = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in floats) ** 0.5
    if norm > 0:
        floats = [x / norm for x in floats]
    return floats


def test_postgres_recall_issues_pgvector_distance_operator(
        tmp_path, pg_dsn, monkeypatch):
    """Postgres `intent_aware_recall` issues SQL with pgvector `<=>`."""
    from memman.store.postgres import PostgresCluster

    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store='hnsw_smoke', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hnsw_smoke', data_dir='')
    backend.meta.set(META_KEY, active_fingerprint().to_json())

    for i in range(N_INSIGHTS):
        ins = Insight(
            id=f'hs-{i:02d}',
            content=f'document {i} alpha bravo charlie',
            category='fact',
            importance=3,
            entities=[],
            source='hnsw-smoke',
            access_count=0,
            created_at=None,
            updated_at=None,
            deleted_at=None,
            last_accessed_at=None,
            effective_importance=0.0)
        backend.nodes.insert(ins)
        backend.nodes.update_embedding(
            ins.id, _gaussian_unit(i), 'voyage-3-lite')

    captured_sql: list[str] = []
    real_execute = psycopg.Cursor.execute

    def spy(self, query, *args, **kwargs):
        captured_sql.append(str(query))
        return real_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, 'execute', spy)

    try:
        qvec = _gaussian_unit(seed=999)
        intent_aware_recall(
            backend, query='document alpha',
            query_vec=qvec, query_entities=[],
            limit=5, intent_override='GENERAL')
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            cluster.drop_store(store='hnsw_smoke', data_dir='')
        except Exception:
            pass
        cluster.close()

    pgvector_ops = [s for s in captured_sql if '<=>' in s]
    assert pgvector_ops, (
        f'pgvector distance operator <=> never appeared in '
        f'{len(captured_sql)} SQL statements during Postgres recall. '
        f'HNSW is dead code. Sample SQL: {captured_sql[:5]}')
