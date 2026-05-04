"""Phase 3.6 -- HNSW-uses-pgvector smoke test.

Asserts that a Postgres `intent_aware_recall` invocation issues at
least one SQL statement containing the pgvector distance operator
`<=>`. If true, the production recall path is using the HNSW index
backed by pgvector. If false, the recall path is falling through to
`backend.nodes.iter_embeddings_as_vecs()` + Python-side cosine
(`vector_search_from_cache`) and the pgvector index is dead code.

This test is currently `xfail`: at HEAD the pipeline reads
`session.snapshot` (None on Postgres), exits the `recall_session`
context immediately, and routes all reads through Backend verbs on
the primary connection rather than through `RecallSession` verbs.
`PostgresRecallSession.vector_anchors()` is implemented but never
called. The Phase 1a docstring in `src/memman/store/backend.py`
(class `RecallSession`) promises typed verbs land in Phase 1b but
they were never added; this test gates that future delivery.

When the Phase 1b verbs are wired into `intent_aware_recall`, this
test should start passing. The `xfail` will then convert to an
XPASS warning at collection — that's the signal to remove the
`xfail` marker.
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


@pytest.mark.xfail(
    reason='HNSW dead-strand: pipeline reads session.snapshot=None on '
           'Postgres then exits recall_session, routing all reads '
           'through Backend verbs (Python-side cosine via '
           'vector_search_from_cache). PostgresRecallSession.'
           'vector_anchors() is orphaned. Phase 1b RecallSession verbs '
           '(keyword_anchors, vector_anchors, neighbors, hydrate, '
           'similarity, causal_neighbors) per backend.py:401-407 '
           'docstring promise were never delivered. This test gates '
           'that future delivery; when it XPASSes, remove the xfail.',
    strict=True)
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
