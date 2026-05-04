"""Phase 3 -- float32-vs-float64 ranking parity gate.

For each of 20 query vectors against a 60-insight synthetic corpus,
the top-5 set on Postgres (float32 via pgvector) must intersect the
top-5 set on SQLite (float64 numpy) at >= 4/5. This validates the
DB-MIGRATION.md Risks-and-open-notes claim that float-precision
differences below the noise floor on real embeddings do not visibly
perturb retrieval rank.

Gate spec at DB-MIGRATION.md:1560.
"""

import random

import pytest
from memman.embed.fingerprint import META_KEY, active_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight

N_INSIGHTS = 60
N_QUERIES = 20
EMBEDDING_DIM = 512
PARITY_FLOOR = 4

pytestmark = pytest.mark.postgres


def _seeded_vec(seed: int) -> list[float]:
    """Deterministic 512-dim unit vector (Gaussian-then-normalized).

    Uses `random.Random.gauss` instead of SHA256-bytes-as-float32 to
    guarantee finite values; pgvector rejects NaN.
    """
    rng = random.Random(seed)
    floats = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in floats) ** 0.5
    if norm > 0:
        floats = [x / norm for x in floats]
    return floats


def _populate(backend) -> None:
    """Seed 60 insights with deterministic 512-dim embeddings."""
    for i in range(N_INSIGHTS):
        ins = Insight(
            id=f'corp-{i:03d}',
            content=f'corpus document number {i} alpha bravo charlie',
            category='fact',
            importance=3,
            entities=[],
            source='parity-test',
            access_count=0,
            created_at=None,
            updated_at=None,
            deleted_at=None,
            last_accessed_at=None,
            effective_importance=0.0)
        backend.nodes.insert(ins)
        backend.nodes.update_embedding(
            ins.id, _seeded_vec(i), 'voyage-3-lite')


def _top5_ids(backend, qvec) -> set[str]:
    """Return the top-5 ids by intent-aware recall on the given backend."""
    result = intent_aware_recall(
        backend, query='corpus document',
        query_vec=qvec, query_entities=[],
        limit=5, intent_override='GENERAL')
    return {r['insight'].id for r in result['results'][:5]}


def test_float32_float64_top5_intersection_geq_4_across_20_queries(
        tmp_path, pg_dsn):
    """sqlite top-5 ∩ postgres top-5 >= 4 for each of 20 query vectors."""
    from memman.store.postgres import PostgresCluster
    from memman.store.sqlite import SqliteCluster

    sqlite_cluster = SqliteCluster()
    sqlite_backend = sqlite_cluster.open(
        store='parity', data_dir=str(tmp_path / 'memman'))
    sqlite_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(sqlite_backend)

    postgres_cluster = PostgresCluster(dsn=pg_dsn)
    try:
        postgres_cluster.drop_store(store='parity_test', data_dir='')
    except Exception:
        pass
    postgres_backend = postgres_cluster.open(
        store='parity_test', data_dir='')
    postgres_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(postgres_backend)

    try:
        failures = []
        for q in range(N_QUERIES):
            qvec = _seeded_vec(1000 + q)
            sqlite_top = _top5_ids(sqlite_backend, qvec)
            postgres_top = _top5_ids(postgres_backend, qvec)
            intersection = len(sqlite_top & postgres_top)
            if intersection < PARITY_FLOOR:
                failures.append(
                    f'query {q}: intersection={intersection}/5, '
                    f'sqlite={sorted(sqlite_top)}, '
                    f'postgres={sorted(postgres_top)}')
        assert not failures, (
            f'{len(failures)}/{N_QUERIES} queries below parity floor '
            f'{PARITY_FLOOR}/5:\n' + '\n'.join(failures))
    finally:
        try:
            sqlite_backend.close()
        except Exception:
            pass
        sqlite_cluster.close()
        try:
            postgres_backend.close()
        except Exception:
            pass
        try:
            postgres_cluster.drop_store(
                store='parity_test', data_dir='')
        except Exception:
            pass
        postgres_cluster.close()
