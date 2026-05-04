"""Phase 3 -- float32-vs-float64 ranking parity gate.

For each of 20 query vectors against a 60-insight synthetic corpus,
the top-5 set on Postgres (float32 via pgvector) must intersect the
top-5 set on SQLite (float64 numpy) at >= 4/5. This validates the
DB-MIGRATION.md Risks-and-open-notes claim that float-precision
differences below the noise floor on real embeddings do not visibly
perturb retrieval rank.

Gate spec at DB-MIGRATION.md:1560.

The corpus is structured (20 topic centers x 3 insights per topic)
not random-against-random, so cosines for the matching topic clear
`VECTOR_SEARCH_MIN_SIM=0.10`. A random corpus would produce cosines
near 0 in 512-dim space, which the bi-encoder threshold filters out
and starves the parity check of signal.
"""

import random

import pytest
from memman.embed.fingerprint import META_KEY, active_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight

N_TOPICS = 12
INSIGHTS_PER_TOPIC = 5
N_INSIGHTS = N_TOPICS * INSIGHTS_PER_TOPIC
N_QUERIES = 20
EMBEDDING_DIM = 512
CORPUS_NOISE = 0.02
QUERY_NOISE = 0.05
PARITY_FLOOR = 4

pytestmark = pytest.mark.postgres


def _unit(vec: list[float]) -> list[float]:
    """Normalize to unit length."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def _gaussian_unit(seed: int) -> list[float]:
    """Deterministic 512-dim unit Gaussian vector."""
    rng = random.Random(seed)
    return _unit([rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)])


def _perturb(vec: list[float], seed: int, scale: float) -> list[float]:
    """Add Gaussian noise then re-normalize."""
    rng = random.Random(seed)
    noisy = [x + rng.gauss(0.0, scale) for x in vec]
    return _unit(noisy)


def _populate(backend, topic_centers: list[list[float]]) -> None:
    """Insert 3 perturbed corpus vectors per topic, 60 insights total."""
    for t_idx, center in enumerate(topic_centers):
        for k in range(INSIGHTS_PER_TOPIC):
            ins_id = f't{t_idx:02d}-i{k}'
            ins = Insight(
                id=ins_id,
                content=f'topic {t_idx} insight {k} alpha bravo charlie',
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
            vec = _perturb(center, seed=t_idx * 100 + k, scale=CORPUS_NOISE)
            backend.nodes.update_embedding(ins_id, vec, 'voyage-3-lite')


def _top5_ids(backend, qvec) -> set[str]:
    """Return the top-5 ids by intent-aware recall on the given backend."""
    result = intent_aware_recall(
        backend, query='topic insight',
        query_vec=qvec, query_entities=[],
        limit=5, intent_override='GENERAL')
    return {r['insight'].id for r in result['results'][:5]}


def test_float32_float64_top5_intersection_geq_4_across_20_queries(
        tmp_path, pg_dsn):
    """sqlite top-5 ∩ postgres top-5 >= 4 for each of 20 query vectors."""
    from memman.store.postgres import PostgresCluster
    from memman.store.sqlite import SqliteCluster

    topic_centers = [_gaussian_unit(seed=i) for i in range(N_TOPICS)]

    sqlite_cluster = SqliteCluster()
    sqlite_backend = sqlite_cluster.open(
        store='parity', data_dir=str(tmp_path / 'memman'))
    sqlite_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(sqlite_backend, topic_centers)

    postgres_cluster = PostgresCluster(dsn=pg_dsn)
    try:
        postgres_cluster.drop_store(store='parity_test', data_dir='')
    except Exception:
        pass
    postgres_backend = postgres_cluster.open(
        store='parity_test', data_dir='')
    postgres_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(postgres_backend, topic_centers)

    try:
        failures = []
        for q in range(N_QUERIES):
            qvec = _perturb(
                topic_centers[q % N_TOPICS],
                seed=10000 + q, scale=QUERY_NOISE)
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
