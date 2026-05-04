"""Phase 3 -- cross-backend recall@10 regression gate.

Gate spec at DB-MIGRATION.md:1559: "Cross-backend recall@10 within
documented tolerance against curated query set."

The curated set is constructed in-test: 20 topic centers, each with
3 corpus insights whose vectors are the topic center plus small
Gaussian noise. Query vectors are the topic centers themselves.
For each query, the ground-truth top-3 is the 3 insights of the
matching topic. The 20 topics yield 60 corpus insights and 20
queries. Recall@3 (matches / 3) is computed per query; the gate
asserts the mean recall on both backends is >= 0.95 and the two
backend recalls agree within +-0.05.

Tolerance values, if loosened during development, must be
documented in the Phase 3 retrospective answer to Q2.
"""

import random

import pytest
from memman.embed.fingerprint import META_KEY, active_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight

N_TOPICS = 20
INSIGHTS_PER_TOPIC = 3
EMBEDDING_DIM = 512
NOISE_SCALE = 0.02
RECALL_FLOOR = 0.95
BACKEND_AGREEMENT_TOLERANCE = 0.05

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


def _perturb(vec: list[float], seed: int) -> list[float]:
    """Add small Gaussian noise then re-normalize."""
    rng = random.Random(seed)
    noisy = [x + rng.gauss(0.0, NOISE_SCALE) for x in vec]
    return _unit(noisy)


def _populate(backend, topic_centers: list[list[float]]) -> None:
    """Insert 3 perturbed corpus vectors per topic, 60 insights total."""
    for t_idx, center in enumerate(topic_centers):
        for k in range(INSIGHTS_PER_TOPIC):
            ins_id = f't{t_idx:02d}-i{k}'
            ins = Insight(
                id=ins_id,
                content=f'topic {t_idx} insight {k}',
                category='fact',
                importance=3,
                entities=[],
                source='recall-at-10-test',
                access_count=0,
                created_at=None,
                updated_at=None,
                deleted_at=None,
                last_accessed_at=None,
                effective_importance=0.0)
            backend.nodes.insert(ins)
            vec = _perturb(center, seed=t_idx * 100 + k)
            backend.nodes.update_embedding(ins_id, vec, 'voyage-3-lite')


def _topk_ids(backend, qvec, k) -> list[str]:
    """Return the top-k ids by intent-aware recall on the given backend."""
    result = intent_aware_recall(
        backend, query='topic insight',
        query_vec=qvec, query_entities=[],
        limit=k, intent_override='GENERAL')
    return [r['insight'].id for r in result['results'][:k]]


def _recall_at_3(backend, topic_centers: list[list[float]]) -> float:
    """Recall over 20 queries: (matches / 3) averaged."""
    total = 0.0
    for t_idx, center in enumerate(topic_centers):
        ground_truth = {
            f't{t_idx:02d}-i{k}' for k in range(INSIGHTS_PER_TOPIC)}
        retrieved = set(
            _topk_ids(backend, center, INSIGHTS_PER_TOPIC + 7))
        hits = len(ground_truth & retrieved)
        total += hits / INSIGHTS_PER_TOPIC
    return total / N_TOPICS


def test_cross_backend_recall_at_10_gate(tmp_path, pg_dsn):
    """Both backends recall >= 0.95 of ground truth, agreeing within 0.05."""
    from memman.store.postgres import PostgresCluster
    from memman.store.sqlite import SqliteCluster

    topic_centers = [_gaussian_unit(seed=i) for i in range(N_TOPICS)]

    sqlite_cluster = SqliteCluster()
    sqlite_backend = sqlite_cluster.open(
        store='r10', data_dir=str(tmp_path / 'memman'))
    sqlite_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(sqlite_backend, topic_centers)

    postgres_cluster = PostgresCluster(dsn=pg_dsn)
    try:
        postgres_cluster.drop_store(store='r10_test', data_dir='')
    except Exception:
        pass
    postgres_backend = postgres_cluster.open(
        store='r10_test', data_dir='')
    postgres_backend.meta.set(META_KEY, active_fingerprint().to_json())
    _populate(postgres_backend, topic_centers)

    try:
        sqlite_recall = _recall_at_3(sqlite_backend, topic_centers)
        postgres_recall = _recall_at_3(postgres_backend, topic_centers)

        assert sqlite_recall >= RECALL_FLOOR, (
            f'sqlite recall {sqlite_recall:.3f} below floor {RECALL_FLOOR}')
        assert postgres_recall >= RECALL_FLOOR, (
            f'postgres recall {postgres_recall:.3f} below floor '
            f'{RECALL_FLOOR}')
        delta = abs(sqlite_recall - postgres_recall)
        assert delta <= BACKEND_AGREEMENT_TOLERANCE, (
            f'sqlite recall {sqlite_recall:.3f} vs postgres recall '
            f'{postgres_recall:.3f} differ by {delta:.3f} > '
            f'{BACKEND_AGREEMENT_TOLERANCE}')
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
            postgres_cluster.drop_store(store='r10_test', data_dir='')
        except Exception:
            pass
        postgres_cluster.close()
