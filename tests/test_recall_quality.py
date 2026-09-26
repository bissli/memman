"""Recall quality specification tests.

These tests define CORRECT recall behavior as behavioral invariants.
Assertions target the per-result signals dict, not fragile ordering
that depends on exact weight constants. If a test fails, the code is
wrong -- do not weaken assertions to match broken behavior.

Design constraint: ANCHOR_TOP_K=30 means all insights become recency
anchors when <30 exist. Each fixture inserts 6+ recent fillers to push
test insights below the top-30 recency cutoff.
"""

import random
from datetime import datetime, timezone

import pytest
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight
from tests.conftest import EMBEDDING_DIM, make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)
RECENT = datetime.now(timezone.utc)


def _insert_fillers(backend, count=8):
    """Insert recent filler insights with no keyword overlap to test queries."""
    for i in range(count):
        backend.nodes.insert(make_insight(
            id=f'filler-{i}',
            content=f'unrelated filler content alpha bravo {i}'))


def _find_result(results, insight_id):
    """Return the result dict for a given insight ID, or None."""
    for r in results:
        if r['insight'].id == insight_id:
            return r
    return None


class TestKeywordSignal:
    """Keyword-matching insight gets a positive keyword signal."""

    def test_keyword_match_has_positive_keyword_signal(self, backend):
        """Insight with query keywords scores high keyword signal; others do not."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='kw-match',
            content='Prometheus monitoring Grafana dashboards observability'))
        backend.nodes.insert(make_insight(
            id='kw-miss-1',
            content='SQLite database schema migration patterns'))
        backend.nodes.insert(make_insight(
            id='kw-miss-2',
            content='Docker container orchestration strategy'))

        result = intent_aware_recall(
            backend,
            query='Prometheus monitoring Grafana dashboards',
            query_vec=None, limit=20)

        match = _find_result(result['results'], 'kw-match')
        miss1 = _find_result(result['results'], 'kw-miss-1')
        miss2 = _find_result(result['results'], 'kw-miss-2')

        assert match is not None
        assert match['signals']['keyword'] > 0.5
        if miss1 is not None:
            assert miss1['signals']['keyword'] < 0.1
        if miss2 is not None:
            assert miss2['signals']['keyword'] < 0.1


class TestRelevanceOrderingSurvivesTheLimit:
    """Nothing re-sorts after the limit slice."""

    def test_results_are_score_descending(self, backend):
        """Recall returns rows in descending score order.

        Mutation: adding a reorder after the `results[:limit]` slice -
            a sort on `(created_at, score)` that moves a page already
            cut by score.
        Oracle: the returned rows sorted by `-score` independently,
            compared as an id sequence.
        """
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        for i, word in enumerate(('rollback', 'schema', 'deploy')):
            backend.nodes.insert(make_insight(
                id=f'ord-{i}',
                content=f'database production migration {word}'))
            set_created_at(backend, f'ord-{i}',
                           OLD.replace(year=2024 + i))

        result = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=20)

        got = [r['insight'].id for r in result['results']]
        want = [r['insight'].id
                for r in sorted(result['results'],
                                key=lambda r: -r['score'])]
        assert got == want


_N_TOPICS = 20
_INSIGHTS_PER_TOPIC = 3
_NOISE_SCALE = 0.02
_RECALL_FLOOR = 0.95
_BACKEND_AGREEMENT_TOLERANCE = 0.05


def _unit(vec: list) -> list:
    """Normalize to unit length."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def _gaussian_unit(seed: int) -> list:
    """Deterministic 512-dim unit Gaussian vector."""
    rng = random.Random(seed)
    return _unit([rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)])


def _perturb(vec: list, seed: int) -> list:
    """Add small Gaussian noise then re-normalize."""
    rng = random.Random(seed)
    noisy = [x + rng.gauss(0.0, _NOISE_SCALE) for x in vec]
    return _unit(noisy)


def _populate_recall(backend, topic_centers: list) -> None:
    """Insert 3 perturbed corpus vectors per topic."""
    for t_idx, center in enumerate(topic_centers):
        for k in range(_INSIGHTS_PER_TOPIC):
            ins_id = f't{t_idx:02d}-i{k}'
            ins = Insight(
                id=ins_id,
                content=f'topic {t_idx} insight {k}',
                category='fact',
                created_at=None,
                updated_at=None,
                deleted_at=None)
            backend.nodes.insert(ins)
            vec = _perturb(center, seed=t_idx * 100 + k)
            backend.nodes.update_embedding(ins_id, vec, 'voyage-3-lite')


def _topk_ids(backend, qvec, k) -> list:
    """Return the top-k ids by intent-aware recall on the given backend."""
    result = intent_aware_recall(
        backend, query='topic insight',
        query_vec=qvec,
        limit=k)
    return [r['insight'].id for r in result['results'][:k]]


def _recall_at_3(backend, topic_centers: list) -> float:
    """Recall over 20 queries: (matches / 3) averaged."""
    total = 0.0
    for t_idx, center in enumerate(topic_centers):
        ground_truth = {
            f't{t_idx:02d}-i{k}' for k in range(_INSIGHTS_PER_TOPIC)}
        retrieved = set(
            _topk_ids(backend, center, _INSIGHTS_PER_TOPIC + 7))
        hits = len(ground_truth & retrieved)
        total += hits / _INSIGHTS_PER_TOPIC
    return total / _N_TOPICS


class TestRecallAt10Gate:
    """Cross-backend recall@10 regression gate."""

    pytestmark = pytest.mark.postgres

    def test_cross_backend_recall_at_10_gate(self, tmp_path, pg_dsn):
        """Both backends recall >= 0.95 of ground truth, agreeing within 0.05.
        """
        from memman.store.postgres import drop_postgres_store
        from memman.store.postgres import open_postgres_backend
        from memman.store.sqlite import drop_sqlite_store, open_sqlite_backend

        topic_centers = [_gaussian_unit(seed=i) for i in range(_N_TOPICS)]

        sqlite_data_dir = str(tmp_path / 'memman')
        sqlite_backend = open_sqlite_backend('r10', sqlite_data_dir)
        sqlite_backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
        _populate_recall(sqlite_backend, topic_centers)

        try:
            drop_postgres_store('r10_test', pg_dsn)
        except Exception:
            pass
        postgres_backend = open_postgres_backend('r10_test', pg_dsn)
        postgres_backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
        _populate_recall(postgres_backend, topic_centers)

        try:
            sqlite_recall = _recall_at_3(sqlite_backend, topic_centers)
            postgres_recall = _recall_at_3(postgres_backend, topic_centers)

            assert sqlite_recall >= _RECALL_FLOOR, (
                f'sqlite recall {sqlite_recall:.3f} below floor {_RECALL_FLOOR}')
            assert postgres_recall >= _RECALL_FLOOR, (
                f'postgres recall {postgres_recall:.3f} below floor '
                f'{_RECALL_FLOOR}')
            delta = abs(sqlite_recall - postgres_recall)
            assert delta <= _BACKEND_AGREEMENT_TOLERANCE, (
                f'sqlite recall {sqlite_recall:.3f} vs postgres recall '
                f'{postgres_recall:.3f} differ by {delta:.3f} > '
                f'{_BACKEND_AGREEMENT_TOLERANCE}')
        finally:
            try:
                sqlite_backend.close()
            except Exception:
                pass
            try:
                drop_sqlite_store('r10', sqlite_data_dir)
            except Exception:
                pass
            try:
                postgres_backend.close()
            except Exception:
                pass
            try:
                drop_postgres_store('r10_test', pg_dsn)
            except Exception:
                pass
