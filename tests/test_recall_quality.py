"""Recall quality specification tests.

These tests define CORRECT recall behavior as behavioral invariants.
Assertions target the per-result signals dict, not fragile ordering
that depends on exact weight constants. If a test fails, the code is
wrong -- do not weaken assertions to match broken behavior.

Design constraint: ANCHOR_TOP_K=30 means all insights become recency
anchors when <30 exist. Each fixture inserts 6+ recent fillers to push
test insights below the top-30 recency cutoff, ensuring graph traversal
is actually exercised.
"""

import random
from datetime import datetime, timedelta, timezone

import pytest
from memman.embed.fingerprint import META_KEY, active_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.model import Insight
from tests.conftest import EMBEDDING_DIM, make_edge, make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)
RECENT = datetime.now(timezone.utc)


def _insert_fillers(backend, count=8):
    """Insert recent filler insights with no keyword overlap to test queries."""
    for i in range(count):
        backend.nodes.insert(make_insight(
            id=f'filler-{i}',
            content=f'unrelated filler content alpha bravo {i}',
            importance=3))


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
            content='Prometheus monitoring Grafana dashboards observability',
            importance=3))
        backend.nodes.insert(make_insight(
            id='kw-miss-1',
            content='SQLite database schema migration patterns',
            importance=3))
        backend.nodes.insert(make_insight(
            id='kw-miss-2',
            content='Docker container orchestration strategy',
            importance=3))

        result = intent_aware_recall(
            backend,
            query='Prometheus monitoring Grafana dashboards',
            query_vec=None, query_entities=[], limit=20)

        match = _find_result(result['results'], 'kw-match')
        miss1 = _find_result(result['results'], 'kw-miss-1')
        miss2 = _find_result(result['results'], 'kw-miss-2')

        assert match is not None
        assert match['signals']['keyword'] > 0.5
        if miss1 is not None:
            assert miss1['signals']['keyword'] < 0.1
        if miss2 is not None:
            assert miss2['signals']['keyword'] < 0.1


class TestEntitySignal:
    """Entity-matching insights get positive entity signal."""

    def test_entity_match_has_positive_entity_signal(self, backend):
        """Docker insights score entity signal; Kubernetes-only does not."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='ent-docker-1',
            content='Docker container networking configuration',
            entities=['Docker'], importance=3))
        backend.nodes.insert(make_insight(
            id='ent-docker-2',
            content='Docker image optimization techniques',
            entities=['Docker'], importance=3))
        backend.nodes.insert(make_insight(
            id='ent-k8s',
            content='Kubernetes pod scheduling policies',
            entities=['Kubernetes'], importance=3))

        result = intent_aware_recall(
            backend,
            query='Docker container',
            query_vec=None, query_entities=['Docker'],
            limit=20, intent_override='ENTITY')

        d1 = _find_result(result['results'], 'ent-docker-1')
        d2 = _find_result(result['results'], 'ent-docker-2')
        k8s = _find_result(result['results'], 'ent-k8s')

        assert d1 is not None
        assert d1['signals']['entity'] > 0
        assert d2 is not None
        assert d2['signals']['entity'] > 0
        if k8s is not None:
            assert k8s['signals']['entity'] == 0


class TestGraphTraversal:
    """Graph edges discover insights unreachable by keyword or recency."""

    def test_graph_traversal_discovers_unreachable_insight(self, backend):
        """Insight with no keyword overlap found via graph edges only."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='graph-1',
            content='FastAPI rate limiting design patterns',
            entities=['FastAPI'], importance=3))
        backend.nodes.insert(make_insight(
            id='graph-2',
            content='API throttling middleware implementation',
            importance=3))
        backend.nodes.insert(make_insight(
            id='graph-3',
            content='Redis cache eviction policy tuning',
            importance=3))

        backend.edges.upsert(make_edge(
            source_id='graph-1', target_id='graph-2',
            edge_type='causal', weight=0.8))
        backend.edges.upsert(make_edge(
            source_id='graph-2', target_id='graph-3',
            edge_type='semantic', weight=0.8))

        result = intent_aware_recall(
            backend,
            query='API rate limiting design',
            query_vec=None, query_entities=['FastAPI'],
            limit=20)

        g3 = _find_result(result['results'], 'graph-3')
        assert g3 is not None, 'graph-3 should be discovered via traversal'
        assert g3['signals']['keyword'] == 0.0
        assert g3['signals']['graph'] > 0


class TestWhyIntentCausalOrdering:
    """WHY intent places causes before effects via topological sort."""

    def test_why_intent_causal_ordering(self, backend):
        """Cause insight appears before effect in WHY results."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='why-cause',
            content='Chose SQLite because embedded serverless database',
            importance=4))
        backend.nodes.insert(make_insight(
            id='why-effect',
            content='SQLite chosen enables single-file deployment',
            importance=4))

        backend.edges.upsert(make_edge(
            source_id='why-cause', target_id='why-effect',
            edge_type='causal', weight=0.9))

        result = intent_aware_recall(
            backend,
            query='why SQLite chosen because embedded',
            query_vec=None, query_entities=[],
            limit=20, intent_override='WHY')

        cause = _find_result(result['results'], 'why-cause')
        effect = _find_result(result['results'], 'why-effect')
        assert cause is not None
        assert effect is not None

        cause_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'why-cause')
        effect_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'why-effect')
        assert cause_idx < effect_idx


class TestWhenIntentChronologicalOrdering:
    """WHEN intent returns results newest-first by created_at."""

    def test_when_intent_chronological_ordering(self, backend):
        """Newer insights appear before older ones under WHEN intent."""
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='when-old',
            content='database migration completed for production deploy',
            importance=4))
        backend.nodes.insert(make_insight(
            id='when-mid',
            content='database schema updated production migration',
            importance=4))
        backend.nodes.insert(make_insight(
            id='when-new',
            content='database rollback production migration issue',
            importance=4))
        set_created_at(backend, 'when-old', OLD - timedelta(hours=2))
        set_created_at(backend, 'when-mid', OLD - timedelta(hours=1))
        set_created_at(backend, 'when-new', OLD)

        result = intent_aware_recall(
            backend,
            query='database production migration',
            query_vec=None, query_entities=[],
            limit=20, intent_override='WHEN')

        old = _find_result(result['results'], 'when-old')
        mid = _find_result(result['results'], 'when-mid')
        new = _find_result(result['results'], 'when-new')
        assert old is not None
        assert mid is not None
        assert new is not None

        new_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'when-new')
        mid_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'when-mid')
        old_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'when-old')
        assert new_idx < mid_idx < old_idx

    def test_when_intent_equal_timestamp_tiebreak(self, backend):
        """Same created_at: higher-scoring insight ranks first."""
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        ts = OLD
        backend.nodes.insert(make_insight(
            id='when-tie-hi',
            content='database production migration rollback strategy',
            importance=5))
        backend.nodes.insert(make_insight(
            id='when-tie-lo',
            content='database production migration backup strategy',
            importance=2))
        set_created_at(backend, 'when-tie-hi', ts)
        set_created_at(backend, 'when-tie-lo', ts)

        result = intent_aware_recall(
            backend,
            query='database production migration',
            query_vec=None, query_entities=[],
            limit=20, intent_override='WHEN')

        hi = _find_result(result['results'], 'when-tie-hi')
        lo = _find_result(result['results'], 'when-tie-lo')
        assert hi is not None
        assert lo is not None
        hi_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'when-tie-hi')
        lo_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'when-tie-lo')
        assert hi_idx < lo_idx


class TestWhyIntentGraphWeight:
    """WHY intent weights graph signal higher than GENERAL in final score."""

    def test_why_intent_weights_graph_higher_than_general(self, backend):
        """Same query under WHY vs GENERAL: WHY graph contribution > GENERAL."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='wg-1',
            content='Chose SQLite because embedded serverless database',
            importance=4))
        backend.nodes.insert(make_insight(
            id='wg-2',
            content='SQLite chosen enables single-file deployment',
            importance=4))

        backend.edges.upsert(make_edge(
            source_id='wg-1', target_id='wg-2',
            edge_type='causal', weight=0.9))

        query = 'why SQLite chosen because embedded'
        why_result = intent_aware_recall(
            backend, query=query, query_vec=None,
            query_entities=[], limit=20, intent_override='WHY')
        gen_result = intent_aware_recall(
            backend, query=query, query_vec=None,
            query_entities=[], limit=20, intent_override='GENERAL')

        why_r1 = _find_result(why_result['results'], 'wg-1')
        gen_r1 = _find_result(gen_result['results'], 'wg-1')
        assert why_r1 is not None
        assert gen_r1 is not None

        from memman.search.recall import RERANK_WEIGHTS
        why_w = RERANK_WEIGHTS['WHY']
        gen_w = RERANK_WEIGHTS['GENERAL']
        graph_sig = why_r1['signals']['graph']
        why_graph_contrib = why_w[3] * graph_sig
        gen_graph_contrib = gen_w[3] * graph_sig
        assert why_graph_contrib > gen_graph_contrib


class TestSingletonEntity:
    """Singleton entity still produces a positive entity signal."""

    def test_singleton_entity_positive_signal(self, backend):
        """Unique entity matched by query gets entity signal > 0."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='singleton',
            content='Terraform infrastructure as code provisioning',
            entities=['Terraform'], importance=3))

        result = intent_aware_recall(
            backend,
            query='Terraform infrastructure',
            query_vec=None, query_entities=['Terraform'],
            limit=20, intent_override='ENTITY')

        singleton = _find_result(result['results'], 'singleton')
        assert singleton is not None
        assert singleton['signals']['entity'] > 0

        for r in result['results']:
            if r['insight'].id != 'singleton':
                assert r['signals']['entity'] == 0


class TestImportanceTiebreaker:
    """Higher importance wins when scores are tied."""

    def test_importance_tiebreaker(self, backend):
        """imp=5 ranks before imp=2 with identical content and timestamps."""
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        ts = OLD
        backend.nodes.insert(make_insight(
            id='tie-high',
            content='logging best practices structured output',
            importance=5))
        backend.nodes.insert(make_insight(
            id='tie-low',
            content='logging best practices structured output',
            importance=2))
        set_created_at(backend, 'tie-high', ts)
        set_created_at(backend, 'tie-low', ts)

        result = intent_aware_recall(
            backend,
            query='logging best practices',
            query_vec=None, query_entities=[],
            limit=20, intent_override='GENERAL')

        high = _find_result(result['results'], 'tie-high')
        low = _find_result(result['results'], 'tie-low')
        assert high is not None
        assert low is not None

        high_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'tie-high')
        low_idx = next(
            i for i, r in enumerate(result['results'])
            if r['insight'].id == 'tie-low')
        assert high_idx < low_idx


class TestEntityCaseInsensitive:
    """Entity matching should be case-insensitive at the reranking layer."""

    def test_entity_case_insensitive(self, backend):
        """Lowercase query entity matches PascalCase stored entity."""
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='case-py',
            content='Python type hints and mypy configuration',
            entities=['Python'], importance=3))

        result = intent_aware_recall(
            backend,
            query='python tips',
            query_vec=None, query_entities=['python'],
            limit=20, intent_override='ENTITY')

        py = _find_result(result['results'], 'case-py')
        assert py is not None
        assert py['signals']['entity'] > 0


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


def _topk_ids(backend, qvec, k) -> list:
    """Return the top-k ids by intent-aware recall on the given backend."""
    result = intent_aware_recall(
        backend, query='topic insight',
        query_vec=qvec, query_entities=[],
        limit=k, intent_override='GENERAL')
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
        from memman.store.postgres import PostgresCluster
        from memman.store.sqlite import SqliteCluster

        topic_centers = [_gaussian_unit(seed=i) for i in range(_N_TOPICS)]

        sqlite_cluster = SqliteCluster()
        sqlite_backend = sqlite_cluster.open(
            store='r10', data_dir=str(tmp_path / 'memman'))
        sqlite_backend.meta.set(META_KEY, active_fingerprint().to_json())
        _populate_recall(sqlite_backend, topic_centers)

        postgres_cluster = PostgresCluster(dsn=pg_dsn)
        try:
            postgres_cluster.drop_store(store='r10_test', data_dir='')
        except Exception:
            pass
        postgres_backend = postgres_cluster.open(
            store='r10_test', data_dir='')
        postgres_backend.meta.set(META_KEY, active_fingerprint().to_json())
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
