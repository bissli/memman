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
from datetime import datetime, timezone

import pytest
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
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
            query_vec=None,
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
            query_vec=None,
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


class TestRelevanceOrderingSurvivesTheLimit:
    """Nothing re-sorts after the limit slice, on any intent."""

    @pytest.mark.parametrize('intent', ['WHY', 'WHEN', 'GENERAL'])
    def test_results_are_score_descending(self, backend, intent):
        """Every intent returns rows in descending score order.

        Mutation: reinstating either post-limit re-sort - the WHEN
            sort on `(created_at, score)` or the WHY
            causal-topological sort - both of which reorder a page
            that was already cut by score.
        Oracle: the returned rows sorted by `-score` independently,
            compared as an id sequence.
        """
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        for i, word in enumerate(('rollback', 'schema', 'deploy')):
            backend.nodes.insert(make_insight(
                id=f'ord-{i}',
                content=f'database production migration {word}',
                importance=4))
            set_created_at(backend, f'ord-{i}',
                           OLD.replace(year=2024 + i))

        result = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=20, intent_override=intent)

        got = [r['insight'].id for r in result['results']]
        want = [r['insight'].id
                for r in sorted(result['results'],
                                key=lambda r: -r['score'])]
        assert got == want

    @pytest.mark.parametrize('intent', ['WHY', 'WHEN'])
    def test_a_short_page_is_the_head_of_a_long_one(self, backend, intent):
        """The first n rows of a limit-m recall ARE a limit-n recall.

        Mutation: reinstating either post-limit re-sort. Both run
            AFTER the slice, so they make a page of 3 the three
            newest (or topologically first) of the top 3 rather than
            the head of the top 20 - the exact defect that made
            `limit 5` disagree with the top 5 of `limit 30`.
        Oracle: two independent calls at different limits on one
            store, compared as id sequences.
        """
        from tests.conftest import set_created_at
        _insert_fillers(backend)
        for i, word in enumerate(
                ('rollback', 'schema', 'deploy', 'backup', 'restore')):
            backend.nodes.insert(make_insight(
                id=f'head-{i}',
                content=f'database production migration {word}',
                importance=4))
            set_created_at(backend, f'head-{i}',
                           OLD.replace(year=2024 + i))

        wide = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=20, intent_override=intent)
        narrow = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=3, intent_override=intent)

        assert len(narrow['results']) == 3
        assert ([r['insight'].id for r in narrow['results']]
                == [r['insight'].id for r in wide['results']][:3])


class TestWhyCausalEdgePayload:
    """WHY carries its causal structure in meta, not in row order."""

    def test_causal_edges_cover_returned_pairs_and_keep_direction(
            self, backend):
        """meta.causal_edges holds every returned pair, cause first.

        Mutation: building the list from the symmetrized `bidir`
            adjacency the beam walks (which would add the reverse of
            every pair), from the PRE-slice candidate set (which
            would name ids the caller never received), or omitting
            the intersection with the returned ids.
        Oracle: the edge written directly to the store, plus the
            assertion that its reverse is absent - the store holds
            `cause -> effect` and only that direction.
        """
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='cause-a',
            content='database production migration locked the table',
            importance=4))
        backend.nodes.insert(make_insight(
            id='effect-b',
            content='database production migration timed out queries',
            importance=4))
        backend.edges.upsert(make_edge(
            source_id='cause-a', target_id='effect-b',
            edge_type='causal', weight=0.9))

        result = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=20, intent_override='WHY')

        edges = result['meta']['causal_edges']
        returned = {r['insight'].id for r in result['results']}
        assert 'cause-a' in returned
        assert 'effect-b' in returned
        assert ['cause-a', 'effect-b'] in edges
        assert ['effect-b', 'cause-a'] not in edges
        assert all(src in returned and tgt in returned
                   for src, tgt in edges)

    def test_causal_edges_absent_on_other_intents(self, backend):
        """Only WHY carries the payload; an empty list still counts.

        Mutation: emitting `causal_edges` for every intent, which
            would spend tokens on a key three intents in four cannot
            populate, or omitting the key on a WHY page that happens
            to have no causal pair - "these rows are unrelated" is a
            fact the rows cannot convey.
        Oracle: the key set per intent, on a store with no causal
            edge at all.
        """
        _insert_fillers(backend)
        backend.nodes.insert(make_insight(
            id='lone', content='database production migration notes',
            importance=4))

        why = intent_aware_recall(
            backend, query='database production migration',
            query_vec=None, limit=5, intent_override='WHY')
        assert why['meta']['causal_edges'] == []

        for intent in ('WHEN', 'ENTITY', 'GENERAL'):
            other = intent_aware_recall(
                backend, query='database production migration',
                query_vec=None, limit=5, intent_override=intent)
            assert 'causal_edges' not in other['meta']


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
            query_vec=None,
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
        query_vec=qvec,
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
