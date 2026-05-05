"""Tests for memman.search -- keyword, intent, and recall."""

import pytest
from memman.search.intent import detect_intent, get_weights, intent_from_string
from memman.search.keyword import keyword_search, tokenize
from memman.search.recall import RECALL_HINTS, RERANK_WEIGHTS
from memman.search.recall import get_traversal_params, intent_aware_recall
from memman.store.model import Insight
from tests.conftest import make_insight


class TestKeywordSearch:
    """Tokenization and keyword search ranking."""

    def test_tokenize_english(self):
        """English words are lowercased and split."""
        tokens = tokenize('Go uses SQLite for persistent storage')
        assert 'go' in tokens
        assert 'sqlite' in tokens
        assert 'persistent' in tokens
        assert 'storage' in tokens
        assert 'for' not in tokens

    def test_tokenize_stopwords(self):
        """Common stopwords are filtered out."""
        tokens = tokenize('the quick fox is very fast')
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'very' not in tokens
        assert 'quick' in tokens
        assert 'fox' in tokens
        assert 'fast' in tokens

    def test_tokenize_empty(self):
        """Empty string produces empty set."""
        assert len(tokenize('')) == 0

    def test_tokenize_all_stopwords(self):
        """All-stopword input produces empty set."""
        assert len(tokenize('the is a an')) == 0

    def test_keyword_search_ranking(self):
        """Best match ranks first."""
        insights = [
            Insight(id='1', content='Go language for building CLI tools', importance=3),
            Insight(id='2', content='SQLite database for Go applications', importance=3),
            Insight(id='3', content='Python machine learning framework', importance=3),
            ]
        results = keyword_search(insights, 'Go CLI tools', 10)
        assert len(results) >= 2
        assert results[0][0].id == '1'
        for i in range(1, len(results)):
            assert results[i][1] <= results[i - 1][1]

    def test_keyword_search_limit(self):
        """Limit caps the result count."""
        words = ['common', 'shared', 'words', 'alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'theta']
        insights = [
            Insight(id=str(i),
                    content=' '.join(words[:3 + (i % len(words))]),
                    importance=i + 1)
            for i in range(20)
            ]
        results = keyword_search(insights, 'common shared words', 5)
        assert len(results) <= 5

    def test_keyword_search_importance_tiebreak(self):
        """Higher importance wins on score tie."""
        insights = [
            Insight(id='low', content='Go memory graph', importance=1),
            Insight(id='high', content='Go memory graph', importance=5),
            ]
        results = keyword_search(insights, 'Go memory graph', 10)
        assert len(results) >= 2
        assert results[0][0].id == 'high'

    def test_keyword_search_empty_query(self):
        """Empty query returns empty results."""
        insights = [Insight(id='1', content='some content')]
        results = keyword_search(insights, '', 10)
        assert len(results) == 0

    def test_keyword_search_entities(self):
        """Entities contribute to matching."""
        insights = [
            Insight(id='1', content='something unrelated',
                    entities=['SQLite']),
            ]
        results = keyword_search(insights, 'SQLite', 10)
        assert len(results) > 0


class TestIntentRouting:
    """Intent detection and weight dispatch."""

    def test_detect_why(self):
        """Why-related queries detect WHY intent."""
        for q in ['why did we choose SQLite',
                  'the reason we chose Go because of motivation']:
            assert detect_intent(q) == 'WHY'

    def test_detect_when(self):
        """Time-related queries detect WHEN intent."""
        for q in ['when was the database migrated',
                  'timeline of changes',
                  'what happened before the release']:
            assert detect_intent(q) == 'WHEN'

    def test_detect_entity(self):
        """Entity-related queries detect ENTITY intent."""
        for q in ['what is MAGMA',
                  'who is responsible for the API',
                  'tell me about the graph engine']:
            assert detect_intent(q) == 'ENTITY'

    def test_detect_general(self):
        """Non-specific queries detect GENERAL intent."""
        for q in ['SQLite performance tuning',
                  'graph traversal algorithm']:
            assert detect_intent(q) == 'GENERAL'

    def test_intent_from_string_valid(self):
        """Valid intent strings parse correctly."""
        assert intent_from_string('WHY') == 'WHY'
        assert intent_from_string('why') == 'WHY'
        assert intent_from_string(' When ') == 'WHEN'
        assert intent_from_string('ENTITY') == 'ENTITY'
        assert intent_from_string('general') == 'GENERAL'

    def test_intent_from_string_invalid(self):
        """Invalid intent string raises ValueError."""
        with pytest.raises(ValueError):
            intent_from_string('BOGUS')

    def test_get_weights_known(self):
        """All intents have weights summing to ~1.0."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            w = get_weights(intent)
            assert len(w) > 0
            total = sum(w.values())
            assert 0.99 < total < 1.01

    def test_get_weights_why_prioritizes_causal(self):
        """WHY intent has highest causal weight."""
        w = get_weights('WHY')
        assert w['causal'] > w['temporal']
        assert w['causal'] > w['semantic']
        assert w['causal'] > w['entity']

    def test_get_weights_when_prioritizes_temporal(self):
        """WHEN intent has highest temporal weight."""
        w = get_weights('WHEN')
        assert w['temporal'] > w['causal']
        assert w['temporal'] > w['semantic']
        assert w['temporal'] > w['entity']

    def test_get_weights_entity_prioritizes_entity(self):
        """ENTITY intent has highest entity weight."""
        w = get_weights('ENTITY')
        assert w['entity'] > w['temporal']
        assert w['entity'] > w['causal']

    def test_get_weights_unknown_fallback(self):
        """Unknown intent falls back to GENERAL weights."""
        w = get_weights('NONEXISTENT')
        general = get_weights('GENERAL')
        for k, v in general.items():
            assert w[k] == v

    def test_detect_intent_tie_returns_general(self):
        """Tied WHY and ENTITY scores fall through to GENERAL."""
        assert detect_intent('describe why') == 'GENERAL'


class TestRecallRanking:
    """Beam search, traversal params, reranking, and meta fields."""

    def test_get_traversal_params_known(self):
        """All known intents have valid params."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            beam_width, max_depth, max_visited = get_traversal_params(intent)
            assert beam_width > 0
            assert max_depth > 0
            assert max_visited > 0

    def test_get_traversal_params_why_larger_beam(self):
        """WHY has larger beam width than GENERAL."""
        why_beam, _why_depth, _why_vis = get_traversal_params('WHY')
        gen_beam, _gen_depth, _gen_vis = get_traversal_params('GENERAL')
        assert why_beam > gen_beam

    def test_get_traversal_params_unknown_fallback(self):
        """Unknown intent falls back to GENERAL."""
        assert get_traversal_params('UNKNOWN') == get_traversal_params('GENERAL')

    def test_rerank_weights_all_intents_present(self):
        """Weight dict covers all four intents."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            assert intent in RERANK_WEIGHTS

    def test_rerank_weights_sum_to_one(self):
        """Each intent's weights sum to 1.0."""
        for intent, w in RERANK_WEIGHTS.items():
            assert abs(sum(w) - 1.0) < 1e-9, (
                f'{intent} weights sum={sum(w)}')

    def test_rerank_why_emphasizes_similarity(self):
        """WHY intent weights similarity score highest."""
        w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['WHY']
        assert w_sim > w_gr
        assert w_sim > w_kw
        assert w_sim > w_ent

    def test_rerank_entity_emphasizes_entity(self):
        """ENTITY intent weights entity score highest."""
        w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['ENTITY']
        assert w_ent >= w_kw
        assert w_ent >= w_sim
        assert w_ent >= w_gr

    def test_rerank_general_similarity_highest(self):
        """GENERAL intent weights similarity highest."""
        w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['GENERAL']
        assert w_sim > max(w_kw, w_ent, w_gr)

    def test_hint_field_by_intent(self, backend):
        """Each intent produces its expected hint string."""
        backend.nodes.insert(make_insight(
            id='meta-any',
            content='test content for recall meta fields'))
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            result = intent_aware_recall(
                backend, query='test content recall',
                query_vec=None, query_entities=[],
                limit=5, intent_override=intent)
            assert result['meta']['hint'] == RECALL_HINTS[intent]

    def test_ordering_field_by_intent(self, backend):
        """Ordering field matches intent-specific sort strategy."""
        backend.nodes.insert(make_insight(
            id='meta-ord',
            content='test content for recall ordering'))
        expected = {
            'WHY': 'causal_topological',
            'WHEN': 'chronological',
            'ENTITY': 'score',
            'GENERAL': 'score',
            }
        for intent, ordering in expected.items():
            result = intent_aware_recall(
                backend, query='test content recall',
                query_vec=None, query_entities=[],
                limit=5, intent_override=intent)
            assert result['meta']['ordering'] == ordering

    def test_sparse_flag_present(self, backend):
        """Sparse flag set when results are below half the requested limit."""
        result = intent_aware_recall(
            backend, query='nonexistent query xyz',
            query_vec=None, query_entities=[],
            limit=10, intent_override='GENERAL')
        assert result['meta']['sparse'] is True

    def test_sparse_flag_absent(self, backend):
        """Sparse flag absent when result count meets threshold."""
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'sparse-{i}',
                content=f'common keyword topic alpha {i}'))
        result = intent_aware_recall(
            backend, query='common keyword topic alpha',
            query_vec=None, query_entities=[],
            limit=5, intent_override='GENERAL')
        assert 'sparse' not in result['meta']
