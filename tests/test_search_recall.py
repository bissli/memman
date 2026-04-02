"""Tests for memman.search.recall -- beam search, traversal params, reranking."""

from memman.search.recall import RECALL_HINTS, RERANK_WEIGHTS
from memman.search.recall import get_traversal_params, intent_aware_recall
from memman.store.node import insert_insight
from tests.conftest import make_insight


def test_get_traversal_params_known():
    """All known intents have valid params."""
    for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
        beam_width, max_depth, max_visited = get_traversal_params(intent)
        assert beam_width > 0
        assert max_depth > 0
        assert max_visited > 0


def test_get_traversal_params_why_larger_beam():
    """WHY has larger beam width than GENERAL."""
    why_beam, _why_depth, _why_vis = get_traversal_params('WHY')
    gen_beam, _gen_depth, _gen_vis = get_traversal_params('GENERAL')
    assert why_beam > gen_beam


def test_get_traversal_params_unknown_fallback():
    """Unknown intent falls back to GENERAL."""
    assert get_traversal_params('UNKNOWN') == get_traversal_params('GENERAL')


def test_rerank_weights_all_intents_present():
    """Weight dict covers all four intents."""
    for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
        assert intent in RERANK_WEIGHTS


def test_rerank_weights_sum_to_one():
    """Each intent's weights sum to 1.0."""
    for intent, w in RERANK_WEIGHTS.items():
        assert abs(sum(w) - 1.0) < 1e-9, f'{intent} weights sum={sum(w)}'


def test_rerank_why_emphasizes_similarity():
    """WHY intent weights similarity score highest."""
    w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['WHY']
    assert w_sim > w_gr
    assert w_sim > w_kw
    assert w_sim > w_ent


def test_rerank_entity_emphasizes_entity():
    """ENTITY intent weights entity score highest."""
    w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['ENTITY']
    assert w_ent >= w_kw
    assert w_ent >= w_sim
    assert w_ent >= w_gr


def test_rerank_general_similarity_highest():
    """GENERAL intent weights similarity highest."""
    w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS['GENERAL']
    assert w_sim > max(w_kw, w_ent, w_gr)


class TestRecallMeta:
    """Meta dict fields: hint, ordering, sparse."""

    def _recall(self, db, intent):
        """Run recall with given intent override and a single matching insight."""
        insert_insight(db, make_insight(
            id=f'meta-{intent}',
            content='test content for recall meta fields'))
        return intent_aware_recall(
            db, query='test content recall',
            query_vec=None, query_entities=[],
            limit=5, intent_override=intent)

    def test_hint_field_by_intent(self, tmp_db):
        """Each intent produces its expected hint string."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            result = self._recall(tmp_db, intent)
            assert result['meta']['hint'] == RECALL_HINTS[intent]

    def test_ordering_field_by_intent(self, tmp_db):
        """Ordering field matches intent-specific sort strategy."""
        expected = {
            'WHY': 'causal_topological',
            'WHEN': 'chronological',
            'ENTITY': 'score',
            'GENERAL': 'score',
            }
        for intent, ordering in expected.items():
            result = self._recall(tmp_db, intent)
            assert result['meta']['ordering'] == ordering

    def test_sparse_flag_present(self, tmp_db):
        """Sparse flag set when results are below half the requested limit."""
        result = intent_aware_recall(
            tmp_db, query='nonexistent query xyz',
            query_vec=None, query_entities=[],
            limit=10, intent_override='GENERAL')
        assert result['meta']['sparse'] is True

    def test_sparse_flag_absent(self, tmp_db):
        """Sparse flag absent when result count meets threshold."""
        for i in range(5):
            insert_insight(tmp_db, make_insight(
                id=f'sparse-{i}',
                content=f'common keyword topic alpha {i}'))
        result = intent_aware_recall(
            tmp_db, query='common keyword topic alpha',
            query_vec=None, query_entities=[],
            limit=5, intent_override='GENERAL')
        assert 'sparse' not in result['meta']
