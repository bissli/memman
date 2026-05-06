"""Entity IDF weighting tests."""

import pytest
from memman.graph.entity import entity_idf_weight


@pytest.mark.parametrize(('doc_freq', 'total_docs', 'bound', 'op'), [
    (0, 100, 0.9, '>'),
    (90, 100, 0.1, '>='),
])
def test_branch_returns(doc_freq, total_docs, bound, op):
    """Cover the doc_freq=0 short-circuit and the max(raw, 0.1) clamp.

    `test_monotonic` (using inputs 2, 50, 90) does not exercise the
    `doc_freq <= 0` branch or the floor clamp; these two cases close
    the gap.
    """
    w = entity_idf_weight(doc_freq, total_docs)
    if op == '>':
        assert w > bound
    else:
        assert w >= bound


def test_monotonic():
    """Weight decreases as doc_freq increases."""
    w1 = entity_idf_weight(2, 100)
    w2 = entity_idf_weight(50, 100)
    w3 = entity_idf_weight(90, 100)
    assert w1 > w2 > w3
