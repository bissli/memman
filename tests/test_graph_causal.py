"""Causal signal detection and token overlap tests."""

from memman.graph.causal import has_causal_signal
from memman.graph.causal import suggest_sub_type, token_overlap


def test_has_causal_signal_absent():
    """Texts without causal keywords return False.

    The corpus rejects temporal 'since', dependency verbs, and
    'blocked on'. This is the only line of defense against accidental
    keyword additions to `CAUSAL_PATTERN`.
    """
    cases = [
        'The sky is blue today',
        'Python is a popular language',
        'We had a meeting yesterday',
        'Working on this since yesterday',
        'Available since Python 3.10',
        'This requires Python 3.11',
        'The feature depends on network availability',
        'Blocked on code review',
        ]
    for text in cases:
        assert has_causal_signal(text) is False, (
            f'expected False for: {text}')


def test_suggest_sub_type_prevents_overrides():
    """When 'because' and 'prevents' both present, prevents wins.

    Pins the priority order in `suggest_sub_type` so a refactor that
    reorders the if/elif/else cannot silently change behavior.
    """
    result = suggest_sub_type(
        'blocked the request because it prevents abuse')
    assert result == 'prevents'


def test_token_overlap_basic():
    """Two sets with partial overlap produce intersection / max ratio."""
    a = {'go', 'sqlite', 'database'}
    b = {'go', 'sqlite', 'web', 'server'}
    overlap = token_overlap(a, b)
    expected = 2 / 4
    assert abs(overlap - expected) < 0.001


def test_token_overlap_no_overlap():
    """Completely disjoint sets return zero."""
    a = {'alpha', 'beta'}
    b = {'gamma', 'delta'}
    assert token_overlap(a, b) == 0.0


def test_token_overlap_empty():
    """One or both sets empty returns zero."""
    assert token_overlap(set(), {'a'}) == 0.0
    assert token_overlap({'a'}, set()) == 0.0
    assert token_overlap(set(), set()) == 0.0


def test_token_overlap_identical():
    """Same set against itself returns 1.0."""
    s = {'go', 'sqlite', 'graph'}
    assert token_overlap(s, s) == 1.0


def test_recent_insights_in_llm_prompt(backend):
    """`_build_llm_prompt` plumbing passes recent insights through.

    Catches plumbing regressions where someone passes post-overlap
    `candidates` instead of `recent` to `_build_llm_prompt`, or
    drops the `'RECENT INSIGHTS'` literal section header.
    """
    from datetime import datetime, timezone

    from memman.graph.causal import infer_llm_causal_edges
    from memman.store.model import Insight

    now = datetime.now(timezone.utc)

    def _make(**kw):
        defaults = {
            'id': 'x', 'content': 'x', 'category': 'fact', 'importance': 3,
            'entities': [], 'source': 'test', 'access_count': 0,
            'created_at': now, 'updated_at': now, 'deleted_at': None,
            'last_accessed_at': None, 'effective_importance': 0.0}
        defaults.update(kw)
        return Insight(**defaults)

    for i in range(3):
        backend.nodes.insert(_make(
            id=f'old-{i}',
            content=f'Database optimization technique {i} because of performance'))

    new_ins = _make(
        id='new-1',
        content='Chose Redis because it improves cache hit rate for database queries')
    backend.nodes.insert(new_ins)

    captured_prompts = []

    class MockLLM:
        def complete(self, system, user):
            captured_prompts.append(user)
            return '[]'

    infer_llm_causal_edges(backend, new_ins, MockLLM())

    assert captured_prompts, 'LLM was never called'
    prompt = captured_prompts[0]
    assert 'RECENT INSIGHTS' in prompt, (
        'Recent insights section missing from LLM prompt — '
        'recent list passed as [] instead of actual recent insights')
