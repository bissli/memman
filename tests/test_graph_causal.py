"""Causal signal detection and token overlap tests ported from Go causal_test.go."""

from mnemon.graph.causal import find_causal_signal, has_causal_signal
from mnemon.graph.causal import suggest_sub_type, token_overlap


class TestHasCausalSignalEnglish:
    """English causal keywords detected."""

    def test_has_causal_signal_english(self):
        """Five English phrases with causal keywords all return True."""
        cases = [
            'Chose SQLite because it is embedded',
            'Performance improved, therefore we kept the design',
            'Latency increased due to network issues',
            'The outage was caused by a misconfigured load balancer',
            'We added caching so that response times would improve',
            ]
        for text in cases:
            assert has_causal_signal(text) is True, (
                f'expected True for: {text}')


class TestHasCausalSignalAbsent:
    """Texts without causal keywords return False."""

    def test_has_causal_signal_absent(self):
        """Three sentences without causal signals return False."""
        cases = [
            'The sky is blue today',
            'Python is a popular language',
            'We had a meeting yesterday',
            ]
        for text in cases:
            assert has_causal_signal(text) is False, (
                f'expected False for: {text}')


class TestSuggestSubTypeCauses:
    """'because' keyword yields 'causes' sub_type."""

    def test_suggest_sub_type_causes(self):
        """Text with 'because' maps to causes."""
        result = suggest_sub_type('Chose Go because it compiles fast')
        assert result == 'causes'


class TestSuggestSubTypeEnables:
    """'so that' keyword yields 'enables' sub_type."""

    def test_suggest_sub_type_enables(self):
        """Text with 'so that' maps to enables."""
        result = suggest_sub_type('Added caching so that latency drops')
        assert result == 'enables'


class TestSuggestSubTypePrevents:
    """'prevents' keyword yields 'prevents' sub_type."""

    def test_suggest_sub_type_prevents(self):
        """Text with 'prevents' maps to prevents."""
        result = suggest_sub_type(
            'The rate limiter prevents overload')
        assert result == 'prevents'


class TestSuggestSubTypeDefault:
    """No causal keywords defaults to 'causes'."""

    def test_suggest_sub_type_default(self):
        """Text without any causal keyword returns causes."""
        result = suggest_sub_type('The sky is blue')
        assert result == 'causes'


class TestSuggestSubTypePreventsOverrides:
    """'prevents' takes priority over 'because'."""

    def test_suggest_sub_type_prevents_overrides(self):
        """When both 'because' and 'prevents' present, prevents wins."""
        result = suggest_sub_type(
            'blocked the request because it prevents abuse')
        assert result == 'prevents'


class TestTokenOverlapBasic:
    """Intersection / max ratio computed correctly."""

    def test_token_overlap_basic(self):
        """Two sets with partial overlap produce expected ratio."""
        a = {'go', 'sqlite', 'database'}
        b = {'go', 'sqlite', 'web', 'server'}
        overlap = token_overlap(a, b)
        expected = 2 / 4
        assert abs(overlap - expected) < 0.001


class TestTokenOverlapNoOverlap:
    """Disjoint sets produce 0.0."""

    def test_token_overlap_no_overlap(self):
        """Completely disjoint sets return zero."""
        a = {'alpha', 'beta'}
        b = {'gamma', 'delta'}
        assert token_overlap(a, b) == 0.0


class TestTokenOverlapEmpty:
    """Empty set produces 0.0."""

    def test_token_overlap_empty(self):
        """One or both sets empty returns zero."""
        assert token_overlap(set(), {'a'}) == 0.0
        assert token_overlap({'a'}, set()) == 0.0
        assert token_overlap(set(), set()) == 0.0


class TestTokenOverlapIdentical:
    """Identical sets produce 1.0."""

    def test_token_overlap_identical(self):
        """Same set against itself returns 1.0."""
        s = {'go', 'sqlite', 'graph'}
        assert token_overlap(s, s) == 1.0


class TestHasCausalSignalExpandedKeywords:
    """Expanded keyword list detects enables/prevents/ensures."""

    def test_has_causal_signal_expanded_keywords(self):
        """Genuinely causal keywords detected."""
        cases = [
            'This enables fast iteration on the design',
            'The feature prevents data loss during compaction',
            'This ensures memories survive compaction',
            ]
        for text in cases:
            assert has_causal_signal(text) is True, (
                f'expected True for: {text}')


class TestHasCausalSignalRejectsAmbiguous:
    """Ambiguous temporal/dependency phrases do not trigger causal signal."""

    def test_has_causal_signal_rejects_ambiguous(self):
        """Temporal 'since' and dependency verbs are not causal signals."""
        cases = [
            'Working on this since yesterday',
            'Available since Python 3.10',
            'This requires Python 3.11',
            'The feature depends on network availability',
            'Blocked on code review',
            ]
        for text in cases:
            assert has_causal_signal(text) is False, (
                f'expected False for: {text}')


class TestFindCausalSignalMatch:
    """First causal keyword returned when present."""

    def test_find_causal_signal_match(self):
        """Returns the matched causal keyword string."""
        result = find_causal_signal('Chose Go because it is fast')
        assert result == 'because'


class TestFindCausalSignalNoMatch:
    """Empty string returned when no causal keyword present."""

    def test_find_causal_signal_no_match(self):
        """No keyword in text returns empty string."""
        result = find_causal_signal('The sky is blue')
        assert result == ''


class TestInferLLMCausalIncludesRecent:
    """infer_llm_causal_edges passes recent insights to LLM prompt."""

    def test_recent_insights_in_llm_prompt(self, tmp_db):
        """LLM prompt contains recent insights, not just graph neighbors."""

        from mnemon.graph.causal import infer_llm_causal_edges
        from mnemon.store.node import insert_insight

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc)
        from mnemon.model import Insight

        def _make(**kw):
            defaults = {
                'id': 'x', 'content': 'x', 'category': 'fact', 'importance': 3,
                'tags': [], 'entities': [], 'source': 'test', 'access_count': 0,
                'created_at': now, 'updated_at': now, 'deleted_at': None,
                'last_accessed_at': None, 'effective_importance': 0.0}
            defaults.update(kw)
            return Insight(**defaults)

        for i in range(3):
            insert_insight(tmp_db, _make(
                id=f'old-{i}',
                content=f'Database optimization technique {i} because of performance'))

        new_ins = _make(
            id='new-1',
            content='Chose Redis because it improves cache hit rate for database queries')
        insert_insight(tmp_db, new_ins)

        captured_prompts = []

        class MockLLM:
            def complete(self, system, user):
                captured_prompts.append(user)
                return '[]'

        infer_llm_causal_edges(tmp_db, new_ins, MockLLM())

        assert captured_prompts, 'LLM was never called'
        prompt = captured_prompts[0]
        assert 'RECENT INSIGHTS' in prompt, (
            'Recent insights section missing from LLM prompt — '
            'recent list passed as [] instead of actual recent insights')
