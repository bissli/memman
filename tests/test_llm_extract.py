"""Tests for memman.llm.extract -- query expansion and the slow role.

These tests use FakeLLMClient to test parsing, validation, and error
handling that can't be exercised through the CLI. Normal-path behavior
with real/mocked LLM is covered by test_cli.py and test_memory_system.py;
the three reconcile stages are covered by test_reconcile_stages.py.
"""

import json

import pytest
from memman.llm.extract import QUERY_EXPANSION_SYSTEM, expand_query
from memman.llm.shared import parse_json_response


class TestSlowRole:
    """The slow role resolves from its own env var, with no fallback."""

    def test_unset_metadata_var_raises(self, env_file):
        """Verify an unset slow model fails loudly instead of borrowing one.

        Mutation: a fallback from `MEMMAN_LLM_MODEL_SLOW` to the
            fast model's var, which bills enrichment on a model the
            operator never chose for it.
        Oracle: `ConfigError` raised with the fast var set and the slow
            var cleared.
        """
        from memman.config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL_FAST
        from memman.config import LLM_MODEL_SLOW
        from memman.exceptions import ConfigError
        from memman.llm.client import get_llm_client, reset_role_cache
        env_file(LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
        env_file(LLM_API_KEY, 'k')
        env_file(LLM_MODEL_FAST, 'anthropic/haiku')
        env_file(LLM_MODEL_SLOW, None)
        reset_role_cache()
        with pytest.raises(ConfigError):
            get_llm_client('slow')


class FakeLLMClient:
    """LLMClient that returns canned responses for unit testing."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Record call and return canned response."""
        self.calls.append((system, user, kwargs))
        return self.response


class FailingLLMClient:
    """LLMClient whose every call raises, for passthrough tests."""

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Raise as a wedged transport would."""
        raise ConnectionError('timeout')


class TestExpandQuery:
    """Query expansion parsing and error handling."""

    def test_basic_expansion(self):
        """The model's expansion reaches the caller, not the raw query.

        Mutation: discarding the parsed expansion and returning the
            original query, or normalizing a valid intent to None.
        Oracle: the exact expansion string in the canned response. A
            substring check would pass on the passthrough value.
        """
        response = json.dumps({
            'expanded_query': 'Redis cache configuration settings',
            'keywords': ['Redis', 'cache', 'config'],
            'intent': 'GENERAL',
            })
        client = FakeLLMClient(response)
        result = expand_query(client, 'Redis config')
        assert result['expanded_query'] == 'Redis cache configuration settings'
        assert result['intent'] == 'GENERAL'

    def test_llm_failure_returns_passthrough(self):
        """LLM failure returns original query unchanged."""
        result = expand_query(FailingLLMClient(), 'my query')
        assert result['expanded_query'] == 'my query'
        assert result['intent'] is None

    def test_result_carries_only_the_two_read_keys(self):
        """All four return paths yield exactly expanded_query and intent.

        Mutation: leaving 'keywords' on one of the four return paths,
            or letting an unrequested model key reach the caller.
        Oracle: the key set the sole production caller in `memman.cli`
            reads - `expanded_query` once and `intent` twice.
        """
        parsed_ok = json.dumps({
            'expanded_query': 'alpha beta',
            'keywords': ['alpha'],
            'intent': 'GENERAL',
            })
        cases = {
            'parsed': (FakeLLMClient(parsed_ok), 'alpha'),
            'unparseable': (FakeLLMClient('not json at all'), 'beta'),
            'llm_error': (FailingLLMClient(), 'gamma'),
            }
        for path, (client, query) in cases.items():
            result = expand_query(client, query)
            assert set(result) == {'expanded_query', 'intent'}, path

        cached_client = FakeLLMClient(parsed_ok)
        expand_query(cached_client, 'delta')
        cached = expand_query(cached_client, 'delta')
        assert len(cached_client.calls) == 1, 'second call must be cached'
        assert set(cached) == {'expanded_query', 'intent'}

    def test_prompt_requests_exactly_the_keys_the_parser_keeps(self):
        """The prompt's JSON skeleton names the surviving result keys.

        Mutation: asking the model for a key the parser discards, which
            buys nothing but tokens, or dropping a key the parser reads,
            which kills intent routing with no error.
        Oracle: the skeleton parsed out of QUERY_EXPANSION_SYSTEM against
            a live expand_query result - neither side hand-copied.
        """
        skeleton = QUERY_EXPANSION_SYSTEM[
            QUERY_EXPANSION_SYSTEM.index('{'):
            QUERY_EXPANSION_SYSTEM.rindex('}') + 1]
        requested = set(json.loads(skeleton))
        response = json.dumps({'expanded_query': 'x', 'intent': 'WHY'})
        returned = set(expand_query(FakeLLMClient(response), 'q'))
        assert requested == returned

    def test_invalid_intent_normalized(self):
        """Unknown intent is set to None."""
        response = json.dumps({
            'expanded_query': 'test',
            'keywords': [],
            'entities': [],
            'intent': 'BOGUS',
            })
        client = FakeLLMClient(response)
        result = expand_query(client, 'test')
        assert result['intent'] is None


@pytest.mark.parametrize(('raw', 'expected'), [
    ('{"key": "val"}', {'key': 'val'}),
    ('```json\n{"key": "val"}\n```', {'key': 'val'}),
    ('not json', None),
    ('[1, 2, 3]', None),
    ('[' * 3000, None),
    ('{"actions": ' + '[' * 1200, None),
])
def test_parse_json_response(raw, expected):
    """JSON response parsing strips fences, rejects non-dicts and runaway nesting.

    Mutation: catching `ValueError` alone, so a body nested past the
        recursion limit raises out of every stage reader and fails the
        write instead of reading as no object.
    Oracle: hand-paired rows; the two nested rows read as None.
    """
    assert parse_json_response(raw) == expected
