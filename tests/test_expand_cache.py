"""Tests for the in-process expand_query TTL cache.

Repeated query expansions in one CLI process hit the cache; varying
the MEMMAN_LLM_MODEL_FAST salt invalidates it. Cache key never
triggers a network round-trip (uses raw env-var, not client.model).
"""

from unittest.mock import MagicMock

import pytest
from memman.llm.extract import expand_query, reset_expand_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty expand_query cache."""
    reset_expand_cache()
    yield
    reset_expand_cache()


def test_repeated_query_hits_cache(env_file):
    """The second call with the same query does not invoke the LLM.
    """
    env_file('MEMMAN_LLM_MODEL_FAST', 'model-x')
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "alpha beta", "keywords": ["alpha"],'
        ' "entities": [], "intent": "GENERAL"}')

    first = expand_query(client, 'alpha')
    second = expand_query(client, 'alpha')

    assert first == second
    assert client.complete.call_count == 1


def test_changing_model_invalidates_cache(env_file):
    """Switching MEMMAN_LLM_MODEL_FAST yields a fresh LLM call.
    """
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "x", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    env_file('MEMMAN_LLM_MODEL_FAST', 'fast-A')
    expand_query(client, 'foo')
    env_file('MEMMAN_LLM_MODEL_FAST', 'fast-B')
    expand_query(client, 'foo')

    assert client.complete.call_count == 2


def test_normalized_query_collapses(env_file):
    """Whitespace-only and case differences collapse to one cache entry.
    """
    env_file('MEMMAN_LLM_MODEL_FAST', 'm')
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "ok", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    expand_query(client, 'Hello World')
    expand_query(client, 'hello   world')
    expand_query(client, 'HELLO WORLD')

    assert client.complete.call_count == 1


def test_cache_key_does_not_resolve_model(env_file):
    """Cache keying reads MEMMAN_LLM_MODEL_FAST directly, not the client.

    Historically `client.model` was a lazy property that would fetch a
    model inventory on first access; the cache key avoided it for
    speed. The lazy property is gone (model is required at construction
    now) but the cache-key contract still uses the env var directly.
    """
    env_file('MEMMAN_LLM_MODEL_FAST', 'safe-key')
    client = MagicMock()
    type(client).model = property(
        lambda self: (_ for _ in ()).throw(
            AssertionError('cache must not resolve client.model')))
    client.complete.return_value = (
        '{"expanded_query": "ok", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    expand_query(client, 'sample')
    expand_query(client, 'sample')


def test_cache_failure_is_not_cached(env_file):
    """LLM errors return a passthrough but are not cached.

    Caching errors would freeze a transient outage into permanent bad
    state for the rest of the process.
    """
    env_file('MEMMAN_LLM_MODEL_FAST', 'm')
    client = MagicMock()
    client.complete.side_effect = [
        RuntimeError('transient'),
        ('{"expanded_query": "alpha", "keywords": ["alpha"],'
         ' "entities": [], "intent": "GENERAL"}'),
        ]

    first = expand_query(client, 'alpha')
    assert first['expanded_query'] == 'alpha'

    second = expand_query(client, 'alpha')
    assert second['expanded_query'] == 'alpha'
    assert client.complete.call_count == 2
