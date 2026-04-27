"""Tests for the in-process expand_query TTL cache.

Repeated query expansions in one CLI process hit the cache; varying
the MEMMAN_LLM_MODEL_FAST salt invalidates it. Cache key never
triggers a network round-trip (uses raw env-var, not client.model).
"""

import time
from unittest.mock import MagicMock

import pytest
from memman.llm.extract import expand_query, reset_expand_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty expand_query cache."""
    reset_expand_cache()
    yield
    reset_expand_cache()


def test_repeated_query_hits_cache(monkeypatch):
    """The second call with the same query does not invoke the LLM.
    """
    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'model-x')
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "alpha beta", "keywords": ["alpha"],'
        ' "entities": [], "intent": "GENERAL"}')

    first = expand_query(client, 'alpha')
    second = expand_query(client, 'alpha')

    assert first == second
    assert client.complete.call_count == 1


def test_changing_model_invalidates_cache(monkeypatch):
    """Switching MEMMAN_LLM_MODEL_FAST yields a fresh LLM call.
    """
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "x", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'fast-A')
    expand_query(client, 'foo')
    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'fast-B')
    expand_query(client, 'foo')

    assert client.complete.call_count == 2


def test_normalized_query_collapses(monkeypatch):
    """Whitespace-only and case differences collapse to one cache entry.
    """
    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'm')
    client = MagicMock()
    client.complete.return_value = (
        '{"expanded_query": "ok", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    expand_query(client, 'Hello World')
    expand_query(client, 'hello   world')
    expand_query(client, 'HELLO WORLD')

    assert client.complete.call_count == 1


def test_cache_key_does_not_resolve_model(monkeypatch):
    """Cache keying must not access llm_client.model (would hit network).

    `client.model` is a property that triggers the ZDR endpoint fetch
    on first access. Cache keying uses the raw env var instead.
    """
    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'safe-key')
    client = MagicMock()
    type(client).model = property(
        lambda self: (_ for _ in ()).throw(
            AssertionError('cache must not resolve client.model')))
    client.complete.return_value = (
        '{"expanded_query": "ok", "keywords": [], "entities": [],'
        ' "intent": "GENERAL"}')

    expand_query(client, 'sample')
    expand_query(client, 'sample')


def test_cache_failure_is_not_cached(monkeypatch):
    """LLM errors return a passthrough but are not cached.

    Caching errors would freeze a transient outage into permanent bad
    state for the rest of the process.
    """
    monkeypatch.setenv('MEMMAN_LLM_MODEL_FAST', 'm')
    client = MagicMock()
    client.complete.side_effect = [
        RuntimeError('transient'),
        '{"expanded_query": "alpha", "keywords": ["alpha"],'
        ' "entities": [], "intent": "GENERAL"}',
        ]

    first = expand_query(client, 'alpha')
    assert first['expanded_query'] == 'alpha'

    second = expand_query(client, 'alpha')
    assert second['expanded_query'] == 'alpha'
    assert client.complete.call_count == 2
