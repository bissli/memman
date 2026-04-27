"""Unit tests for the OpenRouter client and install-time model resolver."""

import pytest
from memman.exceptions import ConfigError
from memman.llm import openrouter_models as om
from memman.llm.openrouter_client import OpenRouterClient

_REAL_RESOLVE = om.resolve_latest_in_family

SAMPLE_MODELS = [
    {'id': 'anthropic/claude-haiku-4.5'},
    {'id': 'anthropic/claude-3.5-haiku'},
    {'id': 'anthropic/claude-3-haiku'},
    {'id': 'anthropic/claude-sonnet-4.6'},
    {'id': 'anthropic/claude-sonnet-4.5'},
    {'id': 'openai/gpt-5-mini'},
    {'id': 'meta-llama/llama-4-maverick'},
    ]


@pytest.fixture(autouse=True)
def _undo_global_resolver_mock(monkeypatch):
    """Restore the real `resolve_latest_in_family` for these tests.

    `tests/conftest.py::_mock_apis` patches the resolver to a fake for
    install-path tests; here we want the real implementation so tests
    can patch `_fetch_models` directly. The original is captured at
    module import time.
    """
    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_in_family',
        _REAL_RESOLVE)
    om.clear_cache()
    yield
    om.clear_cache()


def test_resolve_haiku_picks_highest_version(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'haiku') == \
        'anthropic/claude-haiku-4.5'


def test_resolve_sonnet_picks_highest_version(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'sonnet') == \
        'anthropic/claude-sonnet-4.6'


def test_resolve_returns_none_when_family_absent(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: [
        {'id': 'openai/gpt-5-mini'}])
    assert om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'haiku') is None


def test_resolve_returns_none_on_network_failure(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError('no route')

    monkeypatch.setattr(om, '_fetch_models', boom)
    assert om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'haiku') is None


def test_resolve_caches_within_session(monkeypatch):
    calls = {'n': 0}

    def counting(*a, **k):
        calls['n'] += 1
        return SAMPLE_MODELS

    monkeypatch.setattr(om, '_fetch_models', counting)
    om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'haiku')
    om.resolve_latest_in_family(
        'k', 'https://openrouter.ai/api/v1', 'haiku')
    assert calls['n'] == 1


def test_version_sort_key_orders_correctly():
    assert om._version_sort_key('anthropic/claude-haiku-10.0') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_version_sort_key_suffix_outranks_base():
    assert om._version_sort_key('anthropic/claude-haiku-4.5-v2') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_openrouter_client_requires_model():
    with pytest.raises(ConfigError, match='is empty'):
        OpenRouterClient(
            'https://openrouter.ai/api/v1',
            'sk-or-test',
            role_env_var='MEMMAN_LLM_MODEL_FAST',
            model='')


def test_openrouter_client_accepts_model():
    client = OpenRouterClient(
        'https://openrouter.ai/api/v1',
        'sk-or-test',
        role_env_var='MEMMAN_LLM_MODEL_FAST',
        model='anthropic/claude-haiku-4.5')
    assert client.model == 'anthropic/claude-haiku-4.5'
    assert client.endpoint == 'https://openrouter.ai/api/v1'
