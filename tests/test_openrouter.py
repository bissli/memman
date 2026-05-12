"""Unit tests for the OpenRouter install-time model resolver."""

import pytest
from memman.exceptions import ConfigError
from memman.llm import openrouter_models as om
from memman.llm.client import MemmanLLMClient

_REAL_RESOLVE = om.resolve_latest_for_role

SAMPLE_MODELS = [
    {'id': 'anthropic/claude-haiku-4.5'},
    {'id': 'anthropic/claude-3.5-haiku'},
    {'id': 'anthropic/claude-3-haiku'},
    {'id': 'anthropic/claude-sonnet-4.6'},
    {'id': 'anthropic/claude-sonnet-4.5'},
    {'id': 'meta-llama/llama-4-maverick'},
    ]


@pytest.fixture(autouse=True)
def _undo_global_resolver_mock(monkeypatch):
    """Restore the real `resolve_latest_for_role` for these tests."""
    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_for_role',
        _REAL_RESOLVE)
    om.clear_cache()
    yield
    om.clear_cache()


def test_fast_picks_latest_haiku(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_for_role('fast') == 'anthropic/claude-haiku-4.5'


def test_slow_picks_latest_sonnet(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_for_role('slow') == 'anthropic/claude-sonnet-4.6'


def test_unknown_role_returns_none(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_for_role('bogus') is None


def test_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: [
        {'id': 'meta-llama/llama-4-maverick'}])
    assert om.resolve_latest_for_role('fast') is None


def test_returns_none_on_network_failure(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError('no route')

    monkeypatch.setattr(om, '_fetch_models', boom)
    assert om.resolve_latest_for_role('fast') is None


def test_caches_within_session(monkeypatch):
    calls = {'n': 0}

    def counting(*a, **k):
        calls['n'] += 1
        return SAMPLE_MODELS

    monkeypatch.setattr(om, '_fetch_models', counting)
    om.resolve_latest_for_role('fast')
    om.resolve_latest_for_role('fast')
    assert calls['n'] == 1


def test_fetch_models_sends_no_authorization(monkeypatch):
    """OR's /models is public; memman must not send an Authorization header."""
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'data': SAMPLE_MODELS}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            captured['url'] = url
            captured['headers'] = kwargs.get('headers')
            return _Resp()

    monkeypatch.setattr('memman.llm.openrouter_models.httpx.Client', _Client)
    om._fetch_models('https://openrouter.ai/api/v1')
    assert captured['headers'] is None or 'Authorization' not in (
        captured['headers'] or {})


def test_version_sort_key_orders_correctly():
    assert om._version_sort_key('anthropic/claude-haiku-10.0') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_version_sort_key_suffix_outranks_base():
    assert om._version_sort_key('anthropic/claude-haiku-4.5-v2') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_llm_client_requires_model():
    with pytest.raises(ConfigError, match='model is empty'):
        MemmanLLMClient(
            'https://openrouter.ai/api/v1',
            'sk-or-test',
            model='')


def test_llm_client_accepts_model():
    client = MemmanLLMClient(
        'https://openrouter.ai/api/v1',
        'sk-or-test',
        model='anthropic/claude-haiku-4.5')
    assert client.model == 'anthropic/claude-haiku-4.5'
    assert client.endpoint == 'https://openrouter.ai/api/v1'
