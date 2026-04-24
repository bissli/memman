"""Unit tests for the OpenRouter client, cache, and model picker."""

import json
import time

import httpx
import pytest
from memman.llm import openrouter_cache as cache_mod
from memman.llm.openrouter_cache import clear_cache, get_zdr_endpoints
from memman.llm.openrouter_cache import pick_latest_haiku
from memman.llm.openrouter_client import OpenRouterClient

SAMPLE_ZDR = [
    {'model_id': 'anthropic/claude-haiku-4.5'},
    {'model_id': 'anthropic/claude-haiku-4.5', 'provider_name': 'Google'},
    {'model_id': 'anthropic/claude-3.5-haiku'},
    {'model_id': 'anthropic/claude-3-haiku'},
    {'model_id': 'openai/gpt-5-mini'},
    {'model_id': 'meta-llama/llama-4-maverick'},
    ]


@pytest.fixture(autouse=True)
def _reset_cache(tmp_path, monkeypatch):
    """Clear in-process cache and isolate disk cache per test."""
    clear_cache()
    monkeypatch.setenv('MEMMAN_CACHE_DIR', str(tmp_path))


def test_pick_latest_haiku_prefers_4_5():
    """pick_latest_haiku returns the highest-version Anthropic Haiku.
    """
    assert pick_latest_haiku(SAMPLE_ZDR) == 'anthropic/claude-haiku-4.5'


def test_pick_latest_haiku_dedupes_across_providers():
    """Duplicate model_ids from multiple providers are deduped.
    """
    assert pick_latest_haiku(SAMPLE_ZDR) == 'anthropic/claude-haiku-4.5'


def test_pick_latest_haiku_raises_if_no_haiku():
    """An inventory with no Haiku models raises RuntimeError.
    """
    with pytest.raises(RuntimeError, match='no Anthropic Haiku'):
        pick_latest_haiku([
            {'model_id': 'openai/gpt-5-mini'},
            {'model_id': 'google/gemini-2.5-flash'},
            ])


def test_cache_disk_fresh_short_circuits_fetch(tmp_path, monkeypatch):
    """A fresh disk cache returns its data without hitting the network.
    """
    monkeypatch.setattr(cache_mod, '_fetch',
                        lambda: pytest.fail('network should not be called'))
    cache_file = tmp_path / 'openrouter-zdr.json'
    cache_file.write_text(json.dumps(SAMPLE_ZDR))

    data = get_zdr_endpoints()
    assert data == SAMPLE_ZDR


def test_cache_stale_disk_triggers_fetch(tmp_path, monkeypatch):
    """A stale disk cache triggers an HTTP fetch and rewrites the cache.
    """
    fetched = [{'model_id': 'anthropic/claude-haiku-4.5'}]
    monkeypatch.setattr(cache_mod, '_fetch', lambda: fetched)

    cache_file = tmp_path / 'openrouter-zdr.json'
    cache_file.write_text(json.dumps(SAMPLE_ZDR))
    old = time.time() - 90000
    import os
    os.utime(cache_file, (old, old))

    data = get_zdr_endpoints()
    assert data == fetched


def test_cache_stale_disk_fallback_on_network_failure(
        tmp_path, monkeypatch):
    """If HTTP fails, the stale disk cache is served with a warning.
    """
    def _boom():
        raise httpx.ConnectError('no internet')

    monkeypatch.setattr(cache_mod, '_fetch', _boom)
    cache_file = tmp_path / 'openrouter-zdr.json'
    cache_file.write_text(json.dumps(SAMPLE_ZDR))
    import os
    old = time.time() - 90000
    os.utime(cache_file, (old, old))

    data = get_zdr_endpoints()
    assert data == SAMPLE_ZDR


def test_cache_fail_loud_when_no_source_available(tmp_path, monkeypatch):
    """With no disk cache and a failing fetch, RuntimeError is raised.
    """
    monkeypatch.setattr(cache_mod, '_fetch',
                        lambda: (_ for _ in ()).throw(
                            httpx.ConnectError('down')))
    with pytest.raises(RuntimeError, match='no ZDR cache available'):
        get_zdr_endpoints()


def test_client_uses_picked_haiku_when_no_override(monkeypatch):
    """Without MEMMAN_LLM_MODEL, the client picks the latest Haiku.
    """
    monkeypatch.setattr(cache_mod, '_fetch', lambda: SAMPLE_ZDR)
    client = OpenRouterClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-key')
    assert client.model == 'anthropic/claude-haiku-4.5'


def test_client_honors_valid_model_override(monkeypatch):
    """An override that is present in the ZDR list is used as-is.
    """
    monkeypatch.setattr(cache_mod, '_fetch', lambda: SAMPLE_ZDR)
    client = OpenRouterClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-key',
        model='openai/gpt-5-mini')
    assert client.model == 'openai/gpt-5-mini'


def test_client_rejects_non_zdr_override(monkeypatch):
    """An override that is not in the ZDR list fails loudly.
    """
    import click
    monkeypatch.setattr(cache_mod, '_fetch', lambda: SAMPLE_ZDR)
    client = OpenRouterClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-key',
        model='deepseek/deepseek-chat')
    with pytest.raises(click.ClickException, match='not in the current'):
        _ = client.model


def test_client_complete_injects_zdr_provider_block(monkeypatch):
    """complete() sends provider={zdr:true, data_collection:'deny'}.
    """
    monkeypatch.setattr(cache_mod, '_fetch', lambda: SAMPLE_ZDR)
    seen = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        seen['url'] = url
        seen['body'] = json
        return httpx.Response(
            200,
            request=httpx.Request('POST', url),
            json={'choices': [{'message': {'content': 'ok'}}]})

    monkeypatch.setattr(httpx, 'post', _fake_post)
    client = OpenRouterClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-key')
    out = client.complete('sys', 'user')
    assert out == 'ok'
    assert seen['url'].endswith('/chat/completions')
    provider = seen['body']['provider']
    assert provider['zdr'] is True
    assert provider['data_collection'] == 'deny'
    assert provider['allow_fallbacks'] is True
    assert seen['body']['model'] == 'anthropic/claude-haiku-4.5'


def test_get_llm_client_routes_to_openrouter(monkeypatch):
    """MEMMAN_LLM_PROVIDER=openrouter returns an OpenRouterClient.
    """
    monkeypatch.setenv('MEMMAN_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake-key')
    from memman.llm.client import get_llm_client
    c = get_llm_client()
    assert type(c).__name__ == 'OpenRouterClient'


def test_get_llm_client_default_is_anthropic(monkeypatch):
    """Unset MEMMAN_LLM_PROVIDER routes to AnthropicClient.
    """
    monkeypatch.delenv('MEMMAN_LLM_PROVIDER', raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    from memman.llm.client import get_llm_client
    c = get_llm_client()
    assert type(c).__name__ == 'LLMClient'


def test_get_llm_client_rejects_unknown_provider(monkeypatch):
    """Unknown MEMMAN_LLM_PROVIDER raises a clear ClickException.
    """
    import click
    monkeypatch.setenv('MEMMAN_LLM_PROVIDER', 'wat')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-key')
    from memman.llm.client import get_llm_client
    with pytest.raises(click.ClickException, match='unknown'):
        get_llm_client()
