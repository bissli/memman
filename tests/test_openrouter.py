"""Unit tests for the OpenRouter-facing parts of the LLM client."""

import pytest
from memman.exceptions import ConfigError
from memman.llm.client import MemmanLLMClient


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


def _capture_post(monkeypatch):
    """Stub the shared session and return the dict the body lands in."""
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'choices': [{'message': {'content': '{"ok": true}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                }

    class _Session:
        def post(self, url, **kwargs):
            captured['url'] = url
            captured['body'] = kwargs.get('json')
            return _Resp()

    monkeypatch.setattr('memman.llm.client.get_session', lambda _n: _Session())
    return captured


@pytest.mark.no_mock_llm
def test_openrouter_request_carries_the_operator_provider_routing(monkeypatch):
    """A client given provider routing sends it as the body's `provider`.

    Mutation: the routing block is dropped from the request body, so
    every call routes to whichever provider OpenRouter picks and the
    operator's jurisdiction and retention choice goes unenforced with
    no error raised.
    Oracle: the request body captured from a stubbed session.
    """
    captured = _capture_post(monkeypatch)
    routing = {'only': ['amazon-bedrock'], 'data_collection': 'deny'}
    client = MemmanLLMClient(
        'https://openrouter.ai/api/v1', 'sk-or-test',
        model='anthropic/claude-haiku-4.5', provider_routing=routing)
    client.complete('sys', 'user', stage='enrichment')
    assert captured['body']['provider'] == routing


@pytest.mark.no_mock_llm
def test_client_without_routing_sends_no_provider_key(monkeypatch):
    """A vendor-neutral endpoint gets no OpenRouter-only field.

    Mutation: the provider block is attached unconditionally, so an
    Ollama or vLLM shim receives an unknown top-level field it may
    reject.
    Oracle: the request body captured from a stubbed session.
    """
    captured = _capture_post(monkeypatch)
    client = MemmanLLMClient(
        'http://localhost:11434/v1', '', model='llama3')
    client.complete('sys', 'user', stage='enrichment')
    assert 'provider' not in captured['body']
