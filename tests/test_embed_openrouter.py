"""Tests for `memman.embed.openrouter` provider."""

import pytest
from memman import _http, config
from memman.embed.openrouter import Client
from memman.exceptions import ConfigError


def _seed_keys(monkeypatch):
    """Set the three env vars the provider requires."""
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'sk-or-test')
    monkeypatch.setenv(config.OPENROUTER_ENDPOINT,
                       'https://openrouter.ai/api/v1')
    monkeypatch.setenv(config.OPENROUTER_EMBED_MODEL, 'baai/bge-m3')


def test_constructor_reads_config(monkeypatch):
    _seed_keys(monkeypatch)
    client = Client()
    assert client.endpoint == 'https://openrouter.ai/api/v1'
    assert client.model == 'baai/bge-m3'
    assert client.dim == 0
    assert client._api_key == 'sk-or-test'


@pytest.mark.no_default_env
def test_constructor_raises_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv(config.OPENROUTER_ENDPOINT, raising=False)
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'k')
    monkeypatch.setenv(config.OPENROUTER_EMBED_MODEL, 'm')
    with pytest.raises(ConfigError, match='OPENROUTER_ENDPOINT'):
        Client()


@pytest.mark.no_default_env
def test_constructor_raises_when_model_unset(monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_ENDPOINT, 'https://x')
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'k')
    monkeypatch.delenv(config.OPENROUTER_EMBED_MODEL, raising=False)
    with pytest.raises(ConfigError, match='OPENROUTER_EMBED_MODEL'):
        Client()


def _stub_session(monkeypatch, post_fn):
    from memman.embed import openrouter as orem
    monkeypatch.setitem(
        _http._SESSIONS, orem.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


def test_embed_returns_vector(monkeypatch):
    _seed_keys(monkeypatch)
    expected = [0.5] * 1024

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 200
            def json(self_inner):
                return {'data': [{'embedding': expected}]}
        return Resp()

    _stub_session(monkeypatch, fake_post)
    client = Client()
    vec = client.embed('hello')
    assert vec == expected
    assert client.dim == 1024


def test_embed_batch_returns_one_vector_per_input(monkeypatch):
    _seed_keys(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        n = len(json['input'])
        class Resp:
            status_code = 200
            def json(self_inner):
                return {'data': [
                    {'embedding': [float(i)] * 768} for i in range(n)
                    ]}
        return Resp()

    _stub_session(monkeypatch, fake_post)
    client = Client()
    vectors = client.embed_batch(['a', 'b', 'c'])
    assert len(vectors) == 3
    assert all(len(v) == 768 for v in vectors)
    assert client.dim == 768


def test_available_returns_false_on_probe_failure(monkeypatch):
    _seed_keys(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 401
        return Resp()

    _stub_session(monkeypatch, fake_post)
    client = Client()
    assert client.available() is False


def test_provider_registered(monkeypatch):
    """openrouter is in the embed PROVIDERS registry."""
    from memman.embed import PROVIDERS
    assert 'openrouter' in PROVIDERS


def test_provider_factory_returns_client(monkeypatch):
    _seed_keys(monkeypatch)
    from memman.embed import PROVIDERS
    client = PROVIDERS['openrouter']()
    assert client.name == 'openrouter'
