"""Tests for `memman.embed.openrouter` provider."""

import pytest
from memman import _http, config
from memman.embed.openrouter import Client
from memman.exceptions import ConfigError


def _seed_keys(env_file):
    """Write the three env vars the provider requires to the env file."""
    env_file(config.OPENROUTER_API_KEY, 'sk-or-test')
    env_file(config.OPENROUTER_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(config.OPENROUTER_EMBED_MODEL, 'baai/bge-m3')


def test_constructor_reads_config(env_file):
    _seed_keys(env_file)
    client = Client()
    assert client.endpoint == 'https://openrouter.ai/api/v1'
    assert client.model == 'baai/bge-m3'
    assert client.dim == 0
    assert client._api_key == 'sk-or-test'


@pytest.mark.no_default_env
def test_constructor_raises_when_endpoint_unset(env_file):
    env_file(config.OPENROUTER_ENDPOINT, None)
    env_file(config.OPENROUTER_API_KEY, 'k')
    env_file(config.OPENROUTER_EMBED_MODEL, 'm')
    with pytest.raises(ConfigError, match='OPENROUTER_ENDPOINT'):
        Client()


@pytest.mark.no_default_env
def test_constructor_raises_when_model_unset(env_file):
    env_file(config.OPENROUTER_ENDPOINT, 'https://x')
    env_file(config.OPENROUTER_API_KEY, 'k')
    env_file(config.OPENROUTER_EMBED_MODEL, None)
    with pytest.raises(ConfigError, match='OPENROUTER_EMBED_MODEL'):
        Client()


def _stub_session(monkeypatch, post_fn):
    from memman.embed import openrouter as orem
    monkeypatch.setitem(
        _http._SESSIONS, orem.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


def test_embed_returns_vector(monkeypatch, env_file):
    _seed_keys(env_file)
    expected = [0.5] * 1024

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 200

            def json(self):
                return {'data': [{'embedding': expected}]}
        return Resp()

    _stub_session(monkeypatch, fake_post)
    client = Client()
    vec = client.embed('hello')
    assert vec == expected
    assert client.dim == 1024


def test_embed_batch_returns_one_vector_per_input(monkeypatch, env_file):
    _seed_keys(env_file)

    def fake_post(url, headers=None, json=None, timeout=None):
        n = len(json['input'])

        class Resp:
            status_code = 200

            def json(self):
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


def test_available_returns_false_on_probe_failure(monkeypatch, env_file):
    _seed_keys(env_file)

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 401
        return Resp()

    _stub_session(monkeypatch, fake_post)
    client = Client()
    assert client.available() is False


def test_provider_registered(monkeypatch):
    """Openrouter is in the embed PROVIDERS registry."""
    from memman.embed import PROVIDERS
    assert 'openrouter' in PROVIDERS


def test_provider_factory_returns_client(monkeypatch, env_file):
    _seed_keys(env_file)
    from memman.embed import PROVIDERS
    client = PROVIDERS['openrouter']()
    assert client.name == 'openrouter'
