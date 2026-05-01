"""Tests for memman.embed.voyage -- Voyage AI embedding client.

These tests mock httpx.post to test Client logic (error handling,
response parsing, header construction) without hitting the API.
Normal-path embedding behavior is covered by test_cli.py and
test_memory_system.py in both mock and --live modes.
"""

import pytest
from memman.embed import voyage
from memman.embed.voyage import DEFAULT_ENDPOINT, DEFAULT_MODEL, EMBEDDING_DIM
from memman.embed.voyage import Client

_original_embed = Client.embed
_original_embed_batch = Client.embed_batch
_original_available = Client.available


@pytest.fixture
def real_client(monkeypatch):
    """Client with real methods restored (undo autouse mock)."""
    monkeypatch.setattr(Client, 'embed', _original_embed)
    monkeypatch.setattr(Client, 'embed_batch', _original_embed_batch)
    monkeypatch.setattr(Client, 'available', _original_available)
    monkeypatch.setenv('VOYAGE_API_KEY', 'test-key-123')
    return Client()


class TestClientInit:
    """Client initialization and configuration."""

    def test_default_endpoint(self):
        """Client uses default Voyage endpoint."""
        client = Client()
        assert client.endpoint == DEFAULT_ENDPOINT

    def test_default_model(self):
        """Client uses voyage-3-lite model."""
        client = Client()
        assert client.model == DEFAULT_MODEL

    def test_api_key_from_env_file(self, env_file):
        """Client reads VOYAGE_API_KEY from the env file."""
        env_file('VOYAGE_API_KEY', 'real-test-key')
        client = Client()
        assert client._api_key == 'real-test-key'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, env_file):
        """Client raises ConfigError when VOYAGE_API_KEY is absent from file."""
        from memman.exceptions import ConfigError
        env_file('VOYAGE_API_KEY', None)
        with pytest.raises(ConfigError, match='VOYAGE_API_KEY'):
            Client()


class TestAvailable:
    """Availability check behavior."""

    @pytest.mark.no_default_env
    def test_no_key_raises_at_construction(self, env_file):
        """Construction raises before available() can be called."""
        from memman.exceptions import ConfigError
        env_file('VOYAGE_API_KEY', None)
        with pytest.raises(ConfigError):
            Client()

    def test_available_is_memoized(self, monkeypatch):
        """available() calls the HTTP probe at most once per instance.
        """
        monkeypatch.setattr(Client, 'available', _original_available)
        monkeypatch.setenv('VOYAGE_API_KEY', 'probe-key')
        calls = {'n': 0}

        def _mock_post(url, headers=None, json=None, timeout=None):
            calls['n'] += 1

            class Resp:
                status_code = 200
            return Resp()

        from memman import _http
        monkeypatch.setitem(_http._SESSIONS, voyage.__name__, type('FakeClient', (), {'post': staticmethod(_mock_post)})())
        client = Client()
        assert client.available() is True
        assert client.available() is True
        assert client.available() is True
        assert calls['n'] == 1


class TestEmbed:
    """Embedding generation via httpx."""

    def test_embed_returns_vector(self, real_client, monkeypatch):
        """embed() returns vector from API response."""
        expected_vec = [0.1] * EMBEDDING_DIM

        def mock_post(url, headers=None, json=None, timeout=None):
            """Return valid embedding response."""
            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': [{'embedding': expected_vec}]}
            return Resp()

        from memman import _http
        monkeypatch.setitem(_http._SESSIONS, voyage.__name__, type('FakeClient', (), {'post': staticmethod(mock_post)})())
        vec = real_client.embed('test text')
        assert len(vec) == EMBEDDING_DIM
        assert vec == expected_vec

    def test_embed_raises_on_error_status(self, real_client, monkeypatch):
        """embed() raises RuntimeError on non-200 status."""
        def mock_post(url, headers=None, json=None, timeout=None):
            """Return 401 unauthorized."""
            class Resp:
                status_code = 401
            return Resp()

        from memman import _http
        monkeypatch.setitem(_http._SESSIONS, voyage.__name__, type('FakeClient', (), {'post': staticmethod(mock_post)})())
        with pytest.raises(RuntimeError, match='401'):
            real_client.embed('test')

    def test_embed_raises_on_empty_data(self, real_client, monkeypatch):
        """embed() raises RuntimeError on empty embedding data."""
        def mock_post(url, headers=None, json=None, timeout=None):
            """Return 200 with empty data array."""
            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': []}
            return Resp()

        from memman import _http
        monkeypatch.setitem(_http._SESSIONS, voyage.__name__, type('FakeClient', (), {'post': staticmethod(mock_post)})())
        with pytest.raises(RuntimeError, match='0 vectors'):
            real_client.embed('test')


class TestHeaders:
    """Request header construction."""

    def test_bearer_token(self, env_file):
        """Authorization header uses Bearer token."""
        env_file('VOYAGE_API_KEY', 'my-key')
        client = Client()
        headers = client._headers()
        assert headers['Authorization'] == 'Bearer my-key'
        assert headers['Content-Type'] == 'application/json'


class TestUnavailableMessage:
    """Error message content."""

    def test_message_includes_env_var(self):
        """Unavailable message mentions VOYAGE_API_KEY."""
        client = Client()
        assert 'VOYAGE_API_KEY' in client.unavailable_message()


class TestConstants:
    """Module-level constants."""

    def test_embedding_dim(self):
        """Voyage embedding dimension is 512."""
        assert EMBEDDING_DIM == 512
