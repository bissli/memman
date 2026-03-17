"""OpenAI-compatible embedding client and provider factory tests."""


import httpx
from mnemon.embed.openai import Client


class TestAvailableSuccess:
    """Client.available() returns True when model is listed."""

    def test_available_success(self, monkeypatch):
        """Mocked /v1/models response lists the model."""
        def mock_get(url, **kwargs):
            resp = httpx.Response(
                200,
                json={'data': [{'id': 'text-embedding-3-small'}]},
                request=httpx.Request('GET', url))
            return resp

        monkeypatch.setattr(httpx, 'get', mock_get)
        client = Client()
        assert client.available() is True


class TestAvailableFailure:
    """Client.available() returns False on connection error."""

    def test_available_failure(self, monkeypatch):
        """Connection error makes available() return False."""
        def mock_get(url, **kwargs):
            raise httpx.ConnectError('refused')

        monkeypatch.setattr(httpx, 'get', mock_get)
        client = Client()
        assert client.available() is False


class TestEmbedReturnsFloats:
    """Client.embed() returns a list of floats."""

    def test_embed_returns_floats(self, monkeypatch):
        """Mocked embedding response parsed to float list."""
        def mock_post(url, **kwargs):
            resp = httpx.Response(
                200,
                json={'data': [{'embedding': [0.1, 0.2, 0.3]}]},
                request=httpx.Request('POST', url))
            return resp

        monkeypatch.setattr(httpx, 'post', mock_post)
        client = Client()
        result = client.embed('test text')
        assert result == [0.1, 0.2, 0.3]


class TestUnavailableMessage:
    """Client.unavailable_message() returns non-empty string."""

    def test_unavailable_message(self):
        """Message includes endpoint and model info."""
        client = Client()
        msg = client.unavailable_message()
        assert len(msg) > 0
        assert 'api.openai.com' in msg


class TestGetClientOpenai:
    """MNEMON_EMBED_PROVIDER=openai returns openai Client."""

    def test_get_client_openai(self, monkeypatch):
        """Factory returns openai.Client when provider is 'openai'."""
        monkeypatch.setenv('MNEMON_EMBED_PROVIDER', 'openai')
        from mnemon.embed import get_client
        from mnemon.embed.openai import Client as OpenAIClient
        client = get_client()
        assert isinstance(client, OpenAIClient)


class TestGetClientOllama:
    """MNEMON_EMBED_PROVIDER=ollama returns ollama Client."""

    def test_get_client_ollama(self, monkeypatch):
        """Factory returns ollama.Client when provider is 'ollama'."""
        monkeypatch.setenv('MNEMON_EMBED_PROVIDER', 'ollama')
        from mnemon.embed import get_client
        from mnemon.embed.ollama import Client as OllamaClient
        client = get_client()
        assert isinstance(client, OllamaClient)


class TestGetClientAutoDetectNone:
    """Unset provider + unavailable Ollama returns None."""

    def test_get_client_auto_detect_none(self, monkeypatch):
        """No provider set and Ollama unavailable yields None."""
        monkeypatch.delenv('MNEMON_EMBED_PROVIDER', raising=False)

        def mock_get(url, **kwargs):
            raise httpx.ConnectError('refused')

        monkeypatch.setattr(httpx, 'get', mock_get)
        from mnemon.embed import get_client
        client = get_client()
        assert client is None
