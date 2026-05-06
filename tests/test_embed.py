"""Tests for memman.embed -- vector math, Voyage client, OpenRouter client."""

import math

import pytest
from memman import _http, config
from memman.embed import voyage
from memman.embed.openrouter import Client as OpenRouterClient
from memman.embed.vector import cosine_similarity, deserialize_vector
from memman.embed.vector import serialize_vector
from memman.embed.voyage import DEFAULT_ENDPOINT, DEFAULT_MODEL, EMBEDDING_DIM
from memman.embed.voyage import Client as VoyageClient
from memman.exceptions import ConfigError

_original_voyage_embed = VoyageClient.embed
_original_voyage_embed_batch = VoyageClient.embed_batch
_original_voyage_available = VoyageClient.available


class TestEmbedUtils:
    """Cosine similarity and vector serialization utilities."""

    def test_cosine_identical(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_cosine_opposite(self):
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9

    @pytest.mark.parametrize('a,b', [
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([], []),
        (None, None),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
    ])
    def test_cosine_returns_zero(self, a, b):
        """Mismatched dim, empty/None, or zero vector all return 0.0."""
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_scaled(self):
        """Scaled vector has similarity 1.0."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-9

    def test_serialize_deserialize_roundtrip(self):
        """Verify float64 binary blob roundtrip."""
        original = [1.5, -2.7, 0.0, math.pi, float('inf')]
        blob = serialize_vector(original)
        restored = deserialize_vector(blob)
        assert len(restored) == len(original)
        for o, r in zip(original, restored):
            if math.isinf(o):
                assert math.isinf(r)
            else:
                assert o == r

    def test_serialize_empty(self):
        """Empty/None vector produces empty bytes."""
        assert serialize_vector(None) == b''
        assert serialize_vector([]) == b''

    def test_deserialize_empty(self):
        """Empty/None blob returns None."""
        assert deserialize_vector(None) is None
        assert deserialize_vector(b'') is None

    def test_deserialize_invalid_length(self):
        """Blob with length not multiple of 8 returns None."""
        assert deserialize_vector(bytes(7)) is None


class TestVoyageClient:
    """Voyage AI embedding client -- init, availability, embed, headers."""

    @pytest.fixture
    def real_client(self, monkeypatch):
        """Client with real methods restored (undo autouse mock)."""
        monkeypatch.setattr(VoyageClient, 'embed', _original_voyage_embed)
        monkeypatch.setattr(VoyageClient, 'embed_batch', _original_voyage_embed_batch)
        monkeypatch.setattr(VoyageClient, 'available', _original_voyage_available)
        monkeypatch.setenv('VOYAGE_API_KEY', 'test-key-123')
        return VoyageClient()

    def test_api_key_from_env_file(self, env_file):
        """Client reads VOYAGE_API_KEY from the env file."""
        env_file('VOYAGE_API_KEY', 'real-test-key')
        client = VoyageClient()
        assert client._api_key == 'real-test-key'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, env_file):
        """Client raises ConfigError when VOYAGE_API_KEY is absent from file."""
        env_file('VOYAGE_API_KEY', None)
        with pytest.raises(ConfigError, match='VOYAGE_API_KEY'):
            VoyageClient()

    @pytest.mark.no_default_env
    def test_no_key_raises_at_construction(self, env_file):
        """Construction raises before available() can be called."""
        env_file('VOYAGE_API_KEY', None)
        with pytest.raises(ConfigError):
            VoyageClient()

    def test_available_is_memoized(self, monkeypatch):
        """available() calls the HTTP probe at most once per instance."""
        monkeypatch.setattr(VoyageClient, 'available', _original_voyage_available)
        monkeypatch.setenv('VOYAGE_API_KEY', 'probe-key')
        calls = {'n': 0}

        def _mock_post(url, headers=None, json=None, timeout=None):
            calls['n'] += 1

            class Resp:
                status_code = 200
            return Resp()

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (), {'post': staticmethod(_mock_post)})())
        client = VoyageClient()
        assert client.available() is True
        assert client.available() is True
        assert client.available() is True
        assert calls['n'] == 1

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

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (), {'post': staticmethod(mock_post)})())
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

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (), {'post': staticmethod(mock_post)})())
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

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (), {'post': staticmethod(mock_post)})())
        with pytest.raises(RuntimeError, match='0 vectors'):
            real_client.embed('test')

    def test_bearer_token(self, env_file):
        """Authorization header uses Bearer token."""
        env_file('VOYAGE_API_KEY', 'my-key')
        client = VoyageClient()
        headers = client._headers()
        assert headers['Authorization'] == 'Bearer my-key'
        assert headers['Content-Type'] == 'application/json'

    def test_unavailable_message_includes_env_var(self):
        """Unavailable message mentions VOYAGE_API_KEY."""
        client = VoyageClient()
        assert 'VOYAGE_API_KEY' in client.unavailable_message()


def _seed_openrouter_keys(env_file):
    """Write the three env vars the OpenRouter provider requires."""
    env_file(config.OPENROUTER_API_KEY, 'sk-or-test')
    env_file(config.OPENROUTER_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(config.OPENROUTER_EMBED_MODEL, 'baai/bge-m3')


def _stub_openrouter_session(monkeypatch, post_fn):
    """Replace the OpenRouter HTTP session with a fake."""
    from memman.embed import openrouter as orem
    monkeypatch.setitem(
        _http._SESSIONS, orem.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


class TestOpenRouterClient:
    """OpenRouter embedding provider -- config, embed, availability."""

    def test_constructor_reads_config(self, env_file):
        """Client reads endpoint, model, and key from env file."""
        _seed_openrouter_keys(env_file)
        client = OpenRouterClient()
        assert client.endpoint == 'https://openrouter.ai/api/v1'
        assert client.model == 'baai/bge-m3'
        assert client.dim == 0
        assert client._api_key == 'sk-or-test'

    @pytest.mark.no_default_env
    @pytest.mark.parametrize('missing_attr,match', [
        ('OPENROUTER_ENDPOINT', 'OPENROUTER_ENDPOINT'),
        ('OPENROUTER_EMBED_MODEL', 'OPENROUTER_EMBED_MODEL'),
    ])
    def test_raises_when_required_config_missing(
            self, env_file, missing_attr, match):
        """Constructor raises ConfigError when a required env key is absent."""
        present = {
            'OPENROUTER_ENDPOINT': 'https://x',
            'OPENROUTER_EMBED_MODEL': 'm',
            'OPENROUTER_API_KEY': 'k',
        }
        for key, value in present.items():
            env_file(getattr(config, key),
                     None if key == missing_attr else value)
        with pytest.raises(ConfigError, match=match):
            OpenRouterClient()

    def test_embed_returns_vector(self, monkeypatch, env_file):
        """embed() returns vector from API response and updates dim."""
        _seed_openrouter_keys(env_file)
        expected = [0.5] * 1024

        def fake_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 200

                def json(self):
                    return {'data': [{'embedding': expected}]}
            return Resp()

        _stub_openrouter_session(monkeypatch, fake_post)
        client = OpenRouterClient()
        vec = client.embed('hello')
        assert vec == expected
        assert client.dim == 1024

    def test_embed_batch_returns_one_vector_per_input(self, monkeypatch, env_file):
        """embed_batch() returns one vector per input text."""
        _seed_openrouter_keys(env_file)

        def fake_post(url, headers=None, json=None, timeout=None):
            n = len(json['input'])

            class Resp:
                status_code = 200

                def json(self):
                    return {'data': [
                        {'embedding': [float(i)] * 768} for i in range(n)
                        ]}
            return Resp()

        _stub_openrouter_session(monkeypatch, fake_post)
        client = OpenRouterClient()
        vectors = client.embed_batch(['a', 'b', 'c'])
        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)
        assert client.dim == 768

    def test_available_returns_false_on_probe_failure(self, monkeypatch, env_file):
        """available() returns False when the probe endpoint returns 401."""
        _seed_openrouter_keys(env_file)

        def fake_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 401
            return Resp()

        _stub_openrouter_session(monkeypatch, fake_post)
        client = OpenRouterClient()
        assert client.available() is False

    def test_provider_registered(self, monkeypatch):
        """Openrouter is in the embed PROVIDERS registry."""
        from memman.embed import PROVIDERS
        assert 'openrouter' in PROVIDERS

    def test_provider_factory_returns_client(self, monkeypatch, env_file):
        """PROVIDERS['openrouter']() returns a working client."""
        _seed_openrouter_keys(env_file)
        from memman.embed import PROVIDERS
        client = PROVIDERS['openrouter']()
        assert client.name == 'openrouter'
