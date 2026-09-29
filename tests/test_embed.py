"""Tests for memman.embed: vector serialization and the embed clients.
"""

import math

import pytest
from memman import _http, config
from memman.embed import PROVIDERS
from memman.embed import openrouter as orem
from memman.embed import voyage
from memman.embed.openrouter import Client as OpenRouterClient
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.embed.voyage import EMBEDDING_DIM
from memman.embed.voyage import Client as VoyageClient
from memman.exceptions import ConfigError

_original_voyage_embed = VoyageClient.embed
_original_voyage_embed_batch = VoyageClient.embed_batch
_original_voyage_available = VoyageClient.available


class TestEmbedUtils:
    """Vector serialization utilities.
    """

    def test_serialize_deserialize_roundtrip(self):
        """Verify a float64 vector survives serialize then deserialize exactly.

        Mutation: packing float32 ('f'), which loses precision on pi, or mixing
            byte orders between the two functions.
        Oracle: the input list compared element by element, with inf checked by
            isinf.
        """
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
        """Verify None and an empty vector serialize to empty bytes.

        Mutation: dropping the `if not v` guard, so None raises TypeError in
            the pack call.
        Oracle: the literal b'' for both inputs.
        """
        assert serialize_vector(None) == b''
        assert serialize_vector([]) == b''

    def test_deserialize_empty(self):
        """Verify None and empty bytes deserialize to None.

        Mutation: dropping the `if not b` guard, so b'' returns [] and None
            raises.
        Oracle: the literal None for both inputs.
        """
        assert deserialize_vector(None) is None
        assert deserialize_vector(b'') is None

    def test_deserialize_invalid_length(self):
        """Verify a blob whose length is not a multiple of 8 returns None.

        Mutation: dropping the `% 8` guard, so struct.unpack raises on a
            truncated blob.
        Oracle: a 7-byte blob, one short of a whole float64.
        """
        assert deserialize_vector(bytes(7)) is None


class TestVoyageClient:
    """Voyage AI embedding client -- init, availability, embed, headers.
    """

    @pytest.fixture
    def real_client(self, monkeypatch):
        """Client with real methods restored (undo autouse mock).
        """
        monkeypatch.setattr(VoyageClient, 'embed', _original_voyage_embed)
        monkeypatch.setattr(VoyageClient, 'embed_batch', _original_voyage_embed_batch)
        monkeypatch.setattr(VoyageClient, 'available', _original_voyage_available)
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'test-key-123')
        return VoyageClient()

    def test_api_key_from_env_file(self, env_file):
        """Verify the client takes its key from the env file.

        Mutation: reading the key from a constant or a stale cache instead of
            config, so the file value is ignored.
        Oracle: a distinctive key written to the env file and read back from
            the client.
        """
        env_file('MEMMAN_VOYAGE_API_KEY', 'real-test-key')
        client = VoyageClient()
        assert client._api_key == 'real-test-key'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, env_file):
        """Verify construction raises ConfigError naming the missing Voyage key.

        Mutation: reading the key with config.get in place of config.require,
            so a keyless client builds; or moving the key check out of
            __init__ into available() or embed().
        Oracle: ConfigError, whose message contains MEMMAN_VOYAGE_API_KEY,
            raised by VoyageClient() itself.
        """
        env_file('MEMMAN_VOYAGE_API_KEY', None)
        with pytest.raises(ConfigError, match='MEMMAN_VOYAGE_API_KEY'):
            VoyageClient()

    def test_available_is_memoized(self, monkeypatch):
        """Verify available() probes the HTTP endpoint once per instance.

        Mutation: dropping the _availability_cache read, so every call sends
            another billable probe.
        Oracle: a call counter on the stubbed session equals 1 after three
            calls.
        """
        monkeypatch.setattr(VoyageClient, 'available', _original_voyage_available)
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'probe-key')
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
        """Verify embed() returns the vector from the API response.

        Mutation: returning the whole response, or the wrong element of `data`,
            in place of data[0]['embedding'].
        Oracle: the vector the stubbed API was told to return.
        """
        expected_vec = [0.1] * EMBEDDING_DIM

        def mock_post(url, headers=None, json=None, timeout=None):
            """Return valid embedding response.
            """
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
        """Verify embed() raises RuntimeError naming a non-200 status.

        Mutation: dropping the status check, so a 401 body is parsed as an
            embedding.
        Oracle: RuntimeError whose message contains the stubbed status 401.
        """
        def mock_post(url, headers=None, json=None, timeout=None):
            """Return 401 unauthorized.
            """
            class Resp:
                status_code = 401
            return Resp()

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (), {'post': staticmethod(mock_post)})())
        with pytest.raises(RuntimeError, match='401'):
            real_client.embed('test')

    def test_embed_raises_on_empty_data(self, real_client, monkeypatch):
        """Verify embed() raises RuntimeError when the API returns no vectors.

        Mutation: dropping the length check, so an empty `data` array surfaces
            as IndexError.
        Oracle: RuntimeError whose message contains '0 vectors'.
        """
        def mock_post(url, headers=None, json=None, timeout=None):
            """Return 200 with empty data array.
            """
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
        """Verify _headers() sends the key as a Bearer token with a JSON type.

        Mutation: a different auth scheme, or a dropped Content-Type header.
        Oracle: the literal header values for the key written to the env file.
        """
        env_file('MEMMAN_VOYAGE_API_KEY', 'my-key')
        client = VoyageClient()
        headers = client._headers()
        assert headers['Authorization'] == 'Bearer my-key'
        assert headers['Content-Type'] == 'application/json'

    def test_unavailable_message_includes_env_var(self):
        """Verify unavailable_message() names the env var that fixes it.

        Mutation: a message that omits config.VOYAGE_API_KEY, leaving the user
            no fix.
        Oracle: the literal variable name MEMMAN_VOYAGE_API_KEY.
        """
        client = VoyageClient()
        assert 'MEMMAN_VOYAGE_API_KEY' in client.unavailable_message()


def _seed_openrouter_keys(env_file):
    """Write the three env vars the OpenRouter provider requires.
    """
    env_file(config.OPENROUTER_API_KEY, 'sk-or-test')
    env_file(config.OPENROUTER_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(config.OPENROUTER_EMBED_MODEL, 'baai/bge-m3')


def _stub_openrouter_session(monkeypatch, post_fn):
    """Replace the OpenRouter HTTP session with a fake.
    """
    monkeypatch.setitem(
        _http._SESSIONS, orem.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


class TestOpenRouterClient:
    """OpenRouter embedding provider -- config, embed, availability.
    """

    def test_constructor_reads_config(self, env_file):
        """Verify the OpenRouter client reads endpoint, model and key from config.

        Mutation: swapping two config keys, or starting with a non-zero dim.
        Oracle: the distinct literals written to the env file, and dim 0 before
            any embed.
        """
        _seed_openrouter_keys(env_file)
        client = OpenRouterClient()
        assert client.endpoint == 'https://openrouter.ai/api/v1'
        assert client.model == 'baai/bge-m3'
        assert client.dim == 0
        assert client._api_key == 'sk-or-test'

    @pytest.mark.no_default_env
    @pytest.mark.parametrize(('missing_attr', 'match'), [
        ('OPENROUTER_ENDPOINT', 'OPENROUTER_ENDPOINT'),
        ('OPENROUTER_EMBED_MODEL', 'OPENROUTER_EMBED_MODEL'),
    ])
    def test_raises_when_required_config_missing(
            self, env_file, missing_attr, match):
        """Verify construction raises ConfigError for each missing required key.

        Mutation: reading the endpoint or the embed model with config.get in
            place of config.require.
        Oracle: ConfigError naming the one key removed, for each parametrized
            key.
        """
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
        """Verify embed() returns the response vector and records its width.

        Mutation: returning the wrong element, or leaving dim at 0 after the
            first embed.
        Oracle: the vector the stub returns, and dim equal to its length
            (1024).
        """
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
        """Verify embed_batch() returns one vector per input text.

        Mutation: sending only the first text, or returning fewer vectors than
            inputs.
        Oracle: the stub answers with one vector per posted text, so a count of
            3 at width 768 shows all three texts were sent.
        """
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
        """Verify available() returns False when the probe gets a 401.

        Mutation: letting the probe's RuntimeError escape, or returning True
            regardless of the probe.
        Oracle: a stubbed 401 response and the literal False.
        """
        _seed_openrouter_keys(env_file)

        def fake_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 401
            return Resp()

        _stub_openrouter_session(monkeypatch, fake_post)
        client = OpenRouterClient()
        assert client.available() is False

    def test_provider_registered(self):
        """Verify openrouter is registered in PROVIDERS.

        Mutation: dropping the 'openrouter' entry, so the provider cannot be
            selected.
        Oracle: the literal key 'openrouter'.
        """
        assert 'openrouter' in PROVIDERS

    def test_provider_factory_returns_client(self, monkeypatch, env_file):
        """Verify the openrouter factory builds the OpenRouter client.

        Mutation: registering another provider's class under 'openrouter'.
        Oracle: the literal client name 'openrouter'.
        """
        _seed_openrouter_keys(env_file)
        client = PROVIDERS['openrouter']()
        assert client.name == 'openrouter'
