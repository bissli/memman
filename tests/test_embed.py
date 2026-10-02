"""Tests for memman.embed: vector serialization and the embed clients.
"""

import math

import pytest
from memman import _http
from memman.embed import client as embed_client
from memman.embed.client import Client
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.exceptions import ConfigError

EMBEDDING_DIM = 512
MODEL = 'voyageai/voyage-4-lite'


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


def _stub_session(monkeypatch, post_fn):
    """Replace the embed client's HTTP session with a fake.
    """
    monkeypatch.setitem(
        _http._SESSIONS, embed_client.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


@pytest.mark.no_mock_embed
class TestEmbedClient:
    """Embed client -- init, availability, embed, headers.
    """

    def test_api_key_from_env_file(self, env_file):
        """Verify the client takes its key from the env file.

        Mutation: reading the key from a constant or a stale cache instead of
            config, so the file value is ignored.
        Oracle: a distinctive key written to the env file and read back from
            the client.
        """
        env_file('MEMMAN_API_KEY', 'real-test-key')
        client = Client(MODEL)
        assert client._api_key == 'real-test-key'

    def test_constructor_reads_config(self, env_file):
        """Verify the client binds the endpoint and the model it is given.

        Mutation: swapping two config keys, reading the model from config
            instead of the argument, or starting with a non-zero dim.
        Oracle: the distinct literals written to the env file and passed in,
            and dim 0 before any embed.
        """
        env_file('MEMMAN_ENDPOINT', 'https://example.test/v1/')
        client = Client('some/other-model')
        assert client.endpoint == 'https://example.test/v1'
        assert client.model == 'some/other-model'
        assert client.dim == 0

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, env_file):
        """Verify construction raises ConfigError naming the missing key.

        Mutation: reading the key with config.get in place of
            config.api_key_for, so a keyless client builds; or moving the
            key check out of __init__ into available() or embed().
        Oracle: ConfigError, whose message contains MEMMAN_API_KEY, raised
            by Client() itself on a non-loopback endpoint.
        """
        env_file('MEMMAN_ENDPOINT', 'https://openrouter.ai/api/v1')
        env_file('MEMMAN_API_KEY', None)
        with pytest.raises(ConfigError, match='MEMMAN_API_KEY'):
            Client(MODEL)

    @pytest.mark.no_default_env
    def test_missing_endpoint_raises(self, env_file):
        """Verify construction raises ConfigError when the endpoint is unset.

        Mutation: reading the endpoint with config.get in place of
            config.require.
        Oracle: ConfigError naming MEMMAN_ENDPOINT.
        """
        env_file('MEMMAN_ENDPOINT', None)
        env_file('MEMMAN_API_KEY', 'k')
        with pytest.raises(ConfigError, match='MEMMAN_ENDPOINT'):
            Client(MODEL)

    def test_available_is_memoized(self, monkeypatch):
        """Verify available() probes the HTTP endpoint once per instance.

        Mutation: dropping the _availability_cache read, so every call sends
            another billable probe.
        Oracle: a call counter on the stubbed session equals 1 after three
            calls.
        """
        calls = {'n': 0}

        def _mock_post(url, headers=None, json=None, timeout=None):
            calls['n'] += 1

            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': [{'embedding': [0.1] * 4}]}
            return Resp()

        _stub_session(monkeypatch, _mock_post)
        client = Client(MODEL)
        assert client.available() is True
        assert client.available() is True
        assert client.available() is True
        assert calls['n'] == 1

    def test_available_returns_false_on_probe_failure(self, monkeypatch):
        """Verify available() returns False when the probe gets a 401.

        Mutation: letting the probe's RuntimeError escape, or returning True
            regardless of the probe.
        Oracle: a stubbed 401 response and the literal False.
        """

        def fake_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 401
            return Resp()

        _stub_session(monkeypatch, fake_post)
        assert Client(MODEL).available() is False

    def test_embed_returns_vector(self, monkeypatch):
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

        _stub_session(monkeypatch, mock_post)
        vec = Client(MODEL).embed('test text')
        assert len(vec) == EMBEDDING_DIM
        assert vec == expected_vec

    def test_embed_records_dim(self, monkeypatch):
        """Verify embed() records the response width as dim.

        Mutation: leaving dim at 0 after the first embed.
        Oracle: dim equal to the stub vector length (1024).
        """

        def fake_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 200

                def json(self):
                    return {'data': [{'embedding': [0.5] * 1024}]}
            return Resp()

        _stub_session(monkeypatch, fake_post)
        client = Client(MODEL)
        client.embed('hello')
        assert client.dim == 1024

    def test_embed_batch_returns_one_vector_per_input(self, monkeypatch):
        """Verify embed_batch() returns one vector per input text.

        Mutation: sending only the first text, or returning fewer vectors than
            inputs.
        Oracle: the stub answers with one vector per posted text, so a count of
            3 at width 768 shows all three texts were sent.
        """

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
        client = Client(MODEL)
        vectors = client.embed_batch(['a', 'b', 'c'])
        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)
        assert client.dim == 768

    def test_embed_raises_on_error_status(self, monkeypatch):
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

        _stub_session(monkeypatch, mock_post)
        with pytest.raises(RuntimeError, match='401'):
            Client(MODEL).embed('test')

    def test_embed_raises_on_empty_data(self, monkeypatch):
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

        _stub_session(monkeypatch, mock_post)
        with pytest.raises(RuntimeError, match='0 vectors'):
            Client(MODEL).embed('test')

    def test_embed_raises_on_null_embedding(self, monkeypatch):
        """Verify embed() raises RuntimeError for a row with no embedding.

        Mutation: dropping the None check, so a null row reaches the caller
            as a vector.
        Oracle: RuntimeError whose message contains 'no embedding'.
        """

        def mock_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': [{'embedding': None}]}
            return Resp()

        _stub_session(monkeypatch, mock_post)
        with pytest.raises(RuntimeError, match='no embedding'):
            Client(MODEL).embed('test')

    def test_bearer_token(self, monkeypatch, env_file):
        """Verify the request sends the key as a Bearer token with a JSON type.

        Mutation: a different auth scheme, or a dropped Content-Type header.
        Oracle: the literal header values for the key written to the env file.
        """
        env_file('MEMMAN_API_KEY', 'my-key')
        captured = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            captured['headers'] = headers

            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': [{'embedding': [0.1]}]}
            return Resp()

        _stub_session(monkeypatch, mock_post)
        Client(MODEL).embed('x')
        assert captured['headers']['Authorization'] == 'Bearer my-key'
        assert captured['headers']['Content-Type'] == 'application/json'

    def test_unavailable_message_includes_env_var(self):
        """Verify unavailable_message() names the env var that fixes it.

        Mutation: a message that omits config.API_KEY, leaving the user
            no fix.
        Oracle: the literal variable name MEMMAN_API_KEY.
        """
        client = Client(MODEL)
        assert 'MEMMAN_API_KEY' in client.unavailable_message()


@pytest.mark.no_mock_embed
class TestPrepare:
    """prepare() learns dim with one probe.
    """

    def test_prepare_sets_dim(self, monkeypatch):
        """prepare() probes the endpoint and caches dim on the client.

        Mutation: `prepare()` not storing the probed length in `dim`.
        Oracle: the 1536-length vector the stubbed `embed` returns.
        """

        def _fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(Client, 'embed', _fake_embed)
        ec = Client(MODEL)
        assert ec.dim == 0
        ec.prepare()
        assert ec.dim == 1536

    def test_prepare_idempotent(self, monkeypatch):
        """Subsequent prepare() calls do not re-probe.

        Mutation: `prepare()` calling `embed` on every call, paying a
            network probe each time.
        Oracle: a counting stub that must see exactly one call.
        """

        call_count = {'n': 0}

        def _counting_embed(self, text):
            call_count['n'] += 1
            return [0.2] * 8

        monkeypatch.setattr(Client, 'embed', _counting_embed)
        ec = Client(MODEL)
        ec.prepare()
        ec.prepare()
        assert call_count['n'] == 1
