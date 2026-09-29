"""Tests for memman.rerank.voyage, the one shipped rerank client.
"""

import pytest
from memman import _http, config
from memman.exceptions import ConfigError
from memman.rerank import voyage


@pytest.mark.no_mock_rerank
class TestVoyageClient:
    """Voyage rerank Client behavior.
    """

    def test_default_model(self, monkeypatch):
        """Client uses rerank-3-lite when MEMMAN_VOYAGE_RERANK_MODEL is unset.

        Mutation: `DEFAULT_MODEL` drifting from the value the install
        default map ships, so the two disagree on which reranker runs
        and which one runs depends on whether a config file exists.
        Oracle: the model literal, pinned in both places at once.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-1')
        monkeypatch.delenv('MEMMAN_VOYAGE_RERANK_MODEL', raising=False)
        client = voyage.Client()
        assert client.model == voyage.DEFAULT_MODEL == 'rerank-3-lite'
        assert (config.INSTALL_DEFAULTS[config.VOYAGE_RERANK_MODEL]
                == 'rerank-3-lite')

    def test_configured_model_overrides(self, env_file):
        """MEMMAN_VOYAGE_RERANK_MODEL overrides the default.

        Mutation: `Client.__init__` ignoring the configured model and
        always using `DEFAULT_MODEL`.
        Oracle: the literal `rerank-2.5` written to the env file.
        """
        env_file('MEMMAN_VOYAGE_RERANK_MODEL', 'rerank-2.5')
        client = voyage.Client()
        assert client.model == 'rerank-2.5'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, monkeypatch):
        """Missing VOYAGE_API_KEY raises ConfigError at construction.

        Mutation: `Client.__init__` reading the key with `config.get`
        instead of `config.require`, so a missing key fails later at
        the first HTTP call.
        Oracle: `pytest.raises(ConfigError)` naming the env var.
        """
        monkeypatch.delenv('MEMMAN_VOYAGE_API_KEY', raising=False)
        with pytest.raises(ConfigError, match='MEMMAN_VOYAGE_API_KEY'):
            voyage.Client()

    def test_rerank_returns_index_score_pairs(self, monkeypatch):
        """rerank() returns sorted (index, score) tuples.

        Mutation: `rerank` returning scores without their original
        index, dropping `top_k` from the request body, or posting to
        the wrong path.
        Oracle: a canned response body, and the captured request URL
        and JSON.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-3')
        captured: dict = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            captured['url'] = url
            captured['json'] = json

            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'data': [
                        {'index': 1, 'relevance_score': 0.9},
                        {'index': 0, 'relevance_score': 0.4},
                        ]}
            return Resp()

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (),
                 {'post': staticmethod(mock_post)})())
        client = voyage.Client()
        out = client.rerank('q', ['doc-a', 'doc-b'], top_k=2)
        assert out == [(1, 0.9), (0, 0.4)]
        assert captured['url'].endswith('/v1/rerank')
        assert captured['json']['query'] == 'q'
        assert captured['json']['documents'] == ['doc-a', 'doc-b']
        assert captured['json']['top_k'] == 2

    def test_rerank_empty_documents_short_circuits(self, monkeypatch):
        """Empty document list returns [] without HTTP call.

        Mutation: removing the `if not documents` guard in `rerank`,
        so an empty pool bills a rerank request.
        Oracle: a stub `post` that raises if it is called.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-4')

        def mock_post(*a, **kw):
            raise AssertionError('should not call HTTP for empty docs')

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (),
                 {'post': staticmethod(mock_post)})())
        client = voyage.Client()
        assert client.rerank('q', []) == []

    def test_rerank_raises_on_error_status(self, monkeypatch):
        """Non-200 status raises RuntimeError.

        Mutation: `rerank` parsing the body of a non-200 response
        instead of raising, so a Voyage outage reads as an empty
        result.
        Oracle: a stub response with status 503; the error must name it.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-5')

        def mock_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 503
            return Resp()

        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (),
                 {'post': staticmethod(mock_post)})())
        client = voyage.Client()
        with pytest.raises(RuntimeError, match='503'):
            client.rerank('q', ['d1'], top_k=1)

    def test_available_uses_key_presence(self, monkeypatch):
        """available() returns True when VOYAGE_API_KEY is set.

        Mutation: `available` probing the endpoint or returning False
        regardless of the key.
        Oracle: `True` with only the key set and no HTTP stub.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-6')
        assert voyage.Client().available() is True

    def test_unavailable_message_mentions_env_var(self, monkeypatch):
        """Unavailable message mentions VOYAGE_API_KEY.

        Mutation: `unavailable_message` naming a different setting, so
        an operator sets the wrong variable.
        Oracle: the literal env var name in the message.
        """
        monkeypatch.setenv('MEMMAN_VOYAGE_API_KEY', 'rk-7')
        client = voyage.Client()
        assert 'MEMMAN_VOYAGE_API_KEY' in client.unavailable_message()
