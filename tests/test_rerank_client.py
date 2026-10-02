"""Tests for memman.rerank.client, the one shipped rerank client.
"""

import pytest
from memman import _http, config
from memman.exceptions import ConfigError
from memman.rerank import client as rerank_client


def _stub_session(monkeypatch, post_fn):
    """Replace the rerank client's HTTP session with a fake.
    """
    monkeypatch.setitem(
        _http._SESSIONS, rerank_client.__name__,
        type('FakeClient', (), {'post': staticmethod(post_fn)})())


@pytest.mark.no_mock_rerank
class TestRerankClient:
    """Rerank Client behavior.
    """

    def test_default_model(self):
        """Client uses the install-default rerank model.

        Mutation: the install default map drifting from the model the
        shipped docs name, so which reranker runs depends on whether a
        config file exists.
        Oracle: the model literal, pinned in the install defaults.
        """
        client = rerank_client.Client()
        assert client.model == 'voyageai/rerank-3-lite'
        assert (config.INSTALL_DEFAULTS[config.RERANK_MODEL]
                == 'voyageai/rerank-3-lite')

    def test_configured_model_overrides(self, env_file):
        """MEMMAN_RERANK_MODEL overrides the default.

        Mutation: `Client.__init__` ignoring the configured model and
        always using the install default.
        Oracle: the literal `voyageai/rerank-2.5` written to the env file.
        """
        env_file('MEMMAN_RERANK_MODEL', 'voyageai/rerank-2.5')
        assert rerank_client.Client().model == 'voyageai/rerank-2.5'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, env_file):
        """Missing MEMMAN_API_KEY raises ConfigError at construction.

        Mutation: `Client.__init__` reading the key with `config.get`
        instead of `config.api_key_for`, so a missing key fails later at
        the first HTTP call.
        Oracle: `pytest.raises(ConfigError)` naming the env var.
        """
        env_file('MEMMAN_ENDPOINT', 'https://openrouter.ai/api/v1')
        env_file('MEMMAN_RERANK_MODEL', 'voyageai/rerank-3-lite')
        env_file('MEMMAN_API_KEY', None)
        with pytest.raises(ConfigError, match='MEMMAN_API_KEY'):
            rerank_client.Client()

    def test_rerank_returns_index_score_pairs(self, monkeypatch):
        """rerank() returns sorted (index, score) tuples.

        Mutation: `rerank` returning scores without their original
        index, dropping `top_n` from the request body, reading `data`
        instead of `results`, or posting to the wrong path.
        Oracle: a canned response body, and the captured request URL
        and JSON.
        """
        captured: dict = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            captured['url'] = url
            captured['json'] = json

            class Resp:
                status_code = 200

                def json(self_inner):
                    return {'results': [
                        {'index': 1, 'relevance_score': 0.9},
                        {'index': 0, 'relevance_score': 0.4},
                        ]}
            return Resp()

        _stub_session(monkeypatch, mock_post)
        out = rerank_client.Client().rerank('q', ['doc-a', 'doc-b'], top_n=2)
        assert out == [(1, 0.9), (0, 0.4)]
        assert captured['url'].endswith('/rerank')
        assert captured['json']['query'] == 'q'
        assert captured['json']['documents'] == ['doc-a', 'doc-b']
        assert captured['json']['top_n'] == 2

    def test_rerank_empty_documents_short_circuits(self, monkeypatch):
        """Empty document list returns [] without HTTP call.

        Mutation: removing the `if not documents` guard in `rerank`,
        so an empty pool bills a rerank request.
        Oracle: a stub `post` that raises if it is called.
        """

        def mock_post(*a, **kw):
            raise AssertionError('should not call HTTP for empty docs')

        _stub_session(monkeypatch, mock_post)
        assert rerank_client.Client().rerank('q', []) == []

    def test_rerank_raises_on_error_status(self, monkeypatch):
        """Non-200 status raises RuntimeError.

        Mutation: `rerank` parsing the body of a non-200 response
        instead of raising, so an outage reads as an empty result.
        Oracle: a stub response with status 503; the error must name it.
        """

        def mock_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 503
            return Resp()

        _stub_session(monkeypatch, mock_post)
        with pytest.raises(RuntimeError, match='503'):
            rerank_client.Client().rerank('q', ['d1'], top_n=1)
