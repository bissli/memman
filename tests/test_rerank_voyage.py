"""Tests for memman.rerank.voyage and the rerank provider registry."""

import pytest
from memman.rerank import RERANKERS, Reranker, get_client, voyage


class TestProviderRegistry:
    """Top-level rerank/__init__ behavior."""

    def test_voyage_registered(self):
        """The voyage factory is in RERANKERS."""
        assert 'voyage' in RERANKERS

    def test_get_client_defaults_to_voyage(self, monkeypatch):
        """When MEMMAN_RERANK_PROVIDER is unset, get_client returns voyage.
        """
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-defaults')
        monkeypatch.delenv('MEMMAN_RERANK_PROVIDER', raising=False)
        client = get_client()
        assert client.name == 'voyage'

    def test_get_client_unknown_provider_raises(self, env_file):
        """Unknown provider name surfaces as ConfigError."""
        from memman.exceptions import ConfigError
        env_file('MEMMAN_RERANK_PROVIDER', 'nosuch')
        with pytest.raises(ConfigError, match='nosuch'):
            get_client()

    def test_protocol_attributes(self):
        """Reranker Protocol declares name + model + the three methods.
        """
        assert hasattr(Reranker, 'rerank')
        assert hasattr(Reranker, 'available')
        assert hasattr(Reranker, 'unavailable_message')


class TestVoyageClient:
    """Voyage rerank Client behavior."""

    def test_default_model(self, monkeypatch):
        """Client uses rerank-2.5-lite when MEMMAN_VOYAGE_RERANK_MODEL is unset.
        """
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-1')
        monkeypatch.delenv('MEMMAN_VOYAGE_RERANK_MODEL', raising=False)
        client = voyage.Client()
        assert client.model == voyage.DEFAULT_MODEL == 'rerank-2.5-lite'

    def test_configured_model_overrides(self, env_file):
        """MEMMAN_VOYAGE_RERANK_MODEL overrides the default."""
        env_file('MEMMAN_VOYAGE_RERANK_MODEL', 'rerank-2.5')
        client = voyage.Client()
        assert client.model == 'rerank-2.5'

    @pytest.mark.no_default_env
    def test_missing_api_key_raises(self, monkeypatch):
        """Missing VOYAGE_API_KEY raises ConfigError at construction."""
        from memman.exceptions import ConfigError
        monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
        with pytest.raises(ConfigError, match='VOYAGE_API_KEY'):
            voyage.Client()

    def test_rerank_returns_index_score_pairs(self, monkeypatch):
        """rerank() returns sorted (index, score) tuples."""
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-3')
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

        from memman import _http
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
        """Empty document list returns [] without HTTP call."""
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-4')

        def mock_post(*a, **kw):
            raise AssertionError('should not call HTTP for empty docs')

        from memman import _http
        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (),
                 {'post': staticmethod(mock_post)})())
        client = voyage.Client()
        assert client.rerank('q', []) == []

    def test_rerank_raises_on_error_status(self, monkeypatch):
        """Non-200 status raises RuntimeError."""
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-5')

        def mock_post(url, headers=None, json=None, timeout=None):
            class Resp:
                status_code = 503
            return Resp()

        from memman import _http
        monkeypatch.setitem(
            _http._SESSIONS, voyage.__name__,
            type('FakeClient', (),
                 {'post': staticmethod(mock_post)})())
        client = voyage.Client()
        with pytest.raises(RuntimeError, match='503'):
            client.rerank('q', ['d1'], top_k=1)

    def test_available_uses_key_presence(self, monkeypatch):
        """available() returns True when VOYAGE_API_KEY is set."""
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-6')
        assert voyage.Client().available() is True

    def test_unavailable_message_mentions_env_var(self, monkeypatch):
        """Unavailable message mentions VOYAGE_API_KEY."""
        monkeypatch.setenv('VOYAGE_API_KEY', 'rk-7')
        client = voyage.Client()
        assert 'VOYAGE_API_KEY' in client.unavailable_message()
