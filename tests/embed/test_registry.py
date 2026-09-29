"""Tests for the (provider, model)-keyed embedder registry.

`get_for` returns a client bound to the requested pair, surfaces an
unknown-provider `ConfigError`, and falls back to a placeholder
when credentials are missing.
"""

import threading

import pytest
from memman import config
from memman.embed import PROVIDERS, registry
from memman.embed.registry import get_for
from memman.exceptions import ConfigError, EmbedCredentialError


class TestGetFor:
    """`get_for(provider, model)` returns a client bound to the pair.
    """

    def test_returns_voyage_for_voyage_pair(self):
        """Voyage + voyage-3-lite returns a Voyage client at 512 dim.

        Mutation: `get_for` returning the env-default client, or one
            whose `prepare()` never ran.
        Oracle: the requested provider and model, and voyage-3-lite's
            512 dimension.
        """
        ec = get_for('voyage', 'voyage-3-lite')
        assert ec.name == 'voyage'
        assert ec.model == 'voyage-3-lite'
        assert ec.dim == 512

    def test_unknown_provider_raises_config_error(self):
        """Unknown provider name raises ConfigError listing known names.

        Mutation: `get_for` raising a bare `KeyError`, or omitting the
            registered names from the message.
        Oracle: the message must hold the bad name and `voyage`.
        """
        with pytest.raises(ConfigError) as excinfo:
            get_for('totally-unknown', 'whatever')
        assert 'totally-unknown' in str(excinfo.value)
        assert 'voyage' in str(excinfo.value)

    def test_overrides_model_when_provider_default_differs(
            self, env_file):
        """`get_for` sets the requested model over the env default.

        Mutation: `get_for` skipping the `client.model = model`
            override, so the env default model wins.
        Oracle: the model name passed in, which differs from the env
            file's `baai/bge-m3`.
        """
        env_file('MEMMAN_OPENROUTER_EMBED_MODEL', 'baai/bge-m3')
        ec = get_for('openrouter', 'totally-different-model')
        assert ec.name == 'openrouter'
        assert ec.model == 'totally-different-model'

    def test_get_for_returns_same_instance_when_cached(self):
        """Repeat calls with identical args return the cached client.

        Mutation: `get_for` dropping the cache write, so each call
            builds and probes a new client.
        Oracle: object identity of the two returned clients.
        """
        first = get_for('voyage', 'voyage-3-lite')
        second = get_for('voyage', 'voyage-3-lite')
        assert first is second

    def test_get_for_calls_factory_and_prepare_once_under_contention(
            self, monkeypatch):
        """Concurrent first-call workers do not double-probe the provider.

        Mutation: removing the lock or the second cache check inside
            it, so racing threads each run `factory()` and `prepare()`.
        Oracle: call counters that must read 1 after 8 threads.
        """
        registry.reset_for_tests()

        prepare_calls = 0
        factory_calls = 0

        class _Stub:
            name = 'stubprov'
            model = 'stubmodel'
            dim = 0
            _availability_cache = None

            def prepare(self):
                nonlocal prepare_calls
                prepare_calls += 1

        def _factory():
            nonlocal factory_calls
            factory_calls += 1
            return _Stub()

        monkeypatch.setitem(PROVIDERS, 'stubprov', _factory)

        results: list = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    registry.get_for('stubprov', 'stubmodel')))
            for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert factory_calls == 1
        assert prepare_calls == 1
        assert all(r is results[0] for r in results)


class TestLazyCredentialing:
    """Missing creds yield a placeholder; only embed() raises.
    """

    @pytest.mark.no_default_env
    def test_placeholder_when_creds_missing(self, tmp_path, monkeypatch):
        """`get_for` returns a placeholder when the client constructor
        raises ConfigError. Construction does not raise; only embed() does.

        Mutation: `get_for` letting the constructor's `ConfigError`
            escape instead of returning a placeholder.
        Oracle: the placeholder's `name`, `model`, and `available()`
            False, with no credentials in the isolated data dir.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        config.reset_file_cache()
        ec = get_for('openai', 'text-embedding-3-small')
        assert ec.name == 'openai'
        assert ec.model == 'text-embedding-3-small'
        assert ec.available() is False

    @pytest.mark.no_default_env
    def test_placeholder_embed_raises_credential_error(
            self, tmp_path, monkeypatch):
        """A placeholder's embed() raises EmbedCredentialError with the cause.

        Mutation: the placeholder's `embed()` returning a vector or
            raising a different error type.
        Oracle: `EmbedCredentialError` whose message names `openai`.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        config.reset_file_cache()
        ec = get_for('openai', 'text-embedding-3-small')
        with pytest.raises(EmbedCredentialError) as excinfo:
            ec.embed('hello')
        assert 'openai' in str(excinfo.value).lower()

    @pytest.mark.no_default_env
    def test_placeholder_embed_batch_raises_credential_error(
            self, tmp_path, monkeypatch):
        """Calling embed_batch() also raises EmbedCredentialError.

        Mutation: the placeholder's `embed_batch()` returning an empty
            list, so a batch drain silently embeds nothing.
        Oracle: `EmbedCredentialError` from a two-text batch.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        config.reset_file_cache()
        ec = get_for('openai', 'text-embedding-3-small')
        with pytest.raises(EmbedCredentialError):
            ec.embed_batch(['a', 'b'])
