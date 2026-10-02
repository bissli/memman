"""Tests for the model-keyed embedder registry.

`get_for` returns a client bound to the requested model, caches it per
model, and falls back to a placeholder when credentials are missing.
"""

import threading

import pytest
from memman import config
from memman.embed import registry
from memman.embed.client import Client
from memman.embed.registry import get_for
from memman.exceptions import EmbedCredentialError

MODEL = 'voyageai/voyage-4-lite'


class TestGetFor:
    """`get_for(model)` returns a client bound to the model.
    """

    def test_returns_client_bound_to_model(self, monkeypatch):
        """The client carries the requested model and a probed dim.

        Mutation: `get_for` returning a client bound to the env-default
            model, or one whose `prepare()` never ran.
        Oracle: the requested model, and the 1024-length vector the
            stubbed `embed` returns.
        """
        monkeypatch.setattr(Client, 'embed', lambda self, text: [0.0] * 1024)
        ec = get_for('some/requested-model')
        assert isinstance(ec, Client)
        assert ec.model == 'some/requested-model'
        assert ec.dim == 1024

    def test_get_for_returns_same_instance_when_cached(self):
        """Repeat calls with the same model return the cached client.

        Mutation: `get_for` dropping the cache write, so each call
            builds and probes a new client.
        Oracle: object identity of the two returned clients.
        """
        first = get_for(MODEL)
        second = get_for(MODEL)
        assert first is second

    def test_get_for_caches_per_model(self):
        """Distinct models get distinct clients.

        Mutation: a cache keyed on a constant, so the second model
            receives the first model's client.
        Oracle: two models whose clients differ and carry their own
            model names.
        """
        first = get_for('model/a')
        second = get_for('model/b')
        assert first is not second
        assert (first.model, second.model) == ('model/a', 'model/b')

    def test_get_for_calls_constructor_and_prepare_once_under_contention(
            self, monkeypatch):
        """Concurrent first-call workers do not double-probe the endpoint.

        Mutation: removing the lock or the second cache check inside
            it, so racing threads each run the constructor and `prepare()`.
        Oracle: call counters that must read 1 after 8 threads.
        """
        registry.reset_for_tests()

        prepare_calls = 0
        factory_calls = 0

        class _Stub:
            model = 'stubmodel'
            dim = 0
            _availability_cache = None

            def __init__(self, model):
                nonlocal factory_calls
                factory_calls += 1

            def prepare(self):
                nonlocal prepare_calls
                prepare_calls += 1

        monkeypatch.setattr(registry, 'Client', _Stub)

        results: list = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    registry.get_for('stubmodel')))
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
        Oracle: the placeholder's `model` and `available()` False,
            with no credentials in the isolated data dir.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        config.reset_file_cache()
        ec = get_for(MODEL)
        assert ec.model == MODEL
        assert ec.available() is False

    @pytest.mark.no_default_env
    def test_placeholder_embed_raises_credential_error(
            self, tmp_path, monkeypatch):
        """A placeholder's embed() raises EmbedCredentialError with the cause.

        Mutation: the placeholder's `embed()` returning a vector or
            raising a different error type.
        Oracle: `EmbedCredentialError` whose message names the model.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        config.reset_file_cache()
        ec = get_for(MODEL)
        with pytest.raises(EmbedCredentialError) as excinfo:
            ec.embed('hello')
        assert MODEL in str(excinfo.value)

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
        ec = get_for(MODEL)
        with pytest.raises(EmbedCredentialError):
            ec.embed_batch(['a', 'b'])
