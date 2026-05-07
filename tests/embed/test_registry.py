"""Tests for the (provider, model)-keyed embedder registry.

`get_for` returns a client bound to the requested pair, surfaces an
unknown-provider `ConfigError`, and falls back to a placeholder
when credentials are missing.
"""

import pytest


class TestGetFor:
    """`get_for(provider, model)` returns a client bound to the pair."""

    def test_returns_voyage_for_voyage_pair(self):
        """Voyage + voyage-3-lite returns a Voyage client at 512 dim."""
        from memman.embed.registry import get_for
        ec = get_for('voyage', 'voyage-3-lite')
        assert ec.name == 'voyage'
        assert ec.model == 'voyage-3-lite'
        assert ec.dim == 512

    def test_unknown_provider_raises_config_error(self):
        """Unknown provider name raises ConfigError listing known names."""
        from memman.embed.registry import get_for
        from memman.exceptions import ConfigError
        with pytest.raises(ConfigError) as excinfo:
            get_for('totally-unknown', 'whatever')
        assert 'totally-unknown' in str(excinfo.value)
        assert 'voyage' in str(excinfo.value)

    def test_overrides_model_when_provider_default_differs(
            self, env_file):
        """get_for('openrouter', 'custom-model') returns a client whose
        model attribute is the requested override, not the env default.
        """
        from memman.embed.registry import get_for
        env_file('MEMMAN_OPENROUTER_EMBED_MODEL', 'baai/bge-m3')
        ec = get_for('openrouter', 'totally-different-model')
        assert ec.name == 'openrouter'
        assert ec.model == 'totally-different-model'

    def test_get_for_returns_same_instance_when_cached(self):
        """Repeat calls with identical args return the cached client.

        The lru_cache on `get_for` ensures `factory()` + `prepare()`
        run only once per (provider, model) pair per process.
        """
        from memman.embed.registry import get_for
        first = get_for('voyage', 'voyage-3-lite')
        second = get_for('voyage', 'voyage-3-lite')
        assert first is second


class TestLazyCredentialing:
    """Missing creds yield a placeholder; only embed() raises."""

    @pytest.mark.no_default_env
    def test_placeholder_when_creds_missing(self, tmp_path, monkeypatch):
        """get_for returns a placeholder when the underlying client's
        constructor raises ConfigError (missing creds). Construction
        does not raise; only embed() does.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        from memman import config
        config.reset_file_cache()
        from memman.embed.registry import get_for
        ec = get_for('openai', 'text-embedding-3-small')
        assert ec.name == 'openai'
        assert ec.model == 'text-embedding-3-small'
        assert ec.available() is False

    @pytest.mark.no_default_env
    def test_placeholder_embed_raises_credential_error(
            self, tmp_path, monkeypatch):
        """Calling embed() on a placeholder raises EmbedCredentialError
        with the underlying reason embedded in the message.
        """
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        from memman import config
        config.reset_file_cache()
        from memman.embed.registry import get_for
        from memman.exceptions import EmbedCredentialError
        ec = get_for('openai', 'text-embedding-3-small')
        with pytest.raises(EmbedCredentialError) as excinfo:
            ec.embed('hello')
        assert 'openai' in str(excinfo.value).lower()

    @pytest.mark.no_default_env
    def test_placeholder_embed_batch_raises_credential_error(
            self, tmp_path, monkeypatch):
        """Calling embed_batch() also raises EmbedCredentialError."""
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(tmp_path))
        from memman import config
        config.reset_file_cache()
        from memman.embed.registry import get_for
        from memman.exceptions import EmbedCredentialError
        ec = get_for('openai', 'text-embedding-3-small')
        with pytest.raises(EmbedCredentialError):
            ec.embed_batch(['a', 'b'])
