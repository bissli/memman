"""Tests for the (provider, model)-keyed embedder registry.

The registry shifts runtime authority from the env-resolved global
provider to per-(provider, model) lookup. Used by `_StoreContext`
to bind a per-store embedder from the store's stored fingerprint.

Behaviors covered:
- `get_for(provider, model)` returns a client whose model matches
- `get_active()` matches the env-resolved client
- Lazy credentialing: missing creds yield a placeholder; only
  `embed()` / `embed_batch()` raise `EmbedCredentialError`
- Unknown provider raises `ConfigError`
"""

import pytest


class TestGetActive:
    """`get_active()` mirrors the env-resolved active client."""

    def test_returns_voyage_by_default(self):
        """Default env points at voyage; get_active() returns voyage."""
        from memman.embed.registry import get_active
        ec = get_active()
        assert ec.name == 'voyage'
        assert ec.model == 'voyage-3-lite'
        assert ec.dim == 512


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
