"""Tests for the `prepare()` Protocol method on embedder clients.

`prepare()` is the explicit dim-probe contract: callers can rely on
`ec.dim` being populated after `prepare()` returns, without having
to know whether the underlying provider needs a probe. Voyage knows
its dim at construction (no-op `prepare()`); openai_compat,
openrouter, and ollama lazily fetch and cache.

The registry calls `prepare()` after construction so consumers such as
the swap workflow can read `ec.dim` directly.
"""

from memman.embed.openai_compat import Client as OpenAIClient
from memman.embed.registry import get_for
from memman.embed.voyage import Client


class TestVoyagePrepare:
    """Voyage knows its dim at construction; prepare() is a no-op.
    """

    def test_dim_is_set_before_prepare(self):
        """Voyage exposes `dim=512` immediately after construction.

        Mutation: `Client.__init__` leaving `dim` at 0 for the default
            model.
        Oracle: the documented voyage-3-lite dimension, 512.
        """
        ec = Client()
        assert ec.dim == 512

    def test_prepare_is_idempotent(self):
        """Calling prepare() repeatedly does not change `dim`.

        Mutation: `prepare()` resetting or re-probing `dim` on a client
            that already knows it.
        Oracle: the constant 512 before and after two calls.
        """
        ec = Client()
        ec.prepare()
        ec.prepare()
        assert ec.dim == 512

    def test_model_resolves_from_config(self, env_file):
        """`Client.model` reads from MEMMAN_VOYAGE_EMBED_MODEL when set.

        Mutation: `Client` ignoring the configured model and using a
            hardcoded default.
        Oracle: the model name the test wrote to the env file.
        """
        env_file('MEMMAN_VOYAGE_EMBED_MODEL', 'voyage-3-lite')
        ec = Client()
        assert ec.model == 'voyage-3-lite'
        assert ec.dim == 512

    def test_non_default_model_starts_with_dim_zero(self, env_file, monkeypatch):
        """Non-default model: dim=0 at construction so prepare() probes.

        Mutation: `Client` assuming dim 512 for a model it does not
            know, or `prepare()` skipping the probe.
        Oracle: the 1024-length vector the stubbed `embed` returns.
        """
        env_file('MEMMAN_VOYAGE_EMBED_MODEL', 'voyage-3-large')

        def _fake_embed(self, text):
            return [0.0] * 1024

        monkeypatch.setattr(
            'memman.embed.voyage.Client.embed', _fake_embed)
        ec = Client()
        assert ec.model == 'voyage-3-large'
        assert ec.dim == 0
        ec.prepare()
        assert ec.dim == 1024


class TestOpenAIPrepare:
    """openai_compat's prepare() performs the dim probe lazily.
    """

    def test_prepare_sets_dim(self, monkeypatch):
        """prepare() probes the endpoint and caches dim on the client.

        Mutation: `prepare()` not storing the probed length in `dim`.
        Oracle: the 1536-length vector the stubbed `embed` returns.
        """

        def _fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _fake_embed)
        ec = OpenAIClient()
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

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _counting_embed)
        ec = OpenAIClient()
        ec.prepare()
        ec.prepare()
        assert call_count['n'] == 1


class TestRegistryCallsPrepare:
    """get_for(provider, model) returns a client whose dim is populated.
    """

    def test_voyage_dim_populated(self):
        """Voyage client returned by registry has dim ready to read.

        Mutation: `get_for` skipping `prepare()` on a new client.
        Oracle: the documented voyage-3-lite dimension, 512.
        """
        ec = get_for('voyage', 'voyage-3-lite')
        assert ec.dim == 512

    def test_openai_dim_populated(self, monkeypatch):
        """Openai client returned by registry has dim populated via probe.

        Mutation: `get_for` skipping `prepare()`, leaving `dim` at 0
            for a provider that probes lazily.
        Oracle: the 1536-length vector the stubbed `embed` returns.
        """

        def _fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _fake_embed)
        ec = get_for('openai', 'text-embedding-3-small')
        assert ec.dim == 1536
