"""Tests for the `prepare()` Protocol method on embedder clients.

`prepare()` is the explicit dim-probe contract: callers can rely on
`ec.dim` being populated after `prepare()` returns, without having
to know whether the underlying provider needs a probe. Voyage knows
its dim at construction (no-op `prepare()`); openai_compat,
openrouter, and ollama lazily fetch and cache.

The registry calls `prepare()` after construction so consumers
(e.g., A2's swap workflow) can read `ec.dim` directly.
"""


class TestVoyagePrepare:
    """Voyage knows its dim at construction; prepare() is a no-op."""

    def test_dim_is_set_before_prepare(self):
        """Voyage exposes `dim=512` immediately after construction."""
        from memman.embed.voyage import Client
        ec = Client()
        assert ec.dim == 512

    def test_prepare_is_idempotent(self):
        """Calling prepare() repeatedly does not change `dim`."""
        from memman.embed.voyage import Client
        ec = Client()
        ec.prepare()
        ec.prepare()
        assert ec.dim == 512


class TestOpenAIPrepare:
    """openai_compat's prepare() performs the dim probe lazily."""

    def test_prepare_sets_dim(self, monkeypatch):
        """prepare() probes the endpoint and caches dim on the client."""
        from memman.embed.openai_compat import Client

        def _fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _fake_embed)
        ec = Client()
        assert ec.dim == 0
        ec.prepare()
        assert ec.dim == 1536

    def test_prepare_idempotent(self, monkeypatch):
        """Subsequent prepare() calls do not re-probe."""
        from memman.embed.openai_compat import Client

        call_count = {'n': 0}

        def _counting_embed(self, text):
            call_count['n'] += 1
            return [0.2] * 8

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _counting_embed)
        ec = Client()
        ec.prepare()
        ec.prepare()
        assert call_count['n'] == 1


class TestRegistryCallsPrepare:
    """get_for(provider, model) returns a client whose dim is populated."""

    def test_voyage_dim_populated(self):
        """Voyage client returned by registry has dim ready to read."""
        from memman.embed.registry import get_for
        ec = get_for('voyage', 'voyage-3-lite')
        assert ec.dim == 512

    def test_openai_dim_populated(self, monkeypatch):
        """Openai client returned by registry has dim populated via probe."""

        def _fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(
            'memman.embed.openai_compat.Client.embed', _fake_embed)
        from memman.embed.registry import get_for
        ec = get_for('openai', 'text-embedding-3-small')
        assert ec.dim == 1536
