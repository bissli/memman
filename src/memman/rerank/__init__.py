"""Rerank provider protocol, registry, and selector.

`Reranker` is the structural contract every reranker provider must
satisfy. Concrete clients (`rerank/voyage.py`) register themselves in
the `RERANKERS` dict via a zero-arg factory. `get_client()` resolves
the active provider by looking up `MEMMAN_RERANK_PROVIDER`.

Mirrors the embed-provider pattern so adding a second reranker
(Cohere, Qwen3-Reranker) is one file plus one registry entry.
"""

from collections.abc import Callable
from typing import Protocol

from memman import config
from memman.exceptions import ConfigError


class Reranker(Protocol):
    """Structural contract every reranker client must satisfy."""

    name: str
    model: str

    def available(self) -> bool:
        """Return True when the reranker's API is reachable.
        """
        ...

    def rerank(self, query: str, documents: list[str],
               top_k: int | None = None) -> list[tuple[int, float]]:
        """Score each document against the query.

        Returns a list of `(original_index, relevance_score)` tuples
        sorted by score descending. `top_k` truncates the returned
        list when set.
        """
        ...

    def unavailable_message(self) -> str:
        """Return a user-facing message explaining why unavailable.
        """
        ...


def _voyage_factory() -> Reranker:
    """Build the registered Voyage rerank client."""
    from memman.rerank.voyage import Client
    return Client()


RERANKERS: dict[str, Callable[[], Reranker]] = {
    'voyage': _voyage_factory,
    }


def get_client() -> Reranker:
    """Return the rerank client for the configured provider.

    Routes by `MEMMAN_RERANK_PROVIDER`. Defaults to `voyage` when the
    var is unset so existing installs continue to work without
    re-running `memman install`. Raises `ConfigError` when the
    provider name is unknown.
    """
    raw = config.get(config.RERANK_PROVIDER) or 'voyage'
    name = raw.lower()
    factory = RERANKERS.get(name)
    if factory is None:
        known = ', '.join(sorted(RERANKERS)) or '(none)'
        raise ConfigError(
            f'unknown {config.RERANK_PROVIDER}={name!r};'
            f' registered providers: {known}')
    return factory()
