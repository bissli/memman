"""Embed client protocol and the env-configured client.

`EmbeddingProvider` is the structural contract the embed client and
the registry's credential-missing placeholder both satisfy. A store
binds its client by the model in its fingerprint; `get_client()`
builds the one `MEMMAN_EMBED_MODEL` names, for seeding a fresh store.
"""

from typing import Protocol

from memman import config
from memman.embed.client import Client


class EmbeddingProvider(Protocol):
    """Structural contract every embedding client must satisfy.
    """

    model: str
    dim: int

    def prepare(self) -> None:
        """Populate `dim` with a one-token embed; a no-op once known.
        """
        ...

    def available(self) -> bool:
        """Return True when the endpoint serves the model.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for many texts in one round-trip.
        """
        ...

    def unavailable_message(self) -> str:
        """Return a user-facing message explaining why unavailable.
        """
        ...


def get_client() -> EmbeddingProvider:
    """Embed client for `MEMMAN_EMBED_MODEL` on the shared endpoint.

    Raises
    ------
    ConfigError
        `MEMMAN_EMBED_MODEL` or `MEMMAN_ENDPOINT` is unset, or
        `MEMMAN_API_KEY` is blank on a non-loopback endpoint.
    """
    return Client(config.require(config.EMBED_MODEL))
