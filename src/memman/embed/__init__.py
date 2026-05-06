"""Embed provider protocol, registry, and selector.

`EmbeddingProvider` is the structural contract every provider class
must satisfy: a `name`/`model`/`dim` triple plus `available()` and
`embed(text)` methods. Concrete clients
(`embed/voyage.py`, `embed/openai_compat.py`, `embed/ollama.py`)
register themselves in the `PROVIDERS` dict via a zero-arg factory.

`get_client()` resolves the active provider by looking up
`MEMMAN_EMBED_PROVIDER` in the registry and invoking its factory.
Unknown providers surface as `ConfigError`.

Adding a new provider = drop `embed/<name>.py` implementing the
Protocol, plus one `PROVIDERS[<name>] = factory` line here.
"""

from collections.abc import Callable
from typing import Protocol

from memman import config
from memman.exceptions import ConfigError


class EmbeddingProvider(Protocol):
    """Structural contract every embedding client must satisfy.
    """

    name: str
    model: str
    dim: int

    def prepare(self) -> None:
        """Eagerly populate `dim` so callers can read it directly.

        For providers that know their dim at construction (Voyage),
        this is a no-op. For probe-only providers (`openai_compat`,
        `openrouter`, `ollama`), `prepare()` performs a one-token
        embed and caches the resulting dim on the client. Idempotent
        once dim is populated.
        """
        ...

    def available(self) -> bool:
        """Return True when the provider's API is reachable.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for many texts in one round-trip.

        Used by the worker pipeline to batch enriched-text re-embeds
        across all facts in a row, replacing N HTTP calls with one.
        """
        ...

    def unavailable_message(self) -> str:
        """Return a user-facing message explaining why unavailable.
        """
        ...


def _voyage_factory() -> EmbeddingProvider:
    """Build the registered Voyage client (factory indirection).
    """
    from memman.embed.voyage import Client
    return Client()


def _openai_factory() -> EmbeddingProvider:
    """Build the registered OpenAI-compatible client.
    """
    from memman.embed.openai_compat import Client
    return Client()


def _ollama_factory() -> EmbeddingProvider:
    """Build the registered Ollama client.
    """
    from memman.embed.ollama import Client
    return Client()


def _openrouter_factory() -> EmbeddingProvider:
    """Build the registered OpenRouter embed client."""
    from memman.embed.openrouter import Client
    return Client()


PROVIDERS: dict[str, Callable[[], EmbeddingProvider]] = {
    'voyage': _voyage_factory,
    'openai': _openai_factory,
    'openrouter': _openrouter_factory,
    'ollama': _ollama_factory,
    }


def get_client() -> EmbeddingProvider:
    """Return the embed client for the configured provider.

    Routes by `MEMMAN_EMBED_PROVIDER`. Raises `ConfigError` when the
    var is unset (run `memman install`) or the provider name is unknown.
    """
    raw = config.get(config.EMBED_PROVIDER)
    if not raw:
        raise ConfigError(
            f'{config.EMBED_PROVIDER} is not set;'
            ' run `memman install` to populate the env file')
    name = raw.lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        known = ', '.join(sorted(PROVIDERS)) or '(none)'
        raise ConfigError(
            f'unknown {config.EMBED_PROVIDER}={name!r};'
            f' registered providers: {known}')
    return factory()
