"""Per-(provider, model) embedder registry.

`get_for(provider, model)` constructs an embedder bound to the
requested pair. If the provider constructor raises `ConfigError`
(missing creds), returns a `_PlaceholderEmbedder` so processes can
open multiple stores even when one provider's creds are absent.
"""

from memman.embed import PROVIDERS, EmbeddingProvider
from memman.exceptions import ConfigError, EmbedCredentialError


def get_for(provider: str, model: str) -> EmbeddingProvider:
    """Return an embedder client bound to (provider, model).

    Constructs a fresh client via `PROVIDERS[provider]()` and, when
    the requested model differs from the client's default, overrides
    the client's `model` attribute and resets cached state. When the
    constructor raises `ConfigError` (missing creds), returns a
    placeholder whose `embed()` raises `EmbedCredentialError`.
    """
    factory = PROVIDERS.get(provider)
    if factory is None:
        known = ', '.join(sorted(PROVIDERS)) or '(none)'
        raise ConfigError(
            f'unknown embed provider {provider!r};'
            f' registered: {known}')
    try:
        client = factory()
    except ConfigError as exc:
        return _PlaceholderEmbedder(provider, model, str(exc))
    if client.model != model:
        client.model = model
        client.dim = 0
        client._availability_cache = None
    client.prepare()
    return client


class _PlaceholderEmbedder:
    """Stand-in for an embedder whose creds are absent.

    Exposes name/model/dim like a real client, returns False from
    `available()`, and raises `EmbedCredentialError` on `embed()`
    and `embed_batch()`. The drain converts that into a structured
    `embedder_credential_missing` trace event and a failed queue
    row.
    """

    def __init__(self, provider: str, model: str, reason: str) -> None:
        """Bind the placeholder to a (provider, model) and record
        the underlying ConfigError reason for downstream messages.
        """
        self.name = provider
        self.model = model
        self.dim = 0
        self._reason = reason

    def prepare(self) -> None:
        """No-op: placeholder has no probe to run."""
        return

    def available(self) -> bool:
        """Always False; placeholder cannot probe an absent provider."""
        return False

    def embed(self, text: str) -> list[float]:
        """Raise EmbedCredentialError on any embed attempt."""
        raise EmbedCredentialError(
            f'embed provider {self.name!r} cannot run: {self._reason}')

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Raise EmbedCredentialError on any embed attempt."""
        raise EmbedCredentialError(
            f'embed provider {self.name!r} cannot run: {self._reason}')

    def unavailable_message(self) -> str:
        """Return the underlying ConfigError reason."""
        return self._reason
