"""Per-(provider, model) embedder registry.

`get_for(provider, model)` constructs an embedder bound to the
requested pair. If the provider constructor raises `ConfigError`
(missing creds), returns a `_PlaceholderEmbedder` so processes can
open multiple stores even when one provider's creds are absent.

The result is cached per `(provider, model)` for the lifetime of
the process so `factory()` + `prepare()` (which may issue a network
probe) only run once per pair. The cache uses an explicit
`threading.Lock` rather than `functools.lru_cache` because two
drain workers cold-starting on the same key can both miss with
`lru_cache` and both run the (network-issuing) `factory()` /
`prepare()` pair; the lock + double-check pattern collapses that
to one. Tests share one process, so the autouse fixture in
`tests/conftest.py` calls `reset_for_tests()` between tests to
keep credential-missing flows reproducible.
"""

import threading

from memman.embed import PROVIDERS, EmbeddingProvider
from memman.exceptions import ConfigError, EmbedCredentialError

_GET_FOR_LOCK = threading.Lock()
_GET_FOR_CACHE: dict[tuple[str, str], EmbeddingProvider] = {}


def get_for(provider: str, model: str) -> EmbeddingProvider:
    """Return an embedder client bound to (provider, model).

    Constructs a fresh client via `PROVIDERS[provider]()` and, when
    the requested model differs from the client's default, overrides
    the client's `model` attribute and resets cached state. When the
    constructor raises `ConfigError` (missing creds), returns a
    placeholder whose `embed()` raises `EmbedCredentialError`.

    Cached per `(provider, model)` for the process lifetime under an
    explicit lock to prevent duplicate provider probes when two
    drain workers cold-start on the same key concurrently.
    """
    key = (provider, model)
    cached = _GET_FOR_CACHE.get(key)
    if cached is not None:
        return cached
    with _GET_FOR_LOCK:
        cached = _GET_FOR_CACHE.get(key)
        if cached is not None:
            return cached
        factory = PROVIDERS.get(provider)
        if factory is None:
            known = ', '.join(sorted(PROVIDERS)) or '(none)'
            raise ConfigError(
                f'unknown embed provider {provider!r};'
                f' registered: {known}')
        try:
            client = factory()
        except ConfigError as exc:
            placeholder = _PlaceholderEmbedder(
                provider, model, str(exc))
            _GET_FOR_CACHE[key] = placeholder
            return placeholder
        if client.model != model:
            client.model = model
            client.dim = 0
            client._availability_cache = None
        client.prepare()
        _GET_FOR_CACHE[key] = client
        return client


def reset_for_tests() -> None:
    """Drop the cached entries (test fixture only)."""
    with _GET_FOR_LOCK:
        _GET_FOR_CACHE.clear()


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
