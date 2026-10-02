"""Per-model embedder registry.

`get_for(model, dim)` constructs an embedder bound to the model. If the
constructor raises `ConfigError` (missing endpoint or key), it returns
a `_PlaceholderEmbedder`, so a process can still open stores.

The result is cached per model for the process lifetime, so the
constructor and `prepare()` (which probes the network) run once per
model.
"""

import threading

from memman.embed import EmbeddingProvider
from memman.embed.client import Client
from memman.exceptions import ConfigError, EmbedCredentialError

_GET_FOR_LOCK = threading.Lock()
_GET_FOR_CACHE: dict[str, EmbeddingProvider] = {}


def get_for(model: str, dim: int = 0) -> EmbeddingProvider:
    """Embed client bound to `model`, built and prepared once per process.

    Parameters
    ----------
    model : str
        Model id the store's fingerprint names.
    dim : int, default 0
        Vector width the store already records; a positive value skips
        the billed `prepare()` probe. 0 probes.

    Returns
    -------
    EmbeddingProvider
        The cached client, or a placeholder whose `embed()` raises
        `EmbedCredentialError` when the constructor raised
        `ConfigError`.
    """
    cached = _GET_FOR_CACHE.get(model)
    if cached is not None:
        return cached
    # A dict and lock stand in for `functools.lru_cache`: two drain
    # workers cold-starting on one model can both miss an `lru_cache`
    # and both run the network-issuing prepare. The lock
    # with a re-check collapses that to one run.
    with _GET_FOR_LOCK:
        cached = _GET_FOR_CACHE.get(model)
        if cached is not None:
            return cached
        try:
            client = Client(model)
        except ConfigError as exc:
            placeholder = _PlaceholderEmbedder(model, str(exc))
            _GET_FOR_CACHE[model] = placeholder
            return placeholder
        client.dim = dim
        client.prepare()
        _GET_FOR_CACHE[model] = client
        return client


def reset_for_tests() -> None:
    """Drop the cached entries.

    The autouse fixture in `tests/conftest.py` calls this between
    tests so credential-missing flows stay reproducible.
    """
    with _GET_FOR_LOCK:
        _GET_FOR_CACHE.clear()


class _PlaceholderEmbedder:
    """Stand-in for an embedder whose creds are absent.

    Exposes model/dim like a real client, returns False from
    `available()`, and raises `EmbedCredentialError` on `embed()`
    and `embed_batch()`. The drain converts that into a structured
    `embedder_credential_missing` trace event and a failed queue
    row.
    """

    def __init__(self, model: str, reason: str) -> None:
        """Bind to a model and keep the ConfigError reason.
        """
        self.model = model
        self.dim = 0
        self._reason = reason

    def prepare(self) -> None:
        """No-op: the placeholder has no probe to run.
        """
        return

    def available(self) -> bool:
        """Always False; the placeholder has no endpoint to probe.
        """
        return False

    def embed(self, text: str) -> list[float]:
        """Raise EmbedCredentialError on any embed attempt.
        """
        raise EmbedCredentialError(
            f'embed model {self.model!r} cannot run: {self._reason}')

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Raise EmbedCredentialError on any embed attempt.
        """
        raise EmbedCredentialError(
            f'embed model {self.model!r} cannot run: {self._reason}')

    def unavailable_message(self) -> str:
        """Return the underlying ConfigError reason.
        """
        return self._reason
