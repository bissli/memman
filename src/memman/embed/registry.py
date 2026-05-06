"""Per-(provider, model) embedder registry.

Shifts runtime authority from the env-resolved global provider to
per-store binding. `get_for(provider, model)` constructs an embedder
bound to the requested pair; the caller (typically `_StoreContext`)
reads the store's stored fingerprint and asks for the matching
client.

Lazy credentialing: if the underlying provider's `__init__` raises
`ConfigError` because credentials are missing, `get_for` returns a
`_PlaceholderEmbedder` that satisfies the structural Protocol but
raises `EmbedCredentialError` on `embed()` / `embed_batch()`. This
lets a process open multiple stores -- including ones bound to
provider X without X's creds -- without crashing at open time.
"""

from memman.embed import PROVIDERS, EmbeddingProvider, get_client
from memman.exceptions import ConfigError, EmbedCredentialError


def get_active() -> EmbeddingProvider:
    """Return the env-resolved active client.

    Backwards-compatible path for call sites that don't have a
    per-store fingerprint to bind from.
    """
    return get_client()


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
    if getattr(client, 'model', None) != model:
        client.model = model
        client.dim = 0
        if hasattr(client, '_availability_cache'):
            client._availability_cache = None
    if hasattr(client, 'prepare'):
        try:
            client.prepare()
        except Exception:
            pass
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
