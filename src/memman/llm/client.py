"""LLM provider protocol, registry, and per-role selector.

`LLMProvider` is the structural contract every provider class must
satisfy: a `.complete(system, user) -> str` method. Concrete clients
register themselves in the `PROVIDERS` dict via a factory that takes a
role and reads role-specific env vars.

Two roles exist:

- `fast` — used on the synchronous CLI hot path (recall query
  expansion, doctor's connectivity probe). Reads `MEMMAN_LLM_MODEL_FAST`.
- `slow` — used in the scheduler-driven worker (fact extraction,
  reconciliation, enrichment, causal inference) and the operator
  `graph rebuild` sweep. Reads `MEMMAN_LLM_MODEL_SLOW`.

Routing the recall path to a small/fast model and the worker to a
larger/slow/reasoning model means switching the worker model never
adds latency to interactive commands.

`get_llm_client(role)` resolves the active provider via
`MEMMAN_LLM_PROVIDER` and returns a per-role cached client.
"""

from collections.abc import Callable
from typing import Protocol

from memman import config
from memman.exceptions import ConfigError

ROLE_FAST = 'fast'
ROLE_SLOW = 'slow'
VALID_ROLES = frozenset({ROLE_FAST, ROLE_SLOW})


class LLMProvider(Protocol):
    """Structural contract for any LLM client used by memman.

    The single `complete` method takes the system + user prompts and
    returns the assistant's response text. Streaming is deliberately
    out of scope — memman's pipeline reads each response as a whole.
    """

    def complete(self, system: str, user: str) -> str:
        """Run a completion and return the assistant response text."""
        ...


def _openrouter_factory(role: str) -> LLMProvider:
    """Build the registered OpenRouter client for a role.

    Imported lazily so starting up memman does not touch
    provider-specific env vars unless that provider is actually selected.
    """
    from memman.llm.openrouter_client import get_openrouter_client
    return get_openrouter_client(role)


PROVIDERS: dict[str, Callable[[str], LLMProvider]] = {
    'openrouter': _openrouter_factory,
    }

_ROLE_CACHE: dict[str, LLMProvider] = {}


def get_llm_client(role: str) -> LLMProvider:
    """Return a cached LLM client for the given role.

    `role` must be one of `'fast'` or `'slow'`. Routes by
    `MEMMAN_LLM_PROVIDER` (default: 'openrouter'). Raises `ConfigError`
    when the provider name is unknown or the selected provider's
    required env vars are missing.
    """
    if role not in VALID_ROLES:
        raise ValueError(
            f'unknown LLM role {role!r}; valid roles: {sorted(VALID_ROLES)}')
    cached = _ROLE_CACHE.get(role)
    if cached is not None:
        return cached
    raw = config.get(config.LLM_PROVIDER)
    if not raw:
        raise ConfigError(
            f'{config.LLM_PROVIDER} is not set;'
            ' run `memman install` to populate the env file')
    name = raw.lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        known = ', '.join(sorted(PROVIDERS)) or '(none)'
        raise ConfigError(
            f'unknown {config.LLM_PROVIDER}={name!r};'
            f' registered providers: {known}')
    client = factory(role)
    _ROLE_CACHE[role] = client
    return client


def reset_role_cache() -> None:
    """Drop cached per-role clients. Used by tests that swap env vars."""
    _ROLE_CACHE.clear()
