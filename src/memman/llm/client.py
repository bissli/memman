"""LLM provider protocol, registry, and selector.

`LLMProvider` is the structural contract every provider class must
satisfy: a `.complete(system, user) -> str` method. Concrete clients
(currently only OpenRouter) register themselves in the `PROVIDERS`
dict via a zero-arg factory that reads provider-specific env vars.

`get_llm_client()` resolves the active provider by looking up
`MEMMAN_LLM_PROVIDER` in the registry and invoking its factory.
Unknown providers and missing API keys both surface as `ConfigError`
— the CLI layer (`cli._get_llm_client_or_fail`) re-wraps as
`click.ClickException`.

Adding a new provider = write a class in `llm/<name>_client.py` that
implements `.complete(...)` using helpers from `llm.shared`, plus one
`PROVIDERS[<name>] = factory` line here.
"""

import os
from collections.abc import Callable
from typing import Protocol

from memman import config
from memman.exceptions import ConfigError


class LLMProvider(Protocol):
    """Structural contract for any LLM client used by memman.

    The single `complete` method takes the system + user prompts and
    returns the assistant's response text. Streaming is deliberately
    out of scope — memman's pipeline reads each response as a whole.
    """

    def complete(self, system: str, user: str) -> str:
        """Run a completion and return the assistant response text."""
        ...


def _openrouter_factory() -> LLMProvider:
    """Build the registered OpenRouter client (factory indirection).

    Imported lazily so starting up memman does not fetch the ZDR cache
    or touch provider-specific env vars unless that provider is
    actually selected.
    """
    from memman.llm.openrouter_client import get_openrouter_client
    return get_openrouter_client()


PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    'openrouter': _openrouter_factory,
    }


def get_llm_client() -> LLMProvider:
    """Return the LLM client for the configured provider.

    Routes by `MEMMAN_LLM_PROVIDER` (default: 'openrouter'). Raises
    `ConfigError` when the provider name is unknown or the selected
    provider's required env vars are missing.
    """
    name = os.environ.get(
        config.LLM_PROVIDER, config.DEFAULT_LLM_PROVIDER).lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        known = ', '.join(sorted(PROVIDERS)) or '(none)'
        raise ConfigError(
            f'unknown {config.LLM_PROVIDER}={name!r};'
            f' registered providers: {known}')
    return factory()
