"""Backend-namespaced env key validation.

`PostgresBackendConfig` declares the `MEMMAN_POSTGRES_` namespace and
the explicit set of canonical keys it owns within it. `_validate(env)`
scans the input dict for keys that fall in that namespace and raises
`ConfigError` on any unknown key, with a `did you mean` hint when one
is close. The scan runs whatever backend a store runs, including
sqlite, so a typo in an inactive namespace is still caught.

Bare canonical keys (e.g., `MEMMAN_POSTGRES_DSN`) are rejected: the
per-store routing model requires the per-store suffixed form
(`MEMMAN_POSTGRES_DSN_<store>`) or the cross-store fallback
(`MEMMAN_DEFAULT_POSTGRES_DSN`). The canonical list survives only
as the difflib candidate set used to build the suggestion.

Cross-backend keys (`MEMMAN_OPENROUTER_API_KEY`, `MEMMAN_DEFAULT_BACKEND`,
`MEMMAN_DEFAULT_POSTGRES_DSN`, `MEMMAN_EMBED_PROVIDER`, etc.) belong
to no namespace and are never scanned. They remain governed by the
flat `INSTALLABLE_KEYS` membership check at `config set`.
"""

import difflib
from dataclasses import dataclass

from memman.store.db import valid_store_name
from memman.store.errors import ConfigError


@dataclass
class PostgresBackendConfig:
    """Owns the `MEMMAN_POSTGRES_*` namespace.

    The one canonical key is `MEMMAN_POSTGRES_DSN`, used in its
    per-store form `MEMMAN_POSTGRES_DSN_<store>`.
    """

    NAMESPACE_PREFIX = 'MEMMAN_POSTGRES_'
    OWNED_KEYS = frozenset({'MEMMAN_POSTGRES_DSN'})

    @classmethod
    def _validate(cls, env: dict) -> None:
        """Reject bare or unknown `MEMMAN_POSTGRES_*` keys in `env`.

        Parameters
        ----------
        env : dict
            Env keys to scan; keys outside the namespace are ignored.

        Raises
        ------
        ConfigError
            On the first bare, unknown, or badly suffixed key, with a
            `did you mean` hint when a canonical key is close.
        """
        _validate_namespace(
            env, cls.NAMESPACE_PREFIX, cls.OWNED_KEYS)


def _validate_namespace(
        env: dict, prefix: str, owned: frozenset) -> None:
    """Reject every key under `prefix` that is not a per-store key.

    Parameters
    ----------
    env : dict
        Env keys to scan.
    prefix : str
        Namespace prefix that selects the keys to check.
    owned : frozenset
        Canonical keys. A key passes only as `<owned>_<store>` with a
        valid store name; a bare canonical key fails.

    Raises
    ------
    ConfigError
        On the first failing key. The hint is suffixed with `_<store>`
        so it points at the per-store form.
    """
    candidates = [k for k in env if k.startswith(prefix)]
    for key in candidates:
        if key in owned:
            raise ConfigError(
                f'{key!r} is not a valid bare key under the per-store'
                f' routing model; did you mean {key + "_<store>"!r}?')
        canonical = _strip_store_suffix(key, owned)
        if canonical is not None:
            suffix = key[len(canonical) + 1:]
            if not valid_store_name(suffix):
                raise ConfigError(
                    f'invalid store-name suffix in {key!r};'
                    f' suffix {suffix!r} is not a valid store name')
            continue
        suggestions = difflib.get_close_matches(
            key, owned, n=1, cutoff=0.6)
        if suggestions:
            raise ConfigError(
                f'unknown {prefix} key {key!r};'
                f' did you mean {suggestions[0] + "_<store>"!r}?')
        raise ConfigError(
            f'unknown {prefix} key {key!r}')


def _strip_store_suffix(key: str, owned: frozenset) -> str | None:
    """Return the owned key that `key` extends as `<owned>_<suffix>`.

    Parameters
    ----------
    key : str
        Env key to test.
    owned : frozenset
        Canonical keys.

    Returns
    -------
    str or None
        The longest matching owned key, or None when none matches.
        The suffix is not checked.
    """
    for canonical in sorted(owned, key=len, reverse=True):
        if key.startswith(canonical + '_'):
            return canonical
    return None


def validate_all(env: dict) -> None:
    """Validate `env` against the Postgres backend namespace.

    Runs whatever backend a store uses, sqlite included, so a typo in
    an inactive namespace still fails.

    Parameters
    ----------
    env : dict
        Merged env to scan.

    Raises
    ------
    ConfigError
        On a bare, unknown, or badly suffixed `MEMMAN_POSTGRES_*` key.
    """
    PostgresBackendConfig._validate(env)
