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

from memman.store.errors import ConfigError


@dataclass
class PostgresBackendConfig:
    """Owns the `MEMMAN_POSTGRES_*` namespace.

    Today: `MEMMAN_POSTGRES_DSN_<store>` (per-store DSN). Add new
    keys here as the postgres backend grows them.
    """

    NAMESPACE_PREFIX = 'MEMMAN_POSTGRES_'
    OWNED_KEYS = frozenset({'MEMMAN_POSTGRES_DSN'})

    @classmethod
    def _validate(cls, env: dict) -> None:
        """Reject unknown `MEMMAN_POSTGRES_*` keys in `env`.

        Pulls a `did you mean` hint from `difflib.get_close_matches`
        when one is sufficiently close. Raises `ConfigError`
        immediately on the first unknown key.
        """
        _validate_namespace(
            env, cls.NAMESPACE_PREFIX, cls.OWNED_KEYS)


def _validate_namespace(
        env: dict, prefix: str, owned: frozenset) -> None:
    """Common namespace scan.

    Iterates `env` keys with the namespace prefix. Bare canonical
    keys (members of `owned` like `MEMMAN_POSTGRES_DSN`) are
    rejected -- the per-store routing model requires the suffixed
    form. Per-store-suffixed keys (e.g. `MEMMAN_POSTGRES_DSN_<store>`)
    are accepted when the canonical `<owned-key>_<suffix>` form
    decomposes to (a) a known canonical key and (b) a syntactically
    valid store-name suffix.

    The error message includes a `did you mean` hint pulled from
    `difflib.get_close_matches`, suffixed with `_<store>` so the
    suggestion points at the per-store form rather than the
    rejected bare canonical.
    """
    from memman.store.db import valid_store_name

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
    """Return the canonical owned key when `key == '<owned>_<suffix>'`.

    Returns None when no owned key prefixes `key` with a trailing
    underscore. The suffix syntactic check is the caller's
    responsibility -- this helper only handles the canonical lookup.

    Iterates owned keys longest-first so a hypothetical future
    second canonical key that prefixes another (e.g.
    `MEMMAN_POSTGRES_DSN` and `MEMMAN_POSTGRES_DSN_BACKUP`) matches
    the more specific one first.
    """
    for canonical in sorted(owned, key=len, reverse=True):
        if key.startswith(canonical + '_'):
            return canonical
    return None


def validate_all(env: dict) -> None:
    """Validate `env` against the Postgres backend namespace.

    Catches typos in `MEMMAN_POSTGRES_*` (e.g. a
    `MEMMAN_POSTGRES_DSN_typo=...`) whatever backend a store runs,
    including sqlite. Used by `factory.open_backend` so a single
    open-time call covers this regardless of the active backend.
    """
    PostgresBackendConfig._validate(env)
