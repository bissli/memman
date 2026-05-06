"""Backend-namespaced env key validation.

Each backend declares its `MEMMAN_<NS>_` namespace and the explicit
set of keys it owns within that namespace. `_validate(env)` scans
the input dict for keys that fall in the backend's namespace and
raises `ConfigError` on any unknown key, with a `did you mean`
hint when one is close.

Cross-backend keys (`OPENROUTER_API_KEY`, `MEMMAN_BACKEND`,
`MEMMAN_EMBED_PROVIDER`, etc.) belong to neither dataclass and are
never scanned by either validator. They remain governed by the
flat `INSTALLABLE_KEYS` membership check at `config set`.
"""

import difflib
from dataclasses import dataclass

from memman import config
from memman.store.errors import ConfigError


@dataclass
class PostgresBackendConfig:
    """Owns the `MEMMAN_PG_*` namespace.

    Today: `MEMMAN_PG_DSN`. Add new keys here as the postgres
    backend grows them.
    """

    NAMESPACE_PREFIX = 'MEMMAN_PG_'
    OWNED_KEYS = frozenset({config.PG_DSN})

    @classmethod
    def _validate(cls, env: dict) -> None:
        """Reject unknown `MEMMAN_PG_*` keys in `env`.

        Pulls a `did you mean` hint from `difflib.get_close_matches`
        when one is sufficiently close. Raises `ConfigError`
        immediately on the first unknown key.
        """
        _validate_namespace(
            env, cls.NAMESPACE_PREFIX, cls.OWNED_KEYS)


@dataclass
class SqliteBackendConfig:
    """Owns the `MEMMAN_SQLITE_*` namespace.

    Today the namespace is empty -- SQLite has no backend-specific
    keys. The class exists so a future SQLite-specific knob can be
    added without changing the validation surface.
    """

    NAMESPACE_PREFIX = 'MEMMAN_SQLITE_'
    OWNED_KEYS = frozenset()

    @classmethod
    def _validate(cls, env: dict) -> None:
        """Reject unknown `MEMMAN_SQLITE_*` keys in `env`.
        """
        _validate_namespace(
            env, cls.NAMESPACE_PREFIX, cls.OWNED_KEYS)


def _validate_namespace(
        env: dict, prefix: str, owned: frozenset) -> None:
    """Common namespace scan.

    Iterates `env` keys with the namespace prefix; any key not in
    `owned` raises `ConfigError`. Per-store-suffixed keys (e.g.
    `MEMMAN_PG_DSN_<store>`) are accepted when the canonical
    `<owned-key>_<suffix>` form decomposes to (a) a known canonical
    key and (b) a syntactically valid store-name suffix.

    The error message includes a `did you mean` hint pulled from
    `difflib.get_close_matches` when one is sufficiently close.
    """
    from memman.store.db import valid_store_name

    candidates = [k for k in env if k.startswith(prefix)]
    for key in candidates:
        if key in owned:
            continue
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
                f' did you mean {suggestions[0]!r}?')
        raise ConfigError(
            f'unknown {prefix} key {key!r}')


def _strip_store_suffix(key: str, owned: frozenset) -> str | None:
    """Return the canonical owned key when `key == '<owned>_<suffix>'`.

    Returns None when no owned key prefixes `key` with a trailing
    underscore. The suffix syntactic check is the caller's
    responsibility -- this helper only handles the canonical lookup.
    """
    for canonical in owned:
        if key.startswith(canonical + '_'):
            return canonical
    return None


_REGISTRY: dict[str, type] = {
    'postgres': PostgresBackendConfig,
    'sqlite': SqliteBackendConfig,
    }


def validate_for(backend: str, env: dict) -> None:
    """Validate `env` against the named backend's config dataclass.

    Unknown backends are tolerated -- `factory.open_cluster` will
    surface that itself. This function only validates the namespace
    when the backend is registered.
    """
    cls = _REGISTRY.get(backend.lower())
    if cls is None:
        return
    cls._validate(env)
