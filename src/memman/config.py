"""Central env-var configuration for memman.

This module owns the canonical list of environment variables memman
reads. Every env var name is a module-level constant so call sites
import the name rather than repeating the literal string.

Runtime resolution: the env file at `<MEMMAN_DATA_DIR>/env`
(default `~/.memman/env`) is the canonical, global source of truth
for all `INSTALLABLE_KEYS`. Shell environment variables are NOT
consulted at runtime for installable keys -- this prevents stale
shell exports from silently overriding values the user committed
via `memman install`.

`config.get(name)` and `config.get_bool(name)` read the env file only.
There is no fallback to `os.environ` and no code-default fallback at
runtime; if the file lacks the key, `get` returns `None` and `require`
raises `ConfigError`. `INSTALL_DEFAULTS` exists only for install-time
file population.

Install-time resolution (one-time seed): `collect_install_knobs`
fills the env file using the precedence
`file > os.environ > INSTALL_DEFAULTS`. Shell environment variables
are read at install time only as a seed for keys missing from the
file -- existing file values are sticky and never overridden by a
later shell export.

Process-control vars (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`,
`MEMMAN_WORKER`, `MEMMAN_SCHEDULER_KIND`, `MEMMAN_DEBUG`,
`MEMMAN_AUTHOR`) are NOT installable, never written to the env file,
and are read directly from `os.environ` by their owners (`is_worker`,
`trace.is_enabled`, `resolve_author`, etc.). They do not flow through
`get()`.

Tuning vars (`MEMMAN_EMBED_SWAP_BATCH_SIZE`,
`MEMMAN_EMBED_SWAP_INDEX_TIMEOUT`, `MEMMAN_REINDEX_TIMEOUT`) are
runtime knobs read directly from `os.environ` by their owners; they
are also never persisted to the env file but are surfaced by
`enumerate_effective_config` so operators can see them in
`memman config show`.

`INSTALLABLE_KEYS` is the single source of truth for what `memman
install` persists to `~/.memman/env`. Adding a new global knob is
one tuple entry plus an `INSTALL_DEFAULTS` row when a default exists.
"""

import getpass
import os
from pathlib import Path
from typing import Any

DATA_DIR = 'MEMMAN_DATA_DIR'
STORE = 'MEMMAN_STORE'
LLM_ENDPOINT = 'MEMMAN_LLM_ENDPOINT'
LLM_API_KEY = 'MEMMAN_LLM_API_KEY'
LLM_MODEL = 'MEMMAN_LLM_MODEL'
LLM_PROVIDER_ONLY = 'MEMMAN_LLM_PROVIDER_ONLY'
LLM_DATA_COLLECTION = 'MEMMAN_LLM_DATA_COLLECTION'
LLM_ZDR = 'MEMMAN_LLM_ZDR'
LLM_MAX_INPUT_PRICE = 'MEMMAN_LLM_MAX_INPUT_PRICE'
LLM_MAX_OUTPUT_PRICE = 'MEMMAN_LLM_MAX_OUTPUT_PRICE'
EMBED_PROVIDER = 'MEMMAN_EMBED_PROVIDER'
RERANK_PROVIDER = 'MEMMAN_RERANK_PROVIDER'
RERANK_ENABLED = 'MEMMAN_RERANK_ENABLED'
OPENROUTER_ENDPOINT = 'MEMMAN_OPENROUTER_ENDPOINT'
DEBUG = 'MEMMAN_DEBUG'
WORKER = 'MEMMAN_WORKER'
SCHEDULER_KIND = 'MEMMAN_SCHEDULER_KIND'
LOG_LEVEL = 'MEMMAN_LOG_LEVEL'
DEFAULT_BACKEND = 'MEMMAN_DEFAULT_BACKEND'
DEFAULT_PG_DSN = 'MEMMAN_DEFAULT_POSTGRES_DSN'
INTERVAL = 'MEMMAN_INTERVAL'
BACKUP_CRON = 'MEMMAN_BACKUP_CRON'
BACKUP_TARGET = 'MEMMAN_BACKUP_TARGET'
BACKUP_KEEP = 'MEMMAN_BACKUP_KEEP'

EMBED_SWAP_BATCH_SIZE = 'MEMMAN_EMBED_SWAP_BATCH_SIZE'
EMBED_SWAP_INDEX_TIMEOUT = 'MEMMAN_EMBED_SWAP_INDEX_TIMEOUT'
REINDEX_TIMEOUT = 'MEMMAN_REINDEX_TIMEOUT'

AUTHOR = 'MEMMAN_AUTHOR'


def resolve_author() -> str:
    """Return the author for the current write.

    Returns
    -------
    str
        `MEMMAN_AUTHOR` from `os.environ` when set and non-empty;
        `getpass.getuser()` otherwise.

    Notes
    -----
    - Called at `remember` and `replace` time in the agent's shell,
      where direnv has exported `MEMMAN_AUTHOR`.
    - Never called at drain time: the scheduler subprocess runs under
      systemd without the directory's environment, so the author must
      be read from the queue row, not resolved again.
    """
    return os.environ.get(AUTHOR) or getpass.getuser()


def BACKEND_FOR(store: str) -> str:
    """Per-store backend env-key name: `MEMMAN_BACKEND_<store>`."""
    return f'MEMMAN_BACKEND_{store}'


def RERANK_ENABLED_FOR(store: str) -> str:
    """Per-store rerank-toggle env key: `MEMMAN_RERANK_ENABLED_<store>`."""
    return f'MEMMAN_RERANK_ENABLED_{store}'


def env_key_for(backend: str, key: str, store: str) -> str:
    """Per-store env-key name for a backend descriptor key.

    Returns `MEMMAN_<BACKEND>_<KEY>_<store>`. Used by the registry-
    driven dispatch in `store.factory` so backend additions do not
    require a new module-level helper alongside `BACKEND_FOR`.
    """
    return f'MEMMAN_{backend.upper()}_{key.upper()}_{store}'


def _pg_dsn_prefix() -> str:
    return f'MEMMAN_{"postgres".upper()}_DSN_'


PER_STORE_KEY_SPECS: tuple[tuple[str, bool], ...] = (
    ('MEMMAN_BACKEND_', False),
    (_pg_dsn_prefix(), True),
    ('MEMMAN_RERANK_ENABLED_', False),
    )


OPENROUTER_API_KEY = 'MEMMAN_OPENROUTER_API_KEY'
VOYAGE_API_KEY = 'MEMMAN_VOYAGE_API_KEY'

OPENAI_EMBED_API_KEY = 'MEMMAN_OPENAI_EMBED_API_KEY'
OPENAI_EMBED_ENDPOINT = 'MEMMAN_OPENAI_EMBED_ENDPOINT'
OPENAI_EMBED_MODEL = 'MEMMAN_OPENAI_EMBED_MODEL'
OLLAMA_HOST = 'MEMMAN_OLLAMA_HOST'
OLLAMA_EMBED_MODEL = 'MEMMAN_OLLAMA_EMBED_MODEL'
OLLAMA_MAX_INPUT_CHARS = 'MEMMAN_OLLAMA_MAX_INPUT_CHARS'
OPENROUTER_EMBED_MODEL = 'MEMMAN_OPENROUTER_EMBED_MODEL'
VOYAGE_EMBED_MODEL = 'MEMMAN_VOYAGE_EMBED_MODEL'
VOYAGE_RERANK_MODEL = 'MEMMAN_VOYAGE_RERANK_MODEL'

ENV_FILENAME = 'env'

TRUTHY = frozenset({'1', 'true', 'yes', 'on'})

SECRET_VARS = frozenset({
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    LLM_API_KEY,
    OPENAI_EMBED_API_KEY,
    DEFAULT_PG_DSN,
    })

INSTALLABLE_KEYS = (
    LLM_ENDPOINT,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER_ONLY,
    LLM_DATA_COLLECTION,
    LLM_ZDR,
    LLM_MAX_INPUT_PRICE,
    LLM_MAX_OUTPUT_PRICE,
    EMBED_PROVIDER,
    RERANK_PROVIDER,
    RERANK_ENABLED,
    OPENROUTER_ENDPOINT,
    LOG_LEVEL,
    OPENAI_EMBED_API_KEY,
    OPENAI_EMBED_ENDPOINT,
    OPENAI_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MAX_INPUT_CHARS,
    OPENROUTER_EMBED_MODEL,
    VOYAGE_EMBED_MODEL,
    VOYAGE_RERANK_MODEL,
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    DEFAULT_BACKEND,
    DEFAULT_PG_DSN,
    INTERVAL,
    BACKUP_CRON,
    BACKUP_TARGET,
    BACKUP_KEEP,
    )


NATIVE_INSTALL_KEY_FALLBACKS: dict[str, str] = {
    OPENROUTER_API_KEY: 'OPENROUTER_API_KEY',
    VOYAGE_API_KEY: 'VOYAGE_API_KEY',
    OPENAI_EMBED_API_KEY: 'OPENAI_API_KEY',
    }


def _shell_seed_value(key: str) -> str:
    """Install-time shell lookup with vendor-native-name fallback.

    Checks `os.environ[key]` (memman-prefixed) first; if empty, falls
    back to the vendor's documented native name (e.g. `VOYAGE_API_KEY`)
    via `NATIVE_INSTALL_KEY_FALLBACKS`. Returns the stripped string, or
    '' when neither is set.

    Do NOT call from runtime paths -- runtime resolution is file-only
    via `config.get`. This helper exists for `collect_install_knobs`
    and the wizard's prompt-skip / pre-fill logic only.
    """
    value = os.environ.get(key, '').strip()
    if value:
        return value
    fallback = NATIVE_INSTALL_KEY_FALLBACKS.get(key)
    if not fallback:
        return ''
    return os.environ.get(fallback, '').strip()


def required_install_keys(embed: str) -> set[str]:
    """Return the API-key env vars install must populate for the embed provider.

    LLM-side authentication is endpoint-driven; the wizard enforces the
    "API key required for non-loopback endpoints" rule directly during
    install rather than at this config layer. Experimental embed
    providers return an empty set (doctor warns separately).
    """
    from memman.embed import PROVIDER_REQUIRED_KEYS
    return set(PROVIDER_REQUIRED_KEYS.get(embed, ()))


def is_openrouter_endpoint(url: str) -> bool:
    """Return True when `url` points at the OpenRouter API.

    Normalizes the URL: parses with `urlparse`, lowercases the host,
    strips a leading `www.`, and matches `openrouter.ai` or any
    `*.openrouter.ai` subdomain (regional shards like `eu.openrouter.ai`).
    Resilient to trailing slash and scheme variation.
    """
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ''
    host = host.lower().removeprefix('www.')
    return host == 'openrouter.ai' or host.endswith('.openrouter.ai')


def is_loopback_endpoint(url: str) -> bool:
    """Return True when `url` resolves to the local machine.

    Used by the wizard to decide whether to require `MEMMAN_LLM_API_KEY`
    on install: loopback endpoints (Ollama, local vLLM, LiteLLM proxy)
    typically do not need auth.
    """
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or '').lower()
    if host in {'localhost', '127.0.0.1', '::1'}:
        return True
    return host.endswith('.localhost')


INSTALL_DEFAULTS: dict[str, str] = {
    LLM_ENDPOINT: 'https://openrouter.ai/api/v1',
    LLM_MODEL: 'qwen/qwen3-235b-a22b-2507',
    LLM_PROVIDER_ONLY: 'amazon-bedrock,azure,google-vertex',
    LLM_DATA_COLLECTION: 'deny',
    LLM_ZDR: 'true',
    LLM_MAX_INPUT_PRICE: '0.25',
    LLM_MAX_OUTPUT_PRICE: '1.00',
    EMBED_PROVIDER: 'voyage',
    RERANK_PROVIDER: 'voyage',
    RERANK_ENABLED: 'true',
    OPENROUTER_ENDPOINT: 'https://openrouter.ai/api/v1',
    LOG_LEVEL: 'WARNING',
    OPENAI_EMBED_ENDPOINT: 'https://api.openai.com',
    OPENAI_EMBED_MODEL: 'text-embedding-3-small',
    OLLAMA_HOST: 'http://localhost:11434',
    OLLAMA_EMBED_MODEL: 'nomic-embed-text',
    OLLAMA_MAX_INPUT_CHARS: '1500',
    OPENROUTER_EMBED_MODEL: 'baai/bge-m3',
    VOYAGE_EMBED_MODEL: 'voyage-3-lite',
    VOYAGE_RERANK_MODEL: 'rerank-3-lite',
    DEFAULT_BACKEND: 'sqlite',
    INTERVAL: '60',
    BACKUP_KEEP: '7',
    }

_PROCESS_CONTROL_VARS = (
    DATA_DIR, STORE, WORKER, DEBUG, SCHEDULER_KIND, AUTHOR)

_TUNING_VARS = (
    EMBED_SWAP_BATCH_SIZE,
    EMBED_SWAP_INDEX_TIMEOUT,
    REINDEX_TIMEOUT,
    )

_DIRECT_ENV_VARS = frozenset(_PROCESS_CONTROL_VARS + _TUNING_VARS)

_ALL_VARS = INSTALLABLE_KEYS + _PROCESS_CONTROL_VARS + _TUNING_VARS


_FILE_CACHE: dict[str, str] | None = None
_FILE_CACHE_PATH: str | None = None


def env_file_path(data_dir: str | None = None) -> Path:
    """Return the path to the env file under the given data dir.

    When `data_dir` is omitted, falls back to `MEMMAN_DATA_DIR` from
    `os.environ`, then to `~/.memman`. The env-file location must not
    flow through the resolver itself - that would be circular.
    """
    if data_dir:
        return Path(data_dir) / ENV_FILENAME
    env_data_dir = os.environ.get(DATA_DIR)
    if env_data_dir:
        return Path(env_data_dir) / ENV_FILENAME
    return Path.home() / '.memman' / ENV_FILENAME


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse `KEY=VALUE` env file. Missing file -> empty dict.

    Matches systemd `EnvironmentFile=` semantics: blank lines and
    `#`-prefixed comments are skipped, lines without `=` are skipped,
    and a single matching pair of surrounding `'` or `"` is stripped
    from each value. No `${VAR}` expansion.
    """
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _load_file_cache() -> dict[str, str]:
    """Lazy-load the env-file cache. Reload when the path changes."""
    global _FILE_CACHE, _FILE_CACHE_PATH
    path = env_file_path()
    path_str = str(path)
    if _FILE_CACHE is None or _FILE_CACHE_PATH != path_str:
        _FILE_CACHE = parse_env_file(path)
        _FILE_CACHE_PATH = path_str
    return _FILE_CACHE


def reset_file_cache() -> None:
    """Drop the cached env-file contents.

    Tests call this when they mutate the file mid-process; production
    callers don't need it (each CLI invocation is a fresh process).
    """
    global _FILE_CACHE, _FILE_CACHE_PATH
    _FILE_CACHE = None
    _FILE_CACHE_PATH = None


def get(name: str) -> str | None:
    """Return the resolved value for env var `name`.

    Reads from `<MEMMAN_DATA_DIR>/env` only. Shell environment is
    never consulted; the env file is the canonical, global source of
    truth for installable settings. Empty strings are treated as
    "not set." Returns `None` when the file lacks the key.

    Process-control vars (`DATA_DIR`, `STORE`, `WORKER`, `DEBUG`) are
    never persisted to the file; their owners read `os.environ`
    directly and must not call `get()` for them.
    """
    file_value = _load_file_cache().get(name)
    if file_value is not None and file_value != '':
        return file_value
    return None


def require(name: str) -> str:
    """Return the resolved value for `name` or raise `ConfigError`.

    Use at every call site that needs a value. After `memman install`
    the env file holds every `INSTALLABLE_KEYS` entry, so `require`
    succeeds; raising means install was never run, the file was
    corrupted, or a required-but-optional key is being read on a
    provider that doesn't have it set.
    """
    from memman.exceptions import ConfigError
    value = get(name)
    if value is None:
        raise ConfigError(
            f'{name} is not set in {env_file_path()};'
            ' run `memman install` to populate the env file')
    return value


def get_scoped(name: str, data_dir: str | None = None) -> str | None:
    """Read `name` from the env file under `data_dir`.

    Parameters
    ----------
    name : str
        Env key to read.
    data_dir : str or None, default None
        Directory holding the env file. None defers to `get`, which
        resolves against `MEMMAN_DATA_DIR`.

    Returns
    -------
    str or None
        The value, or None when unset or empty.

    Notes
    -----
    - Pairs with `get_store_backend` / `get_store_pg_dsn` so a caller
      holding a `data_dir` resolves the per-store key and its default
      from ONE file. Mixing `get` with those helpers read per-store
      keys from an explicit directory and defaults from the ambient
      one, which routed a store to the wrong backend under
      `memman --data-dir`.
    """
    if data_dir is None:
        return get(name)
    raw = parse_env_file(env_file_path(data_dir)).get(name)
    return raw or None


def get_store_backend(
        store: str, data_dir: str | None = None) -> str | None:
    """Read `MEMMAN_BACKEND_<store>` from the env file; None if absent.

    Read-only helper -- no fallback to `MEMMAN_DEFAULT_BACKEND`.
    Callers that want default-fallback behavior compose
    `get_store_backend(store) or get(DEFAULT_BACKEND)` explicitly so
    the data flow stays visible.
    """
    if data_dir is None:
        return get(BACKEND_FOR(store))
    file_values = parse_env_file(env_file_path(data_dir))
    raw = file_values.get(BACKEND_FOR(store))
    return raw or None


def get_store_rerank_enabled(
        store: str, data_dir: str | None = None) -> bool | None:
    """Read `MEMMAN_RERANK_ENABLED_<store>` from the env file; None if absent.

    Read-only helper -- no fallback to the global `MEMMAN_RERANK_ENABLED`.
    Callers that want default-fallback behavior compose
    `get_store_rerank_enabled(store) ?? get_bool(RERANK_ENABLED, default=True)`
    explicitly so the data flow stays visible.
    """
    key = RERANK_ENABLED_FOR(store)
    if data_dir is None:
        raw = get(key)
    else:
        raw = parse_env_file(env_file_path(data_dir)).get(key) or None
    if raw is None or raw == '':
        return None
    return raw.strip().lower() in TRUTHY


def get_store_pg_dsn(
        store: str, data_dir: str | None = None) -> str | None:
    """Read `MEMMAN_POSTGRES_DSN_<store>` from the env file; None if absent.
    """
    key = env_key_for('postgres', 'DSN', store)
    if data_dir is None:
        return get(key)
    file_values = parse_env_file(env_file_path(data_dir))
    raw = file_values.get(key)
    return raw or None


def get_bool(name: str, default: bool = False) -> bool:
    """Return True when env var `name` resolves to a truthy string.

    Reads via `get`, so file-only resolution. The `default` argument
    is preserved for callers that need to distinguish "unset" from
    "explicit off" without re-implementing the truthy check.
    """
    raw = get(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def is_worker() -> bool:
    """Return True when running under the scheduler-triggered worker.

    Reads `MEMMAN_WORKER` directly from `os.environ` because it is a
    transient subprocess flag (set by the unit and by `cli.py` when
    spawning children). Never flows through the env file.
    """
    return os.environ.get(WORKER) == '1'


def enumerate_effective_config(redact: bool = True) -> dict[str, Any]:
    """Return a dict of every known env var name and current value.

    Installable keys resolve via `get` (env file only). Process-control
    and tuning vars are read directly from `os.environ` because they
    are never persisted to the file. Unset/empty vars map to None.
    Secret vars are replaced with '***REDACTED***' unless
    `redact=False`. Returned dict is sorted by key for stable
    diagnostic output.
    """
    out: dict[str, Any] = {}
    for name in _ALL_VARS:
        if name in _DIRECT_ENV_VARS:
            raw = os.environ.get(name)
        else:
            raw = get(name)
        if raw is None or raw == '':
            out[name] = None
            continue
        if redact and name in SECRET_VARS:
            out[name] = '***REDACTED***'
            continue
        out[name] = raw
    return dict(sorted(out.items()))


def effective_source(name: str) -> str:
    """Return where `name` resolves from: 'env', 'file', or 'unset'.

    Process-control and tuning vars read `os.environ` and report 'env'
    when set. All other keys (the installable ones) report 'file' when
    present in the env file, 'unset' otherwise. Shell-env values for
    installable keys are invisible to the runtime resolver and are
    NOT reported here.

    Diagnostic helper for `memman doctor` / `memman config show`.
    """
    if name in _DIRECT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is not None and raw != '':
            return 'env'
        return 'unset'
    file_value = _load_file_cache().get(name)
    if file_value is not None and file_value != '':
        return 'file'
    return 'unset'


def collect_install_knobs(data_dir: str) -> dict[str, str]:
    """Build the dict of values to persist to `~/.memman/env` at install.

    Precedence per key: existing env file > `os.environ` >
    `INSTALL_DEFAULTS`. The shell environment is consulted at install
    time only as a one-time seed for keys missing from the file --
    existing file values are sticky and a later shell export never
    overrides them. Once written, runtime resolution reads only the
    file (`config.get` does not consult `os.environ` for installable
    keys).

    Raises `ConfigError` (via the caller's import) when a mandatory
    secret is missing from both the file and the shell env, or when
    a non-OpenRouter endpoint has no `MEMMAN_LLM_MODEL`: the shipped
    default is an OpenRouter id that endpoint would reject.
    """
    from memman.exceptions import ConfigError

    file_values = parse_env_file(env_file_path(data_dir))

    knobs: dict[str, str] = {}
    needs_resolve: set[str] = set()
    for key in INSTALLABLE_KEYS:
        file_value = file_values.get(key, '').strip()
        if file_value:
            knobs[key] = file_value
            continue
        env_value = _shell_seed_value(key)
        if env_value:
            knobs[key] = env_value
            continue
        needs_resolve.add(key)

    endpoint = knobs.get(LLM_ENDPOINT) or INSTALL_DEFAULTS[LLM_ENDPOINT]
    if LLM_MODEL in needs_resolve and not is_openrouter_endpoint(endpoint):
        raise ConfigError(
            f'{LLM_MODEL} is required for the non-OpenRouter endpoint'
            f' {endpoint}; export it or add it to'
            f' {env_file_path(data_dir)} and re-run install')

    for key in list(needs_resolve):
        if key in INSTALL_DEFAULTS:
            knobs[key] = INSTALL_DEFAULTS[key]
            needs_resolve.discard(key)

    if (not knobs.get(LLM_API_KEY)
            and is_openrouter_endpoint(knobs.get(LLM_ENDPOINT, ''))
            and knobs.get(OPENROUTER_API_KEY)):
        knobs[LLM_API_KEY] = knobs[OPENROUTER_API_KEY]

    chosen_embed = knobs.get(EMBED_PROVIDER) or INSTALL_DEFAULTS[EMBED_PROVIDER]
    for required in sorted(required_install_keys(chosen_embed)):
        if not knobs.get(required):
            raise ConfigError(
                f'{required} is required; export it or add it to'
                f' {env_file_path(data_dir)} and re-run install')

    return knobs
