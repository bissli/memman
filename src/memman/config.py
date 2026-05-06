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
`file > os.environ > OpenRouter resolver > INSTALL_DEFAULTS`. Shell
environment variables are read at install time only as a seed for
keys missing from the file -- existing file values are sticky and
never overridden by a later shell export.

Process-control vars (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`,
`MEMMAN_WORKER`, `MEMMAN_SCHEDULER_KIND`, `MEMMAN_DEBUG`) are NOT
installable, never written to the env file, and are read directly
from `os.environ` by their owners (`is_worker`, `trace.is_enabled`,
etc.). They do not flow through `get()`.

`INSTALLABLE_KEYS` is the single source of truth for what `memman
install` persists to `~/.memman/env`. Adding a new global knob is
one tuple entry plus an `INSTALL_DEFAULTS` row when a default exists.
"""

import os
from pathlib import Path
from typing import Any

DATA_DIR = 'MEMMAN_DATA_DIR'
STORE = 'MEMMAN_STORE'
LLM_PROVIDER = 'MEMMAN_LLM_PROVIDER'
LLM_MODEL_FAST = 'MEMMAN_LLM_MODEL_FAST'
LLM_MODEL_SLOW_CANONICAL = 'MEMMAN_LLM_MODEL_SLOW_CANONICAL'
LLM_MODEL_SLOW_METADATA = 'MEMMAN_LLM_MODEL_SLOW_METADATA'
EMBED_PROVIDER = 'MEMMAN_EMBED_PROVIDER'
RERANK_PROVIDER = 'MEMMAN_RERANK_PROVIDER'
OPENROUTER_ENDPOINT = 'MEMMAN_OPENROUTER_ENDPOINT'
DEBUG = 'MEMMAN_DEBUG'
WORKER = 'MEMMAN_WORKER'
LOG_LEVEL = 'MEMMAN_LOG_LEVEL'
DEFAULT_BACKEND = 'MEMMAN_DEFAULT_BACKEND'
DEFAULT_PG_DSN = 'MEMMAN_DEFAULT_PG_DSN'


def BACKEND_FOR(store: str) -> str:
    """Per-store backend env-key name: `MEMMAN_BACKEND_<store>`."""
    return f'MEMMAN_BACKEND_{store}'


def PG_DSN_FOR(store: str) -> str:
    """Per-store DSN env-key name: `MEMMAN_PG_DSN_<store>`."""
    return f'MEMMAN_PG_DSN_{store}'


OPENROUTER_API_KEY = 'OPENROUTER_API_KEY'
VOYAGE_API_KEY = 'VOYAGE_API_KEY'

OPENAI_EMBED_API_KEY = 'MEMMAN_OPENAI_EMBED_API_KEY'
OPENAI_EMBED_ENDPOINT = 'MEMMAN_OPENAI_EMBED_ENDPOINT'
OPENAI_EMBED_MODEL = 'MEMMAN_OPENAI_EMBED_MODEL'
OLLAMA_HOST = 'MEMMAN_OLLAMA_HOST'
OLLAMA_EMBED_MODEL = 'MEMMAN_OLLAMA_EMBED_MODEL'
OPENROUTER_EMBED_MODEL = 'MEMMAN_OPENROUTER_EMBED_MODEL'
VOYAGE_RERANK_MODEL = 'MEMMAN_VOYAGE_RERANK_MODEL'

ENV_FILENAME = 'env'

TRUTHY = frozenset({'1', 'true', 'yes', 'on'})

SECRET_VARS = frozenset({
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    OPENAI_EMBED_API_KEY,
    DEFAULT_PG_DSN,
    })

INSTALLABLE_KEYS = (
    LLM_PROVIDER,
    LLM_MODEL_FAST,
    LLM_MODEL_SLOW_CANONICAL,
    LLM_MODEL_SLOW_METADATA,
    EMBED_PROVIDER,
    RERANK_PROVIDER,
    OPENROUTER_ENDPOINT,
    LOG_LEVEL,
    OPENAI_EMBED_API_KEY,
    OPENAI_EMBED_ENDPOINT,
    OPENAI_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_EMBED_MODEL,
    OPENROUTER_EMBED_MODEL,
    VOYAGE_RERANK_MODEL,
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    DEFAULT_BACKEND,
    DEFAULT_PG_DSN,
    )

MANDATORY_INSTALL_KEYS = (
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    )

INSTALL_DEFAULTS: dict[str, str] = {
    LLM_PROVIDER: 'openrouter',
    LLM_MODEL_FAST: 'anthropic/claude-haiku-4.5',
    LLM_MODEL_SLOW_CANONICAL: 'anthropic/claude-sonnet-4.6',
    LLM_MODEL_SLOW_METADATA: 'anthropic/claude-sonnet-4.6',
    EMBED_PROVIDER: 'voyage',
    RERANK_PROVIDER: 'voyage',
    OPENROUTER_ENDPOINT: 'https://openrouter.ai/api/v1',
    LOG_LEVEL: 'WARNING',
    OPENAI_EMBED_ENDPOINT: 'https://api.openai.com',
    OPENAI_EMBED_MODEL: 'text-embedding-3-small',
    OLLAMA_HOST: 'http://localhost:11434',
    OLLAMA_EMBED_MODEL: 'nomic-embed-text',
    OPENROUTER_EMBED_MODEL: 'baai/bge-m3',
    VOYAGE_RERANK_MODEL: 'rerank-2.5-lite',
    DEFAULT_BACKEND: 'sqlite',
    }

_PROCESS_CONTROL_VARS = (DATA_DIR, STORE, WORKER, DEBUG)

_ALL_VARS = INSTALLABLE_KEYS + _PROCESS_CONTROL_VARS


_FILE_CACHE: dict[str, str] | None = None
_FILE_CACHE_PATH: str | None = None


def env_file_path(data_dir: str | None = None) -> Path:
    """Return the path to the env file under the given data dir.

    When `data_dir` is omitted, falls back to `MEMMAN_DATA_DIR` from
    `os.environ`, then to `~/.memman`. The env-file location must not
    flow through the resolver itself — that would be circular.
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


def get_store_pg_dsn(
        store: str, data_dir: str | None = None) -> str | None:
    """Read `MEMMAN_PG_DSN_<store>` from the env file; None if absent.
    """
    if data_dir is None:
        return get(PG_DSN_FOR(store))
    file_values = parse_env_file(env_file_path(data_dir))
    raw = file_values.get(PG_DSN_FOR(store))
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
    vars (`DATA_DIR`, `STORE`, `WORKER`, `DEBUG`) are read directly
    from `os.environ` because they are never persisted to the file.
    Unset/empty vars map to None. Secret vars are replaced with
    '***REDACTED***' unless `redact=False`. Returned dict is sorted by
    key for stable diagnostic output.
    """
    direct_only = {DATA_DIR, STORE, WORKER, DEBUG}
    out: dict[str, Any] = {}
    for name in _ALL_VARS:
        if name in direct_only:
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

    Process-control vars (`DATA_DIR`, `STORE`, `WORKER`, `DEBUG`) read
    `os.environ` and report 'env' when set. All other keys (the
    installable ones) report 'file' when present in the env file,
    'unset' otherwise. Shell-env values for installable keys are
    invisible to the runtime resolver and are NOT reported here.

    Diagnostic helper for `memman doctor` / `memman config show`.
    """
    if name in {DATA_DIR, STORE, WORKER, DEBUG}:
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

    Precedence per key: existing env file > `os.environ` > OpenRouter
    live resolver (FAST/SLOW only) > `INSTALL_DEFAULTS`. The shell
    environment is consulted at install time only as a one-time seed
    for keys missing from the file -- existing file values are sticky
    and a later shell export never overrides them. Once written,
    runtime resolution reads only the file (`config.get` does not
    consult `os.environ` for installable keys).

    Raises `ConfigError` (via the caller's import) when a mandatory
    secret is missing from both the file and the shell env.
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
        env_value = os.environ.get(key, '').strip()
        if env_value:
            knobs[key] = env_value
            continue
        needs_resolve.add(key)

    provider = knobs.get(LLM_PROVIDER) or INSTALL_DEFAULTS.get(LLM_PROVIDER)
    if provider == 'openrouter':
        from memman.llm.openrouter_models import resolve_latest_in_family
        api_key = knobs.get(OPENROUTER_API_KEY, '')
        endpoint = (
            knobs.get(OPENROUTER_ENDPOINT)
            or INSTALL_DEFAULTS[OPENROUTER_ENDPOINT])
        for role_key, family in (
                (LLM_MODEL_FAST, 'haiku'),
                (LLM_MODEL_SLOW_CANONICAL, 'sonnet'),
                (LLM_MODEL_SLOW_METADATA, 'sonnet')):
            if role_key not in needs_resolve or not api_key:
                continue
            resolved = resolve_latest_in_family(api_key, endpoint, family)
            if resolved:
                knobs[role_key] = resolved
                needs_resolve.discard(role_key)

    for key in list(needs_resolve):
        if key in INSTALL_DEFAULTS:
            knobs[key] = INSTALL_DEFAULTS[key]
            needs_resolve.discard(key)

    for required in MANDATORY_INSTALL_KEYS:
        if not knobs.get(required):
            raise ConfigError(
                f'{required} is required; export it or add it to'
                f' {env_file_path(data_dir)} and re-run install')

    return knobs
