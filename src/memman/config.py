"""Central env-var configuration for memman.

This module owns the canonical list of environment variables memman
reads. Every env var name is a module-level constant so call sites
import the name rather than repeating the literal string. A single
`enumerate_effective_config()` helper walks the whole list for
diagnostics (`doctor`, scheduler trace events).

Design notes:
- Dependency-free on purpose (only `os` and `typing`). Safe to import
  from anywhere without creating a cycle.
- `os.environ.get(...)` is still used at call sites so tests that
  `monkeypatch.setenv` work without further indirection; this module
  standardizes the *names* and typed helpers, not the read mechanism.
- Secrets are redacted in `enumerate_effective_config` by default; the
  list in `_SECRET_VARS` is the single source of truth for what counts.
"""

import os
from typing import Any

DATA_DIR = 'MEMMAN_DATA_DIR'
STORE = 'MEMMAN_STORE'
LLM_PROVIDER = 'MEMMAN_LLM_PROVIDER'
LLM_MODEL = 'MEMMAN_LLM_MODEL'
EMBED_PROVIDER = 'MEMMAN_EMBED_PROVIDER'
OPENROUTER_ENDPOINT = 'MEMMAN_OPENROUTER_ENDPOINT'
CACHE_DIR = 'MEMMAN_CACHE_DIR'
DEBUG = 'MEMMAN_DEBUG'
WORKER = 'MEMMAN_WORKER'
LOG_LEVEL = 'MEMMAN_LOG_LEVEL'

OPENROUTER_API_KEY = 'OPENROUTER_API_KEY'
VOYAGE_API_KEY = 'VOYAGE_API_KEY'

OPENAI_EMBED_API_KEY = 'MEMMAN_OPENAI_EMBED_API_KEY'
OPENAI_EMBED_ENDPOINT = 'MEMMAN_OPENAI_EMBED_ENDPOINT'
OPENAI_EMBED_MODEL = 'MEMMAN_OPENAI_EMBED_MODEL'
OLLAMA_HOST = 'MEMMAN_OLLAMA_HOST'
OLLAMA_EMBED_MODEL = 'MEMMAN_OLLAMA_EMBED_MODEL'

DEFAULT_LLM_PROVIDER = 'openrouter'
DEFAULT_EMBED_PROVIDER = 'voyage'
DEFAULT_LOG_LEVEL = 'WARNING'

TRUTHY = frozenset({'1', 'true', 'yes', 'on'})

_SECRET_VARS = frozenset({
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    OPENAI_EMBED_API_KEY,
    })

_ALL_VARS = (
    DATA_DIR,
    STORE,
    LLM_PROVIDER,
    LLM_MODEL,
    EMBED_PROVIDER,
    OPENROUTER_ENDPOINT,
    CACHE_DIR,
    DEBUG,
    WORKER,
    LOG_LEVEL,
    OPENROUTER_API_KEY,
    VOYAGE_API_KEY,
    OPENAI_EMBED_API_KEY,
    OPENAI_EMBED_ENDPOINT,
    OPENAI_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_EMBED_MODEL,
    )


def get_bool(name: str) -> bool:
    """Return True when env var `name` holds a truthy string value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in TRUTHY


def is_worker() -> bool:
    """Return True when running under the scheduler-triggered worker.
    """
    return os.environ.get(WORKER) == '1'


def enumerate_effective_config(redact: bool = True) -> dict[str, Any]:
    """Return a dict of every known env var name and current value.

    Unset or empty vars map to None. Secret vars are replaced with
    '***REDACTED***' unless `redact=False`. The returned dict is sorted
    by key for stable diagnostic output.
    """
    out: dict[str, Any] = {}
    for name in _ALL_VARS:
        raw = os.environ.get(name)
        if raw is None or raw == '':
            out[name] = None
            continue
        if redact and name in _SECRET_VARS:
            out[name] = '***REDACTED***'
            continue
        out[name] = raw
    return dict(sorted(out.items()))
