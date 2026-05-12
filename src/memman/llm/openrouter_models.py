"""OpenRouter model resolution at install time.

`memman install` calls `resolve_latest_for_role` once per role to pick
the current latest slug from OpenRouter's `/v1/models` endpoint when
the configured LLM endpoint points at OpenRouter. The endpoint is
public (no API key required) and the resolved id is written to
`~/.memman/env`; runtime reads from the file. No runtime queries
against OpenRouter, no on-disk cache.

The wire-format ids returned here keep OR's `vendor/slug` prefix
unchanged -- non-OR endpoints handle their own slugs via the wizard's
interactive prompt rather than this resolver.

The TTL cache exists only to dedupe the three intra-install calls
(one per LLM role) and to absorb a trivial retry during a single
install command.
"""

import logging
import re
import time

import cachetools
import httpx
from memman import trace

logger = logging.getLogger('memman')

MODELS_PATH = '/models'
FETCH_TIMEOUT_SECONDS = 10.0
DEFAULT_TTL_SECONDS = 3600
DEFAULT_ENDPOINT = 'https://openrouter.ai/api/v1'

_inventory_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=8, ttl=DEFAULT_TTL_SECONDS)


def _version_sort_key(model_id: str) -> tuple:
    """Extract a sortable tuple from a vendor-prefixed model id.

    Handles `claude-haiku-4.5`, `claude-sonnet-10.0`,
    `claude-haiku-4.5-v2`. Returns a tuple sorting newer-first under
    descending sort. Non-numeric suffixes (e.g. `-v2`) outrank the
    bare base when both parse to the same numeric tuple.
    """
    tail = model_id.split('/', 1)[-1].lower()
    nums = tuple(int(n) for n in re.findall(r'\d+', tail))
    has_suffix = bool(re.search(r'-[a-z]+\d*$', tail))
    return (nums, 1 if has_suffix else 0, model_id)


_ROLE_PATTERNS: dict[str, re.Pattern] = {
    'fast': re.compile(r'^anthropic/claude-haiku-\d'),
    'slow': re.compile(r'^anthropic/claude-sonnet-\d'),
    }


def _fetch_models(endpoint: str) -> list[dict]:
    """GET the public model list from OpenRouter (no auth required)."""
    url = f'{endpoint.rstrip("/")}{MODELS_PATH}'
    trace.event('openrouter_models_request', url=url)
    t0 = time.monotonic()
    with httpx.Client() as client:
        resp = client.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get('data')
    if not isinstance(data, list):
        raise RuntimeError(
            f'unexpected OpenRouter /models shape: {type(payload).__name__}')
    trace.event(
        'openrouter_models_response',
        status=resp.status_code,
        elapsed_ms=elapsed_ms,
        model_count=len(data))
    return data


def resolve_latest_for_role(
        role: str, endpoint: str = DEFAULT_ENDPOINT) -> str | None:
    """Return the latest OR-formatted model slug for `role`, or None.

    `role` is `'fast'` or `'slow'`. Returns None when no rule exists,
    when OR's catalog has no match, or when the network call fails --
    the caller falls back to `INSTALL_DEFAULTS`. Only fires when the
    install endpoint points at OpenRouter (callers guard via
    `config.is_openrouter_endpoint`).
    """
    pattern = _ROLE_PATTERNS.get(role)
    if pattern is None:
        return None
    cache_key = (endpoint, role)
    cached = _inventory_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        models = _fetch_models(endpoint)
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'openrouter /models fetch failed ({exc}); '
            f'cannot resolve latest {role!r}')
        return None

    candidates: list[str] = []
    seen: set[str] = set()
    for entry in models:
        mid = entry.get('id') or entry.get('model_id') or ''
        if not pattern.match(mid):
            continue
        if mid in seen:
            continue
        seen.add(mid)
        candidates.append(mid)

    if not candidates:
        logger.warning(f'no {role!r} match in openrouter inventory')
        return None

    candidates.sort(key=_version_sort_key, reverse=True)
    picked = candidates[0]
    _inventory_cache[cache_key] = picked
    return picked


def clear_cache() -> None:
    """Wipe the in-memory cache (used by tests)."""
    _inventory_cache.clear()
