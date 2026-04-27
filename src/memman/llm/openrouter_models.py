"""OpenRouter model resolution at install time.

`memman install` calls `resolve_latest_in_family` once per role to pick
the current latest haiku/sonnet from OpenRouter's full model inventory.
The resolved id is written to `~/.memman/env`; runtime reads from the
file. No runtime queries against OpenRouter, no on-disk cache, no ZDR
filter (OpenRouter enforces ZDR per request).

The TTL cache exists only to dedupe the two intra-install calls
(`family='haiku'` and `family='sonnet'`) and to absorb a trivial retry
during a single install command.
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

_inventory_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=4, ttl=DEFAULT_TTL_SECONDS)


def _version_sort_key(model_id: str) -> tuple:
    """Extract a sortable tuple from an Anthropic model id.

    Handles `claude-haiku-4.5`, `claude-sonnet-10.0`, `claude-haiku-4.5-v2`.
    Returns a tuple sorting newer-first under descending sort.
    Non-numeric suffixes (e.g. `-v2`) outrank the bare base when both
    parse to the same numeric tuple.
    """
    tail = model_id.split('/', 1)[-1].lower()
    nums = tuple(int(n) for n in re.findall(r'\d+', tail))
    has_suffix = bool(re.search(r'-[a-z]+\d*$', tail))
    return (nums, 1 if has_suffix else 0, model_id)


def _fetch_models(api_key: str, endpoint: str) -> list[dict]:
    """GET the full model list from OpenRouter."""
    url = f'{endpoint.rstrip("/")}{MODELS_PATH}'
    trace.event('openrouter_models_request', url=url)
    headers = {'Authorization': f'Bearer {api_key}'}
    t0 = time.monotonic()
    with httpx.Client() as client:
        resp = client.get(
            url, headers=headers, timeout=FETCH_TIMEOUT_SECONDS)
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


def resolve_latest_in_family(
        api_key: str, endpoint: str, family: str) -> str | None:
    """Return the latest Anthropic model id matching `family`, or None.

    `family` is a substring like `'haiku'` or `'sonnet'` (case-insensitive).
    Picks the highest-versioned `anthropic/...` id whose tail contains
    the family substring. Returns None on any network or parse failure
    so the caller (install) can fall back to its hard-coded default.
    """
    cache_key = (endpoint, family.lower())
    cached = _inventory_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        models = _fetch_models(api_key, endpoint)
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'openrouter /models fetch failed ({exc}); '
            f'cannot resolve latest {family!r}')
        return None

    candidates: list[str] = []
    seen: set[str] = set()
    needle = family.lower()
    for entry in models:
        mid = entry.get('id') or entry.get('model_id') or ''
        if not mid.startswith('anthropic/'):
            continue
        if needle not in mid.lower():
            continue
        if mid in seen:
            continue
        seen.add(mid)
        candidates.append(mid)

    if not candidates:
        logger.warning(
            f'no anthropic/{family} model in openrouter inventory')
        return None

    candidates.sort(key=_version_sort_key, reverse=True)
    picked = candidates[0]
    _inventory_cache[cache_key] = picked
    return picked


def clear_cache() -> None:
    """Wipe the in-memory cache (used by tests)."""
    _inventory_cache.clear()
