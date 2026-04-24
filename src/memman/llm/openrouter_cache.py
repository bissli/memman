"""OpenRouter ZDR endpoint-list cache.

Two-tier: cachetools TTLCache in-process for a single worker run, and a
JSON file on disk for cross-invocation persistence (cron fires a new
process every 15 min). 24h TTL on both tiers. Stale-disk fallback when
the network is unavailable.
"""

import json
import logging
import os
import time
from pathlib import Path

import cachetools
import httpx

logger = logging.getLogger('memman')

ZDR_ENDPOINT_URL = 'https://openrouter.ai/api/v1/endpoints/zdr'
DEFAULT_TTL_SECONDS = 86400
FETCH_TIMEOUT_SECONDS = 10.0
CACHE_FILENAME = 'openrouter-zdr.json'

_memory_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=1, ttl=DEFAULT_TTL_SECONDS)


def default_cache_dir() -> str:
    """Return ~/.memman/cache (via MEMMAN_CACHE_DIR override)."""
    env = os.environ.get('MEMMAN_CACHE_DIR')
    if env:
        return env
    return str(Path.home() / '.memman' / 'cache')


def cache_file_path(cache_dir: str | None = None) -> Path:
    """Return the JSON cache file path."""
    d = cache_dir or default_cache_dir()
    return Path(d) / CACHE_FILENAME


def _fetch() -> list[dict]:
    """GET the ZDR endpoints list from OpenRouter."""
    resp = httpx.get(ZDR_ENDPOINT_URL, timeout=FETCH_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get('data')
    if not isinstance(data, list):
        raise RuntimeError(
            f'unexpected OpenRouter response shape: {type(payload).__name__}')
    return data


def _read_disk(path: Path, ttl: int) -> tuple[list[dict] | None, bool]:
    """Return (endpoints, is_fresh) from the disk cache."""
    if not path.exists():
        return None, False
    age = time.time() - path.stat().st_mtime
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f'disk cache unreadable at {path}: {exc}')
        return None, False
    if not isinstance(data, list):
        return None, False
    return data, age < ttl


def _write_disk(path: Path, data: list[dict]) -> None:
    """Write the ZDR list to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def get_zdr_endpoints(
        cache_dir: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        force_refresh: bool = False,
        ) -> list[dict]:
    """Return the list of ZDR-capable OpenRouter endpoints.

    Tries in-memory cache, then disk (if fresh), then HTTP fetch.
    On network failure with a stale disk cache, serves the stale data
    with a warning. Raises only when all sources are unavailable.
    """
    if not force_refresh and 'data' in _memory_cache:
        return _memory_cache['data']

    path = cache_file_path(cache_dir)

    if not force_refresh:
        disk_data, fresh = _read_disk(path, ttl_seconds)
        if disk_data is not None and fresh:
            _memory_cache['data'] = disk_data
            return disk_data

    try:
        endpoints = _fetch()
        _write_disk(path, endpoints)
        _memory_cache['data'] = endpoints
        return endpoints
    except (httpx.HTTPError, RuntimeError) as exc:
        disk_data, _ = _read_disk(path, ttl=10 ** 12)
        if disk_data is not None:
            logger.warning(
                f'OpenRouter fetch failed ({exc}); serving stale disk cache')
            _memory_cache['data'] = disk_data
            return disk_data
        raise RuntimeError(
            f'no ZDR cache available: fetch failed ({exc}) and no disk'
            f' cache at {path}') from exc


def clear_cache() -> None:
    """Wipe the in-memory cache (used by tests)."""
    _memory_cache.clear()


def pick_latest_haiku(endpoints: list[dict]) -> str:
    """Pick the latest Anthropic Haiku model ID from the ZDR list.

    Sorts Anthropic Haiku model IDs descending by string — works for
    the current ID scheme (claude-haiku-4.5 sorts above claude-3.5-haiku
    lexicographically because '4' > '3').
    """
    haikus = []
    seen = set()
    for ep in endpoints:
        mid = ep.get('model_id', '')
        if 'haiku' not in mid.lower():
            continue
        if not mid.startswith('anthropic/'):
            continue
        if mid in seen:
            continue
        seen.add(mid)
        haikus.append(mid)
    if not haikus:
        raise RuntimeError(
            'no Anthropic Haiku model in ZDR inventory; refusing to proceed')
    haikus.sort(reverse=True)
    return haikus[0]
