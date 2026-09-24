"""LLM query expansion."""

import logging

import cachetools
from memman import config, trace
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient
from memman.llm.shared import parse_json_response

logger = logging.getLogger('memman')

QUERY_EXPANSION_SYSTEM = (
    'Expand a search query for a personal memory system.\n\n'
    'Return JSON:\n'
    '{"expanded_query": "original plus synonyms and related terms",\n'
    ' "intent": "WHY|WHEN|ENTITY|GENERAL"}\n\n'
    'Keep expanded_query under 50 words.')


_EXPAND_CACHE_TTL = 300
_EXPAND_CACHE_MAX = 256


def _normalize_for_cache(query: str) -> str:
    """Lowercase + collapse whitespace; nothing else."""
    return ' '.join(query.lower().split())


def _expand_cache_key(query: str) -> str:
    """Salt with the configured fast-model id.

    The model id is resolved at install time and persisted to
    `~/.memman/env`, so `config.require` always returns a real value
    here. Reaching this with an unset key means install was never run,
    which is a `ConfigError` upstream callers handle.
    """
    import hashlib
    salt = config.require(config.LLM_MODEL_FAST)
    digest = hashlib.sha256(
        f'{_normalize_for_cache(query)}|{salt}'.encode())
    return digest.hexdigest()[:16]


_expand_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=_EXPAND_CACHE_MAX, ttl=_EXPAND_CACHE_TTL)


def reset_expand_cache() -> None:
    """Drop cached query expansions. Used by tests that swap env vars."""
    _expand_cache.clear()


def expand_query(
        llm_client: MemmanLLMClient,
        query: str) -> dict:
    """Expand recall query with synonyms and related terms.

    Returns dict with: expanded_query, intent.
    On failure: passthrough with original query. Repeated calls with
    the same query in the same process hit a `cachetools.TTLCache`
    keyed by sha256(normalized_query | $MEMMAN_LLM_MODEL_FAST). Cache
    lives only for the duration of one CLI invocation (memman is a
    one-shot CLI), so persistence across processes is left to the
    LLM provider's own response cache.
    """
    cache_key = _expand_cache_key(query)
    cached = _expand_cache.get(cache_key)
    if cached is not None:
        trace.event(
            'query_expand_result', outcome='cache_hit', **cached)
        return dict(cached)

    trace.event('query_expand_start', query=query)
    try:
        raw = llm_client.complete(
            QUERY_EXPANSION_SYSTEM, query,
            stage=llm_usage.STAGE_QUERY_EXPANSION)
    except Exception as exc:
        logger.debug('LLM query expansion failed, using passthrough')
        trace.event(
            'query_expand_result',
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {'expanded_query': query, 'intent': None}

    parsed = parse_json_response(raw)
    if parsed is None:
        return {'expanded_query': query, 'intent': None}

    expanded = parsed.get('expanded_query', query)
    if not isinstance(expanded, str) or not expanded.strip():
        expanded = query

    intent = parsed.get('intent')
    if intent not in {'WHY', 'WHEN', 'ENTITY', 'GENERAL'}:
        intent = None

    result = {
        'expanded_query': expanded,
        'intent': intent,
        }
    _expand_cache[cache_key] = dict(result)
    trace.event('query_expand_result', outcome='ok', **result)
    return result
