"""Shared HTTP utilities for memman provider clients.

Per-subsystem `httpx.Client` pools (`get_session`) so each provider gets
its own connection pool; debug tracing keys events by subsystem name
and one provider's 429 should not wedge another. `post_with_retry`
implements the canonical retry policy used by both the LLM client and
the embed clients (Voyage, OpenAI-compat, OpenRouter-embed, Ollama).
"""

import logging
import time

import httpx

logger = logging.getLogger('memman')

_SESSIONS: dict[str, httpx.Client] = {}

ENRICHMENT_TIMEOUT = 10.0
WORKER_TIMEOUT = 60.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
RETRY_BACKOFF = (1.0, 2.0, 4.0)
MAX_RETRIES = 3


def get_session(name: str) -> httpx.Client:
    """Return the per-subsystem `httpx.Client`, creating it lazily."""
    client = _SESSIONS.get(name)
    if client is None:
        client = httpx.Client()
        _SESSIONS[name] = client
    return client


def reset_sessions() -> None:
    """Drop every cached session. Used by tests that swap transports."""
    for client in _SESSIONS.values():
        try:
            client.close()
        except httpx.HTTPError as exc:
            logger.debug(f'http session close failed: {exc}')
    _SESSIONS.clear()


def post_with_retry(
        session: httpx.Client,
        url: str,
        *,
        headers: dict | None = None,
        json: dict | None = None,
        timeout: float | None = None,
        ) -> httpx.Response:
    """POST with retry on transient HTTP errors.

    Retries `MAX_RETRIES` times on `RETRYABLE_STATUS_CODES` (429 + 5xx),
    sleeping `RETRY_BACKOFF[i]` seconds between attempts. Non-retryable
    errors raise immediately. The final attempt's response (success or
    the last 4xx/5xx) is returned; callers still call `raise_for_status`
    when they want to surface failures.
    """
    last_resp: httpx.Response | None = None
    for attempt in range(MAX_RETRIES):
        resp = session.post(
            url, headers=headers, json=json, timeout=timeout)
        last_resp = resp
        if resp.status_code < 400:
            return resp
        if resp.status_code not in RETRYABLE_STATUS_CODES:
            return resp
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    assert last_resp is not None
    return last_resp
