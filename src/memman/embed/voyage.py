"""Voyage AI HTTP client for embedding generation.

Uses a module-level `httpx.Client` so repeated embed calls in one
worker drain reuse the TLS connection. Tests that need to intercept
the HTTP layer monkeypatch `_CLIENT` directly with a stand-in that
implements `.post(url, headers=..., json=..., timeout=...)`.
"""

import logging
import os
import time

import httpx
from memman import config, trace

logger = logging.getLogger('memman')

DEFAULT_MODEL = 'voyage-3-lite'
DEFAULT_ENDPOINT = 'https://api.voyageai.com'
EMBEDDING_DIM = 512

_CLIENT: httpx.Client | None = None


def _session() -> httpx.Client:
    """Return the module-level httpx.Client, creating it lazily."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client()
    return _CLIENT


class Client:
    """HTTP client for Voyage AI embedding API."""

    name = 'voyage'

    def __init__(self) -> None:
        self.endpoint = DEFAULT_ENDPOINT
        self.model = DEFAULT_MODEL
        self.dim = EMBEDDING_DIM
        self._api_key = os.environ.get(config.VOYAGE_API_KEY) or ''
        self._availability_cache: bool | None = None

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            }

    def available(self) -> bool:
        """Check if the embedding endpoint is reachable.

        Memoized per instance: the probe sends a billable 1-token
        embed, so repeat calls in the same process short-circuit to
        the cached result. A False result is cached too (keyless
        clients don't suddenly acquire a key mid-process).
        """
        if self._availability_cache is not None:
            return self._availability_cache
        if not self._api_key:
            self._availability_cache = False
            return False
        try:
            resp = _session().post(
                f'{self.endpoint}/v1/embeddings',
                headers=self._headers(),
                json={'model': self.model, 'input': ['test']},
                timeout=5.0)
            result = resp.status_code == 200
        except Exception:
            result = False
        self._availability_cache = result
        return result

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text via Voyage API."""
        url = f'{self.endpoint}/v1/embeddings'
        headers = self._headers()
        body = {'model': self.model, 'input': [text]}
        trace.event(
            'embed_request',
            provider='voyage',
            url=url,
            model=self.model,
            input_len=len(text),
            headers=trace.redact_headers(headers))
        t0 = time.monotonic()
        resp = _session().post(url, headers=headers, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'embed_response',
                provider='voyage',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'Voyage returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if not items or 'embedding' not in items[0]:
            trace.event(
                'embed_response',
                provider='voyage',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='empty')
            raise RuntimeError('empty embedding returned')
        vec = items[0]['embedding']
        trace.event(
            'embed_response',
            provider='voyage',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            dim=len(vec),
            usage=data.get('usage'))
        return vec

    def unavailable_message(self) -> str:
        """Return error message when Voyage is not available."""
        return (
            f'Voyage not available at {self.endpoint}'
            f' -- set {config.VOYAGE_API_KEY} to enable embeddings')
