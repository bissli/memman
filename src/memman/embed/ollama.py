"""Ollama embedding client (local).

Talks to a local Ollama server's `/api/embeddings` endpoint.
Host configurable via `MEMMAN_OLLAMA_HOST` (default
`http://localhost:11434`). Model by `MEMMAN_OLLAMA_EMBED_MODEL`
(default `nomic-embed-text`, 768-dim).

`dim` is discovered from the first successful embed call.

Suggested semantic threshold (0.70) is a starting point only —
empirical recalibration on real data is required.
"""

import logging
import os
import time

import httpx
from memman import config, trace

logger = logging.getLogger('memman')

DEFAULT_MODEL = 'nomic-embed-text'
DEFAULT_HOST = 'http://localhost:11434'

_CLIENT: httpx.Client | None = None


def _session() -> httpx.Client:
    """Return the module-level httpx.Client, creating it lazily."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client()
    return _CLIENT


class Client:
    """HTTP client for Ollama's `/api/embeddings` endpoint."""

    name = 'ollama'

    def __init__(self) -> None:
        self.host = (
            os.environ.get(config.OLLAMA_HOST) or DEFAULT_HOST)
        self.model = (
            os.environ.get(config.OLLAMA_EMBED_MODEL) or DEFAULT_MODEL)
        self.dim = 0
        self._availability_cache: bool | None = None

    def available(self) -> bool:
        """Probe the host with a 1-token embed and cache the dim.
        """
        if self._availability_cache is not None:
            return self._availability_cache
        try:
            vec = self.embed('test')
            self.dim = len(vec)
            result = True
        except Exception:
            result = False
        self._availability_cache = result
        return result

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text via Ollama API."""
        url = f'{self.host}/api/embeddings'
        body = {'model': self.model, 'prompt': text}
        trace.event(
            'embed_request',
            provider='ollama',
            url=url,
            model=self.model,
            input_len=len(text))
        t0 = time.monotonic()
        resp = _session().post(url, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'embed_response',
                provider='ollama',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'Ollama returned status {resp.status_code}')
        data = resp.json()
        vec = data.get('embedding')
        if not vec:
            trace.event(
                'embed_response',
                provider='ollama',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='empty')
            raise RuntimeError('empty embedding returned')
        if self.dim == 0:
            self.dim = len(vec)
        trace.event(
            'embed_response',
            provider='ollama',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            dim=len(vec))
        return vec

    def unavailable_message(self) -> str:
        """Return error message when Ollama is not available."""
        return (
            f'Ollama not available at {self.host}'
            f' -- start Ollama or set {config.OLLAMA_HOST}')
