"""Ollama embedding client (local).

Talks to a local Ollama server's `/api/embeddings` endpoint. Host and
model are read from the env-or-file resolver (populated at install
time from `INSTALL_DEFAULTS`).

`dim` is discovered from the first successful embed call.

Suggested semantic threshold (0.70) is a starting point only —
empirical recalibration on real data is required.
"""

import logging
import time

from memman import config, trace
from memman._http import get_session, post_with_retry
from memman.exceptions import ConfigError

logger = logging.getLogger('memman')


class Client:
    """HTTP client for Ollama's `/api/embeddings` endpoint."""

    name = 'ollama'

    def __init__(self) -> None:
        host = config.get(config.OLLAMA_HOST)
        model = config.get(config.OLLAMA_EMBED_MODEL)
        if not host or not model:
            raise ConfigError(
                f'{config.OLLAMA_HOST} or {config.OLLAMA_EMBED_MODEL}'
                ' is unset; run `memman install` to populate the env file')
        self.host = host
        self.model = model
        self.dim = 0
        self._availability_cache: bool | None = None

    def prepare(self) -> None:
        """Probe the host with a 1-token embed and cache `dim`.

        Idempotent. Failures are swallowed; the next embed() call
        will surface the actual error.
        """
        if self.dim:
            return
        try:
            vec = self.embed('test')
            self.dim = len(vec)
        except Exception:
            return

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
        resp = post_with_retry(
            get_session(__name__), url, json=body, timeout=30.0)
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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts. Ollama's /api/embeddings is single-input only,
        so this calls embed() per text.
        """
        return [self.embed(t) for t in texts]

    def unavailable_message(self) -> str:
        """Return error message when Ollama is not available."""
        return (
            f'Ollama not available at {self.host}'
            f' -- start Ollama or set {config.OLLAMA_HOST}')
