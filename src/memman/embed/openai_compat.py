"""OpenAI-compatible embedding client (OpenAI, OpenRouter, vLLM,
LiteLLM, any other provider exposing `/v1/embeddings`).

Endpoint, model, and API key are read from the env-or-file resolver
(populated at install time from `INSTALL_DEFAULTS`).

`dim` is discovered from the first successful embed call and
cached on the client; `available()` performs the discovery probe.

Suggested semantic threshold (0.75) is a starting point only —
empirical recalibration on real data is required before relying on
edge density assumptions.
"""

import logging
import time

from memman import config, trace
from memman._http import get_session, post_with_retry
from memman.exceptions import ConfigError

logger = logging.getLogger('memman')


class Client:
    """HTTP client for OpenAI-compatible `/v1/embeddings` endpoints."""

    name = 'openai'

    def __init__(self) -> None:
        endpoint = config.get(config.OPENAI_EMBED_ENDPOINT)
        model = config.get(config.OPENAI_EMBED_MODEL)
        if not endpoint or not model:
            raise ConfigError(
                f'{config.OPENAI_EMBED_ENDPOINT} or'
                f' {config.OPENAI_EMBED_MODEL} is unset; run'
                ' `memman install` to populate the env file')
        self.endpoint = endpoint
        self.model = model
        self.dim = 0
        self._api_key = config.get(config.OPENAI_EMBED_API_KEY) or ''
        self._availability_cache: bool | None = None

    def prepare(self) -> None:
        """Probe the endpoint with a 1-token embed and cache `dim`.

        Idempotent: if `dim` is already non-zero, returns immediately.
        Failures are swallowed so a missing api_key or unreachable
        endpoint leaves `dim=0` rather than raising; the next embed()
        will surface the actual error.
        """
        if self.dim:
            return
        try:
            vec = self.embed('test')
            self.dim = len(vec)
        except Exception:
            return

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            }

    def available(self) -> bool:
        """Probe the endpoint with a 1-token embed and cache the dim.
        """
        if self._availability_cache is not None:
            return self._availability_cache
        if not self._api_key:
            self._availability_cache = False
            return False
        try:
            vec = self.embed('test')
            self.dim = len(vec)
            result = True
        except Exception:
            result = False
        self._availability_cache = result
        return result

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text via OpenAI-compatible API."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in one HTTP round-trip."""
        if not texts:
            return []
        url = f'{self.endpoint}/v1/embeddings'
        headers = self._headers()
        body = {'model': self.model, 'input': texts}
        trace.event(
            'embed_request',
            provider='openai',
            url=url,
            model=self.model,
            batch_size=len(texts),
            input_lens=[len(t) for t in texts],
            headers=trace.redact_headers(headers))
        t0 = time.monotonic()
        resp = post_with_retry(
            get_session(__name__), url,
            headers=headers, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'embed_response',
                provider='openai',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'OpenAI-compatible endpoint returned status'
                f' {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if len(items) != len(texts):
            trace.event(
                'embed_response',
                provider='openai',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='length_mismatch',
                expected=len(texts),
                got=len(items))
            raise RuntimeError(
                f'OpenAI-compatible endpoint returned {len(items)}'
                f' vectors for {len(texts)} inputs')
        vectors = [item.get('embedding') for item in items]
        if any(v is None for v in vectors):
            raise RuntimeError('OpenAI endpoint returned a row with no embedding')
        if self.dim == 0 and vectors:
            self.dim = len(vectors[0])
        trace.event(
            'embed_response',
            provider='openai',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            batch_size=len(vectors),
            dim=self.dim,
            usage=data.get('usage'))
        return vectors

    def unavailable_message(self) -> str:
        """Return error message when the endpoint is not available."""
        return (
            f'OpenAI-compatible endpoint not available at'
            f' {self.endpoint} -- set'
            f' {config.OPENAI_EMBED_API_KEY} to enable embeddings')
