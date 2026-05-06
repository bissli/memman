"""OpenRouter embedding client.

OpenRouter exposes an OpenAI-schema `/embeddings` endpoint. Unlike the
generic `openai_compat` provider, this one shares its endpoint and
API key with the LLM client (`OPENROUTER_API_KEY`,
`MEMMAN_OPENROUTER_ENDPOINT`), so the user does not duplicate
credentials when running OpenRouter for both modalities. The model id
lives in `MEMMAN_OPENROUTER_EMBED_MODEL` and ships a
ZDR-compliant default (`baai/bge-m3`).

`dim` is discovered from the first successful embed call.
"""

import logging
import time

from memman import config, trace
from memman._http import get_session, post_with_retry

logger = logging.getLogger('memman')


class Client:
    """HTTP client for OpenRouter's `/embeddings` endpoint."""

    name = 'openrouter'

    def __init__(self) -> None:
        self.endpoint = config.require(config.OPENROUTER_ENDPOINT)
        self._api_key = config.require(config.OPENROUTER_API_KEY)
        self.model = config.require(config.OPENROUTER_EMBED_MODEL)
        self.dim = 0
        self._availability_cache: bool | None = None

    def prepare(self) -> None:
        """Probe the endpoint with a 1-token embed and cache `dim`.

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

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            'HTTP-Referer': 'https://github.com/bissli/memman',
            'X-Title': 'memman',
            }

    def available(self) -> bool:
        """Probe the endpoint with a 1-token embed and cache the dim."""
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
        """Generate embedding for text via OpenRouter."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in one HTTP round-trip."""
        if not texts:
            return []
        url = f'{self.endpoint.rstrip("/")}/embeddings'
        headers = self._headers()
        body = {
            'model': self.model,
            'input': texts,
            'encoding_format': 'float',
            }
        trace.event(
            'embed_request',
            provider='openrouter',
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
                provider='openrouter',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'OpenRouter embed returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if len(items) != len(texts):
            trace.event(
                'embed_response',
                provider='openrouter',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='length_mismatch',
                expected=len(texts),
                got=len(items))
            raise RuntimeError(
                f'OpenRouter returned {len(items)} vectors for'
                f' {len(texts)} inputs')
        vectors = [item.get('embedding') for item in items]
        if any(v is None for v in vectors):
            raise RuntimeError('OpenRouter returned a row with no embedding')
        if self.dim == 0 and vectors:
            self.dim = len(vectors[0])
        trace.event(
            'embed_response',
            provider='openrouter',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            batch_size=len(vectors),
            dim=self.dim,
            usage=data.get('usage'))
        return vectors

    def unavailable_message(self) -> str:
        """Return error message when OpenRouter embed is not available."""
        return (
            f'OpenRouter embed not available at {self.endpoint}'
            f' (model={self.model}); verify {config.OPENROUTER_API_KEY}'
            f' and that {self.model!r} is reachable under your'
            ' account privacy policy')
