"""Voyage AI cross-encoder reranker client.

Mirrors `memman.embed.voyage.Client` for embeddings: API key from
`VOYAGE_API_KEY`, model from `MEMMAN_VOYAGE_RERANK_MODEL` (default
`rerank-2.5-lite`), shared httpx session via `_http.get_session`.
"""

import logging
import time

from memman import config, trace
from memman._http import get_session, post_with_retry

logger = logging.getLogger('memman')

DEFAULT_MODEL = 'rerank-2.5-lite'
DEFAULT_ENDPOINT = 'https://api.voyageai.com'


class Client:
    """HTTP client for Voyage AI rerank API."""

    name = 'voyage'

    def __init__(self) -> None:
        self.endpoint = DEFAULT_ENDPOINT
        self.model = (
            config.get(config.VOYAGE_RERANK_MODEL) or DEFAULT_MODEL)
        self._api_key = config.require(config.VOYAGE_API_KEY)
        self._availability_cache: bool | None = None

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            }

    def available(self) -> bool:
        """Return True when API key is set.

        We do not probe the endpoint here because the rerank API costs
        per call and a cheap probe would still bill. Treat the key as
        the availability signal; failures surface at rerank time.
        """
        if self._availability_cache is not None:
            return self._availability_cache
        self._availability_cache = bool(self._api_key)
        return self._availability_cache

    def rerank(self, query: str, documents: list[str],
               top_k: int | None = None) -> list[tuple[int, float]]:
        """Score (query, document) pairs via Voyage cross-encoder.

        Returns sorted (original_index, relevance_score) tuples.
        Empty `documents` short-circuits without an HTTP call.
        """
        if not documents:
            return []
        url = f'{self.endpoint}/v1/rerank'
        headers = self._headers()
        body: dict = {
            'model': self.model, 'query': query, 'documents': documents}
        if top_k is not None:
            body['top_k'] = top_k
        trace.event(
            'rerank_request',
            provider='voyage',
            url=url,
            model=self.model,
            n_docs=len(documents),
            query_len=len(query),
            headers=trace.redact_headers(headers))
        t0 = time.monotonic()
        resp = post_with_retry(
            get_session(__name__), url,
            headers=headers, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'rerank_response',
                provider='voyage',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'Voyage rerank returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        trace.event(
            'rerank_response',
            provider='voyage',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            n_items=len(items),
            usage=data.get('usage'))
        return [
            (int(d['index']), float(d['relevance_score'])) for d in items]

    def unavailable_message(self) -> str:
        """Return error message when Voyage rerank is not available."""
        return (
            f'Voyage rerank not available at {self.endpoint}'
            f' -- set {config.VOYAGE_API_KEY} to enable reranking')
