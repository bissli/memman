"""Voyage AI HTTP client for embedding generation."""

import logging
import os

import httpx

logger = logging.getLogger('memman')

DEFAULT_MODEL = 'voyage-3-lite'
DEFAULT_ENDPOINT = 'https://api.voyageai.com'
EMBEDDING_DIM = 512


class Client:
    """HTTP client for Voyage AI embedding API."""

    def __init__(self) -> None:
        self.endpoint = DEFAULT_ENDPOINT
        self.model = DEFAULT_MODEL
        self._api_key = os.environ.get('VOYAGE_API_KEY') or ''

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            }

    def available(self) -> bool:
        """Check if the embedding endpoint is reachable."""
        if not self._api_key:
            return False
        try:
            resp = httpx.post(
                f'{self.endpoint}/v1/embeddings',
                headers=self._headers(),
                json={'model': self.model, 'input': ['test']},
                timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text via Voyage API."""
        resp = httpx.post(
            f'{self.endpoint}/v1/embeddings',
            headers=self._headers(),
            json={'model': self.model, 'input': [text]},
            timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f'Voyage returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if not items or 'embedding' not in items[0]:
            raise RuntimeError('empty embedding returned')
        return items[0]['embedding']

    def unavailable_message(self) -> str:
        """Return error message when Voyage is not available."""
        return (
            f'Voyage not available at {self.endpoint}'
            f' -- set VOYAGE_API_KEY to enable embeddings')
