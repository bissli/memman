"""OpenAI-compatible HTTP client for embedding generation.

Covers: OpenAI, LM Studio, llama.cpp, vLLM, Groq, Mistral,
and Ollama's OpenAI-compat mode.
"""

import logging
import os

import httpx

logger = logging.getLogger('mnemon')

DEFAULT_MODEL = 'text-embedding-3-small'
DEFAULT_ENDPOINT = 'https://api.openai.com'


class Client:
    """HTTP client for OpenAI-compatible embedding API."""

    def __init__(self) -> None:
        self.endpoint = os.environ.get(
            'MNEMON_EMBED_ENDPOINT', DEFAULT_ENDPOINT)
        self.model = os.environ.get(
            'MNEMON_EMBED_MODEL', DEFAULT_MODEL)
        self._api_key = os.environ.get('MNEMON_EMBED_API_KEY', '')

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional auth."""
        headers: dict[str, str] = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'
        return headers

    def available(self) -> bool:
        """Check if the embedding endpoint is reachable and lists the model."""
        try:
            resp = httpx.get(
                f'{self.endpoint}/v1/models',
                headers=self._headers(),
                timeout=5.0)
            if resp.status_code != 200:
                return False
            models = resp.json().get('data', [])
            return any(m.get('id') == self.model for m in models)
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text via OpenAI-compatible API."""
        resp = httpx.post(
            f'{self.endpoint}/v1/embeddings',
            headers=self._headers(),
            json={'model': self.model, 'input': text},
            timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f'embedding endpoint returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if not items or 'embedding' not in items[0]:
            raise RuntimeError('empty embedding returned')
        return items[0]['embedding']

    def unavailable_message(self) -> str:
        """Return error message when endpoint is not available."""
        return (
            f'OpenAI-compatible endpoint not available at'
            f' {self.endpoint} for model {self.model}')
