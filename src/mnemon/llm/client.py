"""LLM HTTP client for causal inference."""

import logging
import os

import httpx

logger = logging.getLogger('mnemon')

LLM_ENDPOINT_VAR = 'MNEMON_LLM_ENDPOINT'
LLM_API_KEY_VAR = 'MNEMON_LLM_API_KEY'
LLM_MODEL_VAR = 'MNEMON_LLM_MODEL'
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'


class LLMClient:
    """Thin HTTP client for LLM inference."""

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            model: str = DEFAULT_MODEL,
            ) -> None:
        """Initialize with endpoint, API key, and model name."""
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """Send a completion request and return the text response."""
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            }
        body = {
            'model': self.model,
            'max_tokens': 1024,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
            }
        resp = httpx.post(
            f'{self.endpoint}/v1/messages',
            headers=headers,
            json=body,
            timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        return data['content'][0]['text']


def get_llm_client() -> LLMClient | None:
    """Return an LLMClient from env vars, or None if not configured."""
    endpoint = os.environ.get(LLM_ENDPOINT_VAR)
    api_key = os.environ.get(LLM_API_KEY_VAR)
    if not endpoint or not api_key:
        return None
    model = os.environ.get(LLM_MODEL_VAR, DEFAULT_MODEL)
    return LLMClient(endpoint, api_key, model)
