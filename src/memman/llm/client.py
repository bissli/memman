"""LLM HTTP client and JSON response parsing."""

import json
import logging
import os
import time

import click
import httpx

logger = logging.getLogger('memman')

LLM_ENDPOINT_VAR = 'MEMMAN_LLM_ENDPOINT'
LLM_API_KEY_VAR = 'MEMMAN_LLM_API_KEY'
LLM_MODEL_VAR = 'MEMMAN_LLM_MODEL'
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'
DEFAULT_ENDPOINT = 'https://api.anthropic.com'
ENRICHMENT_TIMEOUT = 10.0
MAX_RETRIES = 3
RETRY_BACKOFF = (1.0, 2.0)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)


class LLMClient:
    """Thin HTTP client for LLM inference."""

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            model: str = DEFAULT_MODEL,
            max_tokens: int = 1024,
            timeout: float = ENRICHMENT_TIMEOUT,
            ) -> None:
        """Initialize with endpoint, API key, and model name."""
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        """Send a completion request and return the text response."""
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            }
        body = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
            }
        for attempt in range(MAX_RETRIES):
            resp = httpx.post(
                f'{self.endpoint}/v1/messages',
                headers=headers,
                json=body,
                timeout=self.timeout)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                if (resp.status_code in RETRYABLE_STATUS_CODES
                        and attempt < MAX_RETRIES - 1):
                    delay = RETRY_BACKOFF[min(
                        attempt, len(RETRY_BACKOFF) - 1)]
                    logger.debug(
                        f'LLM {resp.status_code}, retry '
                        f'{attempt + 1}/{MAX_RETRIES - 1} '
                        f'in {delay}s')
                    time.sleep(delay)
                    continue
                raise
            data = resp.json()
            return data['content'][0]['text']


def strip_code_fences(raw: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = raw.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        text = text.removesuffix('```').strip()
    return text


def parse_json_response(raw: str) -> dict | None:
    """Parse JSON dict from LLM response, handling code blocks."""
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_json_list_response(raw: str) -> list | None:
    """Parse JSON list from LLM response, handling code blocks."""
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def get_llm_client() -> LLMClient:
    """Return an LLMClient from env vars.

    Raises click.ClickException if no API key is configured.
    """
    endpoint = os.environ.get(LLM_ENDPOINT_VAR, DEFAULT_ENDPOINT)
    api_key = (os.environ.get(LLM_API_KEY_VAR)
               or os.environ.get('ANTHROPIC_API_KEY'))
    if not api_key:
        raise click.ClickException(
            'ANTHROPIC_API_KEY or MEMMAN_LLM_API_KEY must be set')
    model = os.environ.get(LLM_MODEL_VAR, DEFAULT_MODEL)
    return LLMClient(endpoint, api_key, model)
