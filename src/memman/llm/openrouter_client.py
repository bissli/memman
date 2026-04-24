"""OpenRouter client with ZDR enforcement and dynamic Haiku selection.

Speaks OpenAI-schema /v1/chat/completions. Every request forces the
`provider.zdr=true` and `provider.data_collection="deny"` fields so
OpenRouter routes only to endpoints with formal zero-data-retention
agreements.

Model selection is dynamic: on first call the client fetches (or reuses
the cached) ZDR endpoint list and picks the latest Anthropic Haiku
available. `MEMMAN_LLM_MODEL` overrides the auto-pick, but the override
is still validated against the ZDR list — the client refuses to send
to a non-ZDR endpoint.
"""

import logging
import os
import time

import httpx
from memman import config, trace
from memman.exceptions import ConfigError
from memman.llm.openrouter_cache import get_zdr_endpoints, pick_latest_haiku
from memman.llm.shared import ENRICHMENT_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
from memman.llm.shared import RETRYABLE_STATUS_CODES, safe_json

logger = logging.getLogger('memman')

DEFAULT_OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1'

_CLIENT: httpx.Client | None = None


def _session() -> httpx.Client:
    """Return the module-level httpx.Client, creating it lazily."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client()
    return _CLIENT


class OpenRouterClient:
    """OpenAI-schema LLM client that enforces ZDR routing."""

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            model: str | None = None,
            max_tokens: int = 1024,
            timeout: float = ENRICHMENT_TIMEOUT,
            ) -> None:
        """Initialize with endpoint and API key; model resolved lazily."""
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self._model_override = model
        self._resolved_model: str | None = None
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def model(self) -> str:
        """Return the resolved model ID, picking lazily if unset."""
        if self._resolved_model is None:
            self._resolved_model = self._resolve_model()
        return self._resolved_model

    def _resolve_model(self) -> str:
        """Pick the model ID to use, validating against the ZDR cache."""
        endpoints = get_zdr_endpoints()
        if self._model_override:
            available = {e.get('model_id') for e in endpoints}
            if self._model_override not in available:
                raise ConfigError(
                    f'{config.LLM_MODEL}={self._model_override!r} is not in'
                    ' the current OpenRouter ZDR inventory; refusing to'
                    ' route via non-ZDR endpoints')
            logger.debug(
                f'openrouter model override: {self._model_override}')
            trace.event(
                'llm_model_resolved',
                source='override',
                model=self._model_override,
                endpoint_count=len(endpoints))
            return self._model_override
        picked = pick_latest_haiku(endpoints)
        logger.debug(f'openrouter auto-picked model: {picked}')
        trace.event(
            'llm_model_resolved',
            source='auto',
            model=picked,
            endpoint_count=len(endpoints))
        return picked

    def complete(self, system: str, user: str) -> str:
        """Send a chat-completion request with ZDR enforced."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/bissli/memman',
            'X-Title': 'memman',
            }
        body = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
                ],
            'provider': {
                'zdr': True,
                'data_collection': 'deny',
                'allow_fallbacks': True,
                },
            }

        url = f'{self.endpoint}/chat/completions'
        for attempt in range(MAX_RETRIES):
            trace.event(
                'llm_request',
                provider='openrouter',
                url=url,
                attempt=attempt + 1,
                headers=trace.redact_headers(headers),
                body=body)
            t0 = time.monotonic()
            resp = _session().post(
                url, headers=headers, json=body, timeout=self.timeout)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                trace.event(
                    'llm_response',
                    provider='openrouter',
                    status=resp.status_code,
                    elapsed_ms=elapsed_ms,
                    body=safe_json(resp),
                    error='http_status')
                if (resp.status_code in RETRYABLE_STATUS_CODES
                        and attempt < MAX_RETRIES - 1):
                    delay = RETRY_BACKOFF[min(
                        attempt, len(RETRY_BACKOFF) - 1)]
                    logger.debug(
                        f'openrouter {resp.status_code}, retry'
                        f' {attempt + 1}/{MAX_RETRIES - 1} in {delay}s')
                    time.sleep(delay)
                    continue
                raise
            data = resp.json()
            trace.event(
                'llm_response',
                provider='openrouter',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                body=data)
            choices = data.get('choices') or []
            if not choices:
                raise RuntimeError(
                    f'openrouter returned no choices: {data!r}')
            try:
                return choices[0]['message']['content']
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f'openrouter response missing message.content'
                    f' ({exc}): {data!r}') from exc


def get_openrouter_client() -> OpenRouterClient:
    """Build an OpenRouter client from environment variables."""
    endpoint = os.environ.get(
        config.OPENROUTER_ENDPOINT, DEFAULT_OPENROUTER_ENDPOINT)
    api_key = os.environ.get(config.OPENROUTER_API_KEY)
    if not api_key:
        raise ConfigError(
            f'{config.OPENROUTER_API_KEY} must be set'
            f' when {config.LLM_PROVIDER}=openrouter')
    model_override = os.environ.get(config.LLM_MODEL)
    return OpenRouterClient(endpoint, api_key, model=model_override)
