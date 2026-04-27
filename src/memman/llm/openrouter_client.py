"""OpenRouter client.

Speaks OpenAI-schema /v1/chat/completions. The model id is resolved at
`memman install` time (`memman.llm.openrouter_models.resolve_latest_in_family`)
and persisted to `~/.memman/env`; the runtime client reads it via
`config.get` and sends it through unchanged. Routing/privacy policy is
configured at the OpenRouter account level (see
https://openrouter.ai/settings/privacy).
"""

import logging
import time

import httpx
from memman import config, trace
from memman.exceptions import ConfigError
from memman.llm.shared import ENRICHMENT_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
from memman.llm.shared import RETRYABLE_STATUS_CODES, safe_json

logger = logging.getLogger('memman')

_CLIENT: httpx.Client | None = None


def _session() -> httpx.Client:
    """Return the module-level httpx.Client, creating it lazily."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client()
    return _CLIENT


class OpenRouterClient:
    """OpenAI-schema LLM client for OpenRouter."""

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            role_env_var: str,
            model: str,
            max_tokens: int = 1024,
            timeout: float = ENRICHMENT_TIMEOUT,
            ) -> None:
        """Initialize with endpoint, API key, and an explicit model id.

        `role_env_var` is the env var name (`MEMMAN_LLM_MODEL_FAST` or
        `MEMMAN_LLM_MODEL_SLOW`) the model came from; surfaced in error
        messages so the user knows which knob to tune.
        """
        if not model:
            raise ConfigError(
                f'{role_env_var} is empty; run `memman install` to'
                ' populate it or export it manually')
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.role_env_var = role_env_var
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        """Send a chat-completion request."""
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


_ROLE_ENV_VARS = {
    'fast': config.LLM_MODEL_FAST,
    'slow': config.LLM_MODEL_SLOW,
    }


def get_openrouter_client(role: str) -> OpenRouterClient:
    """Build an OpenRouter client for a role from configured values."""
    endpoint = config.get(config.OPENROUTER_ENDPOINT)
    if not endpoint:
        raise ConfigError(
            f'{config.OPENROUTER_ENDPOINT} is not set;'
            ' run `memman install` to populate the env file')
    api_key = config.get(config.OPENROUTER_API_KEY)
    if not api_key:
        raise ConfigError(
            f'{config.OPENROUTER_API_KEY} must be set'
            f' when {config.LLM_PROVIDER}=openrouter')
    role_env_var = _ROLE_ENV_VARS[role]
    model = config.get(role_env_var)
    if not model:
        raise ConfigError(
            f'{role_env_var} is not set; run `memman install`'
            ' to resolve and persist the latest model id')
    return OpenRouterClient(
        endpoint, api_key, role_env_var=role_env_var, model=model)
