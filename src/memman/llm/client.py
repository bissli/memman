"""OpenAI-compatible LLM client for any vendor's `/chat/completions` shim.

memman speaks one wire protocol: OpenAI's `/chat/completions`. Every
frontier vendor exposes an OpenAI-compat endpoint -- OpenRouter
natively, Anthropic at `/v1`, Google at `/v1beta/openai`, OpenAI of
course, plus Groq / DeepSeek / Mistral / Cerebras / Ollama / vLLM /
LiteLLM / HuggingFace which speak it natively. Users switch vendors
by editing `MEMMAN_LLM_ENDPOINT` (and `MEMMAN_LLM_API_KEY` plus the
three role-model slugs).

Three roles exist:

- `fast` -- synchronous CLI hot path (recall query expansion, doctor's
  connectivity probe). Reads `MEMMAN_LLM_MODEL_FAST`.
- `slow_canonical` -- canonical-content path (fact extraction,
  reconciliation). Reads `MEMMAN_LLM_MODEL_SLOW_CANONICAL`.
- `slow_metadata` -- derived-metadata path (enrichment, causal-edge
  inference). Reads `MEMMAN_LLM_MODEL_SLOW_METADATA`.

Routing the recall path to a small/fast model and the worker to a
larger/slow/reasoning model means switching the worker model never
adds latency to interactive commands. Splitting the slow worker into
canonical vs metadata leaves a knob for tuning enrichment cost
separately from the load-bearing extraction prompt.
"""

import logging
import time

import httpx
from memman import config, trace
from memman._http import ENRICHMENT_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
from memman._http import RETRYABLE_STATUS_CODES, WORKER_TIMEOUT, get_session
from memman.exceptions import ConfigError
from memman.llm.shared import safe_json

logger = logging.getLogger('memman')

ROLE_FAST = 'fast'
ROLE_SLOW_CANONICAL = 'slow_canonical'
ROLE_SLOW_METADATA = 'slow_metadata'
VALID_ROLES = frozenset({ROLE_FAST, ROLE_SLOW_CANONICAL, ROLE_SLOW_METADATA})

_ROLE_ENV_VARS = {
    ROLE_FAST: config.LLM_MODEL_FAST,
    ROLE_SLOW_CANONICAL: config.LLM_MODEL_SLOW_CANONICAL,
    ROLE_SLOW_METADATA: config.LLM_MODEL_SLOW_METADATA,
    }

FAST_MAX_TOKENS = 1024
WORKER_MAX_TOKENS = 4096

# Per-role output budget + read timeout. `fast` is the recall hot
# path and stays tight. The worker roles emit JSON that scales with
# input size (canonical rewrite, enrichment entity/keyword lists);
# a small cap truncates large insights mid-JSON and the parse fails,
# so they get a larger token budget and a longer timeout.
_ROLE_LIMITS = {
    ROLE_FAST: (FAST_MAX_TOKENS, ENRICHMENT_TIMEOUT),
    ROLE_SLOW_CANONICAL: (WORKER_MAX_TOKENS, WORKER_TIMEOUT),
    ROLE_SLOW_METADATA: (WORKER_MAX_TOKENS, WORKER_TIMEOUT),
    }

_OR_ATTRIBUTION_HEADERS = {
    'HTTP-Referer': 'https://github.com/bissli/memman',
    'X-Title': 'memman',
    }


class MemmanLLMClient:
    """OpenAI-schema LLM client for any endpoint with a `/chat/completions` shim."""

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            model: str,
            *,
            max_tokens: int = 1024,
            timeout: float = ENRICHMENT_TIMEOUT,
            extra_headers: dict[str, str] | None = None,
            ) -> None:
        """Initialize with endpoint, API key, and an explicit model id.

        `api_key` may be empty: in that case the `Authorization` header
        is omitted, supporting auth-less endpoints (Ollama, local
        vLLM/LiteLLM). `extra_headers` is merged on top of the standard
        headers and is used to attach attribution headers for known
        endpoints (OpenRouter).
        """
        if not model:
            raise ConfigError(
                'model is empty; run `memman install` to populate the'
                ' role-specific model env var or export it manually')
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = dict(extra_headers) if extra_headers else {}

    def complete(self, system: str, user: str, *,
                 temperature: float | None = None) -> str:
        """Send a chat-completion request.

        `temperature`: pass a float (typically 0.0) to pin sampling and
        get deterministic outputs across runs. When None, the provider's
        default temperature is used.
        """
        headers: dict[str, str] = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        headers.update(self.extra_headers)
        body: dict = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
                ],
            }
        if temperature is not None:
            body['temperature'] = temperature

        url = f'{self.endpoint}/chat/completions'
        for attempt in range(MAX_RETRIES):
            trace.event(
                'llm_request',
                endpoint=self.endpoint,
                url=url,
                attempt=attempt + 1,
                headers=trace.redact_headers(headers),
                body=body)
            t0 = time.monotonic()
            resp = get_session(__name__).post(
                url, headers=headers, json=body, timeout=self.timeout)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                trace.event(
                    'llm_response',
                    endpoint=self.endpoint,
                    status=resp.status_code,
                    elapsed_ms=elapsed_ms,
                    body=safe_json(resp),
                    error='http_status')
                if (resp.status_code in RETRYABLE_STATUS_CODES
                        and attempt < MAX_RETRIES - 1):
                    delay = RETRY_BACKOFF[min(
                        attempt, len(RETRY_BACKOFF) - 1)]
                    logger.debug(
                        f'llm {resp.status_code}, retry'
                        f' {attempt + 1}/{MAX_RETRIES - 1} in {delay}s')
                    time.sleep(delay)
                    continue
                raise
            data = resp.json()
            trace.event(
                'llm_response',
                endpoint=self.endpoint,
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                body=data)
            choices = data.get('choices') or []
            if not choices:
                raise RuntimeError(
                    f'llm returned no choices: {data!r}')
            try:
                return choices[0]['message']['content']
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f'llm response missing message.content'
                    f' ({exc}): {data!r}') from exc


_ROLE_CACHE: dict[str, MemmanLLMClient] = {}


def get_llm_client(role: str) -> MemmanLLMClient:
    """Return a cached `MemmanLLMClient` for the given role.

    `role` must be one of `'fast'`, `'slow_canonical'`, or
    `'slow_metadata'`. Reads `MEMMAN_LLM_ENDPOINT`,
    `MEMMAN_LLM_API_KEY`, and the role's model env var from the
    canonical env file. Raises `ConfigError` when a required value is
    missing. OpenRouter endpoints automatically receive memman's
    attribution headers; other endpoints do not.
    """
    if role not in VALID_ROLES:
        raise ValueError(
            f'unknown LLM role {role!r}; valid roles: {sorted(VALID_ROLES)}')
    cached = _ROLE_CACHE.get(role)
    if cached is not None:
        return cached
    endpoint = config.get(config.LLM_ENDPOINT)
    if not endpoint:
        raise ConfigError(
            f'{config.LLM_ENDPOINT} is not set;'
            ' run `memman install` to populate the env file')
    role_env_var = _ROLE_ENV_VARS[role]
    model = config.get(role_env_var)
    if not model:
        raise ConfigError(
            f'{role_env_var} is not set; run `memman install`'
            ' to resolve and persist the model id')
    api_key = config.get(config.LLM_API_KEY) or ''
    extra: dict[str, str] = {}
    if config.is_openrouter_endpoint(endpoint):
        extra.update(_OR_ATTRIBUTION_HEADERS)
    max_tokens, timeout = _ROLE_LIMITS[role]
    client = MemmanLLMClient(
        endpoint, api_key, model, max_tokens=max_tokens,
        timeout=timeout, extra_headers=extra or None)
    _ROLE_CACHE[role] = client
    return client


def reset_role_cache() -> None:
    """Drop cached per-role clients. Used by tests that swap env vars."""
    _ROLE_CACHE.clear()
