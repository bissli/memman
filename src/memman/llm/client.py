"""OpenAI-compatible LLM client for any vendor's `/chat/completions` shim.

memman speaks one wire protocol: OpenAI's `/chat/completions`. Every
frontier vendor exposes an OpenAI-compat endpoint -- OpenRouter
natively, Anthropic at `/v1`, Google at `/v1beta/openai`, OpenAI of
course, plus Groq / DeepSeek / Mistral / Cerebras / Ollama / vLLM /
LiteLLM / HuggingFace which speak it natively. The endpoint and key
are the shared `MEMMAN_ENDPOINT` and `MEMMAN_API_KEY`, which the
embed and rerank clients also use.

One model serves every call - enrichment and doctor's connectivity
probe - and `MEMMAN_LLM_MODEL` names it.
"""

import logging
import time

import httpx
from memman import config, trace
from memman._http import ENRICHMENT_TIMEOUT, MAX_RETRIES
from memman._http import OPENROUTER_ATTRIBUTION_HEADERS, RETRY_BACKOFF
from memman._http import RETRYABLE_STATUS_CODES, WORKER_TIMEOUT, get_session
from memman._http import privacy_routing
from memman.exceptions import ConfigError
from memman.llm import usage as llm_usage
from memman.llm.shared import safe_json

logger = logging.getLogger('memman')

# Enrichment emits JSON that scales with input size. A small cap
# truncates a large insight mid-JSON and the parse fails, so the client
# takes a large token budget and, with WORKER_TIMEOUT, a long timeout.
WORKER_MAX_TOKENS = 4096

EMPTY_RETRY_DELAY = 0.1


class MemmanLLMClient:
    """OpenAI-schema client for any endpoint with `/chat/completions`.
    """

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            model: str,
            *,
            max_tokens: int = 1024,
            timeout: float = ENRICHMENT_TIMEOUT,
            extra_headers: dict[str, str] | None = None,
            provider_routing: dict | None = None,
            ) -> None:
        """Bind the client to an endpoint, API key, and model id.

        Parameters
        ----------
        endpoint : str
            Base URL; a trailing slash is dropped.
        api_key : str
            May be empty: the `Authorization` header is then omitted,
            for auth-less endpoints (Ollama, local vLLM/LiteLLM).
        model : str
            Model id. Raises `ConfigError` when empty.
        max_tokens : int, default 1024
            Completion token cap sent with each request.
        timeout : float, default ENRICHMENT_TIMEOUT
            Per-request timeout in seconds.
        extra_headers : dict[str, str] or None, default None
            Merged over the standard headers (OpenRouter attribution).
        provider_routing : dict or None, default None
            Sent verbatim as the body's `provider` field, and omitted
            when None. Only an OpenRouter endpoint is given one, since
            a vendor-neutral shim rejects an unknown key.
        """
        self.provider_routing = provider_routing
        if not model:
            raise ConfigError(
                'model is empty; run `memman install` to populate'
                ' MEMMAN_LLM_MODEL or export it manually')
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = dict(extra_headers) if extra_headers else {}

    def complete(self, system: str, user: str, *, stage: str) -> str:
        """Send a chat-completion request and return the message content.

        Parameters
        ----------
        system : str
            System prompt.
        user : str
            User prompt.
        stage : str
            Originating pipeline stage from `llm.usage.VALID_STAGES`;
            every attempt's `usage` block is charged to it. Unknown
            stages raise `ValueError` so a typo cannot create a
            silent phantom bucket.

        Returns
        -------
        str
            The first choice's `message.content`.

        Raises
        ------
        ValueError
            On an unknown `stage`.
        httpx.HTTPStatusError
            On a non-retryable status, or a retryable one after the
            last attempt.
        RuntimeError
            When every attempt returns an empty body, or a response
            lacks `message.content` (raised at once, without retry).

        Notes
        -----
        - Makes up to `MAX_RETRIES` attempts. A retryable status sleeps
          `RETRY_BACKOFF`. An empty body (no `choices`, or empty,
          whitespace-only, or null `content`) or a non-JSON 200 sleeps
          `EMPTY_RETRY_DELAY`.
        """
        if stage not in llm_usage.VALID_STAGES:
            raise ValueError(
                f'unknown LLM stage {stage!r};'
                f' valid stages: {sorted(llm_usage.VALID_STAGES)}')
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
        if self.provider_routing:
            body['provider'] = self.provider_routing

        url = f'{self.endpoint}/chat/completions'
        for attempt in range(MAX_RETRIES):
            trace.event(
                'llm_request',
                endpoint=self.endpoint,
                url=url,
                attempt=attempt + 1,
                stage=stage,
                headers=trace.redact_headers(headers),
                body=body)
            t0 = time.monotonic()
            resp = get_session(__name__).post(
                url, headers=headers, json=body, timeout=self.timeout)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                err_body = safe_json(resp)
                usage_block = (
                    err_body.get('usage')
                    if isinstance(err_body, dict) else None)
                # A non-2xx attempt books as `http_errors`, never
                # `calls`: a 429 storm bills nothing.
                llm_usage.record(stage, usage_block, http_error=True)
                trace.event(
                    'llm_response',
                    endpoint=self.endpoint,
                    status=resp.status_code,
                    elapsed_ms=elapsed_ms,
                    stage=stage,
                    usage=usage_block,
                    body=err_body,
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
            # Keep the raw text for the trace and error surfaces: the
            # operator must tell an HTML error page from truncated JSON.
            raw_body = safe_json(resp)
            data = raw_body if isinstance(raw_body, dict) else None
            # Every HTTP-200 attempt is a billed completion, whether
            # malformed, empty, or successful. Record it once per
            # attempt, before branching, so a raise cannot skip the
            # ledger.
            usage_block = (
                data.get('usage') if isinstance(data, dict) else None)
            llm_usage.record(stage, usage_block)
            choices = (
                (data.get('choices') or [])
                if isinstance(data, dict) else [])
            empty_kind = ''
            content = None
            if data is None:
                empty_kind = 'unparseable_body'
            elif not choices:
                empty_kind = 'empty_choices'
            else:
                try:
                    content = choices[0]['message']['content']
                except (KeyError, TypeError) as exc:
                    trace.event(
                        'llm_response',
                        endpoint=self.endpoint,
                        status=resp.status_code,
                        elapsed_ms=elapsed_ms,
                        stage=stage,
                        usage=usage_block,
                        body=raw_body)
                    raise RuntimeError(
                        f'llm response missing message.content'
                        f' ({exc}): {raw_body!r}') from exc
                if content is None or (isinstance(content, str)
                                       and not content.strip()):
                    empty_kind = 'empty_content'
            if not empty_kind:
                trace.event(
                    'llm_response',
                    endpoint=self.endpoint,
                    status=resp.status_code,
                    elapsed_ms=elapsed_ms,
                    stage=stage,
                    usage=usage_block,
                    body=raw_body)
                return content
            trace.event(
                'llm_response',
                endpoint=self.endpoint,
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                stage=stage,
                usage=usage_block,
                body=raw_body,
                error=empty_kind)
            if attempt < MAX_RETRIES - 1:
                logger.debug(
                    f'llm {empty_kind}, retry'
                    f' {attempt + 1}/{MAX_RETRIES - 1}')
                # An empty body signals a flaky endpoint, so the
                # rate-limit RETRY_BACKOFF does not apply: its sleep
                # is charged against the drain timeout, whose
                # maintenance pass needs a minimum budget.
                time.sleep(EMPTY_RETRY_DELAY)
                continue
            raise RuntimeError(
                f'llm returned {empty_kind} after'
                f' {MAX_RETRIES} attempts: {raw_body!r}')


_CLIENT: MemmanLLMClient | None = None


def get_llm_client() -> MemmanLLMClient:
    """Return the cached `MemmanLLMClient` built from the env file.

    Reads `MEMMAN_ENDPOINT`, `MEMMAN_API_KEY`, and `MEMMAN_LLM_MODEL`
    from the canonical env file. Raises `ConfigError` when a required
    value is missing. OpenRouter endpoints automatically receive
    memman's attribution headers and a provider-routing block: the
    shared privacy pin plus the LLM-only vendor pin from
    `MEMMAN_LLM_PROVIDER_ONLY`; other endpoints receive neither.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    endpoint = config.get(config.ENDPOINT)
    if not endpoint:
        raise ConfigError(
            f'{config.ENDPOINT} is not set;'
            ' run `memman install` to populate the env file')
    model = config.get(config.LLM_MODEL)
    if not model:
        raise ConfigError(
            f'{config.LLM_MODEL} is not set; run `memman install`'
            ' to persist the model id')
    api_key = config.get(config.API_KEY) or ''
    extra: dict[str, str] = {}
    routing = privacy_routing(endpoint)
    if config.is_openrouter_endpoint(endpoint):
        extra.update(OPENROUTER_ATTRIBUTION_HEADERS)
        # Notes:
        # - An empty allowlist sends no pin, so OpenRouter picks any
        #   provider serving the model.
        # - A pin that no provider satisfies fails the call outright:
        #   a refusal is recoverable, a silent route to an unapproved
        #   host is not.
        only = [
            name.strip()
            for name in (config.get(config.LLM_PROVIDER_ONLY) or '').split(',')
            if name.strip()]
        if only:
            routing['only'] = only
    _CLIENT = MemmanLLMClient(
        endpoint, api_key, model, max_tokens=WORKER_MAX_TOKENS,
        timeout=WORKER_TIMEOUT, extra_headers=extra or None,
        provider_routing=routing or None)
    return _CLIENT


def reset_client_cache() -> None:
    """Drop the cached client. Used by tests that swap env vars.
    """
    global _CLIENT
    _CLIENT = None
