"""Per-role LLM client output-token and timeout budgets.

The `slow` worker role emits JSON that scales with input size and
must not truncate large insights, so it gets a large token budget and
a long read timeout.
"""

import pytest
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient, get_llm_client, reset_role_cache


def test_slow_role_gets_large_budget():
    """Enrichment role gets headroom so big inputs are not truncated.
    """
    reset_role_cache()
    client = get_llm_client('slow')
    assert client.max_tokens >= 4096
    assert client.timeout >= 60.0


def test_unset_slow_model_var_raises(env_file):
    """Verify an unset slow model fails loudly instead of falling back.

    Mutation: a fallback to a hardcoded default model when
        `MEMMAN_LLM_MODEL` is unset, which bills enrichment on a
        model the operator never chose.
    Oracle: `ConfigError` raised with the slow var cleared.
    """
    from memman.config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL
    from memman.exceptions import ConfigError
    env_file(LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(LLM_API_KEY, 'k')
    env_file(LLM_MODEL, None)
    reset_role_cache()
    with pytest.raises(ConfigError):
        get_llm_client('slow')


class _RecordingSession:
    """An HTTP session that records every request body and answers 200."""

    def __init__(self):
        self.bodies = []

    def post(self, url, headers, json, timeout):
        self.bodies.append(json)
        return _OkResponse()


class _OkResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {'choices': [{'message': {'content': '{}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}


@pytest.mark.no_mock_llm
def test_complete_honors_a_per_call_max_tokens(monkeypatch):
    """Verify `max_tokens` on `complete` overrides the role ceiling for that call.

    Mutation: ignoring the keyword and always sending the role ceiling,
        so a caller's per-call budget from `shared.py` never reaches
        the request.
    Oracle: the recorded request bodies: the role ceiling without the
        keyword, the override with it.
    """
    session = _RecordingSession()
    monkeypatch.setattr('memman.llm.client.get_session', lambda name: session)
    client = MemmanLLMClient('https://llm.example', 'key', 'model', max_tokens=4096)

    client.complete('s', 'u', stage=llm_usage.STAGE_ENRICHMENT)
    client.complete('s', 'u', stage=llm_usage.STAGE_ENRICHMENT, max_tokens=8192)

    assert [b['max_tokens'] for b in session.bodies] == [4096, 8192]
