"""LLM client output-token and timeout budgets.

Enrichment emits JSON that scales with input size and must not
truncate large insights, so the client gets a large token budget and
a long read timeout.
"""

import pytest
from memman.config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL
from memman.exceptions import ConfigError
from memman.llm.client import get_llm_client, reset_client_cache


def test_client_gets_large_budget():
    """Verify the client gets headroom for large enrichment output.

    Mutation: lowering the token budget or read timeout to a provider
        default, which truncates or times out on a large insight.
    Oracle: the 4096-token and 60-second floors set in the module doc.
    """
    reset_client_cache()
    client = get_llm_client()
    assert client.max_tokens >= 4096
    assert client.timeout >= 60.0


def test_unset_model_var_raises(env_file):
    """Verify an unset model fails loudly instead of falling back.

    Mutation: a fallback to a hardcoded default model when
        `MEMMAN_LLM_MODEL` is unset, which bills enrichment on a
        model the operator never chose.
    Oracle: `ConfigError` raised with the model var cleared.
    """
    env_file(LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(LLM_API_KEY, 'k')
    env_file(LLM_MODEL, None)
    reset_client_cache()
    with pytest.raises(ConfigError):
        get_llm_client()
