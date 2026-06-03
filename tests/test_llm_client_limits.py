"""Per-role LLM client output-token and timeout budgets.

The recall hot path (`fast`) stays tight; the worker roles
(`slow_canonical`, `slow_metadata`) emit JSON that scales with input
size and must not truncate large insights, so they get a larger token
budget and a longer read timeout.
"""

from memman.llm.client import get_llm_client, reset_role_cache


def test_fast_role_keeps_tight_budget():
    """Recall hot-path role keeps the small token budget and short timeout.
    """
    reset_role_cache()
    client = get_llm_client('fast')
    assert client.max_tokens == 1024
    assert client.timeout == 10.0


def test_slow_canonical_role_gets_large_budget():
    """Canonical-rewrite role gets headroom so big inputs are not truncated.
    """
    reset_role_cache()
    client = get_llm_client('slow_canonical')
    assert client.max_tokens >= 4096
    assert client.timeout >= 60.0


def test_slow_metadata_role_gets_large_budget():
    """Enrichment role gets headroom so big inputs are not truncated.
    """
    reset_role_cache()
    client = get_llm_client('slow_metadata')
    assert client.max_tokens >= 4096
    assert client.timeout >= 60.0
