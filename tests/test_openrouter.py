"""Unit tests for the OpenRouter-facing parts of the LLM client.
"""

import pytest
from memman.exceptions import ConfigError
from memman.llm.client import MemmanLLMClient


def test_llm_client_requires_model():
    """Verify an empty model raises ConfigError.

    Mutation: accepting an empty model and letting the request go out with
        no model set.
    Oracle: pytest.raises(ConfigError, match='model is empty').
    """
    with pytest.raises(ConfigError, match='model is empty'):
        MemmanLLMClient(
            'https://openrouter.ai/api/v1',
            'sk-or-test',
            model='')


def test_llm_client_accepts_model():
    """Verify a client keeps the given model and endpoint.

    Mutation: rewriting or dropping the model or endpoint in __init__.
    Oracle: the literal model id and endpoint URL passed in.
    """
    client = MemmanLLMClient(
        'https://openrouter.ai/api/v1',
        'sk-or-test',
        model='anthropic/claude-haiku-4.5')
    assert client.model == 'anthropic/claude-haiku-4.5'
    assert client.endpoint == 'https://openrouter.ai/api/v1'
