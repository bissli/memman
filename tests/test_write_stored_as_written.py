"""The drain stores a write's text as the agent wrote it.

No model reads a write before it is stored: nothing judges it
non-durable, rewords it, or picks its category. The stubs below answer
any extraction-shaped call (a reply carrying a `facts` key) with a
skip or a rewrite, so a write path that still asks a model shows the
model's hand in the stored row.
"""

import json

from tests.conftest import _mock_llm_complete, invoke, parse_remember


def _answer_extraction_with(reply):
    """Return a mock `complete` that swaps any extraction reply for `reply`.
    """
    def complete(self, system, user, **kwargs):
        answer = _mock_llm_complete(self, system, user, **kwargs)
        if 'facts' in json.loads(answer):
            return json.dumps(reply)
        return answer
    return complete


def test_drain_stores_a_write_a_model_would_skip(mm_runner, monkeypatch):
    """Verify a status-shaped write lands instead of being dropped.

    Mutation: a model call restored on the write path whose empty
        reply returns the `trivial content` skip.
    Oracle: the stored row's content against the input string, with
        every extraction-shaped reply forced to a skip.
    """
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        _answer_extraction_with({'facts': [], 'skip_reason': 'status'}))
    text = 'All drives verified after the maintenance window'

    stored = parse_remember(invoke(mm_runner, ['remember', text]), mm_runner)

    assert stored.get('content') == text


def test_drain_stores_the_agents_words_and_category(mm_runner, monkeypatch):
    """Verify the stored row keeps the agent's text, category and entities.

    Mutation: the model's rewrite stored in place of the input, its
        category kept over the `--cat` default, or its entities merged
        into the row.
    Oracle: the input string byte for byte, the CLI's `fact` default,
        and an entity the input never names.
    """
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        _answer_extraction_with({
            'facts': [{
                'text': 'Stored rows in goog, and demo-v3 carry bissli.',
                'category': 'decision',
                'entities': ['Bogus'],
                }],
            'skip_reason': None,
            }))
    text = 'Stored rows in goog and demo-v3 carry bissli.'

    stored = parse_remember(invoke(mm_runner, ['remember', text]), mm_runner)

    assert stored.get('content') == text
    assert stored.get('category') == 'fact'
    assert 'Bogus' not in stored.get('entities', [])
