"""Tests for memman.llm.extract -- fact extraction, reconciliation, query expansion.

These tests use FakeLLMClient to test parsing, validation, and error
handling that can't be exercised through the CLI. Normal-path behavior
with real/mocked LLM is covered by test_cli.py and test_memory_system.py.
"""

import json

import pytest
from memman.llm.extract import _strip_line_refs, expand_query, extract_facts
from memman.llm.extract import reconcile_memories
from memman.llm.shared import parse_json_response


@pytest.mark.parametrize('text,expected', [
    ('See cli.py:182 for the fix', 'See for the fix'),
    ('src/pkg/module/file.py:590 has the bug', 'has the bug'),
    ('Error on line 42 of the module', 'Error on of the module'),
    ('See app/page.html:106 for the diff', 'See for the diff'),
    ('Bind to localhost:8080', 'Bind to localhost:8080'),
    ('Reach 10.10.1.247:8000 over VPN', 'Reach 10.10.1.247:8000 over VPN'),
    ('Worker fires daily at 14:18', 'Worker fires daily at 14:18'),
    ('Returns code:404 on miss', 'Returns code:404 on miss'),
    ('Use python:3.11 base image', 'Use python:3.11 base image'),
    ('redis:6379 is the cache', 'redis:6379 is the cache'),
])
def test_strip_line_refs(text, expected):
    """Strip file:line refs but preserve hostnames, ports, timestamps, versions."""
    assert _strip_line_refs(text) == expected


class TestSlowRoleSplit:
    """The two slow roles resolve to independent env vars.

    Both roles must be set explicitly; there is no back-compat fallback
    from one to the other.
    """

    def test_canonical_and_metadata_resolve_independently(self, env_file):
        from memman.config import LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA, OPENROUTER_API_KEY
        from memman.config import OPENROUTER_ENDPOINT
        from memman.llm.openrouter_client import get_openrouter_client
        env_file(OPENROUTER_API_KEY, 'k')
        env_file(OPENROUTER_ENDPOINT, 'https://x')
        env_file(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        env_file(LLM_MODEL_SLOW_METADATA, 'anthropic/haiku')
        canonical = get_openrouter_client('slow_canonical')
        metadata = get_openrouter_client('slow_metadata')
        assert canonical.model == 'anthropic/sonnet'
        assert metadata.model == 'anthropic/haiku'

    def test_unset_metadata_var_raises(self, env_file):
        import pytest
        from memman.config import LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA, OPENROUTER_API_KEY
        from memman.config import OPENROUTER_ENDPOINT
        from memman.exceptions import ConfigError
        from memman.llm.openrouter_client import get_openrouter_client
        env_file(OPENROUTER_API_KEY, 'k')
        env_file(OPENROUTER_ENDPOINT, 'https://x')
        env_file(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        env_file(LLM_MODEL_SLOW_METADATA, None)
        with pytest.raises(ConfigError):
            get_openrouter_client('slow_metadata')


class FakeLLMClient:
    """LLMClient that returns canned responses for unit testing."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        """Record call and return canned response."""
        self.calls.append((system, user))
        return self.response


class TestExtractFacts:
    """Fact extraction parsing and error handling."""

    def test_single_fact_extracted(self):
        """Single fact returned for simple content."""
        response = json.dumps({
            'facts': [{
                'text': 'Uses PostgreSQL for JSONB support',
                'category': 'decision',
                'importance': 4,
                'entities': ['PostgreSQL', 'JSONB'],
                }],
            'skip_reason': None,
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'I chose Postgres for JSONB')
        assert len(facts) == 1
        assert facts[0]['text'] == 'Uses PostgreSQL for JSONB support'
        assert facts[0]['category'] == 'decision'
        assert facts[0]['importance'] == 4
        assert 'PostgreSQL' in facts[0]['entities']

    def test_multi_fact_extraction(self):
        """Multiple facts returned for complex content."""
        response = json.dumps({
            'facts': [
                {'text': 'Migrated from Flask to FastAPI',
                 'category': 'decision', 'importance': 4,
                 'entities': ['Flask', 'FastAPI']},
                {'text': 'FastAPI is faster for async',
                 'category': 'fact', 'importance': 3,
                 'entities': ['FastAPI']},
                ],
            'skip_reason': None,
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'Switched to FastAPI from Flask')
        assert len(facts) == 2
        assert facts[0]['category'] == 'decision'
        assert facts[1]['category'] == 'fact'

    def test_trivial_content_skipped(self):
        """Trivial content returns empty list (skip)."""
        response = json.dumps({
            'facts': [],
            'skip_reason': 'greeting',
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'Hi there')
        assert facts == []

    def test_llm_failure_returns_passthrough(self):
        """Network/timeout error returns content as passthrough fact."""
        class FailingClient:
            def complete(self, system: str, user: str) -> str:
                raise ConnectionError('timeout')
        facts = extract_facts(FailingClient(), 'important fact')
        assert len(facts) == 1
        assert facts[0]['text'] == 'important fact'

    def test_bad_json_returns_passthrough(self):
        """Malformed JSON returns content as passthrough fact."""
        client = FakeLLMClient('not valid json at all')
        facts = extract_facts(client, 'some content')
        assert len(facts) == 1
        assert facts[0]['text'] == 'some content'

    def test_code_block_json_parsed(self):
        """JSON wrapped in markdown code blocks is parsed."""
        inner = json.dumps({
            'facts': [{'text': 'Redis uses LRU', 'category': 'fact',
                       'importance': 3, 'entities': ['Redis']}],
            'skip_reason': None,
            })
        response = f'```json\n{inner}\n```'
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'Redis LRU')
        assert len(facts) == 1
        assert facts[0]['text'] == 'Redis uses LRU'

    def test_importance_clamped_to_range(self):
        """Importance values outside 2-5 are clamped."""
        response = json.dumps({
            'facts': [
                {'text': 'low', 'category': 'fact',
                 'importance': 0, 'entities': []},
                {'text': 'high', 'category': 'fact',
                 'importance': 10, 'entities': []},
                ],
            'skip_reason': None,
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'test')
        assert facts[0]['importance'] == 2
        assert facts[1]['importance'] == 5

    def test_invalid_category_defaults_to_fact(self):
        """Unknown category maps to 'fact'."""
        response = json.dumps({
            'facts': [{'text': 'test', 'category': 'bogus',
                       'importance': 3, 'entities': []}],
            'skip_reason': None,
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'test')
        assert facts[0]['category'] == 'fact'


class TestReconcileMemories:
    """Memory reconciliation parsing and error handling."""

    def test_empty_existing_returns_all_add(self):
        """No existing memories means all facts are ADD."""
        facts = [{'text': 'new fact', 'entities': []}]
        result = reconcile_memories(
            FakeLLMClient('unused'), facts, [])
        assert len(result) == 1
        assert result[0]['action'] == 'ADD'

    def test_update_action_maps_target_id(self):
        """UPDATE with numeric ID maps to real UUID."""
        response = json.dumps({
            'actions': [{
                'fact': 'updated info',
                'action': 'UPDATE',
                'target_id': '0',
                'merged_text': 'merged content',
                'reason': 'supersedes',
                }],
            })
        existing = [('real-uuid-123', 'old info')]
        client = FakeLLMClient(response)
        result = reconcile_memories(
            client, [{'text': 'updated info'}], existing)
        assert result[0]['action'] == 'UPDATE'
        assert result[0]['target_id'] == 'real-uuid-123'
        assert result[0]['merged_text'] == 'merged content'

    def test_invalid_target_id_falls_back_to_add(self):
        """Hallucinated numeric ID causes fallback to ADD."""
        response = json.dumps({
            'actions': [{
                'fact': 'test',
                'action': 'UPDATE',
                'target_id': '99',
                'merged_text': 'merged',
                }],
            })
        existing = [('uuid-1', 'memory 1')]
        client = FakeLLMClient(response)
        result = reconcile_memories(
            client, [{'text': 'test'}], existing)
        assert result[0]['action'] == 'ADD'

    def test_llm_failure_defaults_to_add(self):
        """LLM failure returns all-ADD."""
        class FailingClient:
            def complete(self, system: str, user: str) -> str:
                raise ConnectionError('timeout')
        facts = [{'text': 'a'}, {'text': 'b'}]
        result = reconcile_memories(
            FailingClient(), facts, [('id-1', 'mem 1')])
        assert all(r['action'] == 'ADD' for r in result)

    def test_none_action_preserved(self):
        """NONE action (already captured) is returned."""
        response = json.dumps({
            'actions': [{
                'fact': 'same info',
                'action': 'NONE',
                'target_id': '0',
                'merged_text': None,
                }],
            })
        existing = [('uuid-1', 'same info')]
        client = FakeLLMClient(response)
        result = reconcile_memories(
            client, [{'text': 'same info'}], existing)
        assert result[0]['action'] == 'NONE'


class TestExpandQuery:
    """Query expansion parsing and error handling."""

    def test_basic_expansion(self):
        """Query is expanded with keywords and entities."""
        response = json.dumps({
            'expanded_query': 'Redis cache configuration settings',
            'keywords': ['Redis', 'cache', 'config'],
            'entities': ['Redis'],
            'intent': 'GENERAL',
            })
        client = FakeLLMClient(response)
        result = expand_query(client, 'Redis config')
        assert 'Redis' in result['expanded_query']
        assert result['intent'] == 'GENERAL'

    def test_llm_failure_returns_passthrough(self):
        """LLM failure returns original query unchanged."""
        class FailingClient:
            def complete(self, system: str, user: str) -> str:
                raise ConnectionError('timeout')
        result = expand_query(FailingClient(), 'my query')
        assert result['expanded_query'] == 'my query'
        assert result['intent'] is None

    def test_invalid_intent_normalized(self):
        """Unknown intent is set to None."""
        response = json.dumps({
            'expanded_query': 'test',
            'keywords': [],
            'entities': [],
            'intent': 'BOGUS',
            })
        client = FakeLLMClient(response)
        result = expand_query(client, 'test')
        assert result['intent'] is None


@pytest.mark.parametrize('raw,expected', [
    ('{"key": "val"}', {'key': 'val'}),
    ('```json\n{"key": "val"}\n```', {'key': 'val'}),
    ('not json', None),
    ('[1, 2, 3]', None),
])
def test_parse_json_response(raw, expected):
    """JSON response parsing strips code-block fences, rejects non-dicts."""
    assert parse_json_response(raw) == expected
