"""Tests for memman.llm.extract -- fact extraction, reconciliation, query expansion.

These tests use FakeLLMClient to test parsing, validation, and error
handling that can't be exercised through the CLI. Normal-path behavior
with real/mocked LLM is covered by test_cli.py and test_memory_system.py.
"""

import json

from memman.llm.extract import _strip_line_refs, expand_query, extract_facts
from memman.llm.extract import reconcile_memories
from memman.llm.shared import parse_json_response


class TestStripLineRefs:
    """The post-extraction regex must strip file:line refs only.

    Hostnames, IP:port, timestamps, version pins, and HTTP codes share
    the surface shape `word:digits` but are NOT line references and
    must be preserved.
    """

    def test_strips_module_py_line(self):
        assert _strip_line_refs(
            'See cli.py:182 for the fix') == 'See for the fix'

    def test_strips_path_module_py_line(self):
        assert _strip_line_refs(
            'src/pkg/module/file.py:590 has the bug'
            ) == 'has the bug'

    def test_strips_explicit_line_n(self):
        assert _strip_line_refs(
            'Error on line 42 of the module'
            ) == 'Error on of the module'

    def test_preserves_localhost_port(self):
        assert _strip_line_refs(
            'Bind to localhost:8080'
            ) == 'Bind to localhost:8080'

    def test_preserves_ip_port(self):
        assert _strip_line_refs(
            'Reach 10.10.1.247:8000 over VPN'
            ) == 'Reach 10.10.1.247:8000 over VPN'

    def test_preserves_timestamp(self):
        assert _strip_line_refs(
            'Worker fires daily at 14:18'
            ) == 'Worker fires daily at 14:18'

    def test_preserves_http_code_word(self):
        assert _strip_line_refs(
            'Returns code:404 on miss'
            ) == 'Returns code:404 on miss'

    def test_preserves_version_pin(self):
        assert _strip_line_refs(
            'Use python:3.11 base image'
            ) == 'Use python:3.11 base image'

    def test_preserves_redis_port(self):
        assert _strip_line_refs(
            'redis:6379 is the cache'
            ) == 'redis:6379 is the cache'

    def test_strips_html_extension_line(self):
        assert _strip_line_refs(
            'See app/page.html:106 for the diff'
            ) == 'See for the diff'


class TestSlowRoleSplit:
    """The two slow roles resolve to independent env vars.

    Both roles must be set explicitly; there is no back-compat fallback
    from one to the other.
    """

    def test_canonical_and_metadata_resolve_independently(self, monkeypatch):
        from memman.config import LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA, OPENROUTER_API_KEY
        from memman.config import OPENROUTER_ENDPOINT
        from memman.llm.openrouter_client import get_openrouter_client
        monkeypatch.setenv(OPENROUTER_API_KEY, 'k')
        monkeypatch.setenv(OPENROUTER_ENDPOINT, 'https://x')
        monkeypatch.setenv(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        monkeypatch.setenv(LLM_MODEL_SLOW_METADATA, 'anthropic/haiku')
        from memman import config as _cfg
        _cfg.reset_file_cache()
        canonical = get_openrouter_client('slow_canonical')
        metadata = get_openrouter_client('slow_metadata')
        assert canonical.model == 'anthropic/sonnet'
        assert metadata.model == 'anthropic/haiku'

    def test_unset_metadata_var_raises(self, monkeypatch, tmp_path):
        import pytest
        from memman.config import DATA_DIR, LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA, OPENROUTER_API_KEY
        from memman.config import OPENROUTER_ENDPOINT
        from memman.exceptions import ConfigError
        from memman.llm.openrouter_client import get_openrouter_client
        monkeypatch.setenv(DATA_DIR, str(tmp_path))
        monkeypatch.setenv(OPENROUTER_API_KEY, 'k')
        monkeypatch.setenv(OPENROUTER_ENDPOINT, 'https://x')
        monkeypatch.setenv(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        monkeypatch.delenv(LLM_MODEL_SLOW_METADATA, raising=False)
        from memman import config as _cfg
        _cfg.reset_file_cache()
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


class TestParseJsonResponse:
    """JSON response parsing with code block handling."""

    def test_plain_json(self):
        """Plain JSON dict is parsed."""
        assert parse_json_response('{"key": "val"}') == {'key': 'val'}

    def test_code_block_json(self):
        """JSON in markdown code block is parsed."""
        raw = '```json\n{"key": "val"}\n```'
        assert parse_json_response(raw) == {'key': 'val'}

    def test_invalid_json_returns_none(self):
        """Non-JSON returns None."""
        assert parse_json_response('not json') is None

    def test_list_returns_none(self):
        """JSON list (not dict) returns None."""
        assert parse_json_response('[1, 2, 3]') is None
