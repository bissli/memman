"""Tests for memman.llm.extract -- fact extraction and query expansion.

These tests use FakeLLMClient to test parsing, validation, and error
handling that can't be exercised through the CLI. Normal-path behavior
with real/mocked LLM is covered by test_cli.py and test_memory_system.py;
the three reconcile stages are covered by test_reconcile_stages.py.
"""

import json

import pytest
from memman.llm.extract import FACT_EXTRACTION_SYSTEM, QUERY_EXPANSION_SYSTEM
from memman.llm.extract import _strip_line_refs, expand_query, extract_facts
from memman.llm.shared import parse_json_response


@pytest.mark.parametrize(('text', 'expected'), [
    ('See cli.py:182 for the fix', 'See cli.py for the fix'),
    ('src/pkg/module/file.py:590 has the bug', 'src/pkg/module/file.py has the bug'),
    ('Error on line 42 of the module', 'Error on of the module'),
    ('See app/page.html:106 for the diff', 'See app/page.html for the diff'),
    ('Edit config.yaml:12 then restart', 'Edit config.yaml then restart'),
    ('Bind to localhost:8080', 'Bind to localhost:8080'),
    ('The pool connects to db.example.com:5432 over TLS',
     'The pool connects to db.example.com:5432 over TLS'),
    ('See https://api.example.com:8443/v1 for the API',
     'See https://api.example.com:8443/v1 for the API'),
    ('Reach 192.0.2.1:8000 over VPN', 'Reach 192.0.2.1:8000 over VPN'),
    ('Worker fires daily at 14:18', 'Worker fires daily at 14:18'),
    ('Returns code:404 on miss', 'Returns code:404 on miss'),
    ('Use python:3.11 base image', 'Use python:3.11 base image'),
    ('redis:6379 is the cache', 'redis:6379 is the cache'),
])
def test_strip_line_refs(text, expected):
    """Drop the line number, keep the path; preserve hosts, ports, clocks, tags.

    Mutation: deleting the filename together with its line number (the
    pre-0.34.0 rule); matching any dot-letter run before the colon, which
    reads `db.example.com:5432` as a file and strips its port; or
    broadening the pattern until it also eats a `host:port`, a `HH:MM`
    clock, an image tag, or a `code:404`.
    Oracle: hand-paired input/output rows: the stripping rows keep the
    path, the preserving rows come back unchanged.
    """
    assert _strip_line_refs(text) == expected


class TestSlowRoleSplit:
    """The two slow roles resolve to independent env vars.

    Both roles must be set explicitly; there is no back-compat fallback
    from one to the other.
    """

    def test_canonical_and_metadata_resolve_independently(self, env_file):
        from memman.config import LLM_API_KEY, LLM_ENDPOINT
        from memman.config import LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA
        from memman.llm.client import get_llm_client, reset_role_cache
        env_file(LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
        env_file(LLM_API_KEY, 'k')
        env_file(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        env_file(LLM_MODEL_SLOW_METADATA, 'anthropic/haiku')
        reset_role_cache()
        canonical = get_llm_client('slow_canonical')
        metadata = get_llm_client('slow_metadata')
        assert canonical.model == 'anthropic/sonnet'
        assert metadata.model == 'anthropic/haiku'

    def test_unset_metadata_var_raises(self, env_file):
        import pytest
        from memman.config import LLM_API_KEY, LLM_ENDPOINT
        from memman.config import LLM_MODEL_SLOW_CANONICAL
        from memman.config import LLM_MODEL_SLOW_METADATA
        from memman.exceptions import ConfigError
        from memman.llm.client import get_llm_client, reset_role_cache
        env_file(LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
        env_file(LLM_API_KEY, 'k')
        env_file(LLM_MODEL_SLOW_CANONICAL, 'anthropic/sonnet')
        env_file(LLM_MODEL_SLOW_METADATA, None)
        reset_role_cache()
        with pytest.raises(ConfigError):
            get_llm_client('slow_metadata')


class FakeLLMClient:
    """LLMClient that returns canned responses for unit testing."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Record call and return canned response."""
        self.calls.append((system, user, kwargs))
        return self.response


class FailingLLMClient:
    """LLMClient whose every call raises, for passthrough tests."""

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Raise as a wedged transport would."""
        raise ConnectionError('timeout')


class TestExtractFacts:
    """Fact extraction parsing and error handling."""

    def test_extraction_drops_overlong_entities(self):
        """Fact-extraction entities get the same length guardrail.

        `extract_facts` is the other producer of LLM entities; an
        uncapped blob here lands in entity edges and the embedding
        exactly like an enrichment one.

        Mutation: leaving the entity parse in `extract_facts`
            uncapped (or truncating instead of dropping).
        Oracle: the 250-char entity is absent from the parsed fact,
            with no prefix remnant; the valid entity survives.
        """
        response = json.dumps({
            'facts': [{
                'text': 'Uses Qdrant for vector search',
                'category': 'decision',
                'importance': 4,
                'entities': ['Qdrant', 'z' * 250],
                }],
            'skip_reason': None,
            })
        client = FakeLLMClient(response)
        facts = extract_facts(client, 'chose Qdrant')
        assert facts[0]['entities'] == ['Qdrant']

    def test_extraction_prompt_carries_no_importance(self):
        """Verify the extraction prompt no longer asks for an importance.

        Mutation: the importance ladder, or the `"importance"` key in the
            output schema or an example, returning to the prompt.
        Oracle: the shipped string.
        """
        assert 'importance' not in FACT_EXTRACTION_SYSTEM

    def test_extracted_fact_carries_no_importance_key(self):
        """Verify a model-emitted importance is dropped from the parsed fact.

        Mutation: reading the model's `importance` value into the fact
            (today's clamp branch), so the caller's `--imp` loses to it.
        Oracle: key absence on the parsed fact.
        """
        response = json.dumps({
            'facts': [{'text': 'Uses PostgreSQL for JSONB support',
                       'category': 'decision', 'importance': 5,
                       'entities': ['PostgreSQL']}],
            'skip_reason': None,
            })
        facts = extract_facts(FakeLLMClient(response), 'I chose Postgres')
        assert 'importance' not in facts[0]

    def test_extract_facts_start_traces_the_content(self, monkeypatch):
        """Verify the `extract_facts_start` trace event carries the content.

        Mutation: dropping the `content` kwarg, which leaves the
            extraction-error rate unmeasurable under trace.
        Oracle: the recorded kwargs of the event.
        """
        from memman import trace
        events = []
        monkeypatch.setattr(trace, 'event',
                            lambda name, **kw: events.append((name, kw)))
        response = json.dumps({'facts': [], 'skip_reason': 'greeting'})
        extract_facts(FakeLLMClient(response), 'hi there')
        start = [kw for name, kw in events if name == 'extract_facts_start']
        assert start[0]['content'] == 'hi there'

    def test_extract_facts_traces_a_response_without_facts(self, monkeypatch):
        """Verify a response that decodes to no `facts` still leaves a trace.

        A response cut at the token ceiling decodes to a complete inner
        object with no `facts` key under the last-object reader.

        Mutation: the bare passthrough return on an empty or missing
            `facts` list, which emits no `extract_facts_result` event and
            leaves the extraction-error rate unmeasurable.
        Oracle: the recorded event, carrying the outcome and the raw body.
        """
        from memman import trace
        events = []
        monkeypatch.setattr(trace, 'event',
                            lambda name, **kw: events.append((name, kw)))
        raw = ('{"facts": [{"text": "Redis caches sessions", "category": "fact",'
               ' "entities": []}, {"text": "cut off')
        facts = extract_facts(FakeLLMClient(raw), 'Redis caches sessions')
        results = [kw for name, kw in events if name == 'extract_facts_result']
        assert facts[0]['text'] == 'Redis caches sessions'
        assert results[0]['outcome'] == 'no_facts'
        assert results[0]['raw'] == raw

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
            def complete(self, system: str, user: str, **kwargs) -> str:
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


class TestExpandQuery:
    """Query expansion parsing and error handling."""

    def test_basic_expansion(self):
        """The model's expansion reaches the caller, not the raw query.

        Mutation: discarding the parsed expansion and returning the
            original query, or normalizing a valid intent to None.
        Oracle: the exact expansion string in the canned response. A
            substring check would pass on the passthrough value.
        """
        response = json.dumps({
            'expanded_query': 'Redis cache configuration settings',
            'keywords': ['Redis', 'cache', 'config'],
            'intent': 'GENERAL',
            })
        client = FakeLLMClient(response)
        result = expand_query(client, 'Redis config')
        assert result['expanded_query'] == 'Redis cache configuration settings'
        assert result['intent'] == 'GENERAL'

    def test_llm_failure_returns_passthrough(self):
        """LLM failure returns original query unchanged."""
        result = expand_query(FailingLLMClient(), 'my query')
        assert result['expanded_query'] == 'my query'
        assert result['intent'] is None

    def test_result_carries_only_the_two_read_keys(self):
        """All four return paths yield exactly expanded_query and intent.

        Mutation: leaving 'keywords' on one of the four return paths,
            or letting an unrequested model key reach the caller.
        Oracle: the key set the sole production caller in `memman.cli`
            reads - `expanded_query` once and `intent` twice.
        """
        parsed_ok = json.dumps({
            'expanded_query': 'alpha beta',
            'keywords': ['alpha'],
            'intent': 'GENERAL',
            })
        cases = {
            'parsed': (FakeLLMClient(parsed_ok), 'alpha'),
            'unparseable': (FakeLLMClient('not json at all'), 'beta'),
            'llm_error': (FailingLLMClient(), 'gamma'),
            }
        for path, (client, query) in cases.items():
            result = expand_query(client, query)
            assert set(result) == {'expanded_query', 'intent'}, path

        cached_client = FakeLLMClient(parsed_ok)
        expand_query(cached_client, 'delta')
        cached = expand_query(cached_client, 'delta')
        assert len(cached_client.calls) == 1, 'second call must be cached'
        assert set(cached) == {'expanded_query', 'intent'}

    def test_prompt_requests_exactly_the_keys_the_parser_keeps(self):
        """The prompt's JSON skeleton names the surviving result keys.

        Mutation: asking the model for a key the parser discards, which
            buys nothing but tokens, or dropping a key the parser reads,
            which kills intent routing with no error.
        Oracle: the skeleton parsed out of QUERY_EXPANSION_SYSTEM against
            a live expand_query result - neither side hand-copied.
        """
        skeleton = QUERY_EXPANSION_SYSTEM[
            QUERY_EXPANSION_SYSTEM.index('{'):
            QUERY_EXPANSION_SYSTEM.rindex('}') + 1]
        requested = set(json.loads(skeleton))
        response = json.dumps({'expanded_query': 'x', 'intent': 'WHY'})
        returned = set(expand_query(FakeLLMClient(response), 'q'))
        assert requested == returned

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


@pytest.mark.parametrize(('raw', 'expected'), [
    ('{"key": "val"}', {'key': 'val'}),
    ('```json\n{"key": "val"}\n```', {'key': 'val'}),
    ('not json', None),
    ('[1, 2, 3]', None),
    ('[' * 3000, None),
    ('{"actions": ' + '[' * 1200, None),
])
def test_parse_json_response(raw, expected):
    """JSON response parsing strips fences, rejects non-dicts and runaway nesting.

    Mutation: catching `ValueError` alone, so a body nested past the
        recursion limit raises out of every stage reader and fails the
        write instead of reading as no object.
    Oracle: hand-paired rows; the two nested rows read as None.
    """
    assert parse_json_response(raw) == expected
