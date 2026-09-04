"""Tests for memman.llm.extract -- fact extraction, reconciliation, query expansion.

These tests use FakeLLMClient to test parsing, validation, and error
handling that can't be exercised through the CLI. Normal-path behavior
with real/mocked LLM is covered by test_cli.py and test_memory_system.py.
"""

import json

import pytest
from memman.llm.extract import QUERY_EXPANSION_SYSTEM, RECONCILIATION_SYSTEM
from memman.llm.extract import _strip_line_refs, expand_query, extract_facts
from memman.llm.extract import reconcile_memories
from memman.llm.shared import parse_json_response


@pytest.mark.parametrize(('text', 'expected'), [
    ('See cli.py:182 for the fix', 'See for the fix'),
    ('src/pkg/module/file.py:590 has the bug', 'has the bug'),
    ('Error on line 42 of the module', 'Error on of the module'),
    ('See app/page.html:106 for the diff', 'See for the diff'),
    ('Bind to localhost:8080', 'Bind to localhost:8080'),
    ('Reach 192.0.2.1:8000 over VPN', 'Reach 192.0.2.1:8000 over VPN'),
    ('Worker fires daily at 14:18', 'Worker fires daily at 14:18'),
    ('Returns code:404 on miss', 'Returns code:404 on miss'),
    ('Use python:3.11 base image', 'Use python:3.11 base image'),
    ('redis:6379 is the cache', 'redis:6379 is the cache'),
])
def test_strip_line_refs(text, expected):
    """Strip file:line refs but preserve hostnames, ports, timestamps, versions.

    Mutation: broadening the `file:line` pattern until it also eats a
    `host:port`, a `HH:MM` clock, an image tag, or a `code:404` - the
    colon-digit shape is identical, so a pattern written without a
    filename-extension guard strips all of them.
    Oracle: hand-paired input/output rows, half of which must come
    back unchanged.
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
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Record call and return canned response."""
        self.calls.append((system, user))
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
            def complete(self, system: str, user: str, **kwargs) -> str:
                raise ConnectionError('timeout')
        facts = [{'text': 'a'}, {'text': 'b'}]
        result = reconcile_memories(
            FailingClient(), facts, [('id-1', 'mem 1')])
        assert all(r['action'] == 'ADD' for r in result)

    def test_none_action_carries_the_memory_it_names(self):
        """NONE keeps the id of the memory that already captured the
        fact.

        The id is the whole payload of a NONE verdict: it is what the
        write path bumps instead of storing the restatement. Mapping
        it only for the actions that mutate a row leaves the caller
        unable to tell WHICH memory captured the fact.

        Mutation: nulling `target_id` for any action other than
            UPDATE or DELETE, on the reasoning that only those two
            address a row.
        Oracle: the real uuid behind the numeric id in the canned
            response, which the id map alone can supply.
        """
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
        assert result[0]['target_id'] == 'uuid-1'

    def test_reconcile_prompt_requires_an_id_on_none(self):
        """The prompt asks NONE to name the capturing memory.

        Every other test stubs the LLM, so nothing else can see the
        prompt drop the demand - and a NONE with no id makes the
        corroboration path inert however correct the code is.

        Mutation: restoring the bare 'NONE: fact already captured
            adequately' wording, which names no memory and lets the
            model answer with a null target.
        Oracle: the NONE line of the shipped system prompt, required
            to carry the id placeholder that ADD must not.
        """
        none_line = next(
            ln for ln in RECONCILIATION_SYSTEM.split(chr(10))
            if ln.startswith('- NONE'))
        assert '<id>' in none_line

    @pytest.mark.parametrize('role', ['slow_canonical', 'fast'])
    def test_live_model_names_the_memory_on_none(self, role, request):
        """The configured models answer NONE with the id, not null.

        The prompt line is the load-bearing half of corroboration:
        the write path can only bump a row the model named. So this
        is a claim about the MODEL, and only a live call can hold it.
        `slow_canonical` is the role that reconciles; `fast` is
        carried as the quality floor.

        Mutation: dropping the id demand from the NONE line of
            RECONCILIATION_SYSTEM -- over 16 cases x 2 models x 2
            repeats the id-less wording returned a null target on 39
            of 48 reworded restatements against 4 of 48 for the
            shipped wording, so corroboration goes back to counting
            byte-identical writes alone.
        Oracle: the id of the one shortlist row that restates the
            fact, against a second row that does not.
        """
        if not request.config.getoption('--live'):
            pytest.skip('needs --live: asserts real model compliance')
        from memman.llm.client import get_llm_client
        client = get_llm_client(role)
        existing = [
            ('uuid-unrelated', 'Alice uses vim keybindings.'),
            ('uuid-restated',
             'Redis caches session tokens for the web tier.'),
            ]
        result = reconcile_memories(
            client, [{'text': 'Session tokens are cached in Redis.',
                      'entities': []}], existing)
        assert result[0]['action'] == 'NONE'
        assert result[0]['target_id'] == 'uuid-restated'


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
])
def test_parse_json_response(raw, expected):
    """JSON response parsing strips code-block fences, rejects non-dicts."""
    assert parse_json_response(raw) == expected


def test_reconcile_prompt_offers_supersede_and_never_delete():
    """Verify the shipped prompt names SUPERSEDE and no longer names DELETE.

    Mutation: SUPERSEDE in the parser but not the prompt (the model
        never emits it, so the branch is dead), or DELETE left in the
        prompt (the model discards the contradicting fact, the 5-for-5
        fleet outcome the action is removed for).
    Oracle: the `RECONCILIATION_SYSTEM` text itself: a SUPERSEDE action
        line carrying the id placeholder, the action enum, and no
        DELETE anywhere.
    """
    lines = RECONCILIATION_SYSTEM.split(chr(10))
    assert any(ln.startswith('- SUPERSEDE <id>') for ln in lines)
    assert 'ADD|UPDATE|SUPERSEDE|NONE' in RECONCILIATION_SYSTEM
    assert 'DELETE' not in RECONCILIATION_SYSTEM


def test_supersede_action_maps_target_and_merged_text():
    """Verify SUPERSEDE resolves its numeric id and keeps the merged text.

    Mutation: leaving SUPERSEDE out of the accepted set, so the
        unknown-action fallback turns every contradiction into an ADD
        beside the row it contradicts.
    Oracle: the real uuid behind the numeric id and the canned merged
        text, read back off the parsed action.
    """
    response = json.dumps({
        'actions': [{
            'fact': 'the broker is redis now',
            'action': 'SUPERSEDE',
            'target_id': '0',
            'merged_text': 'the broker is redis now (was kombu)',
            }],
        })
    existing = [('uuid-1', 'the broker is kombu')]
    result = reconcile_memories(
        FakeLLMClient(response), [{'text': 'the broker is redis now'}],
        existing)
    assert result[0]['action'] == 'SUPERSEDE'
    assert result[0]['target_id'] == 'uuid-1'
    assert result[0]['merged_text'] == 'the broker is redis now (was kombu)'


def test_a_delete_emit_is_read_as_supersede():
    """Verify a stray DELETE from the model lands as SUPERSEDE.

    Mutation: dropping the alias, so DELETE falls through the
        unknown-action fallback to ADD and the contradicted peer
        stays live beside the new fact.
    Oracle: the parsed action for a canned DELETE response: SUPERSEDE,
        the mapped uuid, and no merged text.
    """
    response = json.dumps({
        'actions': [{
            'fact': 'postgres was abandoned',
            'action': 'DELETE',
            'target_id': '0',
            'merged_text': None,
            }],
        })
    existing = [('uuid-1', 'we run on postgres')]
    result = reconcile_memories(
        FakeLLMClient(response), [{'text': 'postgres was abandoned'}],
        existing)
    assert result[0] == {
        'fact': 'postgres was abandoned', 'action': 'SUPERSEDE',
        'target_id': 'uuid-1', 'merged_text': None}


@pytest.mark.parametrize('action', ['UPDATE', 'SUPERSEDE', 'NONE'])
@pytest.mark.parametrize('target', [None, '99'])
def test_a_targetless_update_or_supersede_is_read_as_add(action, target):
    """Verify a row-addressing action with no resolvable target becomes ADD.

    Mutation: keeping the action when `target_id` is null, so the plan
        inserts plainly and reports a link that never happened (the
        shipped shape for UPDATE).
    Oracle: parsed action 'ADD' with a null target, for both a null and
        an unmapped numeric id.
    """
    response = json.dumps({
        'actions': [{
            'fact': 'x', 'action': action, 'target_id': target,
            'merged_text': 'merged'}],
        })
    result = reconcile_memories(
        FakeLLMClient(response), [{'text': 'x'}], [('uuid-1', 'mem 1')])
    assert result[0] == {'fact': 'x', 'action': 'ADD', 'target_id': None,
                         'merged_text': None}


@pytest.mark.parametrize('action', ['REPLACE', 'ADD', 'bogus'])
def test_a_normalized_add_drops_the_target_and_merged_text(action):
    """Verify an ADD never carries a target or merged text.

    Mutation: normalizing the action to ADD while keeping the mapped
        `target_id` (the plan then reports `replaced_id` for a row that
        stays current) or the model's `merged_text` (the new row stores
        clauses of a memory the parser never resolved).
    Oracle: the parsed action for an unknown action, and for an ADD
        the model decorated with a target, against a mapped id.
    """
    response = json.dumps({
        'actions': [{
            'fact': 'x', 'action': action, 'target_id': '0',
            'merged_text': 'merged with mem 1'}],
        })
    result = reconcile_memories(
        FakeLLMClient(response), [{'text': 'x'}], [('uuid-1', 'mem 1')])
    assert result[0] == {'fact': 'x', 'action': 'ADD', 'target_id': None,
                         'merged_text': None}
