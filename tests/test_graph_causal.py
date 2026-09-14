"""Causal signal detection and token overlap tests."""

import json
import re
from datetime import datetime, timezone

from memman.graph.causal import infer_llm_causal_edges, suggest_sub_type
from memman.graph.causal import token_overlap
from memman.store.model import Insight


def test_suggest_sub_type_prevents_overrides():
    """When 'because' and 'prevents' both present, prevents wins.

    Pins the priority order in `suggest_sub_type` so a refactor that
    reorders the if/elif/else cannot silently change behavior.
    """
    result = suggest_sub_type(
        'blocked the request because it prevents abuse')
    assert result == 'prevents'


def test_token_overlap_basic():
    """Two sets with partial overlap produce intersection / max ratio."""
    a = {'go', 'sqlite', 'database'}
    b = {'go', 'sqlite', 'web', 'server'}
    overlap = token_overlap(a, b)
    expected = 2 / 4
    assert abs(overlap - expected) < 0.001


def test_token_overlap_no_overlap():
    """Completely disjoint sets return zero."""
    a = {'alpha', 'beta'}
    b = {'gamma', 'delta'}
    assert token_overlap(a, b) == 0.0


def test_token_overlap_empty():
    """One or both sets empty returns zero."""
    assert token_overlap(set(), {'a'}) == 0.0
    assert token_overlap({'a'}, set()) == 0.0
    assert token_overlap(set(), set()) == 0.0


def test_token_overlap_identical():
    """Same set against itself returns 1.0."""
    s = {'go', 'sqlite', 'graph'}
    assert token_overlap(s, s) == 1.0


def test_recent_insights_in_llm_prompt(backend):
    """`_build_llm_prompt` renders a RECENT INSIGHTS section.

    Mutation: dropping the `'RECENT INSIGHTS'` literal section header,
        or passing an empty list where the gate-passing recent rows
        belong.
    Oracle: the section header read back out of the captured prompt.
    """
    from datetime import datetime, timezone

    from memman.graph.causal import infer_llm_causal_edges
    from memman.store.model import Insight

    now = datetime.now(timezone.utc)

    def _make(**kw):
        defaults = {
            'id': 'x', 'content': 'x', 'category': 'fact', 'importance': 3,
            'entities': [], 'source': 'test', 'access_count': 0,
            'created_at': now, 'updated_at': now, 'deleted_at': None,
            'last_accessed_at': None}
        defaults.update(kw)
        return Insight(**defaults)

    for i in range(3):
        backend.nodes.insert(_make(
            id=f'old-{i}',
            content=f'Database optimization technique {i} because of performance'))

    new_ins = _make(
        id='new-1',
        content='Chose Redis because it improves cache hit rate for database queries')
    backend.nodes.insert(new_ins)

    captured_prompts = []

    class MockLLM:
        def complete(self, system, user, **kwargs):
            captured_prompts.append(user)
            return '[]'

    infer_llm_causal_edges(backend, new_ins, MockLLM())

    assert captured_prompts, 'LLM was never called'
    prompt = captured_prompts[0]
    assert 'RECENT INSIGHTS' in prompt, (
        'Recent insights section missing from LLM prompt — '
        'recent list passed as [] instead of actual recent insights')


NEW_FACT = ('Chose Redis for the cache layer because Postgres cache hit '
            'rate degraded under load')
OVERLAPPING_FACT = ('Postgres cache hit rate degraded under sustained '
                    'write load in the Redis evaluation')
UNRELATED_FACT = ('The office coffee machine is descaled every second '
                  'Tuesday by building maintenance')


def _insight(now, **kw):
    """Build an Insight carrying test defaults, overridden by keyword."""
    defaults = {
        'id': 'x', 'content': 'x', 'category': 'fact', 'importance': 3,
        'entities': [], 'source': 'test', 'access_count': 0,
        'created_at': now, 'updated_at': now, 'deleted_at': None,
        'last_accessed_at': None,
        }
    defaults.update(kw)
    return Insight(**defaults)


class _CapturingLLM:
    """Record every user prompt and answer with a canned response.

    Parameters
    ----------
    response : str or callable, default '[]'
        The raw completion, or a callable taking the user prompt and
        returning it.

    Attributes
    ----------
    prompts : list[str]
        Every user prompt received, in call order.
    """

    def __init__(self, response='[]'):
        """Store the canned response and start an empty prompt log.
        """
        self.prompts = []
        self.response = response

    def complete(self, system, user, **kwargs):
        """Record the user prompt and return the canned response.
        """
        self.prompts.append(user)
        if callable(self.response):
            return self.response(user)
        return self.response


def _seed_gate_straddling_rows(backend, now):
    """Insert one row clearing the overlap gate and one failing it.

    Returns the new insight, already stored, whose content the two
    seeded rows straddle.
    """
    backend.nodes.insert(_insight(now, id='pass-1', content=OVERLAPPING_FACT))
    backend.nodes.insert(_insight(now, id='fail-1', content=UNRELATED_FACT))
    new_ins = _insight(now, id='new-1', content=NEW_FACT)
    backend.nodes.insert(new_ins)
    return new_ins


def test_a_recent_row_below_the_overlap_gate_is_not_rendered(backend):
    """A recent row failing MIN_CAUSAL_OVERLAP stays out of the prompt.

    Mutation: handing the unfiltered `recent` list to
        `_build_llm_prompt`, so a row the gate rejected is rendered and
        billed for despite being barred from `valid_ids`.
    Oracle: hand-computed token overlap - 0.0000 for the coffee-machine
        row and 0.7273 for the Postgres row, against the 0.15 gate.
    """
    now = datetime.now(timezone.utc)
    new_ins = _seed_gate_straddling_rows(backend, now)

    client = _CapturingLLM()
    infer_llm_causal_edges(backend, new_ins, client)

    assert client.prompts, 'the LLM was never called'
    assert 'id=pass-1' in client.prompts[0]
    assert 'id=fail-1' not in client.prompts[0]


def test_a_gate_passing_recent_row_is_rendered_once(backend):
    """A recent row clearing the gate appears once, not in both sections.

    Mutation: rendering the candidate list and the recent list without
        removing the rows they share, so every gate-passing recent row
        is sent twice - once as a graph neighbor, once as recent.
    Oracle: the count of the row's id token in the captured prompt,
        against the hand-computed 1.
    """
    now = datetime.now(timezone.utc)
    new_ins = _seed_gate_straddling_rows(backend, now)

    client = _CapturingLLM()
    infer_llm_causal_edges(backend, new_ins, client)

    assert client.prompts[0].count('id=pass-1') == 1


def test_every_id_rendered_in_the_prompt_can_become_an_edge(backend):
    """The prompt shows exactly the rows the edge filter will accept.

    Mutation: rendering any row outside `valid_ids` - the unfiltered
        `recent` list, or a cap applied to the rendered lists but not to
        the candidate set they are derived from.
    Oracle: the model answers with one edge per rendered id and the
        shipped edge filter must return every one of them, so the
        comparand is the prompt itself rather than a second copy of the
        `valid_ids` expression.
    """
    now = datetime.now(timezone.utc)
    new_ins = _seed_gate_straddling_rows(backend, now)

    def _edge_per_rendered_id(prompt):
        """Answer with a high-confidence edge to every id rendered."""
        targets = [
            rid for rid in re.findall(r'id=([\w-]+)', prompt)
            if rid != 'new-1'
            ]
        return json.dumps([
            {
                'source_id': 'new-1',
                'target_id': rid,
                'confidence': 0.9,
                'sub_type': 'causes',
                'rationale': 'test',
                }
            for rid in targets
            ])

    client = _CapturingLLM(_edge_per_rendered_id)
    edges = infer_llm_causal_edges(backend, new_ins, client)

    rendered = {
        rid for rid in re.findall(r'id=([\w-]+)', client.prompts[0])
        if rid != 'new-1'
        }
    assert rendered, 'no context row was rendered at all'
    assert {e.target_id for e in edges} == rendered
