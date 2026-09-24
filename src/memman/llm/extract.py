"""LLM memory reconciliation and query expansion."""

import logging
import re

import cachetools
from memman import config, trace
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient
from memman.llm.shared import complete_parsed, parse_json_response

logger = logging.getLogger('memman')

_WS_COLLAPSE_RE = re.compile(r'\s+')

SCREEN_MAX_TOKENS = 2048
VERDICT_MAX_TOKENS = 2048
MERGE_MAX_TOKENS = 8192

SCREEN_RELATIONS = frozenset({'CONTRADICTS', 'REFINES', 'RESTATES', 'UNRELATED'})
UNJUDGED = 'UNJUDGED'

# Notes:
# - The write path's disposition of every verdict token the text
#   offers, keyed on the `takes only` line of RECONCILIATION_SYSTEM;
#   `tests/test_reconcile_stages.py` pins the keys to that line, so a
#   token added to the text without a row fails the suite.
# - The reader skips an entry whose token is outside the table (the
#   model's stray word), so it can neither write nor block, and a
#   lookup here never sees one.
VERDICT_DISPOSITION = {
    'SUPERSEDE': 'supersede',
    'UPDATE': 'update',
    'NONE': 'none',
    'ADD': 'keep',
    }

PAIRWISE_SCREEN_SYSTEM = """You are a memory manager. ONE EXISTING MEMORY and ONE NEW FACT arrive. The fact is the newer statement. Judge the memory's present-tense claims against the fact and name their relation.

Relations, judged on present-tense claims only:
- CONTRADICTS: the memory asserts something the fact says is no longer so or never was: a changed value or name, a mechanism that was removed or replaced, a decision that was reversed, or a question the memory left open that the fact settles the other way. A memory that reports a dated event, or what was true at a stated time, is not contradicted by a later state. A fact that says something is no longer true, or corrects an earlier statement, contradicts a memory that asserts the old state.
- REFINES: the fact adds compatible detail to the memory's subject and overturns nothing the memory asserts.
- RESTATES: the memory already carries every claim the fact makes.
- UNRELATED: a different subject, or compatible independent claims.

Return ONLY JSON, no commentary:
{"relation": "CONTRADICTS|REFINES|RESTATES|UNRELATED",
 "contradicted_clauses": ["<each clause of the memory the fact overturns, quoted from the memory>"],
 "reason": "brief explanation"}
contradicted_clauses is empty unless the relation is CONTRADICTS."""

RECONCILIATION_SYSTEM = """You are a memory manager. One NEW FACT arrives with a list of EXISTING MEMORIES, each under a numeric id. Judge EVERY memory against the fact and return one action per memory whose state the fact changes. A memory whose state the fact does not change gets no entry.

Relations, judged on present-tense claims only:
- CONTRADICTS: the memory asserts something the fact says is no longer so or never was: a changed value or name, a mechanism that was removed or replaced, a decision that was reversed, or a question the memory left open that the fact settles the other way. A memory that reports a dated event, or what was true at a stated time, is not contradicted by a later state. A fact that says something is no longer true, or corrects an earlier statement, contradicts every memory that asserts the old state.
- REFINES: the fact adds compatible detail to the memory's subject.
- RESTATES: the memory already carries every claim the fact makes.
- UNRELATED: a different subject, or compatible independent claims.

Actions:
- SUPERSEDE <id>: the fact contradicts memory <id>. Name EVERY contradicted memory, not only the closest one. A memory the fact merely repeats or extends is never superseded.
- UPDATE <id>: the fact refines memory <id>. At most one.
- NONE <id>: memory <id> restates the fact. Alone.
- ADD: no memory is contradicted, refined, or restating. Alone.
The action field takes only ADD, UPDATE, SUPERSEDE, or NONE.

Return JSON:
{"actions": [
  {"action": "ADD|UPDATE|SUPERSEDE|NONE",
   "target_id": null for ADD, else the numeric id,
   "reason": "brief explanation"}
 ]}

When one memory restates the fact and another contradicts it, the restating memory takes UPDATE (it is folded into the successor) and the contradicted one takes SUPERSEDE; NONE is for a fact that changes nothing.

Use the numeric IDs shown, not UUIDs. A contradicted memory gets SUPERSEDE, never ADD."""

MERGE_SYSTEM = """You are a memory manager. A NEW FACT arrives with the EXISTING MEMORIES it changes. Each memory is listed under a numeric id with the clauses the fact overturns (CONTRADICTED CLAUSES); a memory listed with no contradicted clauses is one the fact only adds detail to. Write the SUCCESSOR TEXT that replaces every listed memory.

The successor text:
- states the new fact;
- keeps EVERY clause of every listed memory that is not among its contradicted clauses, in that memory's own words where possible;
- drops each contradicted clause and never restates it as true;
- adds nothing that neither the fact nor a listed memory states;
- is one canonical paragraph of prose (no headers, no bullet lists, no back-references such as "memory 0").

Return ONLY JSON, no commentary:
{"merged_text": "<the successor text>"}"""

QUERY_EXPANSION_SYSTEM = (
    'Expand a search query for a personal memory system.\n\n'
    'Return JSON:\n'
    '{"expanded_query": "original plus synonyms and related terms",\n'
    ' "intent": "WHY|WHEN|ENTITY|GENERAL"}\n\n'
    'Keep expanded_query under 50 words.')


def screen_memory(
        llm_client: MemmanLLMClient,
        fact_text: str,
        memory: tuple[str, str]) -> tuple[str, list[str]]:
    """Stage 1: one row's relation to the fact and the clauses overturned.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The fast-worker client; one `complete` call per invocation.
    fact_text : str
        The new write's text, as the agent wrote it.
    memory : tuple[str, str]
        `(real_id, content)`, the row under judgment.

    Returns
    -------
    tuple[str, list[str]]
        The relation, one of `SCREEN_RELATIONS` or `UNJUDGED`, and the
        clauses of the row the fact overturns, quoted from the row;
        empty unless the relation is CONTRADICTS.

    Notes
    -----
    - Fail-open: an LLM error, an unparsed body, a missing or unknown
      relation is `UNJUDGED`, and the keep rule shows an UNJUDGED row
      to stage 2 rather than drop it.
    """
    real_id, content = memory
    body = f'EXISTING MEMORY:\n{content}\n\nNEW FACT:\n{fact_text}'
    try:
        raw = llm_client.complete(
            PAIRWISE_SCREEN_SYSTEM, body,
            stage=llm_usage.STAGE_SCREEN,
            max_tokens=SCREEN_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'screen_result', target_id=real_id, outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return UNJUDGED, []
    parsed = parse_json_response(raw)
    relation = (str(parsed.get('relation', '')).upper()
                if isinstance(parsed, dict) else '')
    if relation not in SCREEN_RELATIONS:
        trace.event(
            'screen_result', target_id=real_id, outcome='unjudged', raw=raw)
        return UNJUDGED, []
    clauses: list[str] = []
    if relation == 'CONTRADICTS':
        quoted = parsed.get('contradicted_clauses')
        if isinstance(quoted, list):
            clauses = [c for c in quoted if isinstance(c, str) and c.strip()]
    trace.event(
        'screen_result', target_id=real_id, outcome='ok',
        relation=relation, clauses=len(clauses))
    return relation, clauses


def judge_memory(
        llm_client: MemmanLLMClient,
        fact_text: str,
        memory: tuple[str, str]) -> str:
    """Stage 2: the write path's verdict on one screened row.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The fast-worker client; one `complete` call per invocation.
    fact_text : str
        The new write's text, as the agent wrote it.
    memory : tuple[str, str]
        `(real_id, content)`, the one row shown under `[0]`.

    Returns
    -------
    str
        `supersede`, `update`, `none`, or `keep` for no write against
        the row.

    Notes
    -----
    - The id map is the measured `first` variant: the first entry
      with a known token that names ANY non-null id decides, whatever
      id it names. The model numbers sections of the one row and judges
      a section (25 rows of 1,221 on the probe), and every such verdict
      is about the row shown.
    - An entry with a null id is skipped: an ADD there neither decides
      nor blocks, a row verdict there is dropped. DELETE reads as
      SUPERSEDE. An entry whose token is outside `VERDICT_DISPOSITION`
      is skipped the same way, so the next known entry decides.
    - Fail-closed: an LLM error or an unparsed body is `keep`.
    """
    real_id, content = memory
    body = f'EXISTING MEMORIES:\n[0] {content}\n\nNEW FACT:\n{fact_text}'
    trace.event('reconcile_start', target_id=real_id)
    try:
        raw = llm_client.complete(
            RECONCILIATION_SYSTEM, body,
            stage=llm_usage.STAGE_RECONCILIATION,
            max_tokens=VERDICT_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'reconcile_result', target_id=real_id, outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return 'keep'

    parsed = parse_json_response(raw)
    if parsed is None or not isinstance(parsed.get('actions'), list):
        trace.event(
            'reconcile_result', target_id=real_id, outcome='parse_error',
            raw=raw)
        return 'keep'

    verdict = 'keep'
    for entry in parsed['actions']:
        if not isinstance(entry, dict) or entry.get('target_id') is None:
            continue
        action = str(entry.get('action', '')).upper()
        if action == 'DELETE':
            action = 'SUPERSEDE'
        if action not in VERDICT_DISPOSITION:
            continue
        verdict = VERDICT_DISPOSITION[action]
        break
    trace.event(
        'reconcile_result', target_id=real_id, outcome='ok', verdict=verdict)
    return verdict


def merge_successor(
        llm_client: MemmanLLMClient,
        fact_text: str,
        target: tuple[str, str, list[str]]) -> str | None:
    """Stage 3: the successor text for one retiring target.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The fast-worker client; one `complete` call per invocation.
    fact_text : str
        The new write's text, as the agent wrote it.
    target : tuple[str, str, list[str]]
        `(real_id, content, clauses)`: the retiring row and the clauses
        the screen quoted as contradicted; empty for an update target
        or a row the screen did not call CONTRADICTS.

    Returns
    -------
    str | None
        The merged text, stripped; None on an LLM error, an unparsed
        body or an empty text, so the caller stores the fact and marks
        the oplog row `(unmerged)`.

    Notes
    -----
    - One target per call: a body listing the row alone keeps clauses
      that a body listing every row drops.
    - The screen can quote, as a contradicted clause, a sentence it
      copied from the NEW FACT rather than from the target memory.
      Handed to the merge, such a clause makes the merge delete the
      fact's own correction from the successor. A clause whose
      normalized text appears in the fact but not in the target
      content is a fact-copied clause and is suppressed before
      rendering.
    """
    real_id, content, clauses = target
    norm_fact = ' '.join(fact_text.lower().split())
    norm_content = ' '.join(content.lower().split())
    # Notes:
    # - A clause found in the fact but not in the target is a fact
    #   sentence the screen copied, not a memory clause; handed to the
    #   merge it makes the merge delete the fact's own correction.
    # - Whitespace and case are collapsed because the screen returns
    #   clauses in the model's own wrapping and casing.
    filtered_clauses = [
        c for c in clauses
        if (nc := ' '.join(c.lower().split())) in norm_content
        or nc not in norm_fact
        ]
    suppressed = len(clauses) - len(filtered_clauses)
    clause_lines = [f'- {clause}' for clause in filtered_clauses] or ['(none)']
    body = (
        f'EXISTING MEMORIES:\n[0] {content}\n'
        'CONTRADICTED CLAUSES of [0]:\n' + '\n'.join(clause_lines)
        + f'\n\nNEW FACT:\n{fact_text}')
    try:
        parsed, raw = complete_parsed(
            llm_client, MERGE_SYSTEM, body,
            stage=llm_usage.STAGE_MERGE,
            max_tokens=MERGE_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'reconcile_merge', target_id=real_id,
            clauses=len(filtered_clauses), suppressed=suppressed,
            outcome='error', error=f'{type(exc).__name__}: {exc}')
        return None
    text = parsed.get('merged_text') if isinstance(parsed, dict) else None
    if not isinstance(text, str) or not text.strip():
        trace.event(
            'reconcile_merge', target_id=real_id,
            clauses=len(filtered_clauses), suppressed=suppressed,
            outcome='no_text', raw=raw)
        return None
    trace.event(
        'reconcile_merge', target_id=real_id,
        clauses=len(filtered_clauses), suppressed=suppressed,
        outcome='ok')
    return text.strip()


_EXPAND_CACHE_TTL = 300
_EXPAND_CACHE_MAX = 256


def _normalize_for_cache(query: str) -> str:
    """Lowercase + collapse whitespace; nothing else."""
    return ' '.join(query.lower().split())


def _expand_cache_key(query: str) -> str:
    """Salt with the configured fast-model id.

    The model id is resolved at install time and persisted to
    `~/.memman/env`, so `config.require` always returns a real value
    here. Reaching this with an unset key means install was never run,
    which is a `ConfigError` upstream callers handle.
    """
    import hashlib
    salt = config.require(config.LLM_MODEL_FAST)
    digest = hashlib.sha256(
        f'{_normalize_for_cache(query)}|{salt}'.encode())
    return digest.hexdigest()[:16]


_expand_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=_EXPAND_CACHE_MAX, ttl=_EXPAND_CACHE_TTL)


def reset_expand_cache() -> None:
    """Drop cached query expansions. Used by tests that swap env vars."""
    _expand_cache.clear()


def expand_query(
        llm_client: MemmanLLMClient,
        query: str) -> dict:
    """Expand recall query with synonyms and related terms.

    Returns dict with: expanded_query, intent.
    On failure: passthrough with original query. Repeated calls with
    the same query in the same process hit a `cachetools.TTLCache`
    keyed by sha256(normalized_query | $MEMMAN_LLM_MODEL_FAST). Cache
    lives only for the duration of one CLI invocation (memman is a
    one-shot CLI), so persistence across processes is left to the
    LLM provider's own response cache.
    """
    cache_key = _expand_cache_key(query)
    cached = _expand_cache.get(cache_key)
    if cached is not None:
        trace.event(
            'query_expand_result', outcome='cache_hit', **cached)
        return dict(cached)

    trace.event('query_expand_start', query=query)
    try:
        raw = llm_client.complete(
            QUERY_EXPANSION_SYSTEM, query,
            stage=llm_usage.STAGE_QUERY_EXPANSION)
    except Exception as exc:
        logger.debug('LLM query expansion failed, using passthrough')
        trace.event(
            'query_expand_result',
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {'expanded_query': query, 'intent': None}

    parsed = parse_json_response(raw)
    if parsed is None:
        return {'expanded_query': query, 'intent': None}

    expanded = parsed.get('expanded_query', query)
    if not isinstance(expanded, str) or not expanded.strip():
        expanded = query

    intent = parsed.get('intent')
    if intent not in {'WHY', 'WHEN', 'ENTITY', 'GENERAL'}:
        intent = None

    result = {
        'expanded_query': expanded,
        'intent': intent,
        }
    _expand_cache[cache_key] = dict(result)
    trace.event('query_expand_result', outcome='ok', **result)
    return result
