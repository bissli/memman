"""LLM-based fact extraction, memory reconciliation, and query expansion."""

import logging
import re

import cachetools
from memman import config, trace
from memman.llm.client import LLMProvider
from memman.llm.shared import parse_json_response

logger = logging.getLogger('memman')

_LINE_REF_RE = re.compile(
    r'\b(?:line \d+|[\w./]+:\d{2,})\b', re.IGNORECASE)
_WS_COLLAPSE_RE = re.compile(r'\s+')


def _strip_line_refs(text: str) -> str:
    """Strip file:line and 'line N' references; collapse whitespace."""
    return _WS_COLLAPSE_RE.sub(' ', _LINE_REF_RE.sub('', text)).strip()


FACT_EXTRACTION_SYSTEM = """You are a personal memory system curator. The user is storing memories for future recall via an upstream fast LLM that emits raw, unpolished content. Your job has THREE steps applied in order.

## Step 1: Skip-judgment

Decide if the input is durable knowledge worth long-term storage. Return {"facts": [], "skip_reason": "..."} for non-durable shapes:

- status updates ("All tests passed", "deployed v1.4 to staging")
- task receipts ("All drives verified after the maintenance window")
- in-progress observations ("migration is 60% complete")
- ephemeral metrics ("queue depth is 4 right now")
- "I just did X" reports
- one-off action confirmations
- greetings, filler, or unintelligible text

Otherwise continue to Step 2.

## Step 2: Canonical-shape rewrite

Rewrite into clean prose, claim-for-claim. The rewrite is strictly 1-to-1: same facts, reshaped surface form. Apply:

- Strip all-caps section markers (REBUILD COST:, KEY FINDING:, ROOT CAUSE:, ARCHITECTURAL CONSTRAINT:)
- Strip back-references (memory [N], see [0], "as mentioned earlier")
- Replace anaphoric openers ("This means", "That implies") with explicit subjects
- Strip transient adverbs (currently, today, "as of YYYY-MM-DD")
- Drop preamble filler ("This is relevant for...", "It is important to note that...")
- Convert past-tense decision narratives ("Rejected X. Chose Y.") into present-tense durable rules
- When the input frames something as non-default state (preserved, enabled, set to, retained, kept), keep that framing - don't flatten to plain present-tense
- Preserve specific names, numbers, and technical terms verbatim. Do not generalize.

## Step 3: Return as a SINGLE atomic fact

Return EXACTLY ONE fact whose `text` is the cleaned content from Step 2. Do NOT split a coherent input into multiple facts. The user remembers in coherent chunks; recall synthesizes across chunks via the graph and similarity. Splitting at storage time loses context.

CRITICAL: even if the input contains 3-5 distinct claims about a single coherent topic (e.g., a multi-section blob describing one system's behavior), return ONE fact whose text covers all the claims in canonical paragraph form. Splitting is not the curator's job.

Output JSON:
{"facts": [{"text": "<cleaned content>", "category": "preference|decision|fact|insight|context", "importance": 2-5, "entities": [...]}], "skip_reason": null}

## Category mapping

- preference: user likes/dislikes/prefers
- decision: explicit choice of X over Y with rationale, or "X uses Y rather than Z"
- fact: how something works (formulas, API behavior, data layout, code patterns)
- insight: lesson learned from experience
- context: project background or user role

A formula or behavior description is a fact, not a decision.

## Importance ladder (sharper anchors)

- imp=2: passing mention, soft preference
- imp=3: useful working knowledge - the DEFAULT for most facts
- imp=4: explicit decision with rationale, or strong user preference
- imp=5: rare cross-codebase invariant - the 1-in-20 fact that defines the entire system's contract ("Postgres is the system of record", "the API contract is JSON-only", "this database is the only source of truth")

**imp=5 is RARE.** Target distribution across a typical 20-fact session: ~10 at imp=3, ~7 at imp=4, ~2 at imp=5, ~1 at imp=2.

**ANTI-RULE: long, detailed, architectural-SOUNDING content is NOT automatically imp=5.** A 4000-character blob describing a workaround procedure, a code-level race condition, a tool docstring specification, or a proposal-rejection rationale is typically imp=4, not imp=5. The length and tone of the input do NOT determine importance - the structural role of the fact does.

**Test before assigning imp=5:** *would this fact still need to be true if we rewrote the entire codebase from scratch?* If yes -> imp=5. If it's just true OF this codebase (specific function, specific procedure, specific decision) -> imp=4 or imp=3.

## CRITICAL DIRECTIVE: Preserve the user's domain vocabulary

The examples below illustrate STYLE transformations only. Never substitute indexing, pipeline, database, or other domain terminology if the user's input did not use it. If the input is about React components, return facts about React components. If about CloudFormation, return facts about CloudFormation. The lessons below are about SHAPE (skip / cleanup / single-fact), not subject matter.

---

## Examples - Category 1: Accepted, barely changed

Input: "Pre-fetch object keys into a set for 1M+ file transfers to avoid 1M HEAD requests."
Output: {"facts": [{"text": "Pre-fetch object keys into a set for 1M+ file transfers to avoid 1M HEAD requests.", "category": "fact", "importance": 3, "entities": ["HEAD requests"]}], "skip_reason": null}

Input: "Avoid materializing all files upfront with list(fs.walk(recursive=True)) because it causes 30-60 minute enumeration delay and ~300MB memory spike."
Output: {"facts": [{"text": "Avoid materializing all files upfront with list(fs.walk(recursive=True)) because it causes 30-60 minute enumeration delay and ~300MB memory spike.", "category": "fact", "importance": 3, "entities": ["fs.walk"]}], "skip_reason": null}

Input: "render_grouped() added a sort_key parameter to enable custom sorting when grouping by non-date partition keys."
Output: {"facts": [{"text": "render_grouped() added a sort_key parameter to enable custom sorting when grouping by non-date partition keys.", "category": "fact", "importance": 3, "entities": ["render_grouped"]}], "skip_reason": null}

Input: "A timeout of 5.0 seconds and max_retries=1 were added to the LLM HTTP client to bound worst-case LLM latency."
Output: {"facts": [{"text": "A timeout of 5.0 seconds and max_retries=1 were added to the LLM HTTP client to bound worst-case LLM latency.", "category": "decision", "importance": 3, "entities": ["LLM HTTP client"]}], "skip_reason": null}

Input: "CLI surface should be minimal. Every flag must earn its existence."
Output: {"facts": [{"text": "CLI surface should be minimal. Every flag must earn its existence.", "category": "preference", "importance": 4, "entities": ["CLI"]}], "skip_reason": null}

## Examples - Category 2: Accepted, rewritten (style cleanup, claims preserved, ONE fact only)

Input: "Decision: rejected the _cleanup_helper() approach for orphan directory migration. Instead, manual rm -rf during rollout. Rationale: aligns with memory [3] preference for manual one-time cleanup over migration code in small-userbase projects."
Output: {"facts": [{"text": "Orphan directory migration uses manual rm -rf during rollout rather than a _cleanup_helper() function; in small-userbase projects, manual one-time cleanup is preferred over migration code.", "category": "decision", "importance": 4, "entities": ["_cleanup_helper", "rm -rf"]}], "skip_reason": null}

Input: "INTENT_PARSER.PY DEPRECATION PATH: intent_parser.py (an LLM-based query parser) has been DELETED. Deletion was safe because all three deprecation conditions were met: (1) the search MCP tool docstring was updated to teach LLM callers to extract metadata filters; (2) callers internalized the new filter-extraction pattern; (3) the local LLM dependency was removed from the runtime."
Output: {"facts": [{"text": "intent_parser.py, an LLM-based query parser, was deleted after its three deprecation conditions were met: the search MCP tool docstring was updated to teach callers to extract metadata filters; callers internalized the new filter-extraction pattern; and the local LLM runtime dependency was removed.", "category": "fact", "importance": 3, "entities": ["intent_parser.py", "MCP"]}], "skip_reason": null}

Input: "The pipeline currently uses Postgres as the system of record. This means the search index is fully recomputable from Postgres, which makes Postgres backups the only durable persistence layer."
Output: {"facts": [{"text": "The pipeline uses Postgres as the system of record; the search index is fully recomputable from Postgres, which makes Postgres backups the only durable persistence layer.", "category": "fact", "importance": 5, "entities": ["Postgres"]}], "skip_reason": null}

(Multi-section synthesis blobs with multiple SHOUTY headers are also single-fact outputs: rewrite as one canonical paragraph, all claims preserved, no decomposition.)

## Examples - Category 3: Skip as non-durable

Input: "All tests passed in the latest CI run after the rebase."
Output: {"facts": [], "skip_reason": "test_run_receipt"}

Input: "Currently processing the backlog at about 12 documents per second."
Output: {"facts": [], "skip_reason": "ephemeral_throughput"}

Input: "Just deployed v1.4 to staging via the release script."
Output: {"facts": [], "skip_reason": "deployment_receipt"}

Input: "Hi there"
Output: {"facts": [], "skip_reason": "greeting"}

## Examples - Edge cases (surface phrasing matches but durability differs)

Input: "All EC2 application updates are deployed via the standard release script, never via direct SSH."
Output: {"facts": [{"text": "All EC2 application updates are deployed via the standard release script, never via direct SSH.", "category": "preference", "importance": 4, "entities": ["EC2", "SSH"]}], "skip_reason": null}

Input: "Every login attempt is verified against the LDAP directory before the session token is issued."
Output: {"facts": [{"text": "Every login attempt is verified against the LDAP directory before the session token is issued.", "category": "fact", "importance": 4, "entities": ["LDAP"]}], "skip_reason": null}

---

Now process the user's input. Return ONLY JSON, no commentary."""

RECONCILIATION_SYSTEM = (
    'You are a memory manager. Compare new facts against existing '
    'memories and decide what to do.\n\n'
    'For each new fact, output one action:\n'
    '- ADD: new information not in existing memories\n'
    '- UPDATE <id>: refines or supersedes memory <id>. '
    'Provide merged text.\n'
    '- DELETE <id>: memory <id> is contradicted by new facts\n'
    '- NONE: fact already captured adequately\n\n'
    'Return JSON:\n'
    '{"actions": [\n'
    '  {"fact": "the fact text",\n'
    '   "action": "ADD|UPDATE|DELETE|NONE",\n'
    '   "target_id": null or "<numeric id>",\n'
    '   "merged_text": "merged content for UPDATE only",\n'
    '   "reason": "brief explanation"}\n'
    ']}\n\n'
    'Use the numeric IDs shown, not UUIDs. '
    'Prefer UPDATE over DELETE+ADD when info evolves.')

QUERY_EXPANSION_SYSTEM = (
    'Expand a search query for a personal memory system.\n\n'
    'Return JSON:\n'
    '{"expanded_query": "original plus synonyms and related terms",\n'
    ' "keywords": ["search", "terms"],\n'
    ' "entities": ["NamedEntity1"],\n'
    ' "intent": "WHY|WHEN|ENTITY|GENERAL"}\n\n'
    'Keep expanded_query under 50 words.')


def extract_facts(
        llm_client: LLMProvider,
        content: str) -> list[dict]:
    """Extract atomic facts via LLM.

    Returns list of dicts with keys: text, category, importance, entities.
    On LLM failure: returns single passthrough fact.
    On LLM skip (skip_reason): returns empty list.
    """
    trace.event('extract_facts_start', content_len=len(content))
    try:
        raw = llm_client.complete(FACT_EXTRACTION_SYSTEM, content)
    except Exception as exc:
        logger.debug('LLM fact extraction failed, using passthrough')
        trace.event(
            'extract_facts_result',
            outcome='passthrough',
            error=f'{type(exc).__name__}: {exc}')
        return _passthrough_fact(content, 'fact', 3)

    parsed = parse_json_response(raw)
    if parsed is None:
        logger.debug('LLM fact extraction parse error, using passthrough')
        trace.event(
            'extract_facts_result',
            outcome='parse_error',
            raw=raw)
        return _passthrough_fact(content, 'fact', 3)

    skip_reason = parsed.get('skip_reason')
    if skip_reason:
        logger.debug(f'LLM skipped: {skip_reason}')
        trace.event(
            'extract_facts_result',
            outcome='skipped',
            skip_reason=skip_reason)
        return []

    raw_facts = parsed.get('facts', [])
    if not isinstance(raw_facts, list) or not raw_facts:
        return _passthrough_fact(content, 'fact', 3)

    facts = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        text = _strip_line_refs(f.get('text', '').strip())
        if not text:
            continue
        importance = f.get('importance', 3)
        try:
            importance = max(2, min(5, int(importance)))
        except (ValueError, TypeError):
            importance = 3
        category = f.get('category', 'fact')
        if category not in {'preference', 'decision', 'fact',
                            'insight', 'context'}:
            category = 'fact'
        entities = f.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        entities = [str(e) for e in entities if e]
        facts.append({
            'text': text,
            'category': category,
            'importance': importance,
            'entities': entities,
            })

    result = facts or _passthrough_fact(content, 'fact', 3)
    trace.event(
        'extract_facts_result',
        outcome='ok',
        fact_count=len(result),
        skip_reason=skip_reason,
        facts=result)
    return result


def _passthrough_fact(
        content: str, category: str,
        importance: int) -> list[dict]:
    """Wrap raw content as a single fact for fallback."""
    return [{
        'text': content,
        'category': category,
        'importance': importance,
        'entities': [],
        }]


def reconcile_memories(
        llm_client: LLMProvider,
        facts: list[dict],
        existing_memories: list[tuple[str, str]]) -> list[dict]:
    """Compare facts against existing memories via LLM.

    Args:
        llm_client: configured LLM client
        facts: extracted facts from extract_facts()
        existing_memories: (real_id, content) pairs. Mapped to numeric
            IDs internally to prevent UUID hallucination.

    Returns list of dicts with keys: fact, action, target_id, merged_text.
    On failure: returns all-ADD.
    """
    if not existing_memories:
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    id_map = {}
    memory_lines = []
    for idx, (real_id, content) in enumerate(existing_memories):
        id_map[str(idx)] = real_id
        memory_lines.append(f'[{idx}] {content}')

    fact_lines = [f'- {f["text"]}' for f in facts]

    prompt = (
        'EXISTING MEMORIES:\n'
        + '\n'.join(memory_lines)
        + '\n\nNEW FACTS:\n'
        + '\n'.join(fact_lines))

    trace.event(
        'reconcile_start',
        fact_count=len(facts),
        existing_count=len(existing_memories))
    try:
        raw = llm_client.complete(RECONCILIATION_SYSTEM, prompt)
    except Exception as exc:
        logger.debug('LLM reconciliation failed, defaulting to ADD')
        trace.event(
            'reconcile_result',
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    parsed = parse_json_response(raw)
    if parsed is None:
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    actions_raw = parsed.get('actions', [])
    if not isinstance(actions_raw, list):
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    results = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        action = a.get('action', 'ADD').upper()
        if action not in {'ADD', 'UPDATE', 'DELETE', 'NONE'}:
            action = 'ADD'
        target_id = None
        if a.get('target_id') is not None:
            numeric_id = str(a['target_id'])
            target_id = id_map.get(numeric_id)
            if target_id is None:
                action = 'ADD'
        merged_text = a.get('merged_text')
        results.append({
            'fact': a.get('fact', ''),
            'action': action,
            'target_id': target_id,
            'merged_text': merged_text,
            })

    if not results:
        trace.event('reconcile_result', outcome='empty',
                    fallback='ADD')
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    trace.event(
        'reconcile_result',
        outcome='ok',
        actions=results)
    return results


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
        llm_client: LLMProvider,
        query: str) -> dict:
    """Expand recall query with synonyms and related terms.

    Returns dict with: expanded_query, keywords, entities, intent.
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
        raw = llm_client.complete(QUERY_EXPANSION_SYSTEM, query)
    except Exception as exc:
        logger.debug('LLM query expansion failed, using passthrough')
        trace.event(
            'query_expand_result',
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {'expanded_query': query, 'keywords': [],
                'entities': [], 'intent': None}

    parsed = parse_json_response(raw)
    if parsed is None:
        return {'expanded_query': query, 'keywords': [],
                'entities': [], 'intent': None}

    expanded = parsed.get('expanded_query', query)
    if not isinstance(expanded, str) or not expanded.strip():
        expanded = query

    keywords = parsed.get('keywords', [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords if k]

    entities = parsed.get('entities', [])
    if not isinstance(entities, list):
        entities = []
    entities = [str(e) for e in entities if e]

    intent = parsed.get('intent')
    if intent not in {'WHY', 'WHEN', 'ENTITY', 'GENERAL'}:
        intent = None

    result = {
        'expanded_query': expanded,
        'keywords': keywords,
        'entities': entities,
        'intent': intent,
        }
    _expand_cache[cache_key] = dict(result)
    trace.event('query_expand_result', outcome='ok', **result)
    return result
