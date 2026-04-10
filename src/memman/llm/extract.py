"""LLM-based fact extraction, memory reconciliation, and query expansion."""

import logging

from memman.llm.client import LLMClient, parse_json_response

logger = logging.getLogger('memman')

FACT_EXTRACTION_SYSTEM = (
    'You are a personal memory system. The user is storing memories '
    'for future recall. Extract atomic facts from the input.\n\n'
    'Return JSON:\n'
    '{"facts": [\n'
    '  {"text": "the fact, preserving key terms from the original",\n'
    '   "category": "preference|decision|fact|insight|context",\n'
    '   "importance": 2-5,\n'
    '   "entities": ["Entity1", "Entity2"]}\n'
    '], "skip_reason": null}\n\n'
    'Rules:\n'
    '- Extract 1-5 atomic facts. Each should stand alone.\n'
    '- ONLY skip with {"facts": [], "skip_reason": "..."} for content '
    'that is truly empty of information: greetings ("hi", "thanks"), '
    'filler ("ok", "sure"), or unintelligible text. '
    'Technical statements, configs, tool names, and general knowledge '
    'are ALL worth storing -- the user chose to remember them.\n'
    '- Preserve specific names, numbers, and technical terms from the '
    'original. Do not generalize or paraphrase away key details.\n'
    '- Category: preference=user likes/dislikes/prefers, '
    'decision=explicit choice of X over Y with rationale, '
    'fact=how something works (formulas, API behavior, data layout, '
    'code patterns), insight=lesson learned from experience, '
    'context=project background or user role. '
    'A formula or behavior description is a fact, not a decision.\n'
    '- Importance: 2=minor detail or passing mention, '
    '3=useful working knowledge (formulas, API behavior, '
    'data characteristics), '
    '4=decision with rationale that changes future behavior, '
    '5=architectural invariant or core user belief. '
    'Most facts should be 3. Reserve 4 for explicit decisions.\n'
    '- Entities: named entities only -- people, tools, libraries, '
    'projects, concepts, languages.\n'
    '- Each fact should be a complete sentence that makes sense alone.\n\n'
    'Examples:\n\n'
    'Input: "Hi there"\n'
    'Output: {"facts": [], "skip_reason": "greeting"}\n\n'
    'Input: "I chose Postgres over MySQL for JSONB support"\n'
    'Output: {"facts": [{"text": "Chose PostgreSQL over MySQL because '
    'of JSONB support", "category": "decision", "importance": 4, '
    '"entities": ["PostgreSQL", "MySQL", "JSONB"]}], '
    '"skip_reason": null}\n\n'
    'Input: "Switched to FastAPI, it\'s faster than Flask for async"\n'
    'Output: {"facts": [{"text": "Switched from Flask to FastAPI", '
    '"category": "decision", "importance": 4, '
    '"entities": ["FastAPI", "Flask"]}, '
    '{"text": "FastAPI is faster than Flask for async workloads", '
    '"category": "fact", "importance": 3, '
    '"entities": ["FastAPI", "Flask"]}], "skip_reason": null}\n\n'
    'Input: "Nginx worker_connections set to 4096 for load balancing"\n'
    'Output: {"facts": [{"text": "Nginx worker_connections set to '
    '4096 for load balancing", "category": "fact", "importance": 3, '
    '"entities": ["Nginx"]}], "skip_reason": null}\n\n'
    'Input: "Gamma PnL = 50 * gamma * r^2 using decimal returns"\n'
    'Output: {"facts": [{"text": "Gamma PnL calculated as '
    '50 * gamma * r^2 where r is the decimal return", '
    '"category": "fact", "importance": 3, '
    '"entities": ["Gamma PnL"]}], "skip_reason": null}')

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
        llm_client: LLMClient,
        content: str) -> list[dict]:
    """Extract atomic facts via LLM.

    Returns list of dicts with keys: text, category, importance, entities.
    On LLM failure: returns single passthrough fact.
    On LLM skip (skip_reason): returns empty list.
    """
    try:
        raw = llm_client.complete(FACT_EXTRACTION_SYSTEM, content)
    except Exception:
        logger.debug('LLM fact extraction failed, using passthrough')
        return _passthrough_fact(content, 'fact', 3)

    parsed = parse_json_response(raw)
    if parsed is None:
        logger.debug('LLM fact extraction parse error, using passthrough')
        return _passthrough_fact(content, 'fact', 3)

    skip_reason = parsed.get('skip_reason')
    if skip_reason:
        logger.debug(f'LLM skipped: {skip_reason}')
        return []

    raw_facts = parsed.get('facts', [])
    if not isinstance(raw_facts, list) or not raw_facts:
        return _passthrough_fact(content, 'fact', 3)

    facts = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        text = f.get('text', '').strip()
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

    return facts or _passthrough_fact(content, 'fact', 3)


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
        llm_client: LLMClient,
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

    try:
        raw = llm_client.complete(RECONCILIATION_SYSTEM, prompt)
    except Exception:
        logger.debug('LLM reconciliation failed, defaulting to ADD')
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
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None}
                for f in facts]

    return results


def expand_query(
        llm_client: LLMClient,
        query: str) -> dict:
    """Expand recall query with synonyms and related terms.

    Returns dict with: expanded_query, keywords, entities, intent.
    On failure: passthrough with original query.
    """
    try:
        raw = llm_client.complete(QUERY_EXPANSION_SYSTEM, query)
    except Exception:
        logger.debug('LLM query expansion failed, using passthrough')
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

    return {
        'expanded_query': expanded,
        'keywords': keywords,
        'entities': entities,
        'intent': intent,
        }
