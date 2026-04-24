"""LLM-based insight enrichment: entities, keywords, summary, semantic facts."""

import logging

from memman import trace
from memman.llm.client import parse_json_response
from memman.model import Insight

logger = logging.getLogger('memman')

ENRICHMENT_SYSTEM_PROMPT = (
    'You are a memory graph enrichment engine. Given a memory insight, '
    'extract structured metadata.\n\n'
    'Return JSON with these fields:\n'
    '{\n'
    '  "entities": ["list of key entities: people, tools, libraries, '
    'projects, concepts, files"],\n'
    '  "keywords": ["search keywords that would help find this insight"],\n'
    '  "summary": "one-sentence summary of the core fact or decision",\n'
    '  "semantic_facts": ["list of 1-3 atomic facts stated or implied"]\n'
    '}\n\n'
    'Focus on precision -- only include entities and facts you are '
    'confident about from the text.')


def enrich_with_llm(insight: Insight, llm_client: object) -> dict:
    """Extract enrichment fields from an insight via LLM.

    Returns a dict with keys: entities, keywords, summary, semantic_facts.
    Pure function -- caller handles all DB writes.
    """
    prompt = f'INSIGHT (id={insight.id[:8]}):\n{insight.content}'
    trace.event(
        'enrich_start',
        insight_id=insight.id,
        content_len=len(insight.content),
        existing_entity_count=len(insight.entities))

    try:
        raw = llm_client.complete(ENRICHMENT_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        logger.debug(f'LLM enrichment unavailable for {insight.id}')
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {}

    parsed = parse_json_response(raw)
    if parsed is None:
        logger.debug(f'LLM enrichment parse error for {insight.id}')
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='parse_error',
            raw=raw)
        return {}

    llm_entities = parsed.get('entities', [])
    if not isinstance(llm_entities, list):
        llm_entities = []
    llm_entities = [str(e) for e in llm_entities if e]

    existing = {e.strip().lower() for e in insight.entities}
    merged = list(insight.entities)
    for e in llm_entities:
        key = e.strip().lower()
        if key not in existing:
            merged.append(e)
            existing.add(key)

    keywords = parsed.get('keywords', [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords if k]

    summary = parsed.get('summary', '')
    if not isinstance(summary, str):
        summary = ''

    facts = parsed.get('semantic_facts', [])
    if not isinstance(facts, list):
        facts = []
    facts = [str(f) for f in facts if f]

    logger.debug(
        f'Enriched {insight.id}: {len(llm_entities)} new entities, '
        f'{len(keywords)} keywords, {len(facts)} facts')

    result = {
        'entities': merged,
        'keywords': keywords,
        'summary': summary,
        'semantic_facts': facts,
        }
    trace.event(
        'enrich_result',
        insight_id=insight.id,
        outcome='ok',
        new_entity_count=len(llm_entities),
        keyword_count=len(keywords),
        fact_count=len(facts),
        summary=summary)
    return result


def build_enriched_text(content: str, keywords: list[str]) -> str:
    """Build keyword-enriched text for re-embedding."""
    if not keywords:
        return content
    return f'{content} [KEYWORDS: {" ".join(keywords)}]'
