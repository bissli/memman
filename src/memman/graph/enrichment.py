"""LLM-based insight enrichment: entities, keywords, summary, semantic facts."""

import logging

from memman import trace
from memman.llm import usage as llm_usage
from memman.llm.shared import complete_parsed, drop_overlong_strings
from memman.store.model import Insight

logger = logging.getLogger('memman')

ENRICHMENT_SYSTEM_PROMPT = (
    'You are a memory graph enrichment engine. Given a memory insight, '
    'extract structured metadata.\n\n'
    'Return JSON with these fields:\n'
    '{\n'
    '  "entities": ["key entities: people, tools, libraries, projects, '
    'concepts, files. Copy each one VERBATIM as it appears in the text. '
    'One entity per element; never join two names into one string. No '
    'type prefix and no parenthetical gloss: write pandas, not '
    'Library: pandas; write us-east-1, not AWS Region: us-east-1. A '
    'string that is not a literal substring of the text does not '
    'belong here"],\n'
    '  "keywords": ["search keywords that would help find this insight"],\n'
    '  "summary": "one-sentence summary of the core fact or decision",\n'
    '  "semantic_facts": ["list of 1-3 atomic facts stated or implied"]\n'
    '}\n\n'
    'Focus on precision -- only include entities and facts you are '
    'confident about from the text.')

# Hard caps on the MODEL's own contribution, so an over-eager LLM
# (e.g. 80+ entities on a large multi-claim blob) cannot inflate the
# entity graph or the keyword-enriched embedding. Enforced in
# post-processing rather than the prompt so prompt_version (and
# stored-row provenance) stays stable.
MAX_ENRICH_ENTITIES = 20
MAX_ENRICH_KEYWORDS = 12


def enrich_with_llm(
        insight: Insight, llm_client: object,
        *, seed_entities: bool = True) -> dict:
    """Extract enrichment fields from an insight via LLM.

    Parameters
    ----------
    insight : Insight
        The row to enrich. Its `entities` seed the returned list.
    llm_client : object
        Anything exposing `complete(system, user, stage=...)`.
    seed_entities : bool, default True
        True unions the model's entities onto the stored list. False
        returns the model's entities alone, so the caller's write
        REPLACES the row's vocabulary.

    Returns
    -------
    dict
        Keys `entities`, `keywords`, `summary`, `semantic_facts`, or
        empty on an LLM or parse failure. Pure function -- the caller
        handles every DB write.

    Notes
    -----
    - A rebuild passes `seed_entities=False`. Stored order decides
      which names reach `create_entity_edges` before
      `MAX_TOTAL_ENTITY_EDGES` is spent, and the seed sits first, so
      a seeded rebuild leaves the new names with no edge at all.
    - `seed_entities=False` drops a caller-supplied `--entity` name
      the model does not return. That is the stated cost of replacing
      a vocabulary; no column, oplog row or queue hint recovers one.
    """
    prompt = f'INSIGHT (id={insight.id[:8]}):\n{insight.content}'
    trace.event(
        'enrich_start',
        insight_id=insight.id,
        content_len=len(insight.content),
        existing_entity_count=len(insight.entities))

    try:
        parsed, raw = complete_parsed(
            llm_client, ENRICHMENT_SYSTEM_PROMPT, prompt,
            stage=llm_usage.STAGE_ENRICHMENT)
    except Exception as exc:
        logger.warning(
            'enrichment LLM call failed for %s (len=%d): %s: %s;'
            ' row stays unenriched',
            insight.id, len(insight.content),
            type(exc).__name__, exc)
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {}

    if parsed is None:
        logger.warning(
            'enrichment body did not decode for %s (len=%d, raw_len=%d)'
            ' on either draw; row stays unenriched',
            insight.id, len(insight.content), len(raw))
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='parse_error',
            raw=raw)
        return {}

    llm_entities = parsed.get('entities', [])
    if not isinstance(llm_entities, list):
        llm_entities = []
    llm_entities = [
        name for name in (str(e).strip() for e in llm_entities)
        if name]
    # Length cap BEFORE the merge loop and before the count cap:
    # `merged` is seeded from insight.entities, which carries the
    # user's --entity values, and 20 valid entities plus 3 over-long
    # ones must yield 20, not 17.
    llm_entities = drop_overlong_strings(
        llm_entities, kind='entity', owner=insight.id)

    # Notes:
    # - The count cap bounds what the MODEL adds, never the seed.
    #   `merged` starts as the user's --entity values, which the CLI
    #   already validated and accepted, so capping the merged list
    #   discarded caller input and spent the model's own budget on
    #   the seed.
    # - An unseeded call caps the model at MAX_ENRICH_ENTITIES on its
    #   own, since every name it returns is an addition.
    seed = list(insight.entities) if seed_entities else []
    existing = {e.strip().lower() for e in seed}
    merged = list(seed)
    added = 0
    for e in llm_entities:
        if added >= MAX_ENRICH_ENTITIES:
            break
        key = e.strip().lower()
        if key not in existing:
            merged.append(e)
            existing.add(key)
            added += 1

    keywords = parsed.get('keywords', [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = drop_overlong_strings(
        [str(k) for k in keywords if k],
        kind='keyword', owner=insight.id)[:MAX_ENRICH_KEYWORDS]

    summary = parsed.get('summary', '')
    if not isinstance(summary, str):
        summary = ''
    if summary and len(summary) >= len(insight.content) * 0.85:
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
