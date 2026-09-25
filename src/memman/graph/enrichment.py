"""LLM-based insight enrichment: keywords, summary."""

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
    '  "keywords": ["search keywords that would help find this insight"],\n'
    '  "summary": "one-sentence summary of the core fact or decision"\n'
    '}\n\n'
    'Focus on precision -- only include keywords you are confident about '
    'from the text.')

# Hard cap on the MODEL's own contribution, so an over-eager LLM
# (e.g. dozens of keywords on a large multi-claim blob) cannot inflate
# the keyword-enriched embedding. Enforced in post-processing rather
# than the prompt so prompt_version (and stored-row provenance) stays
# stable.
MAX_ENRICH_KEYWORDS = 12


def enrich_with_llm(insight: Insight, llm_client: object) -> dict:
    """Extract enrichment fields from an insight via LLM.

    Parameters
    ----------
    insight : Insight
        The row to enrich; its id and content form the prompt.
    llm_client : object
        Anything exposing `complete(system, user, stage=...)`.

    Returns
    -------
    dict
        Keys `keywords` and `summary`, or `{}` when the LLM call fails.

    Notes
    -----
    - A body that decodes on neither draw returns both keys empty. The
      outcome is terminal: retrying it would bill the call on every
      drain of the row's store.
    - Callers stamp `enriched_at` only when a non-empty dict and a
      vector land in the same pass, so `{}` leaves the row to the
      stranded-row sweep. On a re-enrichment the empty keys replace
      the keywords and summary the row held.
    - Pure function -- the caller handles every DB write.
    """
    prompt = f'INSIGHT (id={insight.id[:8]}):\n{insight.content}'
    trace.event(
        'enrich_start',
        insight_id=insight.id,
        content_len=len(insight.content))

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
            ' on either draw; returning empty keywords and summary',
            insight.id, len(insight.content), len(raw))
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='parse_error',
            raw=raw)
        return {'keywords': [], 'summary': ''}

    keywords = parsed.get('keywords', [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = drop_overlong_strings(
        [str(k) for k in keywords if k],
        owner=insight.id)[:MAX_ENRICH_KEYWORDS]

    summary = parsed.get('summary', '')
    if not isinstance(summary, str):
        summary = ''
    if summary and len(summary) >= len(insight.content) * 0.85:
        summary = ''

    logger.debug(f'Enriched {insight.id}: {len(keywords)} keywords')

    result = {
        'keywords': keywords,
        'summary': summary,
        }
    trace.event(
        'enrich_result',
        insight_id=insight.id,
        outcome='ok',
        keyword_count=len(keywords),
        summary=summary)
    return result


def build_enriched_text(content: str, keywords: list[str]) -> str:
    """Text the store embeds for a row: its content plus its keywords.

    Parameters
    ----------
    content : str
        The row's content.
    keywords : list[str]
        Enrichment keywords; empty when enrichment failed or found none.

    Returns
    -------
    str
        `content` alone when `keywords` is empty, else
        `'<content> [KEYWORDS: k1 k2 ...]'`.
    """
    if not keywords:
        return content
    return f'{content} [KEYWORDS: {" ".join(keywords)}]'
