"""LLM-based insight enrichment: a one-sentence summary."""

import logging

from memman import trace
from memman.llm import usage as llm_usage
from memman.llm.shared import complete_parsed
from memman.store.model import Insight

logger = logging.getLogger('memman')

ENRICHMENT_SYSTEM_PROMPT = (
    'You are a memory graph enrichment engine. Given a memory insight, '
    'extract structured metadata.\n\n'
    'Return JSON with this field:\n'
    '{\n'
    '  "summary": "one-sentence summary of the core fact or decision"\n'
    '}')


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
        Key `summary`, or `{}` when the LLM call fails.

    Notes
    -----
    - A body that decodes on neither draw returns an empty summary.
      The outcome is terminal: retrying it would bill the call on
      every drain of the row's store.
    - Callers stamp `enriched_at` only when a non-empty dict and a
      vector land in the same pass, so `{}` leaves the row to the
      stranded-row sweep. On a re-enrichment the empty summary
      replaces the one the row held.
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
            ' on either draw; returning an empty summary',
            insight.id, len(insight.content), len(raw))
        trace.event(
            'enrich_result',
            insight_id=insight.id,
            outcome='parse_error',
            raw=raw)
        return {'summary': ''}

    summary = parsed.get('summary', '')
    if not isinstance(summary, str):
        summary = ''
    if summary and len(summary) >= len(insight.content) * 0.85:
        summary = ''

    trace.event(
        'enrich_result',
        insight_id=insight.id,
        outcome='ok',
        summary=summary)
    return {'summary': summary}
