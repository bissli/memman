"""Enrichment pass over pending rows: summary, vector.
"""

import logging
from collections.abc import Callable
from typing import Any

from memman import trace
from memman.embed import EmbeddingProvider
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient, get_llm_client
from memman.llm.shared import complete_parsed
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')

MAX_ENRICH_BATCH = 20

ENRICHMENT_SYSTEM_PROMPT = (
    'You summarize a memory insight in one sentence.\n\n'
    'Return JSON with this field:\n'
    '{\n'
    '  "summary": "one-sentence summary of the core fact or decision"\n'
    '}')


def enrich_with_llm(
        insight: Insight, llm_client: MemmanLLMClient) -> dict[str, Any]:
    """Summarize an insight via LLM.

    Parameters
    ----------
    insight : Insight
        The row to enrich; its id and content form the prompt.
    llm_client : MemmanLLMClient
        Client whose `complete` runs the call.

    Returns
    -------
    dict
        Key `summary`. `{}` when the LLM call fails. Callers stamp
        `enriched_at` only for a non-empty dict plus a vector, so `{}`
        leaves the row to the stranded-row sweep. A body that decodes
        on neither draw returns `{'summary': ''}`, which on a
        re-enrichment replaces the summary the row held.
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
        # Terminal: a retry would bill the call on every drain of the
        # row's store.
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


def enrich_pending(
        backend: Backend,
        llm_client: MemmanLLMClient | None = None,
        embed_client: EmbeddingProvider | None = None,
        max_batch: int = MAX_ENRICH_BATCH,
        on_progress: Callable[[str, Insight], None] | None = None,
        ) -> int:
    """Enrich and embed up to `max_batch` pending rows.

    Parameters
    ----------
    backend : Backend
        The store to work through.
    llm_client : MemmanLLMClient | None, default None
        Serves the enrichment call. None resolves `get_llm_client()`,
        whose model `compute_prompt_version` stamps on every row this
        pass writes; a client on any other model makes that stamp name
        a model that did not run.
    embed_client : EmbeddingProvider | None, default None
        The store-bound embedder; None stores no vector.
    max_batch : int, default MAX_ENRICH_BATCH
        Most rows one call processes.
    on_progress : Callable[[str, Insight], None] | None, default None
        Called with `'enrich'` before and `'done'` after each row.

    Returns
    -------
    int
        Rows processed; 0 once nothing is pending, which ends a
        caller's loop.

    Notes
    -----
    - Pending means `enrich_attempted_at is null`. Every attempt
      stamps `enrich_attempted_at`, whether or not the enrichment or
      the embed succeeded, so a failing row leaves the pending set
      and a rebuild loop terminates. `enriched_at` is stamped only
      when an enrichment and a vector both land.
    """
    pending_ids = backend.nodes.get_pending_enrich_ids(limit=max_batch)
    if not pending_ids:
        return 0

    from memman.pipeline.remember import compute_prompt_version

    try:
        active_pv: str | None = compute_prompt_version()
    except Exception as exc:
        logger.debug(
            f'compute_prompt_version failed; provenance left null:'
            f' {type(exc).__name__}: {exc}')
        active_pv = None

    processed = 0

    for insight_id in pending_ids:
        insight = backend.nodes.get(insight_id)
        if insight is None:
            continue

        if on_progress:
            on_progress('enrich', insight)

        enrichment: dict[str, Any] = {}
        # Resolved inside the try, so a client that fails to build
        # degrades to an unenriched row exactly as a failed call does.
        # get_llm_client caches its client, so the repeat costs nothing.
        try:
            if llm_client is None:
                llm_client = get_llm_client()
            enrichment = enrich_with_llm(insight, llm_client)
        except Exception:
            enrichment = {}

        new_vec = None
        new_vec_model = ''
        # Embeds on any enrichment: a row the write stored without a
        # vector gets one only here.
        if (enrichment
                and embed_client is not None
                and embed_client.available()):
            try:
                new_vec = embed_client.embed(insight.content)
                new_vec_model = embed_client.model or ''
            except Exception as exc:
                logger.warning(
                    'Re-embed failed for %s: %s; insight will not flip'
                    ' to enriched (retry on next pass)',
                    insight.id, exc)

        with backend.transaction():
            if enrichment:
                backend.nodes.update_enrichment(
                    insight.id, summary=enrichment.get('summary', ''))

            if new_vec is not None:
                backend.nodes.update_embedding(
                    insight.id, new_vec, new_vec_model)

            backend.nodes.stamp_enrich_attempted(insight_id)
            # No vector this pass, no stamp: an embed that failed or
            # could not run leaves the row for the stranded-row sweep.
            if enrichment and new_vec is not None:
                backend.nodes.stamp_enriched(
                    insight_id, prompt_version=active_pv)
        if on_progress:
            on_progress('done', insight)
        processed += 1
        logger.debug(
            f'enrich_pending {insight_id}: enriched={bool(enrichment)}'
            f' embedded={new_vec is not None}')

    return processed
