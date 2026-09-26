"""Enrichment pass over pending rows: keywords, summary, vector."""

import logging
from collections.abc import Callable

from memman.embed import EmbeddingProvider
from memman.llm.client import MemmanLLMClient, get_llm_client
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')

MAX_LINK_BATCH = 20


def link_pending(
        backend: Backend,
        metadata_llm_client: MemmanLLMClient | None = None,
        embed_client: EmbeddingProvider | None = None,
        max_batch: int = MAX_LINK_BATCH,
        on_progress: Callable[[str, Insight], None] | None = None,
        ) -> int:
    """Enrich and embed up to `max_batch` pending rows.

    Parameters
    ----------
    backend : Backend
        The store to work through.
    metadata_llm_client : MemmanLLMClient | None, default None
        Serves the enrichment call; None resolves the `slow` role.
    embed_client : EmbeddingProvider | None, default None
        The store-bound embedder; None stores no vector.
    max_batch : int, default MAX_LINK_BATCH
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
    - Pending means `linked_at is null`. Every attempt stamps
      `linked_at`, whether or not the enrichment or the embed
      succeeded, so a failing row leaves the pending set and a
      rebuild loop terminates. `enriched_at` is stamped only when an
      enrichment and a vector both land.
    - An omitted `metadata_llm_client` is resolved from `slow` HERE
      rather than inherited from a caller's other client:
      `compute_prompt_version` stamps the slow model on every row
      this pass writes, so any other client enriching the row makes
      that stamp name a model that did not run and prices the call
      at the wrong role.
    """
    pending_ids = backend.nodes.get_pending_link_ids(limit=max_batch)
    if not pending_ids:
        return 0

    from memman.graph.enrichment import enrich_with_llm
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

        enrichment: dict = {}
        # Resolved inside the try, so an unresolvable role degrades to
        # an unenriched row exactly as a failed call does.
        # get_llm_client caches per role, so the repeat costs nothing.
        try:
            if metadata_llm_client is None:
                metadata_llm_client = get_llm_client('slow')
            enrichment = enrich_with_llm(insight, metadata_llm_client)
        except Exception:
            enrichment = {}

        keywords = enrichment.get('keywords', [])
        new_vec = None
        new_vec_model = ''
        # Embeds on any enrichment, keywords or none: a row the write
        # stored without a vector gets one only here.
        if (enrichment
                and embed_client is not None
                and embed_client.available()):
            from memman.graph.enrichment import build_enriched_text
            enriched_text = build_enriched_text(
                insight.content, keywords)
            try:
                new_vec = embed_client.embed(enriched_text)
                new_vec_model = embed_client.model or ''
            except Exception as exc:
                logger.warning(
                    'Re-embed failed for %s: %s; insight will not flip'
                    ' to enriched (retry on next pass)',
                    insight.id, exc)

        with backend.transaction():
            if enrichment:
                backend.nodes.update_enrichment(
                    insight.id,
                    keywords=enrichment.get('keywords', []),
                    summary=enrichment.get('summary', ''))

            if new_vec is not None:
                backend.nodes.update_embedding(
                    insight.id, new_vec, new_vec_model)

            backend.nodes.stamp_linked(insight_id)
            # No vector this pass, no stamp: an embed that failed or
            # could not run leaves the row for the stranded-row sweep.
            if enrichment and new_vec is not None:
                backend.nodes.stamp_enriched(
                    insight_id, prompt_version=active_pv)
        if on_progress:
            on_progress('done', insight)
        processed += 1
        logger.debug(
            f'link_pending {insight_id}: enriched={bool(enrichment)}'
            f' embedded={new_vec is not None}')

    return processed
