"""Remember pipeline - single entry point shared by sync CLI and worker.

Structure:

1. Quality check - advisory warnings only.
2. Planning phase - enrich the row, then embed its content once.
   **No DB writes.**
3. Apply phase - one transaction commits the replace link, insert,
   enrichment update, and stamp.

A write adds one row, or replaces the row `replace <id>` names.
Nothing else retires a row.

The apply phase runs only after all LLM and embed work has returned.
A crash during planning leaves the DB untouched, and the retry path
re-runs the whole pipeline.
"""

import logging
from typing import Any

import httpx
from memman.embed import EmbeddingProvider
from memman.exceptions import EmbedCredentialError
from memman.llm.client import get_llm_client
from memman.pipeline.enrich import enrich_with_llm
from memman.search.quality import check_content_quality
from memman.store.backend import Backend
from memman.store.model import Insight, format_timestamp, insight_to_delta_dict

logger = logging.getLogger('memman')


def run_remember(
        backend: Backend,
        insight: Insight,
        ec: EmbeddingProvider,
        replaced_id: str = '',
        ) -> dict[str, Any]:
    """Store one write, enriched and embedded, and return the result.

    Parameters
    ----------
    backend : Backend
        The target store.
    insight : Insight
        The row to store, built by the caller: its `content` is
        stored as written, and no model judges or rewords it.
    ec : EmbeddingProvider
        The store-bound embedder, from `bound_embedder`.
    replaced_id : str, default ''
        The row a `replace` retires; '' for a plain add.

    Returns
    -------
    dict[str, Any]
        `_apply_plan`'s result dict, plus `quality_warnings`.

    Raises
    ------
    EmbedCredentialError
        The store's embed provider has no credentials. An HTTP or
        runtime embed failure is logged instead, and the row is
        stored without a vector.
    """
    quality_warnings = check_content_quality(insight.content)

    llm_client = get_llm_client()
    try:
        enrichment = enrich_with_llm(insight, llm_client)
    except Exception:
        enrichment = {}

    embed_vec = None
    try:
        embed_vec = ec.embed(insight.content)
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

    insight.embedding_model = ec.model

    with backend.transaction():
        result = _apply_plan(
            backend, insight, replaced_id, embed_vec, enrichment,
            llm_client.model)

    result['quality_warnings'] = quality_warnings
    return result


def _apply_plan(
        backend: Backend,
        insight: Insight,
        replaced_id: str,
        embed_vec: list[float] | None,
        enrichment: dict[str, Any],
        summary_model: str,
        ) -> dict[str, Any]:
    """Store one insight. Must be invoked inside a transaction.

    A `replace` retires its target and never deletes it.

    Parameters
    ----------
    backend : Backend
        The target store.
    insight : Insight
        The row to store; its timestamps are refreshed from the stored
        row.
    replaced_id : str
        The row a `replace` retires; '' for a plain add.
    embed_vec : list[float] or None
        The content vector; None stores the row without one.
    enrichment : dict[str, Any]
        Output of `enrich_with_llm`; empty leaves the row unenriched.
    summary_model : str
        The LLM model id stored with the summary; unused when
        `enrichment` is empty.

    Returns
    -------
    dict[str, Any]
        The write's result. A target that is not current (forgotten,
        replaced by an earlier write, or never stored) is reported
        under `target_gone`, and the write degrades to a plain add.
    """
    replaced = False
    target_gone: dict[str, str | None] | None = None
    if replaced_id:
        before_target = backend.nodes.get_include_deleted(replaced_id)
        linked = backend.nodes.mark_replaced(replaced_id, insight.id)
        if linked and before_target is not None:
            replaced = True
            # The predecessor keeps its content behind `replaced_by`,
            # and the successor copies nothing from it.
            backend.oplog.log(
                operation='replace', insight_id=replaced_id,
                detail=f'replaced by {insight.id}',
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(insight))
        else:
            target_gone = {
                'id': replaced_id,
                'replaced_by': (before_target.replaced_by
                                if before_target is not None else None),
                }
            logger.warning(
                f'replace target {replaced_id} is not current;'
                ' degrading to add')
            # Notes:
            # - The row is stored either way, so nothing is lost, but a
            #   caller who ran `replace` to correct one row otherwise
            #   gets a new unlinked row and no sign the correction
            #   missed.
            # - The row is filed against the SUCCESSOR, which is
            #   readable; the requested target may be gone from the
            #   table entirely, and it is named in the detail instead.
            backend.oplog.log(
                operation='target-gone', insight_id=insight.id,
                detail=f'replace target {replaced_id} was not'
                ' current; stored without the link',
                after=insight_to_delta_dict(insight))

    backend.nodes.insert(insight)
    stored = backend.nodes.get(insight.id)
    if stored is not None and stored.created_at is not None:
        insight.created_at = stored.created_at
        insight.updated_at = stored.updated_at

    embedded = embed_vec is not None
    if embed_vec is not None:
        backend.nodes.update_embedding(
            insight.id, embed_vec, insight.embedding_model or '')

    backend.oplog.log(
        operation='remember', insight_id=insight.id, detail=insight.content,
        after=insight_to_delta_dict(insight))

    backend.nodes.stamp_enrich_attempted(insight.id)
    if enrichment:
        backend.nodes.update_enrichment(
            insight.id, summary=enrichment.get('summary', ''),
            summary_model=summary_model)
    # A vectorless row stays unstamped: the stranded-row sweep selects
    # `enriched_at is null`, and it is the only path that embeds the
    # row again.
    if enrichment and embedded:
        backend.nodes.stamp_enriched(insight.id)

    result: dict[str, Any] = {
        'id': insight.id,
        'content': insight.content,
        'action': 'replace' if replaced else 'add',
        'created_at': (
            format_timestamp(insight.created_at)
            if insight.created_at is not None else ''),
        'enrichment': {'summary': enrichment.get('summary', '')},
        'embedded': embedded,
        }
    # `replaced_id` names what this write linked; `target_gone` names
    # the row that now holds the topic, one read away, so a degraded
    # add cannot hide it.
    if replaced:
        result['replaced_id'] = replaced_id
    if target_gone is not None:
        result['target_gone'] = target_gone
    return result
