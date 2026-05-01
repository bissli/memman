"""Graph engine: orchestrates automatic edge creation when insights are stored."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from memman.graph.entity import MAX_ENTITY_LINKS, MAX_TOTAL_ENTITY_EDGES
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import AUTO_SEMANTIC_THRESHOLD, build_embed_cache
from memman.graph.semantic import create_semantic_edges
from memman.graph.temporal import MAX_PROXIMITY_EDGES, MIN_PROXIMITY_WEIGHT
from memman.graph.temporal import TEMPORAL_WINDOW_HOURS, create_temporal_edge
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')


def fast_edges(backend: Backend, insight: Insight) -> dict[str, int]:
    """Run cheap edge generators (temporal + entity).

    LLM causal inference is deferred to link_pending().
    Semantic edges are deferred to link_pending().
    """
    return {
        'temporal': create_temporal_edge(backend, insight),
        'entity': create_entity_edges(backend, insight),
        }


MAX_LINK_BATCH = 20


def link_pending(
        backend: Backend,
        embed_cache: dict[str, list[float]] | None = None,
        llm_client: object | None = None,
        metadata_llm_client: object | None = None,
        embed_client: object | None = None,
        max_batch: int = MAX_LINK_BATCH,
        on_progress: 'Callable[[str, Insight], None] | None' = None,
        ) -> int:
    """Process insights where linked_at IS NULL.

    Creates semantic edges (and optionally LLM causal/enrichment edges)
    for pending insights. Returns the number of insights processed.
    """
    pending_ids = backend.nodes.get_pending_link_ids(limit=max_batch)
    if not pending_ids:
        return 0

    if embed_cache is None:
        embed_cache = build_embed_cache(backend)

    if metadata_llm_client is None:
        metadata_llm_client = llm_client

    from memman.graph.causal import infer_llm_causal_edges
    from memman.graph.enrichment import enrich_with_llm

    processed = 0

    for insight_id in pending_ids:
        insight = backend.nodes.get(insight_id)
        if insight is None:
            continue

        if on_progress:
            on_progress('enrich', insight)

        def _do_causal() -> list:
            with backend.readonly_context() as ro:
                return infer_llm_causal_edges(
                    ro, insight, metadata_llm_client)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_enrich = pool.submit(
                enrich_with_llm, insight, metadata_llm_client)
            fut_causal = pool.submit(_do_causal)
            try:
                enrichment = fut_enrich.result()
            except Exception:
                enrichment = {}
            if on_progress:
                on_progress('causal', insight)
            try:
                causal_edges = fut_causal.result()
            except Exception:
                causal_edges = []

        keywords = enrichment.get('keywords', [])
        new_vec = None
        if (keywords
                and embed_client is not None
                and embed_client.available()):
            from memman.graph.enrichment import build_enriched_text
            enriched_text = build_enriched_text(
                insight.content, keywords)
            try:
                new_vec = embed_client.embed(enriched_text)
            except Exception:
                logger.debug(f'Re-embed failed for {insight.id}')

        def _write_results() -> int:
            if enrichment:
                backend.nodes.update_enrichment(
                    insight.id,
                    keywords=enrichment.get('keywords', []),
                    summary=enrichment.get('summary', ''),
                    semantic_facts=enrichment.get('semantic_facts', []))
                backend.nodes.update_entities(
                    insight.id, enrichment.get('entities', []))
                insight.entities = enrichment.get('entities', [])

            if new_vec is not None:
                from memman.embed.vector import serialize_vector
                if embed_cache is not None:
                    embed_cache[insight.id] = new_vec
                backend.nodes.update_embedding(
                    insight.id, serialize_vector(new_vec),
                    getattr(embed_client, 'model', None) or '')

            backend.edges.delete_auto_for_node(insight.id, 'entity')
            create_entity_edges(backend, insight)

            backend.edges.delete_auto_for_node(insight.id, 'semantic')
            sem_count = create_semantic_edges(
                backend, insight, embed_cache)

            backend.edges.delete_auto_for_node(insight.id, 'causal')
            for edge in causal_edges:
                try:
                    backend.edges.upsert(edge)
                except Exception:
                    pass

            backend.nodes.stamp_linked(insight_id)
            if enrichment:
                backend.nodes.stamp_enriched(insight_id)
            return sem_count

        with backend.transaction():
            semantic_count = _write_results()
        if on_progress:
            on_progress('done', insight)
        processed += 1
        logger.debug(
            f'Linked {insight_id}: {semantic_count} semantic edges')

    return processed


def compute_constants_hash() -> str:
    """Return a short SHA-256 hash of all edge-relevant constants.

    Embed provider/model/dim drift is owned by
    `embed.fingerprint.assert_consistent` (hard-fail) and not
    folded into this hash, which only governs silent edge-reindex
    on graph-shape constant changes.
    """
    blob = json.dumps({
        'auto_semantic_threshold': AUTO_SEMANTIC_THRESHOLD,
        'min_proximity_weight': MIN_PROXIMITY_WEIGHT,
        'temporal_window_hours': TEMPORAL_WINDOW_HOURS,
        'max_entity_links': MAX_ENTITY_LINKS,
        'max_total_entity_edges': MAX_TOTAL_ENTITY_EDGES,
        'max_proximity_edges': MAX_PROXIMITY_EDGES,
        }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def reindex_auto_edges(
        backend: Backend, dry_run: bool = False) -> dict[str, int]:
    """Delete auto-created edges and re-create semantic/entity edges.

    Heuristic causal edges are deleted (replaced by LLM in Tier 3).
    LLM/manual causal edges are preserved.
    """
    semantic_del = backend.edges.count_auto_by_type('semantic')
    entity_del = backend.edges.count_auto_by_type('entity')
    temporal_del = backend.edges.count_low_weight_temporal_proximity(
        min_weight=MIN_PROXIMITY_WEIGHT)
    causal_del = backend.edges.count_auto_by_type('causal')

    if dry_run:
        stats = {
            'semantic_deleted': semantic_del,
            'entity_deleted': entity_del,
            'temporal_pruned': temporal_del,
            'causal_deleted': causal_del,
            'semantic_created': 0,
            'entity_created': 0,
            'dry_run': 1,
            }
        insights = backend.nodes.get_all_active()
        if insights:
            embed_cache = build_embed_cache(backend)
            for insight in insights:
                stats['entity_created'] += create_entity_edges(
                    backend, insight, dry_run=True)
                stats['semantic_created'] += create_semantic_edges(
                    backend, insight, embed_cache, dry_run=True)
        return stats

    stats: dict[str, int] = {
        'semantic_deleted': 0,
        'entity_deleted': 0,
        'temporal_pruned': 0,
        'causal_deleted': 0,
        'semantic_created': 0,
        'entity_created': 0,
        }

    def tx_body() -> None:
        backend.edges.delete_auto_by_type('semantic')
        stats['semantic_deleted'] = semantic_del

        backend.edges.delete_auto_by_type('entity')
        stats['entity_deleted'] = entity_del

        backend.edges.delete_low_weight_temporal_proximity(
            min_weight=MIN_PROXIMITY_WEIGHT)
        stats['temporal_pruned'] = temporal_del

        backend.edges.delete_auto_by_type('causal')
        stats['causal_deleted'] = causal_del

        insights = backend.nodes.get_all_active()
        if not insights:
            return

        embed_cache = build_embed_cache(backend)

        for insight in insights:
            stats['entity_created'] += create_entity_edges(
                backend, insight)
            stats['semantic_created'] += create_semantic_edges(
                backend, insight, embed_cache)

        stats['entity_created'] = backend.edges.count_auto_by_type(
            'entity')
        stats['semantic_created'] = backend.edges.count_auto_by_type(
            'semantic')

        backend.nodes.clear_linked_at()

        backend.oplog.log(
            operation='reindex', insight_id='',
            detail=json.dumps(stats))

    with backend.transaction():
        tx_body()
    return stats


def reindex_if_constants_changed(
        backend: Backend) -> dict[str, int] | None:
    """Reindex auto-edges when the stored constants hash is stale.

    Returns the reindex stats dict on reindex, or None when the stored
    hash already matched (common case). Emits a WARNING when constants
    drifted (operator hint to run `memman graph rebuild`) and a DEBUG
    line on first-time initialization.
    """
    current_hash = compute_constants_hash()
    stored_hash = backend.meta.get('constants_hash')
    if stored_hash == current_hash:
        return None

    stats = reindex_auto_edges(backend)
    backend.meta.set('constants_hash', current_hash)
    if stored_hash is not None:
        logger.warning(
            f'edge constants changed (hash {stored_hash} ->'
            f' {current_hash}); reindexed edges and cleared'
            ' linked_at. Run `memman graph rebuild` to re-embed'
            ' and re-enrich.')
    else:
        logger.debug(
            f'initial constants hash set: {current_hash}')
    return stats
