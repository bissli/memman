"""Graph engine: orchestrates automatic edge creation when insights are stored."""

import hashlib
import json
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from memman.graph.entity import MAX_ENTITY_LINKS, MAX_TOTAL_ENTITY_EDGES
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import AUTO_SEMANTIC_THRESHOLD, build_embed_cache
from memman.graph.semantic import create_semantic_edges
from memman.graph.temporal import MAX_PROXIMITY_EDGES, MIN_PROXIMITY_WEIGHT
from memman.graph.temporal import TEMPORAL_WINDOW_HOURS, create_temporal_edge
from memman.model import Insight, format_timestamp
from memman.store.edge import count_auto_edges_by_type
from memman.store.edge import count_low_weight_temporal_proximity
from memman.store.edge import delete_auto_edges_by_type
from memman.store.edge import delete_auto_edges_for_node
from memman.store.edge import delete_low_weight_temporal_proximity
from memman.store.edge import insert_edge
from memman.store.node import clear_linked_at, get_all_active_insights
from memman.store.node import get_insight_by_id, get_pending_link_ids
from memman.store.node import stamp_enriched, stamp_linked, update_entities

logger = logging.getLogger('memman')


def fast_edges(db: 'DB', insight: Insight) -> dict[str, int]:
    """Run cheap edge generators (temporal + entity).

    LLM causal inference is deferred to link_pending().
    Semantic edges are deferred to link_pending().
    """
    return {
        'temporal': create_temporal_edge(db, insight),
        'entity': create_entity_edges(db, insight),
        }


MAX_LINK_BATCH = 20


def link_pending(
        db: 'DB',
        embed_cache: dict[str, list[float]] | None = None,
        llm_client: object | None = None,
        embed_client: object | None = None,
        max_batch: int = MAX_LINK_BATCH,
        on_progress: 'Callable[[str, Insight], None] | None' = None,
        ) -> int:
    """Process insights where linked_at IS NULL.

    Creates semantic edges (and optionally LLM causal/enrichment edges)
    for pending insights. Returns the number of insights processed.
    """
    pending_ids = get_pending_link_ids(db, max_batch)
    if not pending_ids:
        return 0

    if embed_cache is None:
        embed_cache = build_embed_cache(db)

    from memman.graph.causal import infer_llm_causal_edges
    from memman.graph.enrichment import enrich_with_llm
    from memman.store.db import open_read_only

    now = format_timestamp(datetime.now(timezone.utc))
    processed = 0
    data_dir = str(pathlib.Path(db.path).parent)

    for insight_id in pending_ids:
        insight = get_insight_by_id(db, insight_id)
        if insight is None:
            continue

        if on_progress:
            on_progress('enrich', insight)

        def _do_causal() -> list:
            ro_db = open_read_only(data_dir)
            try:
                return infer_llm_causal_edges(
                    ro_db, insight, llm_client)
            finally:
                ro_db.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_enrich = pool.submit(
                enrich_with_llm, insight, llm_client)
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
                from memman.store.node import update_enrichment
                update_enrichment(
                    db, insight.id,
                    enrichment.get('keywords', []),
                    enrichment.get('summary', ''),
                    enrichment.get('semantic_facts', []))
                update_entities(
                    db, insight.id, enrichment.get('entities', []))
                insight.entities = enrichment.get('entities', [])

            if new_vec is not None:
                from memman.embed.vector import serialize_vector
                from memman.store.node import update_embedding
                if embed_cache is not None:
                    embed_cache[insight.id] = new_vec
                update_embedding(
                    db, insight.id, serialize_vector(new_vec))

            delete_auto_edges_for_node(db, insight.id, 'entity')
            create_entity_edges(db, insight)

            delete_auto_edges_for_node(db, insight.id, 'semantic')
            sem_count = create_semantic_edges(
                db, insight, embed_cache)

            delete_auto_edges_for_node(db, insight.id, 'causal')
            for edge in causal_edges:
                try:
                    insert_edge(db, edge)
                except Exception:
                    pass

            stamp_linked(db, insight_id, now)
            if enrichment:
                stamp_enriched(db, insight_id, now)
            return sem_count

        semantic_count = db.in_transaction(_write_results)
        if on_progress:
            on_progress('done', insight)
        processed += 1
        logger.debug(
            f'Linked {insight_id}: {semantic_count} semantic edges')

    return processed


def compute_constants_hash() -> str:
    """Return a short SHA-256 hash of all edge-relevant constants.

    The embedding model name is included so that a provider or model
    swap (e.g. voyage-3-lite -> voyage-3, 512-dim -> 1024-dim) trips
    the reindex path. Without this, new-dim vectors would cosine
    against old-dim vectors to 0.0 silently.
    """
    from memman.embed.voyage import DEFAULT_MODEL as EMBED_MODEL

    blob = json.dumps({
        'auto_semantic_threshold': AUTO_SEMANTIC_THRESHOLD,
        'min_proximity_weight': MIN_PROXIMITY_WEIGHT,
        'temporal_window_hours': TEMPORAL_WINDOW_HOURS,
        'max_entity_links': MAX_ENTITY_LINKS,
        'max_total_entity_edges': MAX_TOTAL_ENTITY_EDGES,
        'max_proximity_edges': MAX_PROXIMITY_EDGES,
        'embed_model': EMBED_MODEL,
        }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def reindex_auto_edges(
        db: 'DB', dry_run: bool = False) -> dict[str, int]:
    """Delete auto-created edges and re-create semantic/entity edges.

    Heuristic causal edges are deleted (replaced by LLM in Tier 3).
    LLM/manual causal edges are preserved.
    """
    from memman.store.oplog import log_op

    semantic_del = count_auto_edges_by_type(db, 'semantic')
    entity_del = count_auto_edges_by_type(db, 'entity')
    temporal_del = count_low_weight_temporal_proximity(
        db, MIN_PROXIMITY_WEIGHT)
    causal_del = count_auto_edges_by_type(db, 'causal')

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
        insights = get_all_active_insights(db)
        if insights:
            embed_cache = build_embed_cache(db)
            for insight in insights:
                stats['entity_created'] += create_entity_edges(
                    db, insight, dry_run=True)
                stats['semantic_created'] += create_semantic_edges(
                    db, insight, embed_cache, dry_run=True)
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
        delete_auto_edges_by_type(db, 'semantic')
        stats['semantic_deleted'] = semantic_del

        delete_auto_edges_by_type(db, 'entity')
        stats['entity_deleted'] = entity_del

        delete_low_weight_temporal_proximity(db, MIN_PROXIMITY_WEIGHT)
        stats['temporal_pruned'] = temporal_del

        delete_auto_edges_by_type(db, 'causal')
        stats['causal_deleted'] = causal_del

        insights = get_all_active_insights(db)
        if not insights:
            return

        embed_cache = build_embed_cache(db)

        for insight in insights:
            stats['entity_created'] += create_entity_edges(
                db, insight)
            stats['semantic_created'] += create_semantic_edges(
                db, insight, embed_cache)

        stats['entity_created'] = count_auto_edges_by_type(db, 'entity')
        stats['semantic_created'] = count_auto_edges_by_type(
            db, 'semantic')

        clear_linked_at(db)

        log_op(db, 'reindex', '', json.dumps(stats))

    db.in_transaction(tx_body)
    return stats
