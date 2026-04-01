"""Graph engine: orchestrates automatic edge creation when insights are stored."""

import hashlib
import json
import logging
from datetime import datetime, timezone

from mnemon.graph.entity import MAX_ENTITY_LINKS, MAX_TOTAL_ENTITY_EDGES
from mnemon.graph.entity import create_entity_edges
from mnemon.graph.semantic import AUTO_SEMANTIC_THRESHOLD, build_embed_cache
from mnemon.graph.semantic import create_semantic_edges
from mnemon.graph.temporal import MAX_PROXIMITY_EDGES, MIN_PROXIMITY_WEIGHT
from mnemon.graph.temporal import TEMPORAL_WINDOW_HOURS, create_temporal_edge
from mnemon.model import Edge, Insight, format_timestamp, parse_timestamp
from mnemon.store.edge import insert_edge
from mnemon.store.node import get_all_active_insights, get_insight_by_id
from mnemon.store.node import update_entities

logger = logging.getLogger('mnemon')


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
        ) -> int:
    """Process insights where linked_at IS NULL.

    Creates semantic edges (and optionally LLM causal/enrichment edges)
    for pending insights. Returns the number of insights processed.
    """
    rows = db._conn.execute(
        'SELECT id FROM insights'
        ' WHERE linked_at IS NULL AND deleted_at IS NULL'
        ' ORDER BY created_at ASC'
        f' LIMIT {max_batch}'
        ).fetchall()
    if not rows:
        return 0

    if embed_cache is None:
        embed_cache = build_embed_cache(db)

    now = format_timestamp(datetime.now(timezone.utc))
    processed = 0

    for (insight_id,) in rows:
        insight = get_insight_by_id(db, insight_id)
        if insight is None:
            continue

        from mnemon.graph.enrichment import enrich_with_llm
        enrichment = enrich_with_llm(insight, llm_client)

        keywords = enrichment.get('keywords', [])
        new_vec = None
        if (keywords
                and embed_client is not None
                and embed_client.available()):
            from mnemon.graph.enrichment import build_enriched_text
            enriched_text = build_enriched_text(
                insight.content, keywords)
            try:
                new_vec = embed_client.embed(enriched_text)
            except Exception:
                logger.debug(f'Re-embed failed for {insight.id}')

        from mnemon.graph.causal import infer_llm_causal_edges
        causal_edges = infer_llm_causal_edges(
            db, insight, llm_client)

        def _write_results() -> int:
            if enrichment:
                from mnemon.store.node import update_enrichment
                update_enrichment(
                    db, insight.id,
                    enrichment.get('keywords', []),
                    enrichment.get('summary', ''),
                    enrichment.get('semantic_facts', []))
                update_entities(
                    db, insight.id, enrichment.get('entities', []))
                insight.entities = enrichment.get('entities', [])

            if new_vec is not None:
                from mnemon.embed.vector import serialize_vector
                from mnemon.store.node import update_embedding
                embed_cache[insight.id] = new_vec
                update_embedding(
                    db, insight.id, serialize_vector(new_vec))

            db._conn.execute(
                "DELETE FROM edges WHERE (source_id = ? OR target_id = ?)"
                " AND edge_type = 'entity'"
                " AND (json_extract(metadata, '$.created_by') IS NULL"
                "      OR json_extract(metadata, '$.created_by')"
                "         NOT IN ('claude', 'manual'))",
                (insight.id, insight.id))
            create_entity_edges(db, insight)

            db._conn.execute(
                "DELETE FROM edges WHERE (source_id = ? OR target_id = ?)"
                " AND edge_type = 'semantic'"
                " AND (json_extract(metadata, '$.created_by') IS NULL"
                "      OR json_extract(metadata, '$.created_by') = 'auto')",
                (insight.id, insight.id))
            sem_count = create_semantic_edges(
                db, insight, embed_cache)

            db._conn.execute(
                "DELETE FROM edges WHERE (source_id = ? OR target_id = ?)"
                " AND edge_type = 'causal'"
                " AND json_extract(metadata, '$.created_by') = 'llm'",
                (insight.id, insight.id))
            for edge in causal_edges:
                try:
                    insert_edge(db, edge)
                except Exception:
                    pass

            db._conn.execute(
                'UPDATE insights SET linked_at = ? WHERE id = ?',
                (now, insight_id))
            if enrichment:
                db._conn.execute(
                    'UPDATE insights SET enriched_at = ? WHERE id = ?',
                    (now, insight_id))
            return sem_count

        semantic_count = db.in_transaction(_write_results)
        processed += 1
        logger.debug(
            f'Linked {insight_id}: {semantic_count} semantic edges')

    return processed


def compute_constants_hash() -> str:
    """Return a short SHA-256 hash of all edge-relevant constants."""
    blob = json.dumps({
        'auto_semantic_threshold': AUTO_SEMANTIC_THRESHOLD,
        'min_proximity_weight': MIN_PROXIMITY_WEIGHT,
        'temporal_window_hours': TEMPORAL_WINDOW_HOURS,
        'max_entity_links': MAX_ENTITY_LINKS,
        'max_total_entity_edges': MAX_TOTAL_ENTITY_EDGES,
        'max_proximity_edges': MAX_PROXIMITY_EDGES,
        }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def relink_auto_edges(
        db: 'DB', dry_run: bool = False) -> dict[str, int]:
    """Delete auto-created edges and re-create semantic/entity edges.

    Heuristic causal edges are deleted (replaced by LLM in Tier 3).
    LLM/manual causal edges are preserved.
    """
    from mnemon.store.oplog import log_op

    semantic_del = db._conn.execute(
        "SELECT COUNT(*) FROM edges"
        " WHERE edge_type = 'semantic'"
        " AND json_extract(metadata, '$.created_by') = 'auto'"
        ).fetchone()[0]
    entity_del = db._conn.execute(
        "SELECT COUNT(*) FROM edges"
        " WHERE edge_type = 'entity'"
        " AND (json_extract(metadata, '$.created_by') IS NULL"
        "      OR json_extract(metadata, '$.created_by') <> 'claude')"
        ).fetchone()[0]
    temporal_del = db._conn.execute(
        "SELECT COUNT(*) FROM edges"
        " WHERE edge_type = 'temporal'"
        " AND json_extract(metadata, '$.sub_type') = 'proximity'"
        " AND weight < ?",
        (MIN_PROXIMITY_WEIGHT,)).fetchone()[0]
    causal_del = db._conn.execute(
        "SELECT COUNT(*) FROM edges"
        " WHERE edge_type = 'causal'"
        " AND (json_extract(metadata, '$.created_by') IS NULL"
        "      OR json_extract(metadata, '$.created_by')"
        "         NOT IN ('llm', 'claude', 'manual'))"
        ).fetchone()[0]

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
        db._conn.execute(
            "DELETE FROM edges"
            " WHERE edge_type = 'semantic'"
            " AND json_extract(metadata, '$.created_by') = 'auto'")
        stats['semantic_deleted'] = semantic_del

        db._conn.execute(
            "DELETE FROM edges"
            " WHERE edge_type = 'entity'"
            " AND (json_extract(metadata, '$.created_by') IS NULL"
            "      OR json_extract(metadata, '$.created_by') <> 'claude')")
        stats['entity_deleted'] = entity_del

        db._conn.execute(
            "DELETE FROM edges"
            " WHERE edge_type = 'temporal'"
            " AND json_extract(metadata, '$.sub_type') = 'proximity'"
            " AND weight < ?",
            (MIN_PROXIMITY_WEIGHT,))
        stats['temporal_pruned'] = temporal_del

        db._conn.execute(
            "DELETE FROM edges"
            " WHERE edge_type = 'causal'"
            " AND (json_extract(metadata, '$.created_by') IS NULL"
            "      OR json_extract(metadata, '$.created_by')"
            "         NOT IN ('llm', 'claude', 'manual'))")
        stats['causal_deleted'] = causal_del

        manual_entity_rows = db._conn.execute(
            "SELECT source_id, target_id, edge_type, weight,"
            " metadata, created_at FROM edges"
            " WHERE edge_type = 'entity'"
            " AND json_extract(metadata, '$.created_by') = 'claude'"
            ).fetchall()

        insights = get_all_active_insights(db)
        if not insights:
            return

        embed_cache = build_embed_cache(db)

        for insight in insights:
            stats['entity_created'] += create_entity_edges(
                db, insight)
            stats['semantic_created'] += create_semantic_edges(
                db, insight, embed_cache)

        stats['entity_created'] = db._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'entity'"
            " AND (json_extract(metadata, '$.created_by') IS NULL"
            "      OR json_extract(metadata, '$.created_by') <> 'claude')"
            ).fetchone()[0]
        stats['semantic_created'] = db._conn.execute(
            "SELECT COUNT(*) FROM edges"
            " WHERE edge_type = 'semantic'"
            " AND json_extract(metadata, '$.created_by') = 'auto'"
            ).fetchone()[0]

        for row in manual_entity_rows:
            db._conn.execute(
                'UPDATE edges SET metadata = ?,'
                ' weight = MAX(weight, ?)'
                ' WHERE source_id = ? AND target_id = ?'
                ' AND edge_type = ?',
                (row[4], row[3], row[0], row[1], row[2]))
            if db._conn.execute(
                    'SELECT COUNT(*) FROM edges'
                    ' WHERE source_id = ? AND target_id = ?'
                    ' AND edge_type = ?',
                    (row[0], row[1], row[2])).fetchone()[0] == 0:
                insert_edge(db, Edge(
                    source_id=row[0], target_id=row[1],
                    edge_type=row[2], weight=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    created_at=parse_timestamp(row[5])))

        db._conn.execute(
            'UPDATE insights SET linked_at = NULL'
            ' WHERE deleted_at IS NULL')

        log_op(db, 'relink', '', json.dumps(stats))

    db.in_transaction(tx_body)
    return stats
