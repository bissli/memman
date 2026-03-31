"""Graph engine: orchestrates automatic edge creation when insights are stored."""

import hashlib
import json
import logging
from datetime import datetime, timezone

from mnemon.graph.causal import CAUSAL_LOOKBACK, create_causal_edges
from mnemon.graph.entity import ACRONYM_STOPWORDS, ENTITY_PATTERNS
from mnemon.graph.entity import ENTITY_STOPWORDS, MAX_ENTITY_LINKS
from mnemon.graph.entity import MAX_TOTAL_ENTITY_EDGES, TECH_DICTIONARY
from mnemon.graph.entity import create_entity_edges, extract_entities
from mnemon.graph.entity import merge_entities
from mnemon.graph.semantic import AUTO_SEMANTIC_THRESHOLD, build_embed_cache
from mnemon.graph.semantic import create_semantic_edges
from mnemon.graph.temporal import MAX_PROXIMITY_EDGES
from mnemon.graph.temporal import MIN_PROXIMITY_WEIGHT, TEMPORAL_WINDOW_HOURS
from mnemon.graph.temporal import create_temporal_edge
from mnemon.model import Insight, format_timestamp
from mnemon.store.node import get_all_active_insights, get_insight_by_id
from mnemon.store.node import update_entities

logger = logging.getLogger('mnemon')


def _prepare_entities(insight: Insight) -> None:
    """Extract, merge, and filter entities for an insight in place."""
    extracted = extract_entities(insight.content)
    insight.entities = merge_entities(insight.entities, extracted)
    insight.entities = [
        e for e in insight.entities
        if e not in ENTITY_STOPWORDS
        and e not in ACRONYM_STOPWORDS]


def fast_edges(db: 'DB', insight: Insight) -> dict[str, int]:
    """Run cheap edge generators only (temporal + entity + causal).

    Semantic edges are deferred to consolidate_pending().
    """
    _prepare_entities(insight)

    return {
        'temporal': create_temporal_edge(db, insight),
        'entity': create_entity_edges(db, insight),
        'causal': create_causal_edges(db, insight),
        }


MAX_CONSOLIDATION_BATCH = 20


def consolidate_pending(
        db: 'DB',
        embed_cache: dict[str, list[float]] | None = None,
        llm_client: object | None = None,
        max_batch: int = MAX_CONSOLIDATION_BATCH,
        ) -> int:
    """Process insights where consolidated_at IS NULL.

    Creates semantic edges (and optionally LLM causal edges) for pending
    insights. Returns the number of insights processed.
    """
    rows = db._conn.execute(
        'SELECT id FROM insights'
        ' WHERE consolidated_at IS NULL AND deleted_at IS NULL'
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

        _prepare_entities(insight)
        semantic_count = create_semantic_edges(db, insight, embed_cache)

        if llm_client is not None:
            from mnemon.graph.causal import create_llm_causal_edges
            create_llm_causal_edges(db, insight, llm_client)

        db._conn.execute(
            'UPDATE insights SET consolidated_at = ? WHERE id = ?',
            (now, insight_id))
        processed += 1
        logger.debug(
            f'Consolidated {insight_id}: {semantic_count} semantic edges')

    return processed


def compute_constants_hash() -> str:
    """Return a short SHA-256 hash of all edge-relevant constants."""
    blob = json.dumps({
        'auto_semantic_threshold': AUTO_SEMANTIC_THRESHOLD,
        'entity_patterns': [p.pattern for p in ENTITY_PATTERNS],
        'acronym_stopwords': sorted(ACRONYM_STOPWORDS),
        'entity_stopwords': sorted(ENTITY_STOPWORDS),
        'tech_dictionary': sorted(TECH_DICTIONARY),
        'causal_lookback': CAUSAL_LOOKBACK,
        'min_proximity_weight': MIN_PROXIMITY_WEIGHT,
        'temporal_window_hours': TEMPORAL_WINDOW_HOURS,
        'max_entity_links': MAX_ENTITY_LINKS,
        'max_total_entity_edges': MAX_TOTAL_ENTITY_EDGES,
        'max_proximity_edges': MAX_PROXIMITY_EDGES,
        }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def rebuild_auto_edges(
        db: 'DB', dry_run: bool = False) -> dict[str, int]:
    """Delete auto-created edges and re-create semantic/entity/causal edges."""
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

    if dry_run:
        stats = {
            'semantic_deleted': semantic_del,
            'entity_deleted': entity_del,
            'temporal_pruned': temporal_del,
            'semantic_created': 0,
            'entity_created': 0,
            'causal_created': 0,
            'dry_run': 1,
            }
        insights = get_all_active_insights(db)
        if insights:
            embed_cache = build_embed_cache(db)
            for insight in insights:
                _prepare_entities(insight)
                stats['entity_created'] += create_entity_edges(
                    db, insight, dry_run=True)
                stats['semantic_created'] += create_semantic_edges(
                    db, insight, embed_cache, dry_run=True)
                stats['causal_created'] += create_causal_edges(
                    db, insight, dry_run=True)
        return stats

    stats: dict[str, int] = {
        'semantic_deleted': 0,
        'entity_deleted': 0,
        'temporal_pruned': 0,
        'semantic_created': 0,
        'entity_created': 0,
        'causal_created': 0,
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
            _prepare_entities(insight)
            update_entities(db, insight.id, insight.entities)
            stats['entity_created'] += create_entity_edges(
                db, insight)
            stats['semantic_created'] += create_semantic_edges(
                db, insight, embed_cache)
            stats['causal_created'] += create_causal_edges(
                db, insight)

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
                'INSERT OR REPLACE INTO edges'
                ' (source_id, target_id, edge_type, weight,'
                '  metadata, created_at)'
                ' VALUES (?, ?, ?, ?, ?, ?)', row)

        now = format_timestamp(datetime.now(timezone.utc))
        db._conn.execute(
            'UPDATE insights SET consolidated_at = ?'
            ' WHERE deleted_at IS NULL', (now,))

        log_op(db, 'rebuild', '', json.dumps(stats))

    db.in_transaction(tx_body)
    return stats
