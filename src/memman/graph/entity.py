"""Entity edge creation via IDF-weighted co-occurrence."""

import math
from datetime import datetime, timezone

from memman.model import Edge, Insight
from memman.store.edge import count_insights_with_entity
from memman.store.edge import find_insights_with_entity, insert_edge

MAX_ENTITY_LINKS = 5
MAX_TOTAL_ENTITY_EDGES = 50


def entity_idf_weight(doc_freq: int, total_docs: int) -> float:
    """Compute IDF-based weight for an entity edge."""
    if total_docs <= 1 or doc_freq >= total_docs:
        return 0.0
    if doc_freq <= 0:
        return 1.0
    raw = math.log(total_docs / doc_freq) / math.log(total_docs)
    return max(raw, 0.1)


def create_entity_edges(
        db: 'DB', insight: Insight, dry_run: bool = False) -> int:
    """Create entity co-occurrence edges between the insight and existing insights."""
    if not insight.entities:
        return 0

    from memman.store.node import count_active_insights
    total_docs = count_active_insights(db)
    use_idf = total_docs > 5

    now = datetime.now(timezone.utc)
    count = 0

    for entity in insight.entities:
        if count >= MAX_TOTAL_ENTITY_EDGES:
            break
        ids = find_insights_with_entity(
            db, entity, insight.id, MAX_ENTITY_LINKS)
        if not ids:
            continue

        if use_idf:
            doc_freq = count_insights_with_entity(
                db, entity, insight.id) + 1
            weight = entity_idf_weight(doc_freq, total_docs)
            if weight == 0.0:
                continue
        else:
            weight = 1.0

        for target_id in ids:
            if count >= MAX_TOTAL_ENTITY_EDGES:
                break
            if not dry_run:
                try:
                    insert_edge(db, Edge(
                        source_id=insight.id, target_id=target_id,
                        edge_type='entity', weight=weight,
                        metadata={'entity': entity}, created_at=now))
                except Exception:
                    pass
            count += 1
            if not dry_run:
                try:
                    insert_edge(db, Edge(
                        source_id=target_id, target_id=insight.id,
                        edge_type='entity', weight=weight,
                        metadata={'entity': entity}, created_at=now))
                except Exception:
                    pass
            count += 1

    return count
