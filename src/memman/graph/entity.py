"""Entity edge creation via IDF-weighted co-occurrence."""

import math

from memman.store.backend import Backend
from memman.store.model import Edge, Insight

MAX_ENTITY_LINKS = 5
MAX_TOTAL_ENTITY_EDGES = 50


def normalize_entity(e: str) -> str:
    """Normalize entity for matching: lowercase + strip."""
    return e.strip().lower()


def entity_idf_weight(doc_freq: int, total_docs: int) -> float:
    """Compute IDF-based weight for an entity edge."""
    if total_docs <= 1 or doc_freq >= total_docs:
        return 0.0
    if doc_freq <= 0:
        return 1.0
    raw = math.log(total_docs / doc_freq) / math.log(total_docs)
    return max(raw, 0.1)


def create_entity_edges(
        backend: Backend, insight: Insight, dry_run: bool = False) -> int:
    """Create entity co-occurrence edges between the insight and existing insights."""
    if not insight.entities:
        return 0

    total_docs = backend.nodes.count_active()
    use_idf = total_docs > 5

    count = 0

    for entity in insight.entities:
        if count >= MAX_TOTAL_ENTITY_EDGES:
            break
        ids = backend.edges.find_with_entity(
            entity, exclude_id=insight.id, limit=MAX_ENTITY_LINKS)
        if not ids:
            continue

        if use_idf:
            doc_freq = backend.edges.count_with_entity(
                entity, exclude_id=insight.id) + 1
            weight = entity_idf_weight(doc_freq, total_docs)
            if weight == 0.0:
                continue
        else:
            weight = 1.0

        for target_id in ids:
            if count >= MAX_TOTAL_ENTITY_EDGES:
                break
            if not dry_run:
                backend.edges.upsert(Edge(
                    source_id=insight.id, target_id=target_id,
                    edge_type='entity', weight=weight,
                    metadata={'entity': entity}))
            count += 1
            if not dry_run:
                backend.edges.upsert(Edge(
                    source_id=target_id, target_id=insight.id,
                    edge_type='entity', weight=weight,
                    metadata={'entity': entity}))
            count += 1

    return count
