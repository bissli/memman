"""Temporal edge creation (backbone chain + proximity)."""

from memman.store.backend import Backend
from memman.store.model import Edge, Insight

TEMPORAL_WINDOW_HOURS = 4.0
MAX_PROXIMITY_EDGES = 5
MIN_PROXIMITY_WEIGHT = 0.10


def create_temporal_edge(backend: Backend, insight: Insight) -> int:
    """Create backbone and proximity temporal edges for a new insight.

    Re-reads `insight.created_at` from the backend so the proximity
    weight comparison uses the server-stamped value, not whatever was
    on the in-memory dataclass when the caller built it. This makes
    proximity edges deterministic across writers in the presence of
    clock skew (Phase 1a Decision #1: backends own time).
    """
    stored = backend.nodes.get(insight.id)
    if stored is not None and stored.created_at is not None:
        insight = Insight(
            **{**insight.__dict__,
               'created_at': stored.created_at,
               'updated_at': stored.updated_at,
               })

    count = 0

    prev = backend.nodes.get_latest_by_source(
        source=insight.source, exclude_id=insight.id)
    if prev is not None:
        try:
            backend.edges.upsert(Edge(
                source_id=prev.id, target_id=insight.id,
                edge_type='temporal', weight=1.0,
                metadata={'sub_type': 'backbone', 'direction': 'precedes'}))
            count += 1
        except Exception:
            pass
        try:
            backend.edges.upsert(Edge(
                source_id=insight.id, target_id=prev.id,
                edge_type='temporal', weight=1.0,
                metadata={'sub_type': 'backbone', 'direction': 'succeeds'}))
            count += 1
        except Exception:
            pass

    recent = backend.nodes.get_recent_in_window(
        exclude_id=insight.id, window_hours=TEMPORAL_WINDOW_HOURS,
        limit=MAX_PROXIMITY_EDGES)
    if not recent:
        return count

    backbone_id = prev.id if prev else ''

    for near in recent:
        if near.id == backbone_id:
            continue

        hours_diff = abs(
            (insight.created_at - near.created_at).total_seconds() / 3600)
        weight = 1.0 / (1.0 + hours_diff)
        if weight < MIN_PROXIMITY_WEIGHT:
            continue

        meta = {
            'sub_type': 'proximity',
            'hours_diff': f'{hours_diff:.2f}',
            }
        try:
            backend.edges.upsert(Edge(
                source_id=insight.id, target_id=near.id,
                edge_type='temporal', weight=weight,
                metadata=meta))
            count += 1
        except Exception:
            pass
        try:
            backend.edges.upsert(Edge(
                source_id=near.id, target_id=insight.id,
                edge_type='temporal', weight=weight,
                metadata=meta))
            count += 1
        except Exception:
            pass

    return count
