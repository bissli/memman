"""Semantic edge creation and candidate discovery.

AUTO_SEMANTIC_THRESHOLD is model-specific. If the embedding model changes,
recalibrate by computing all pairwise cosine similarities and inspecting
quality at each band. See docs/design/04-graph-model.md.
"""

from memman.embed.vector import cosine_similarity, deserialize_vector
from memman.store.backend import Backend
from memman.store.model import Edge, Insight, format_float

AUTO_SEMANTIC_THRESHOLD = 0.62
MAX_AUTO_SEMANTIC_EDGES = 3


def build_embed_cache(
        backend: Backend) -> dict[str, list[float]] | None:
    """Load all embeddings from the backend into a map."""
    all_embedded = backend.nodes.get_all_embeddings()
    if not all_embedded:
        return None
    cache: dict[str, list[float]] = {}
    for eid, _content, blob in all_embedded:
        v = deserialize_vector(blob)
        if v is not None:
            cache[eid] = v
    return cache or None


def create_semantic_edges(
        backend: Backend, insight: Insight,
        embed_cache: dict[str, list[float]] | None = None,
        dry_run: bool = False) -> int:
    """Auto-create semantic edges for insights with high cosine similarity."""
    if embed_cache is None:
        embed_cache = build_embed_cache(backend)
    if embed_cache is None:
        return 0

    insight_vec = embed_cache.get(insight.id)
    if insight_vec is None:
        return 0

    scored = []
    for eid, other_vec in embed_cache.items():
        if eid == insight.id:
            continue
        cos_sim = cosine_similarity(insight_vec, other_vec)
        if cos_sim >= AUTO_SEMANTIC_THRESHOLD:
            scored.append((eid, cos_sim))

    if not scored:
        return 0

    scored.sort(key=lambda x: x[1], reverse=True)
    if len(scored) > MAX_AUTO_SEMANTIC_EDGES:
        scored = scored[:MAX_AUTO_SEMANTIC_EDGES]

    count = 0
    for eid, sim in scored:
        meta = {
            'created_by': 'auto',
            'cosine': format_float(sim),
            }
        if not dry_run:
            try:
                backend.edges.upsert(Edge(
                    source_id=insight.id, target_id=eid,
                    edge_type='semantic', weight=sim,
                    metadata=meta))
            except Exception:
                pass
        count += 1
        if not dry_run:
            try:
                backend.edges.upsert(Edge(
                    source_id=eid, target_id=insight.id,
                    edge_type='semantic', weight=sim,
                    metadata=meta))
            except Exception:
                pass
        count += 1

    return count
