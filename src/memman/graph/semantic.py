"""Semantic edge creation and candidate discovery.

Callers pass a per-(provider, model, surface) calibrated value resolved
via `memman.graph.engine._resolve_semantic_threshold`. Passing
`threshold=None` short-circuits to zero edges, which is how uncalibrated
fingerprints opt out of auto-semantic-edge creation.
"""

from memman.embed.vector import cosine_similarity
from memman.store.backend import Backend
from memman.store.model import Edge, Insight, format_float

MAX_AUTO_SEMANTIC_EDGES = 3


def create_semantic_edges(
        backend: Backend, insight: Insight,
        embed_cache: dict[str, list[float]] | None = None,
        dry_run: bool = False,
        threshold: float | None = None) -> int:
    """Auto-create semantic edges for insights with high cosine similarity.

    `threshold` is the cosine cutoff. `None` (the default) means the
    calling fingerprint is uncalibrated; the function short-circuits to
    zero edges rather than apply a wrong-model default.
    """
    if threshold is None:
        return 0
    if embed_cache is None:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())
    if not embed_cache:
        return 0

    insight_vec = embed_cache.get(insight.id)
    if insight_vec is None:
        return 0

    scored = []
    for eid, other_vec in embed_cache.items():
        if eid == insight.id:
            continue
        cos_sim = cosine_similarity(insight_vec, other_vec)
        if cos_sim >= threshold:
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
            backend.edges.upsert(Edge(
                source_id=insight.id, target_id=eid,
                edge_type='semantic', weight=sim,
                metadata=meta))
        count += 1
        if not dry_run:
            backend.edges.upsert(Edge(
                source_id=eid, target_id=insight.id,
                edge_type='semantic', weight=sim,
                metadata=meta))
        count += 1

    return count
