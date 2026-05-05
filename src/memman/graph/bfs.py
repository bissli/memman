"""Breadth-first graph traversal.

Wraps `EdgeStore.get_neighborhood` so callers can keep the
existing dict-shaped result and `BFSOptions` ergonomics. The
underlying traversal lives on the Backend Protocol; Postgres
implements it as a recursive CTE.
"""

from dataclasses import dataclass
from typing import Any

from memman.store.backend import Backend


@dataclass
class BFSOptions:
    """Controls BFS traversal behavior."""

    max_depth: int = 2
    max_nodes: int = 0
    edge_filter: str = ''


def bfs(backend: Backend, start_id: str,
        opts: BFSOptions) -> list[dict[str, Any]]:
    """Bounded BFS from start_id.

    Returns dicts with keys `insight`, `hop`, `via_edge`. The
    bounded neighborhood comes from
    `backend.edges.get_neighborhood`; insights are hydrated via
    `backend.nodes.get_many`. Soft-deleted neighbors are dropped.
    """
    triples = backend.edges.get_neighborhood(
        start_id, depth=opts.max_depth,
        edge_filter=opts.edge_filter)
    if opts.max_nodes > 0 and len(triples) > opts.max_nodes:
        triples = triples[:opts.max_nodes]
    if not triples:
        return []
    insights = {
        ins.id: ins for ins in
        backend.nodes.get_many([t[0] for t in triples])
        }
    return [
        {'insight': insights[nid], 'hop': hop, 'via_edge': etype}
        for nid, hop, etype in triples
        if nid in insights
        ]
