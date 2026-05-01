"""Intent-aware recall with beam search, RRF, Kahn's topological sort.

Opens a `RecallSession` via `Backend.recall_session()` which reads
the worker-materialized snapshot when present. Falls back to direct
Backend verb calls when the snapshot is missing,
fingerprint-mismatched, or unreadable. The snapshot path eliminates
`nodes.get_all_active`, `iter_embeddings_as_vecs`, `edges.by_node`,
and `nodes.get` calls from the synchronous recall hot path.
"""

import heapq
import logging
from collections.abc import Callable
from typing import Any

from memman.embed.vector import cosine_similarity
from memman.search.intent import detect_intent, get_weights
from memman.search.keyword import insight_tokens, keyword_search, tokenize
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')

ANCHOR_TOP_K = 30
LAMBDA1 = 1.0
LAMBDA2 = 0.4
RRF_K = 60
VECTOR_SEARCH_MIN_SIM = 0.10
RERANK_SHORTLIST = 100
MIN_RERANK_TOKENS = 2

TRAVERSAL_PARAMS: dict[str, tuple[int, int, int]] = {
    'WHY': (15, 5, 500),
    'WHEN': (10, 5, 400),
    'ENTITY': (10, 4, 400),
    'GENERAL': (10, 4, 500),
    }

RERANK_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    'WHY':     (0.15, 0.10, 0.45, 0.30),
    'WHEN':    (0.20, 0.10, 0.40, 0.30),
    'ENTITY':  (0.20, 0.35, 0.35, 0.10),
    'GENERAL': (0.25, 0.15, 0.45, 0.15),
    }


RECALL_HINTS: dict[str, str] = {
    'WHY': 'Trace the causal chain: earlier results cause later ones',
    'WHEN': 'Results are newest-first: reconstruct the timeline',
    'ENTITY': 'Describe the entity using evidence across these memories',
    'GENERAL': 'Synthesize key points across these related memories',
    }


def get_traversal_params(intent: str) -> tuple[int, int, int]:
    """Return (beam_width, max_depth, max_visited) for the given intent."""
    return TRAVERSAL_PARAMS.get(intent, TRAVERSAL_PARAMS['GENERAL'])


def vector_search_from_cache(
        embed_cache: dict[str, list[float]],
        query_vec: list[float],
        limit: int) -> list[tuple[str, float]]:
    """Cosine similarity search over pre-loaded embeddings."""
    heap_list: list[tuple[float, str]] = []
    for id, vec in embed_cache.items():
        sim = cosine_similarity(query_vec, vec)
        if sim <= VECTOR_SEARCH_MIN_SIM:
            continue
        if limit <= 0 or len(heap_list) < limit:
            heapq.heappush(heap_list, (sim, id))
        elif sim > heap_list[0][0]:
            heapq.heapreplace(heap_list, (sim, id))

    if not heap_list:
        return []

    result = []
    while heap_list:
        sim, id = heapq.heappop(heap_list)
        result.append((id, sim))
    result.reverse()
    return result


def _bidirectional_adjacency(
        directed: dict[str, list[tuple[str, str, float]]],
        ) -> dict[str, list[tuple[str, str, float]]]:
    """Mirror a directional source -> targets map into both directions.

    Beam search walks edges as undirected; the snapshot stores them
    keyed by source. This helper materializes the reverse direction so
    `nid -> incoming + outgoing` is one dict lookup.
    """
    bidir: dict[str, list[tuple[str, str, float]]] = {}
    for source_id, edges in directed.items():
        bidir.setdefault(source_id, []).extend(edges)
        for target_id, etype, weight in edges:
            bidir.setdefault(target_id, []).append(
                (source_id, etype, weight))
    return bidir


def beam_search_from_anchor(
        start_id: str,
        start_score: float,
        weights: dict[str, float],
        params: tuple[int, int, int],
        score_map: dict[str, float],
        via_map: dict[str, str],
        insight_map: dict[str, Insight],
        sim_cache: dict[str, float] | None,
        edges_lookup,
        insight_lookup) -> None:
    """Perform beam search from a single anchor node.

    `edges_lookup(nid) -> iterable of (neighbor_id, edge_type, weight)`
    `insight_lookup(nid) -> Insight | None`
    Both lookups encapsulate either a snapshot dict access or a SQL
    query so callers don't branch on data source.
    """
    beam_width, max_depth, max_visited = params
    visited = {start_id: True}
    total_visited = 1

    current = [(-start_score, start_id, 0)]

    for depth in range(max_depth):
        if not current or total_visited >= max_visited:
            break

        next_items: list[tuple[float, str, int]] = []

        for neg_score, nid, _d in current:
            cur_score = -neg_score

            for neighbor_id, etype, weight in edges_lookup(nid):
                if total_visited >= max_visited:
                    break

                structural = weights.get(etype, 0.0) * weight
                semantic = (
                    sim_cache.get(neighbor_id, 0.0)
                    if sim_cache is not None else 0.0)
                neighbor_score = (
                    cur_score + LAMBDA1 * structural
                    + LAMBDA2 * semantic)

                existing = score_map.get(neighbor_id)
                if existing is None or neighbor_score > existing:
                    score_map[neighbor_id] = neighbor_score
                    via_map[neighbor_id] = etype
                    if neighbor_id not in insight_map:
                        ins = insight_lookup(neighbor_id)
                        if ins is not None:
                            insight_map[neighbor_id] = ins

                if neighbor_id not in visited:
                    visited[neighbor_id] = True
                    total_visited += 1
                    heapq.heappush(
                        next_items,
                        (-neighbor_score, neighbor_id, depth + 1))

        pruned = []
        count = 0
        while next_items and count < beam_width:
            item = heapq.heappop(next_items)
            pruned.append(item)
            count += 1
        current = pruned


def causal_topological_sort(
        results: list[dict[str, Any]],
        causal_edges_lookup: Callable[[str], list[str]]
        ) -> list[dict[str, Any]]:
    """Reorder results so causes appear before effects using Kahn's algorithm.

    `causal_edges_lookup(source_id) -> iterable of target_ids` exposes
    only the source-keyed causal edges, since this sort treats edges as
    strictly directional.
    """
    if len(results) <= 1:
        return results

    id_set = {r['insight'].id for r in results}
    id_to_result = {r['insight'].id: r for r in results}

    adj: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {r['insight'].id: 0 for r in results}

    for r in results:
        rid = r['insight'].id
        for target_id in causal_edges_lookup(rid):
            if target_id in id_set:
                adj.setdefault(rid, []).append(target_id)
                in_degree[target_id] += 1

    heap_list: list[tuple[float, str]] = []
    for r in results:
        rid = r['insight'].id
        if in_degree[rid] == 0:
            heapq.heappush(
                heap_list, (-id_to_result[rid]['score'], rid))

    ordered = []
    while heap_list:
        _neg_score, nid = heapq.heappop(heap_list)
        ordered.append(id_to_result[nid])
        for target in adj.get(nid, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                heapq.heappush(
                    heap_list,
                    (-id_to_result[target]['score'], target))

    if len(ordered) < len(results):
        covered = {r['insight'].id for r in ordered}
        ordered.extend(r for r in results if r['insight'].id not in covered)

    return ordered


def intent_aware_recall(
        backend: Backend, query: str,
        query_vec: list[float] | None,
        query_entities: list[str],
        limit: int,
        intent_override: str | None = None,
        rerank: bool = False) -> dict:
    """Perform MAGMA-aligned intent-aware retrieval.

    Loads the worker-materialized snapshot when present and consumes
    it for all hot-path reads. Falls back to direct Backend verb
    calls when the snapshot is missing or its embedding fingerprint
    doesn't match the active client.

    When `rerank=True` and the query has more than `MIN_RERANK_TOKENS`
    tokens, the top `RERANK_SHORTLIST` candidates by multi-signal score
    are re-scored by Voyage rerank-2.5-lite and the rerank score
    replaces the final ordering. On reranker failure the baseline
    ordering is preserved.
    """
    if intent_override:
        intent = intent_override
        intent_source = 'override'
    else:
        intent = detect_intent(query)
        intent_source = 'auto'

    weights = get_weights(intent)
    params = get_traversal_params(intent)

    with backend.recall_session() as session:
        snapshot = session.snapshot

        if snapshot is not None:
            all_insights = snapshot.insights
            embed_cache = snapshot.embeddings
            bidir = _bidirectional_adjacency(snapshot.adjacency)
            insights_by_id = {i.id: i for i in snapshot.insights}

            def _edges_lookup(nid):
                return bidir.get(nid, ())

            def _insight_lookup(nid):
                return insights_by_id.get(nid)

            def _causal_edges_lookup(source_id):
                return [
                    target for target, etype, _w
                    in snapshot.adjacency.get(source_id, ())
                    if etype == 'causal']
        else:
            all_insights = backend.nodes.get_all_active()
            embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())

            def _edges_lookup(nid):
                for e in backend.edges.by_node(nid):
                    neighbor_id = (
                        e.target_id if e.target_id != nid else e.source_id)
                    yield (neighbor_id, e.edge_type, e.weight)

            def _insight_lookup(nid):
                return backend.nodes.get(nid)

            def _causal_edges_lookup(source_id):
                return [
                    e.target_id
                    for e in backend.edges.by_source_and_type(
                        source_id, 'causal')]

    sim_cache: dict[str, float] = {}
    if query_vec is not None and embed_cache:
        for eid, vec in embed_cache.items():
            s = cosine_similarity(query_vec, vec)
            if s > 0:
                sim_cache[eid] = s

    anchor_map: dict[str, tuple[Insight, float, str]] = {}

    token_cache: dict[str, set[str]] = {}
    keyword_anchors = keyword_search(
        all_insights, query, ANCHOR_TOP_K, token_cache)
    for rank, (ins, _score) in enumerate(keyword_anchors):
        anchor_map[ins.id] = (
            ins, 1.0 / (RRF_K + rank + 1), 'keyword')

    if query_vec is not None and embed_cache:
        vector_hits = vector_search_from_cache(
            embed_cache, query_vec, ANCHOR_TOP_K)
        for rank, (vid, _sim) in enumerate(vector_hits):
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if vid in anchor_map:
                ins, old_score, _via = anchor_map[vid]
                anchor_map[vid] = (
                    ins, old_score + rrf_score, 'hybrid')
            else:
                ins = _insight_lookup(vid)
                if ins is not None:
                    anchor_map[vid] = (ins, rrf_score, 'vector')

    time_sorted = sorted(
        all_insights, key=lambda i: i.created_at, reverse=True)
    time_limit = min(ANCHOR_TOP_K, len(time_sorted))
    for rank in range(time_limit):
        ins = time_sorted[rank]
        rrf_score = 1.0 / (RRF_K + rank + 1)
        if ins.id in anchor_map:
            a_ins, old_score, old_via = anchor_map[ins.id]
            new_via = old_via
            if old_via in {'keyword', 'vector'}:
                new_via = 'hybrid'
            anchor_map[ins.id] = (
                a_ins, old_score + rrf_score, new_via)
        else:
            anchor_map[ins.id] = (ins, rrf_score, 'time')

    max_anchor_score = max(
        (s for _, s, _ in anchor_map.values()), default=0)
    if max_anchor_score > 0:
        anchor_map = {
            k: (ins, s / max_anchor_score, via)
            for k, (ins, s, via) in anchor_map.items()
            }

    anchor_count = len(anchor_map)
    if anchor_count == 0:
        logger.warning(
            f'Zero anchors: all_insights={len(all_insights)}, '
            f'query={query[:80]}')

    score_map: dict[str, float] = {}
    via_map: dict[str, str] = {}
    insight_map: dict[str, Insight] = {}

    for aid, (ins, score, via) in anchor_map.items():
        score_map[aid] = score
        via_map[aid] = via
        insight_map[aid] = ins

    for aid, (ins, score, via) in anchor_map.items():
        beam_search_from_anchor(
            aid, score, weights, params,
            score_map, via_map, insight_map, sim_cache,
            _edges_lookup, _insight_lookup)

    traversed_count = len(score_map)

    query_tokens = tokenize(query)
    query_entity_set = {e.lower() for e in query_entities}

    candidates: list[dict[str, Any]] = []
    graph_min = None
    graph_max = None
    for cid, graph_raw in score_map.items():
        ins = insight_map.get(cid)
        if ins is None:
            continue
        if graph_min is None:
            graph_min = graph_raw
            graph_max = graph_raw
        else:
            graph_min = min(graph_min, graph_raw)
            graph_max = max(graph_max, graph_raw)
        candidates.append({
            'id': cid, 'ins': ins, 'via': via_map.get(cid, ''),
            'graph_raw': graph_raw,
            })

    if graph_min is None:
        graph_min = 0.0
        graph_max = 0.0
    graph_range = graph_max - graph_min
    if graph_range == 0:
        graph_range = 1.0

    for c in candidates:
        kw_score = 0.0
        if query_tokens:
            ct = token_cache.get(c['id'])
            if ct is None:
                ct = insight_tokens(c['ins'])
            intersection = sum(1 for t in query_tokens if t in ct)
            kw_score = intersection / len(query_tokens)

        ent_score = 0.0
        if query_entity_set:
            matched = sum(
                1 for e in c['ins'].entities
                if e.lower() in query_entity_set)
            ent_score = matched / max(1, len(query_entity_set))

        sim_score = 0.0
        if sim_cache is not None:
            sim_score = sim_cache.get(c['id'], 0.0)

        graph_score = (c['graph_raw'] - graph_min) / graph_range

        c['kw_score'] = kw_score
        c['ent_score'] = ent_score
        c['sim_score'] = sim_score
        c['graph_score'] = graph_score

    w_kw, w_ent, w_sim, w_gr = RERANK_WEIGHTS.get(
        intent, RERANK_WEIGHTS['GENERAL'])

    results: list[dict[str, Any]] = []
    for c in candidates:
        final_score = (
            w_kw * c['kw_score'] + w_ent * c['ent_score']
            + w_sim * c['sim_score'] + w_gr * c['graph_score'])
        results.append({
            'insight': c['ins'],
            'score': final_score,
            'intent': intent,
            'via': c['via'],
            'signals': {
                'keyword': c['kw_score'],
                'entity': c['ent_score'],
                'similarity': c['sim_score'],
                'graph': c['graph_score'],
                },
            })

    results.sort(
        key=lambda r: (-r['score'], -r['insight'].importance))

    reranked = False
    if rerank and len(query.split()) > MIN_RERANK_TOKENS:
        shortlist_size = min(RERANK_SHORTLIST, len(results))
        if shortlist_size >= 2:
            try:
                from memman.rerank import get_client as get_rerank_client
                rerank_client = get_rerank_client()
                shortlist = results[:shortlist_size]
                docs = [r['insight'].content for r in shortlist]
                scored = rerank_client.rerank(
                    query, docs, top_k=shortlist_size)
                reordered = []
                for orig_idx, score in scored:
                    r = shortlist[orig_idx]
                    r['score'] = float(score)
                    r['signals']['rerank'] = float(score)
                    reordered.append(r)
                results = reordered + results[shortlist_size:]
                reranked = True
            except Exception as exc:
                logger.warning(
                    f'rerank failed, keeping baseline ordering: {exc}')

    if limit > 0 and len(results) > limit:
        results = results[:limit]

    if intent == 'WHY':
        results = causal_topological_sort(results, _causal_edges_lookup)
    elif intent == 'WHEN':
        results.sort(
            key=lambda r: (r['insight'].created_at, r['score']),
            reverse=True)

    sparse = not results or (limit > 0 and len(results) < limit // 2)

    if intent == 'WHY':
        ordering = 'causal_topological'
    elif intent == 'WHEN':
        ordering = 'chronological'
    else:
        ordering = 'score'

    meta = {
        'intent': intent,
        'intent_source': intent_source,
        'anchor_count': anchor_count,
        'traversed': traversed_count,
        'hint': RECALL_HINTS.get(intent, RECALL_HINTS['GENERAL']),
        'ordering': ordering,
        'reranked': reranked,
        }
    if sparse:
        meta['sparse'] = True

    return {'results': results, 'meta': meta}
