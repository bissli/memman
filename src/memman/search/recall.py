"""Recall with beam search and RRF fusion.

Reads live storage on every request. The candidate universe is
`nodes.get_all_active()` and the graph is one `edges.adjacency()`
read, so recall cannot serve a row the store has deleted or
superseded, and cannot miss one it holds as current.

Notes
-----
- There is deliberately no derived read artifact on this path: a
  materialized copy needs its own staleness check, and without one
  recall serves deleted rows and hides live ones.
- Vector work stays behind `RecallSession`: `vector_anchors` for the
  top-k and `similarities` for the per-candidate cosine. This module
  never holds a whole-store embedding dict.
- Keyword work stays there too, behind `keyword_counts`. This module
  never tokenizes the store: the count comes back per id and fills
  `kw_score` directly.
"""

import heapq
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

from memman import trace
from memman.search.keyword import keyword_search, tokenize
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')

ANCHOR_TOP_K = 30
LAMBDA1 = 1.0
LAMBDA2 = 0.4
RRF_K = 60
RERANK_SHORTLIST = 100
MIN_RERANK_TOKENS = 2

# Notes:
# - The weight per edge type sums to 1.0. That fixes the STRUCTURAL
#   term's scale against the two terms it is summed with in a
#   traversal score, `anchor_rrf + LAMBDA1 * structural + LAMBDA2 *
#   semantic`, neither of which these weights touch.
# - The balance is load-bearing: scaling only the middle term is not
#   a transform the min-max normalization of `graph_score` undoes. It
#   re-ranks rows and changes which nodes the beam keeps at its cut,
#   so changing a weight here changes retrieved order.
# - The types weighted are exactly the edge types the store writes. A
#   type nothing mints would contribute nothing while still consuming
#   the 1.0 budget, shrinking every other type's share.
EDGE_WEIGHTS: dict[str, float] = {
    'temporal': 0.334, 'semantic': 0.333, 'entity': 0.333,
    }

# `(beam_width, max_depth, max_visited)` for every traversal.
TRAVERSAL_PARAMS: tuple[int, int, int] = (10, 4, 500)

# Notes:
# - `(w_kw, w_sim, w_gr)`: the raw row divided by its own sum, so the
#   weights sum to 1.0 and `score` spans one range. Both hold only to
#   within a float ulp, and `sim_score` is an unclamped cosine that
#   can exceed 1 by an ulp.
# - The raw row is the four-weight table's GENERAL row with `w_ent`
#   deleted and the survivors untouched. It carries that table's
#   direction and is not a measured optimum.
# - The division is computed rather than written out because no
#   quotient here has an exact float literal, and a rounded literal
#   turns the row as well as scaling it.
# - `score` is NOT comparable across the rerank shortlist boundary.
#   When the pool exceeds `RERANK_SHORTLIST` the cross-encoder
#   overwrites the head's scores while the tail keeps these blended
#   values; a smaller pool is overwritten whole, leaving no tail.
_RERANK_WEIGHTS_RAW: tuple[float, float, float] = (0.25, 0.45, 0.15)

RERANK_WEIGHTS: tuple[float, float, float] = tuple(
    weight / sum(_RERANK_WEIGHTS_RAW) for weight in _RERANK_WEIGHTS_RAW)


def _bidirectional_adjacency(
        directed: dict[str, list[tuple[str, str, float]]],
        ) -> dict[str, list[tuple[str, str, float]]]:
    """Mirror a directional source -> targets map into both directions.

    Beam search walks edges as undirected; `EdgeStore.adjacency()`
    returns them keyed by source. This helper materializes the reverse
    direction so `nid -> incoming + outgoing` is one dict lookup.
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
        edges_lookup: Callable[[str], Any],
        insight_lookup: Callable[[str], Insight | None],
        phantom_ids: set[str]) -> int:
    """Perform beam search from a single anchor node.

    Parameters
    ----------
    start_id : str
        Anchor insight id the traversal starts from.
    start_score : float
        Anchor's fused RRF score; seeds the running path score.
    weights : dict[str, float]
        Weight per edge type.
    params : tuple[int, int, int]
        `(beam_width, max_depth, max_visited)`.
    score_map : dict[str, float]
        Best path score per node; updated in place.
    via_map : dict[str, str]
        Edge type that produced each node's best score; updated in
        place.
    insight_map : dict[str, Insight]
        Node id -> Insight for every scored node; updated in place.
    sim_cache : dict[str, float] | None
        Query-cosine per node (None when there is no query vector).
    edges_lookup : Callable
        `nid -> iterable of (neighbor_id, edge_type, weight)`,
        read from the pre-built bidirectional adjacency map.
    insight_lookup : Callable
        `nid -> Insight | None`; same encapsulation.
    phantom_ids : set[str]
        Ids an edge referenced but `insight_lookup` could not
        resolve; shared across anchors and updated in place, so one
        dangling edge costs one lookup per recall rather than one per
        anchor.

    Returns
    -------
    int
        Nodes visited from this anchor (the anchor included); equal
        to `max_visited` when the traversal hit its budget.
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

                # An edge can outlive its endpoint row. Resolve the
                # neighbour first: scoring one that does not resolve
                # would spend a visit budget slot and a beam push on a
                # node no result can ever carry, and re-resolve the
                # same miss once per anchor.
                if neighbor_id in phantom_ids:
                    continue
                if neighbor_id not in insight_map:
                    ins = insight_lookup(neighbor_id)
                    if ins is None:
                        phantom_ids.add(neighbor_id)
                        continue
                    insight_map[neighbor_id] = ins

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
    return total_visited


def intent_aware_recall(
        backend: Backend, query: str,
        query_vec: list[float] | None,
        limit: int, *,
        rerank: bool = False,
        category: str = '',
        source: str = '',
        ) -> dict[str, Any]:
    """Rank the store's current rows against a query, best first.

    Parameters
    ----------
    backend : Backend
        Per-store handle exposing the verb surface.
    query : str
        Search text; tokenized for keyword anchors and scoring.
    query_vec : list[float] | None
        Query embedding; None degrades to the keyword/time paths.
    limit : int
        Result cap; `limit <= 0` means unbounded.
    rerank : bool, default False
        Re-score the shortlist with the cross-encoder (see Notes).
    category : str, default ''
        Keep only insights with this exact category ('' = no filter).
    source : str, default ''
        Keep only insights with this exact source ('' = no filter).

    Returns
    -------
    dict[str, Any]
        `{'results': [...], 'meta': {...}}`. Each result carries
        `insight`, `score`, `via` and `signals`; `meta` carries
        `anchor_count` (filtered), `traversed` (deliberately
        unfiltered) and `reranked`.

    Notes
    -----
    - Reads live storage on every call: the candidate universe is
      `nodes.get_all_active()` and the graph is one
      `edges.adjacency()` read, so a row the store has deleted or
      superseded cannot be returned and a row it holds as current
      cannot be hidden.
    - `category`/`source` filter the ANCHOR pools and the final result
      set, never the graph traversal: a hop through a non-matching
      neighbour is correct, and filtering the candidates loop would
      perturb the `graph_min`/`graph_max` normalisation.
    - With a filter and `limit > 0`, the anchor budget becomes
      `max(ANCHOR_TOP_K, limit)`; unfiltered recall keeps
      `ANCHOR_TOP_K` untouched so the ablation harness's
      `anchor_top_k` sweep is never overridden.
    - When `rerank=True` and the query has more than
      `MIN_RERANK_TOKENS` tokens, the top `RERANK_SHORTLIST`
      candidates by multi-signal score are re-scored by the
      configured Voyage reranker; the filter runs before the rerank
      block so the shortlist holds only returnable rows. On reranker
      failure the baseline ordering is preserved.
    - Rows come back in relevance order at every `limit`, so the
      first `n` of a `limit`-`m` recall are the `limit`-`n` recall.
      Nothing re-sorts after the limit slice.
    """
    # Hoisted once: `is_enabled` can fall through to a file read, so
    # calling it per event site is a hot-path regression.
    enabled = trace.is_enabled()

    def _matches(ins: Insight) -> bool:
        return ((not category or ins.category == category)
                and (not source or ins.source == source))

    # Notes:
    # - The `limit <= 0` half matters because a non-positive limit
    #   means unbounded at the slice below.
    # - A bare max() would silently override the ablation harness's
    #   anchor_top_k sweep on every unfiltered rerank config.
    anchor_k = (ANCHOR_TOP_K
                if (limit <= 0 or not (category or source))
                else max(ANCHOR_TOP_K, limit))

    all_insights = backend.nodes.get_all_active()
    insights_by_id = {i.id: i for i in all_insights}

    # Notes:
    # - One projection-only read of the whole edge table, not one
    #   query per frontier node. The per-node form re-reads each
    #   edge several times over and costs a psycopg round-trip
    #   apiece on Postgres.
    # - `adjacency()` skips `metadata`, whose per-row json.loads is
    #   the costliest part of the equivalent `edges.all()` and which
    #   traversal discards.
    bidir = _bidirectional_adjacency(backend.edges.adjacency())
    phantom_ids: set[str] = set()

    def _edges_lookup(nid: str) -> Any:
        return bidir.get(nid, ())

    def _insight_lookup(nid: str) -> Insight | None:
        return insights_by_id.get(nid)

    query_tokens = tokenize(query)

    sim_cache: dict[str, float] = {}
    keyword_counts: dict[str, int] = {}
    with backend.recall_session() as session:
        # Notes:
        # - Counted where the text lives -- one FTS5 probe per token
        #   on SQLite -- rather than tokenizing every active row
        #   per request.
        # - This IS `kw_score`'s numerator, so the scoring loop
        #   needs no whole-store token cache.
        try:
            keyword_counts = session.keyword_counts(query_tokens)
        except Exception as exc:
            logger.warning(
                f'session.keyword_counts failed, keyword signal'
                f' unavailable: {exc}')
        if query_vec is not None:
            # Scored where the vectors live: one matmul on SQLite, one
            # `embedding <=>` query on Postgres. The pipeline needs N
            # scalars, and pulling N x dim floats to compute them
            # would be a whole-store read per recall on both
            # backends.
            try:
                sim_cache = session.similarities(query_vec)
            except Exception as exc:
                logger.warning(
                    f'session.similarities failed, similarity signal'
                    f' unavailable: {exc}')
            try:
                vector_hits = session.vector_anchors(
                    query_vec, k=anchor_k,
                    category=category, source=source)
            except Exception as exc:
                logger.warning(
                    f'session.vector_anchors failed, no vector'
                    f' anchors this request: {exc}')
                vector_hits = []
        else:
            vector_hits = []

    # Anchor selection draws from the filtered pool; traversal keeps
    # the full `insights_by_id` so hops through non-matching rows work.
    anchor_pool = (all_insights if not (category or source)
                   else [i for i in all_insights if _matches(i)])

    anchor_map: dict[str, tuple[Insight, float, str]] = {}

    keyword_anchors = keyword_search(
        anchor_pool, query, anchor_k, keyword_counts)
    for rank, (ins, _score) in enumerate(keyword_anchors):
        anchor_map[ins.id] = (
            ins, 1.0 / (RRF_K + rank + 1), 'keyword')

    for rank, (vid, _sim) in enumerate(vector_hits):
        rrf_score = 1.0 / (RRF_K + rank + 1)
        if vid in anchor_map:
            ins, old_score, _via = anchor_map[vid]
            anchor_map[vid] = (
                ins, old_score + rrf_score, 'hybrid')
        else:
            looked = _insight_lookup(vid)
            if looked is not None:
                anchor_map[vid] = (looked, rrf_score, 'vector')

    time_sorted = sorted(
        anchor_pool, key=lambda i: i.created_at, reverse=True)
    time_limit = min(anchor_k, len(time_sorted))
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

    if enabled:
        # vector_hits against anchor_k is the measurement Phase 1
        # deferred: whether a selective --cat/--source filter makes
        # the vector scan return fewer than k anchors.
        trace.event(
            'recall_anchors',
            anchor_k=anchor_k,
            keyword_hits=len(keyword_anchors),
            vector_hits=len(vector_hits),
            time_hits=time_limit,
            fused_pool=anchor_count,
            via_counts=dict(Counter(
                via for _, _, via in anchor_map.values())),
            filtered=bool(category or source))

    score_map: dict[str, float] = {}
    via_map: dict[str, str] = {}
    insight_map: dict[str, Insight] = {}

    for aid, (ins, score, via) in anchor_map.items():
        score_map[aid] = score
        via_map[aid] = via
        insight_map[aid] = ins

    visited_total = 0
    capped_anchors = 0
    for aid, (ins, score, via) in anchor_map.items():
        visited = beam_search_from_anchor(
            aid, score, EDGE_WEIGHTS, TRAVERSAL_PARAMS,
            score_map, via_map, insight_map, sim_cache,
            _edges_lookup, _insight_lookup, phantom_ids)
        visited_total += visited
        if visited >= TRAVERSAL_PARAMS[2]:
            capped_anchors += 1

    traversed_count = len(score_map)
    if enabled:
        trace.event(
            'recall_traversal',
            visited=visited_total,
            capped_anchors=capped_anchors,
            max_visited=TRAVERSAL_PARAMS[2],
            traversed=traversed_count)

    candidates: list[dict[str, Any]] = []
    graph_min: float | None = None
    graph_max: float | None = None
    for cid, graph_raw in score_map.items():
        cid_ins = insight_map.get(cid)
        if cid_ins is None:
            continue
        if graph_min is None or graph_max is None:
            graph_min = graph_raw
            graph_max = graph_raw
        else:
            graph_min = min(graph_min, graph_raw)
            graph_max = max(graph_max, graph_raw)
        candidates.append({
            'id': cid, 'ins': cid_ins, 'via': via_map.get(cid, ''),
            'graph_raw': graph_raw,
            })

    if graph_min is None or graph_max is None:
        graph_min = 0.0
        graph_max = 0.0
    graph_range = graph_max - graph_min
    if graph_range == 0:
        graph_range = 1.0

    for c in candidates:
        kw_score = 0.0
        if query_tokens:
            kw_score = (keyword_counts.get(c['id'], 0)
                        / len(query_tokens))

        sim_score = sim_cache.get(c['id'], 0.0)

        graph_score = (c['graph_raw'] - graph_min) / graph_range

        c['kw_score'] = kw_score
        c['sim_score'] = sim_score
        c['graph_score'] = graph_score

    w_kw, w_sim, w_gr = RERANK_WEIGHTS

    results: list[dict[str, Any]] = []
    for c in candidates:
        final_score = (
            w_kw * c['kw_score'] + w_sim * c['sim_score']
            + w_gr * c['graph_score'])
        results.append({
            'insight': c['ins'],
            'score': final_score,
            'via': c['via'],
            'signals': {
                'keyword': c['kw_score'],
                'similarity': c['sim_score'],
                'graph': c['graph_score'],
                },
            })

    results.sort(
        key=lambda r: (-r['score'], -r['insight'].importance))

    # Filter after the weighted-sum sort (so graph_min/graph_max
    # normalisation saw the full pool) and BEFORE rerank (so the
    # cross-encoder shortlist holds only returnable rows).
    if category or source:
        results = [r for r in results if _matches(r['insight'])]

    reranked = False
    if rerank and len(query.split()) > MIN_RERANK_TOKENS:
        shortlist_size = min(RERANK_SHORTLIST, len(results))
        if shortlist_size >= 2:
            try:
                from memman.rerank import get_client as get_rerank_client
                rerank_client = get_rerank_client()
                shortlist = results[:shortlist_size]
                docs = [r['insight'].content for r in shortlist]
                before_ids = [r['insight'].id for r in shortlist]
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
                if enabled:
                    # Movement is diffed by ID: the reranker replaces
                    # every score, so a score diff always says "all
                    # moved" and could never justify or kill the
                    # cross-encoder.
                    moved = sum(
                        1 for bid, r in zip(before_ids, reordered)
                        if bid != r['insight'].id)
                    trace.event(
                        'recall_rerank',
                        shortlist=shortlist_size,
                        moved=moved)
            except Exception as exc:
                logger.warning(
                    f'rerank failed, keeping baseline ordering: {exc}')

    if limit > 0 and len(results) > limit:
        results = results[:limit]

    meta: dict[str, Any] = {
        'anchor_count': anchor_count,
        'traversed': traversed_count,
        'reranked': reranked,
        }

    return {'results': results, 'meta': meta}
