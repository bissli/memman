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
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from memman import trace
from memman.embed.vector import cosine_similarity
from memman.search.intent import detect_intent, get_weights
from memman.search.keyword import insight_tokens, keyword_search, tokenize
from memman.store.backend import Backend
from memman.store.model import Insight

if TYPE_CHECKING:
    from memman.embed.fingerprint import Fingerprint

logger = logging.getLogger('memman')

ANCHOR_TOP_K = 30
LAMBDA1 = 1.0
LAMBDA2 = 0.4
RRF_K = 60
VECTOR_SEARCH_MIN_SIM = 0.10
RERANK_SHORTLIST = 100
MIN_RERANK_TOKENS = 2

# Notes:
# - MMR_LAMBDA weighs relevance against the one-shot diversity
#   penalty: score = lam * rel - (1 - lam) * max pool similarity;
#   1.0 disables the term.
# - 1.0 is the MEASURED value, not a default: the recall_ablation
#   mmr sweep (2026-08-05, lambda in {0.5..0.9} x rerank on/off, 12
#   queries, search-store sandbox) showed that under the default
#   cross-encoder rerank the final output is byte-identical at 0.9
#   and within 1.6 redundancy points at 0.5, while rerank-off gains
#   (0.655 -> 0.504 redundancy at 0.5) rewrite most of the top-10
#   with no relevance oracle to certify them. See
#   experiments/recall_ablation/README.md for the sweep record.
# - MMR_POOL bounds the gram matrix (O(n^2)); it must exceed
#   RERANK_SHORTLIST or MMR provably cannot change shortlist
#   membership whenever rerank is on (the default).
MMR_LAMBDA = 1.0
MMR_POOL = 200

TRAVERSAL_PARAMS: dict[str, tuple[int, int, int]] = {
    'WHY': (15, 5, 500),
    'WHEN': (10, 5, 400),
    'ENTITY': (10, 4, 400),
    'GENERAL': (10, 4, 500),
    }

# Notes:
# - `(w_kw, w_sim, w_gr)` per intent, each raw row divided by its own
#   sum, so every shipped row sums to 1.0 and `score` carries one
#   range at every intent. Both hold only to within a float ulp:
#   WHEN sums to 0.9999999999999999, and `sim_score` is an unclamped
#   cosine that can return 1 + 1 ulp, so the true bound is
#   [0, 1 + 4.5e-16]. One range is not one meaning - the mix behind
#   a 0.7 still differs per intent - and `graph_score` is min-max
#   normalized over the query's own pool, so no score compares
#   across queries at any intent.
# - The raw rows are the pre-0.23.0 four-weight table with `w_ent`
#   deleted and the survivors untouched, which left them summing to
#   WHY 0.90, WHEN 0.90, ENTITY 0.65, GENERAL 0.85. The shipped rows
#   inherit their DIRECTION from that table; only the scale is new.
#   They are not a measured optimum: the sweeps recorded under
#   `experiments/quality_matrix/results/sweep_rerank/` flag the
#   shipped arm as beaten by the grid peak in all six WHEN and WHY
#   runs, and ENTITY and GENERAL have no grid at all.
# - The division is computed rather than written out because no
#   quotient here has an exact float literal - 0.45/0.90 is 1/2, yet
#   computes to 0.5000000000000001, because the raw row sums to
#   0.8999999999999999. A rounded literal turns a row as well as
#   scaling it;
#   `experiments/recall_ablation/verify_weight_rounding.py` prices
#   that turn in returned positions.
# - `score` is NOT comparable across the rerank shortlist boundary.
#   When the pool exceeds `RERANK_SHORTLIST` the cross-encoder
#   overwrites the head's scores while the tail keeps these blended
#   values; a smaller pool is overwritten whole, leaving no tail.
_RERANK_WEIGHTS_RAW: dict[str, tuple[float, float, float]] = {
    'WHY':     (0.15, 0.45, 0.30),
    'WHEN':    (0.20, 0.40, 0.30),
    'ENTITY':  (0.20, 0.35, 0.10),
    'GENERAL': (0.25, 0.45, 0.15),
    }

RERANK_WEIGHTS: dict[str, tuple[float, float, float]] = {
    intent: (kw / (kw + sim + gr),
             sim / (kw + sim + gr),
             gr / (kw + sim + gr))
    for intent, (kw, sim, gr) in _RERANK_WEIGHTS_RAW.items()
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
        edges_lookup: Callable[[str], Any],
        insight_lookup: Callable[[str], Insight | None]) -> int:
    """Perform beam search from a single anchor node.

    Parameters
    ----------
    start_id : str
        Anchor insight id the traversal starts from.
    start_score : float
        Anchor's fused RRF score; seeds the running path score.
    weights : dict[str, float]
        Intent-adaptive weight per edge type.
    params : tuple[int, int, int]
        `(beam_width, max_depth, max_visited)` for the intent.
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
        `nid -> iterable of (neighbor_id, edge_type, weight)`;
        encapsulates snapshot dict access or a SQL query.
    insight_lookup : Callable
        `nid -> Insight | None`; same encapsulation.

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
    return total_visited


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
        limit: int, *,
        fingerprint: 'Fingerprint',
        intent_override: str | None = None,
        rerank: bool = False,
        rerank_weights_override: dict[
            str, tuple[float, float, float]] | None = None,
        category: str = '',
        source: str = '',
        min_score: float = 0.0,
        ) -> dict[str, Any]:
    """Perform MAGMA-aligned intent-aware retrieval.

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
    fingerprint : Fingerprint
        Active embedding fingerprint; gates snapshot reuse.
    intent_override : str | None, default None
        Force an intent instead of `detect_intent(query)`.
    rerank : bool, default False
        Re-score the shortlist with the cross-encoder (see Notes).
    rerank_weights_override : dict | None, default None
        Read the intent's `(w_kw, w_sim, w_gr)` from this dict
        instead of module-level `RERANK_WEIGHTS`.
    category : str, default ''
        Keep only insights with this exact category ('' = no filter).
    source : str, default ''
        Keep only insights with this exact source ('' = no filter).
    min_score : float, default 0.0
        Relevance floor on `kw_score + sim_score`; 0.0 = no filter.

    Returns
    -------
    dict[str, Any]
        `{'results': [...], 'meta': {...}}`; `meta.anchor_count` is
        the filtered anchor count, `meta.traversed` is deliberately
        unfiltered, and `meta.sparse` flags a low-confidence result
        set (see Notes).

    Notes
    -----
    - Loads the worker-materialized snapshot when present; falls back
      to direct Backend verb calls when it is missing or its embedding
      fingerprint does not match.
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
      candidates by multi-signal score are re-scored by Voyage
      rerank-2.5-lite; the filter runs before the rerank block so the
      shortlist holds only returnable rows. On reranker failure the
      baseline ordering is preserved.
    - `min_score` thresholds `kw_score + sim_score`, never the blended
      score: `graph_score` is min-max normalized, so the top candidate
      of any query scores 1.0 there and a blended floor would sit at
      `w_gr`, which moves per intent. Its range is therefore 0.0-2.0.
    - `meta.sparse` marks a low-confidence result set. It fires on an
      empty set, on fewer than `limit // 2` rows, and when no
      candidate matched a query token -- the case an unscoped query
      hits, where the recency-anchor channel returns newest-first rows
      that match nothing. The token test reads the candidate pool as
      scored, before `category`/`source` filtering, `min_score`, MMR,
      rerank and the limit slice, so a recall whose returned rows were
      reached by graph from a match is not called irrelevant.
    - The keyword channel alone carries that last arm, and no
      similarity term belongs in it. `sim_cache` holds a cosine only
      when it is strictly positive and the lookup defaults to 0.0, so
      an exactly-zero similarity means either that the row carries no
      embedding or that its cosine was non-positive -- never that a
      real relevance was measured and found small. The matching and
      non-matching populations also overlap on similarity, so no floor
      separates them. Measured by
      `experiments/recall_ablation/verify_sparse_rule.py`.
    """
    if intent_override:
        intent = intent_override
        intent_source = 'override'
    else:
        intent = detect_intent(query)
        intent_source = 'auto'

    weights = get_weights(intent)
    params = get_traversal_params(intent)
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

    with backend.recall_session(fingerprint) as session:
        snapshot = session.snapshot

        if snapshot is not None:
            all_insights = snapshot.insights
            embed_cache = snapshot.embeddings
            bidir = _bidirectional_adjacency(snapshot.adjacency)
            insights_by_id = {i.id: i for i in snapshot.insights}

            def _edges_lookup(nid: str) -> Any:
                return bidir.get(nid, ())

            def _insight_lookup(nid: str) -> Insight | None:
                return insights_by_id.get(nid)

            def _causal_edges_lookup(source_id: str) -> list[str]:
                return [
                    target for target, etype, _w
                    in snapshot.adjacency.get(source_id, ())
                    if etype == 'causal']
        else:
            all_insights = backend.nodes.get_all_active()
            embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())
            try:
                session._embed_cache = embed_cache  # type: ignore[attr-defined]
                session._meta_cache = {  # type: ignore[attr-defined]
                    i.id: (i.category, i.source) for i in all_insights}
            except AttributeError:
                pass

            def _edges_lookup(nid: str) -> Any:
                for e in backend.edges.by_node(nid):
                    neighbor_id = (
                        e.target_id if e.target_id != nid else e.source_id)
                    yield (neighbor_id, e.edge_type, e.weight)

            def _insight_lookup(nid: str) -> Insight | None:
                return backend.nodes.get(nid)

            def _causal_edges_lookup(source_id: str) -> list[str]:
                return [
                    e.target_id
                    for e in backend.edges.by_source_and_type(
                        source_id, 'causal')]

        if query_vec is not None:
            try:
                vector_hits = session.vector_anchors(
                    query_vec, k=anchor_k,
                    min_sim=VECTOR_SEARCH_MIN_SIM,
                    category=category, source=source)
            except Exception as exc:
                logger.warning(
                    f'session.vector_anchors failed, falling back: {exc}')
                vector_hits = []
        else:
            vector_hits = []

    sim_cache: dict[str, float] = {}
    if query_vec is not None and embed_cache:
        for eid, vec in embed_cache.items():
            s = cosine_similarity(query_vec, vec)
            if s > 0:
                sim_cache[eid] = s

    # Anchor selection draws from the filtered pool; traversal keeps
    # the full `insights_by_id` so hops through non-matching rows work.
    anchor_pool = (all_insights if not (category or source)
                   else [i for i in all_insights if _matches(i)])

    anchor_map: dict[str, tuple[Insight, float, str]] = {}

    token_cache: dict[str, set[str]] = {}
    keyword_anchors = keyword_search(
        anchor_pool, query, anchor_k, token_cache)
    for rank, (ins, _score) in enumerate(keyword_anchors):
        anchor_map[ins.id] = (
            ins, 1.0 / (RRF_K + rank + 1), 'keyword')

    if vector_hits and not embed_cache:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())

    if query_vec is not None and not vector_hits and embed_cache:
        # Filter the INPUT dict, never the returned hits: a post-cut
        # filter reproduces the under-fill this change removes.
        anchor_embed_cache = embed_cache
        if category or source:
            eligible_ids = {i.id for i in anchor_pool}
            anchor_embed_cache = {
                eid: vec for eid, vec in embed_cache.items()
                if eid in eligible_ids}
        vector_hits = vector_search_from_cache(
            anchor_embed_cache, query_vec, anchor_k)

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
            intent=intent,
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
            aid, score, weights, params,
            score_map, via_map, insight_map, sim_cache,
            _edges_lookup, _insight_lookup)
        visited_total += visited
        if visited >= params[2]:
            capped_anchors += 1

    traversed_count = len(score_map)
    if enabled:
        trace.event(
            'recall_traversal',
            visited=visited_total,
            capped_anchors=capped_anchors,
            max_visited=params[2],
            traversed=traversed_count)

    query_tokens = tokenize(query)

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
            ct = token_cache.get(c['id'])
            if ct is None:
                ct = insight_tokens(c['ins'])
            intersection = sum(1 for t in query_tokens if t in ct)
            kw_score = intersection / len(query_tokens)

        sim_score = 0.0
        if sim_cache is not None:
            sim_score = sim_cache.get(c['id'], 0.0)

        graph_score = (c['graph_raw'] - graph_min) / graph_range

        c['kw_score'] = kw_score
        c['sim_score'] = sim_score
        c['graph_score'] = graph_score

    rerank_table = (rerank_weights_override
                    if rerank_weights_override is not None
                    else RERANK_WEIGHTS)
    w_kw, w_sim, w_gr = rerank_table.get(
        intent, RERANK_WEIGHTS['GENERAL'])

    results: list[dict[str, Any]] = []
    for c in candidates:
        final_score = (
            w_kw * c['kw_score'] + w_sim * c['sim_score']
            + w_gr * c['graph_score'])
        results.append({
            'insight': c['ins'],
            'score': final_score,
            'intent': intent,
            'via': c['via'],
            'signals': {
                'keyword': c['kw_score'],
                'similarity': c['sim_score'],
                'graph': c['graph_score'],
                },
            })

    results.sort(
        key=lambda r: (-r['score'], -r['insight'].importance))

    # Read the keyword evidence off the UNFILTERED pool: a category or
    # source filter can drop the row that matched and keep the rows it
    # reached by graph, and judging relevance on the survivors alone
    # would then call a working filtered recall irrelevant.
    pool_matched_a_token = any(
        r['signals']['keyword'] > 0.0 for r in results)

    # Filter after the weighted-sum sort (so graph_min/graph_max
    # normalisation saw the full pool) and BEFORE rerank (so the
    # cross-encoder shortlist holds only returnable rows).
    if category or source:
        results = [r for r in results if _matches(r['insight'])]

    if min_score > 0.0:
        results = [
            r for r in results
            if r['signals']['keyword'] + r['signals']['similarity']
            >= min_score]

    # One-shot MMR diversity: score every candidate once against the
    # whole pool, then sort once -- NOT greedy iterative MMR (an
    # O(k*n) selection loop and a different algorithm). Runs between
    # the filter (only returnable rows) and the rerank block (so it
    # can change shortlist membership).
    if MMR_LAMBDA < 1.0 and len(results) > 1 and embed_cache:
        pool = results[:MMR_POOL]
        vec_rows = [
            (i, embed_cache[r['insight'].id])
            for i, r in enumerate(pool)
            if r['insight'].id in embed_cache]
        if len(vec_rows) > 1:
            # Ragged dims (mid-model-swap, a partial reembed, or a
            # short blob) would make np.array raise; off-modal rows
            # join the unembedded set and hold their positions.
            modal_dim = Counter(
                len(v) for _, v in vec_rows).most_common(1)[0][0]
            vec_rows = [
                (i, v) for i, v in vec_rows if len(v) == modal_dim]
        if len(vec_rows) > 1:
            mx = np.array([v for _, v in vec_rows], dtype=np.float64)
            norms = np.linalg.norm(mx, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            unit = mx / norms
            gram = unit @ unit.T
            # Zero the diagonal so a candidate's self-similarity is
            # excluded from its own max.
            np.fill_diagonal(gram, 0.0)
            max_sim = gram.max(axis=1)
            # Only embedded rows are re-sorted, each holding one of
            # the slots the embedded set occupied; a vector-less row
            # keeps its relevance position. Scoring it instead would
            # hand it a zero penalty -- the maximum diversity bonus
            # -- and float exactly the degraded rows to the head.
            embedded = [idx for idx, _v in vec_rows]
            mmr_by_idx = {
                idx: (MMR_LAMBDA * pool[idx]['score']
                      - (1.0 - MMR_LAMBDA) * float(ms))
                for (idx, _v), ms in zip(vec_rows, max_sim)}
            reordered_iter = iter(sorted(
                embedded, key=lambda i: mmr_by_idx[i], reverse=True))
            embedded_set = set(embedded)
            pool = [
                pool[next(reordered_iter)] if i in embedded_set
                else pool[i]
                for i in range(len(pool))]
            results = pool + results[MMR_POOL:]

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

    if intent == 'WHY':
        results = causal_topological_sort(results, _causal_edges_lookup)
    elif intent == 'WHEN':
        results.sort(
            key=lambda r: (r['insight'].created_at, r['score']),
            reverse=True)

    sparse = (
        not results
        or (limit > 0 and len(results) < limit // 2)
        or not pool_matched_a_token)

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
