"""Recall over RRF-fused keyword, vector and recency anchors.

Reads live storage on every request. The candidate universe is
`nodes.get_all_active()`, so recall cannot serve a row the store has
deleted or superseded, and cannot miss one it holds as current.

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

import logging
from collections import Counter
from typing import Any

from memman import trace
from memman.search.keyword import keyword_search, tokenize
from memman.store.backend import Backend
from memman.store.model import Insight

logger = logging.getLogger('memman')

ANCHOR_TOP_K = 30
RRF_K = 60
RERANK_SHORTLIST = 100
MIN_RERANK_TOKENS = 2

# Notes:
# - `(w_kw, w_sim, w_anchor)`: the raw row divided by its own sum, so
#   the weights sum to 1.0 and `score` spans one range. Both hold only
#   to within a float ulp, and `sim_score` is an unclamped cosine that
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
        `anchor_count` and `reranked`.

    Notes
    -----
    - Reads live storage on every call: the candidate universe is
      `nodes.get_all_active()`, so a row the store has deleted or
      superseded cannot be returned and a row it holds as current
      cannot be hidden.
    - The candidates are exactly the fused anchors. `category` and
      `source` filter every channel before its cut, so each candidate
      is returnable.
    - The keyword and recency channels take `ANCHOR_TOP_K` rows each;
      with a filter and `limit > 0` that becomes
      `max(ANCHOR_TOP_K, limit)`. The vector channel takes at least
      `RERANK_SHORTLIST`, so the reranker sees up to a full shortlist
      of query neighbors.
    - `signals['anchor']` is the min-max of the fused RRF score: the
      one term that carries recency into `score`, so a recent row with
      no keyword or vector match still outranks an older one.
    - When `rerank=True` and the query has more than
      `MIN_RERANK_TOKENS` tokens, the top `RERANK_SHORTLIST`
      candidates by multi-signal score are re-scored by the
      configured Voyage reranker. On reranker failure the baseline
      ordering is preserved.
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
    vector_k = max(RERANK_SHORTLIST, anchor_k)

    all_insights = backend.nodes.get_all_active()
    insights_by_id = {i.id: i for i in all_insights}

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
                    query_vec, k=vector_k,
                    category=category, source=source)
            except Exception as exc:
                logger.warning(
                    f'session.vector_anchors failed, no vector'
                    f' anchors this request: {exc}')
                vector_hits = []
        else:
            vector_hits = []

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
            looked = insights_by_id.get(vid)
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
        # vector_hits against vector_k shows whether a selective
        # --cat/--source filter makes the vector scan return fewer
        # than k anchors.
        trace.event(
            'recall_anchors',
            anchor_k=anchor_k,
            vector_k=vector_k,
            keyword_hits=len(keyword_anchors),
            vector_hits=len(vector_hits),
            time_hits=time_limit,
            fused_pool=anchor_count,
            via_counts=dict(Counter(
                via for _, _, via in anchor_map.values())),
            filtered=bool(category or source))

    anchor_scores = [s for _, s, _ in anchor_map.values()]
    anchor_min = min(anchor_scores, default=0.0)
    anchor_range = max(anchor_scores, default=0.0) - anchor_min
    if anchor_range == 0:
        anchor_range = 1.0

    w_kw, w_sim, w_anchor = RERANK_WEIGHTS

    results: list[dict[str, Any]] = []
    for cid, (ins, anchor_raw, via) in anchor_map.items():
        kw_score = 0.0
        if query_tokens:
            kw_score = keyword_counts.get(cid, 0) / len(query_tokens)
        sim_score = sim_cache.get(cid, 0.0)
        anchor_score = (anchor_raw - anchor_min) / anchor_range
        results.append({
            'insight': ins,
            'score': (w_kw * kw_score + w_sim * sim_score
                      + w_anchor * anchor_score),
            'via': via,
            'signals': {
                'keyword': kw_score,
                'similarity': sim_score,
                'anchor': anchor_score,
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
        'reranked': reranked,
        }

    return {'results': results, 'meta': meta}
