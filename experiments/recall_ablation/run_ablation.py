"""Ablation sweep for memman recall quality.

Runs each query in queries.json through the selected configurations
of intent_aware_recall on a target store and emits per-(config,
query, rank) rows to results.csv. Each row also carries a
`redundancy` column: the mean over the returned rows of each row's
max pairwise cosine against the other returned rows -- the number a
diversity term (MMR) exists to reduce.

Configurations swept:
    baseline              ANCHOR_TOP_K=30, current weights (production)
    anchor_60             ANCHOR_TOP_K=60, current weights
    anchor_100            ANCHOR_TOP_K=100, current weights
    weights_v2            ANCHOR_TOP_K=30, retuned weights
    anchor_60_weights_v2  ANCHOR_TOP_K=60, retuned weights
    expand_only           ANCHOR_TOP_K=30, current weights, query expanded by LLM
    rerank_voyage         baseline + post-hoc Voyage rerank-2.5-lite on top-100
    mmr_l{NN}[_rerank]    baseline + one-shot MMR at lambda 0.NN,
                          with and without the post-hoc rerank
    mmr_after_l{NN}_rerank  the spec's alternative placement: rerank the
                          full top-100 shortlist FIRST, then the one-shot
                          MMR re-sort over the reranked list (reranker
                          relevance as the score term), then the limit
                          slice

Select a subset with `--configs name1,name2`; point `--data-dir` at a
sandbox copy to sweep without touching live stores.

`_WEIGHTS_V2_RAW` bumps WHY graph 0.30->0.45 and GENERAL similarity
0.45->0.55, with the other slots of those rows adjusted to compensate,
and keeps the ENTITY row's own retune (kw 0.20->0.15, sim 0.35->0.25).
Those deltas describe the RAW rows. Both that table and the shipped
one are then divided by each row's own sum by the same expression, so
every arm departs from production exactly where its raw row does and
the exported `WEIGHTS_V2` sums to 1.0 like the shipped table.
"""

import argparse
import contextlib
import csv
import json
import os
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from memman.embed import EmbeddingProvider
from memman.embed.fingerprint import Fingerprint, bound_embedder
from memman.embed.fingerprint import stored_fingerprint
from memman.embed.vector import cosine_similarity
from memman.search import recall as recall_mod
from memman.search.recall import intent_aware_recall
from memman.store.db import default_data_dir, open_read_only, store_dir
from memman.store.sqlite import SqliteBackend

# Notes:
# - `(w_kw, w_sim, w_gr)`, matching the shipped three-signal blend.
#   Each row is the historical four-slot candidate with the entity
#   slot dropped and the other three left untouched, so every arm
#   still departs from production exactly where it always did.
_WEIGHTS_V2_RAW: dict[str, tuple[float, float, float]] = {
    'WHY':     (0.10, 0.35, 0.45),
    'WHEN':    (0.20, 0.40, 0.30),
    'ENTITY':  (0.15, 0.25, 0.10),
    'GENERAL': (0.20, 0.55, 0.15),
    }

WEIGHTS_V2: dict[str, tuple[float, float, float]] = {
    intent: (kw / (kw + sim + gr),
             sim / (kw + sim + gr),
             gr / (kw + sim + gr))
    for intent, (kw, sim, gr) in _WEIGHTS_V2_RAW.items()
    }


MMR_SWEEP = (0.5, 0.7, 0.8, 0.9)

CONFIGS = [
    {'name': 'baseline',             'anchor_top_k':  30, 'weights': None},
    {'name': 'anchor_60',            'anchor_top_k':  60, 'weights': None},
    {'name': 'anchor_100',           'anchor_top_k': 100, 'weights': None},
    {'name': 'weights_v2',           'anchor_top_k':  30, 'weights': WEIGHTS_V2},
    {'name': 'anchor_60_weights_v2', 'anchor_top_k':  60, 'weights': WEIGHTS_V2},
    {'name': 'expand_only',          'anchor_top_k':  30, 'weights': None,
     'expand': True},
    {'name': 'rerank_voyage',        'anchor_top_k':  30, 'weights': None,
     'rerank': True},
    ]
CONFIGS += [
    cfg
    for _lam in MMR_SWEEP
    for cfg in (
        {'name': f'mmr_l{int(_lam * 100):02d}', 'anchor_top_k': 30,
         'weights': None, 'mmr_lambda': _lam},
        {'name': f'mmr_l{int(_lam * 100):02d}_rerank',
         'anchor_top_k': 30, 'weights': None, 'mmr_lambda': _lam,
         'rerank': True},
        {'name': f'mmr_after_l{int(_lam * 100):02d}_rerank',
         'anchor_top_k': 30, 'weights': None, 'rerank': True,
         'mmr_after_rerank': _lam},
        )]


@contextlib.contextmanager
def overridden(anchor_top_k: int | None, weights: dict | None,
               mmr_lambda: float | None = None) -> Iterator[None]:
    """Temporarily monkey-patch recall module constants.
    """
    saved_atk = recall_mod.ANCHOR_TOP_K
    saved_w = dict(recall_mod.RERANK_WEIGHTS)
    saved_lam = recall_mod.MMR_LAMBDA
    if anchor_top_k is not None:
        recall_mod.ANCHOR_TOP_K = anchor_top_k
    if weights is not None:
        recall_mod.RERANK_WEIGHTS = weights
    if mmr_lambda is not None:
        recall_mod.MMR_LAMBDA = mmr_lambda
    try:
        yield
    finally:
        recall_mod.ANCHOR_TOP_K = saved_atk
        recall_mod.RERANK_WEIGHTS = saved_w
        recall_mod.MMR_LAMBDA = saved_lam


def voyage_rerank(query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
    """Call Voyage rerank-2.5-lite. Returns list of (orig_index, score).
    """
    key = os.environ['VOYAGE_API_KEY']
    body = {'model': 'rerank-2.5-lite', 'query': query,
            'documents': docs, 'top_k': top_k}
    req = urllib.request.Request(
        'https://api.voyageai.com/v1/rerank',
        data=json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {key}',
                 'Content-Type': 'application/json'},
        method='POST')
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return [(d['index'], d['relevance_score']) for d in resp['data']]


def expand_via_llm(query: str) -> str:
    """Call expand_query once for the expand_only config. Returns the
    expanded query string, or the original on failure.
    """
    try:
        from memman.llm.client import get_llm_client
        from memman.llm.extract import expand_query
        client = get_llm_client('fast')
        result = expand_query(client, query)
        return result.get('expanded_query', query)
    except Exception as exc:
        print(f'  WARN expand_only: LLM expansion failed ({exc}); '
              f'using original query')
        return query


def run_one(backend: SqliteBackend, fingerprint: Fingerprint,
            query: str, expanded_query: str,
            query_vec_cache: dict, embed_client: EmbeddingProvider,
            config: dict, limit: int, embeddings: dict) -> list[dict]:
    """Run a single (query, config) and return top-`limit` result rows.
    """
    use_query = expanded_query if config.get('expand') else query
    if use_query not in query_vec_cache:
        query_vec_cache[use_query] = embed_client.embed(use_query)
    qvec = query_vec_cache[use_query]

    fetch_limit = 200 if config.get('rerank') else limit

    with overridden(config['anchor_top_k'], config['weights'],
                    config.get('mmr_lambda')):
        t0 = time.perf_counter()
        resp = intent_aware_recall(
            backend, use_query, qvec, fetch_limit,
            fingerprint=fingerprint)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    results = resp['results']

    if config.get('rerank'):
        shortlist = results[:100]
        after_lam = config.get('mmr_after_rerank')
        if len(shortlist) >= 2:
            docs = [r['insight'].content for r in shortlist]
            try:
                t_r = time.perf_counter()
                # The after-rerank placement needs the WHOLE reranked
                # shortlist (membership of the final slice may
                # change), and the reranker's relevance scores.
                top_k = len(docs) if after_lam is not None else limit
                scored = voyage_rerank(use_query, docs, top_k=top_k)
                elapsed_ms += (time.perf_counter() - t_r) * 1000
                results = [shortlist[i] for i, _ in scored]
                if after_lam is not None and len(results) > 1:
                    # Notes:
                    # - The spec's alternative placement: one-shot
                    #   MMR over the RERANKED list, before the limit
                    #   slice.
                    # - Mirrors production semantics (unembedded
                    #   rows hold their slots) so the two placements
                    #   are comparable; the relevance term is the
                    #   reranker score, the only one available here.
                    rr_scores = [s for _, s in scored]
                    embedded = [
                        i for i, r in enumerate(results)
                        if r['insight'].id in embeddings]
                    if len(embedded) > 1:
                        vec_by_idx = {
                            i: embeddings[results[i]['insight'].id]
                            for i in embedded}
                        mmr_by_idx = {}
                        for i in embedded:
                            max_sim = max(
                                cosine_similarity(
                                    vec_by_idx[i], vec_by_idx[j])
                                for j in embedded if j != i)
                            mmr_by_idx[i] = (
                                after_lam * rr_scores[i]
                                - (1.0 - after_lam) * max_sim)
                        reordered = iter(sorted(
                            embedded, key=lambda i: mmr_by_idx[i],
                            reverse=True))
                        embedded_set = set(embedded)
                        results = [
                            results[next(reordered)]
                            if i in embedded_set else results[i]
                            for i in range(len(results))]
            except Exception as exc:
                print(f'  WARN rerank_voyage: rerank call failed '
                      f'({exc}); using unranked top-K')
                results = shortlist[:limit]
        else:
            results = shortlist

    results = results[:limit]
    redundancy = _redundancy(
        [r['insight'].id for r in results], embeddings)
    redundancy_cell = (
        round(redundancy, 4) if redundancy is not None else '')
    rows = []
    for rank, r in enumerate(results, start=1):
        ins = r['insight']
        rows.append({
            'config': config['name'],
            'query': query,
            'rank': rank,
            'insight_id': ins.id,
            'score': round(r.get('score', 0.0), 6),
            'intent': r.get('intent', ''),
            'via': r.get('via', ''),
            'elapsed_ms': round(elapsed_ms, 1),
            'redundancy': redundancy_cell,
            'content': ins.content[:140].replace('\n', ' '),
            })
    return rows


def _redundancy(ids: list[str], embeddings: dict) -> float | None:
    """Mean over rows of each row's max pairwise cosine to the rest.

    The number a diversity term exists to reduce: 1.0 means every
    returned row has a near-duplicate in the same result set.
    Returns None (an empty CSV cell) when fewer than two returned
    rows are embedded -- "unmeasurable" must stay distinguishable
    from a measured 0.0, or partial-coverage stores drag a config's
    mean toward the best possible value.
    """
    vecs = [embeddings[i] for i in ids if i in embeddings]
    if len(vecs) < 2:
        return None
    per_row_max = []
    for i, vi in enumerate(vecs):
        best = max(
            cosine_similarity(vi, vj)
            for j, vj in enumerate(vecs) if j != i)
        per_row_max.append(best)
    return sum(per_row_max) / len(per_row_max)


def jaccard(a: list[str], b: list[str]) -> float:
    """Set-overlap ratio of two id lists (1.0 when both are empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def main() -> None:
    """Run the sweep: recall per (query, config), csv + summaries."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', default='search')
    ap.add_argument('--data-dir', default=default_data_dir())
    ap.add_argument('--queries', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument(
        '--configs', default='',
        help='comma-separated config names (default: all)')
    args = ap.parse_args()

    here = Path(__file__).parent
    queries_path = Path(args.queries) if args.queries else here / 'queries.json'
    out_path = Path(args.out) if args.out else here / 'results.csv'

    wanted = {n.strip() for n in args.configs.split(',') if n.strip()}
    configs = [c for c in CONFIGS if not wanted or c['name'] in wanted]

    with Path(queries_path).open() as f:
        queries = json.load(f)

    sdir = store_dir(args.data_dir, args.store)
    print(f'opening store at {sdir}')
    backend = SqliteBackend(open_read_only(sdir))
    fingerprint = stored_fingerprint(backend)
    embed_client = bound_embedder(backend)
    print(f'embed: {embed_client.name} model={embed_client.model} '
          f'dim={embed_client.dim}')
    embeddings = dict(backend.nodes.iter_embeddings_as_vecs())

    query_vec_cache: dict[str, list[float]] = {}
    expanded_cache: dict[str, str] = {}

    if any(c.get('expand') for c in configs):
        print('pre-computing LLM query expansion (one call per unique query)...')
        for q in queries:
            expanded_cache[q['query']] = expand_via_llm(q['query'])

    all_rows = []
    for q in queries:
        query = q['query']
        kind = q.get('kind', '')
        expected_intent = q.get('expected_intent', '')
        print(f'\n=== [{kind}] {query!r}  expected_intent={expected_intent}')
        for cfg in configs:
            try:
                rows = run_one(
                    backend, fingerprint, query,
                    expanded_cache.get(query, query),
                    query_vec_cache, embed_client, cfg, args.limit,
                    embeddings)
            except Exception as exc:
                print(f'  ERROR {cfg["name"]}: {exc}')
                continue
            all_rows.extend(rows)
            ids = [r['insight_id'] for r in rows]
            elapsed = rows[0]['elapsed_ms'] if rows else 0
            print(f'  {cfg["name"]:22s} top={len(rows):2d}  '
                  f'elapsed={elapsed:5.0f}ms  '
                  f'redund={rows[0]["redundancy"] if rows else "-"}  '
                  f'top1={ids[0][:8] if ids else "-"}')

    with Path(out_path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'config', 'query', 'rank', 'insight_id', 'score',
            'intent', 'via', 'elapsed_ms', 'redundancy', 'content'])
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nwrote {len(all_rows)} rows to {out_path}')

    if 'baseline' not in {c['name'] for c in configs}:
        print('\n(baseline config not in this run; skipping the'
              ' Jaccard-vs-baseline summaries)')
        return

    print('\n=== TOP-10 JACCARD OVERLAP vs BASELINE per query ===')
    by_qc: dict[tuple[str, str], list[str]] = {}
    for r in all_rows:
        by_qc.setdefault((r['query'], r['config']), []).append(
            r['insight_id'])

    queries_str = sorted({r['query'] for r in all_rows})
    configs_str = [
        c['name'] for c in configs if c['name'] != 'baseline']

    print(f'{"query":50s} ' + ' '.join(f'{c[:12]:>12s}' for c in configs_str))
    rows_summary = []
    for q in queries_str:
        base = by_qc.get((q, 'baseline'), [])
        line = f'{q[:50]:50s} '
        rec = {'query': q}
        for cfg in configs_str:
            ids = by_qc.get((q, cfg), [])
            j = jaccard(base, ids)
            rec[cfg] = j
            line += f'{j:>12.2f} '
        print(line)
        rows_summary.append(rec)

    print('\n=== AGGREGATE: 1 - mean(Jaccard) per config (higher = changed more) ===')
    for cfg in configs_str:
        avg = sum(r[cfg] for r in rows_summary) / max(1, len(rows_summary))
        print(f'  {cfg:22s} mean Jaccard={avg:.3f}  '
              f'mean change={1-avg:.3f}')


if __name__ == '__main__':
    main()
