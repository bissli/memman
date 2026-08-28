# Recall ablation harness

Sweeps `memman recall` over hard-coded configurations on a target store and emits per-`(config, query, rank)` rows. Use this to A/B new `RERANK_WEIGHTS`, `ANCHOR_TOP_K`, `MMR_LAMBDA` values, or other recall tunables against your own corpus. This harness is the arbiter of shipped retrieval constants ("measured or not shipped"), which is why it is tracked while its data artifacts (`results*.csv`, `sandbox/`) are not.

## Run

```bash
poetry run python run_ablation.py --store NAME --limit 10
```

Flags:

- `--store NAME` (default `search`) — target store; must already contain data.
- `--data-dir PATH` (default `~/.memman`) — point at a sandbox copy to sweep without touching live stores.
- `--limit N` (default `10`) — results per query.
- `--queries PATH` — query set JSON; defaults to `queries.json` next to the script.
- `--out PATH` — output CSV; defaults to `results.csv` next to the script.
- `--configs a,b,c` — subset of config names to run (default: all).

The configurations are hard-coded inside `run_ablation.py` (see the module docstring). To add a new config, edit the `CONFIGS` list. Each output row carries a `redundancy` column — the mean over returned rows of each row's max pairwise cosine against the rest — the quantity a diversity term exists to reduce. To compare runs against labeled relevance scores, feed the output CSV into `experiments/eval/`.

Rerank configs call the Voyage rerank endpoint directly and read `VOYAGE_API_KEY` from the environment (export it from `MEMMAN_VOYAGE_API_KEY` in `~/.memman/env`).

## Entity signal retired (2026-08-28, memman 0.23.0)

The blend lost its fourth term. `ent_score = matched_entities / query_entities_count` and its per-intent weight are gone; `RERANK_WEIGHTS` is now `(w_kw, w_sim, w_graph)`.

Why, in the order the evidence landed:

- The term was fed only by Step 0's LLM expansion, which became opt-in on 2026-04-28. From that date it was identically 0.0 on the default path, so no harness ever measured it while live - every call site in this harness and in `experiments/quality_matrix/` passes an empty entity list.
- Fed deliberately (by seeding the set from the query text), it measured indistinguishable from noise against 30 LLM-judged queries: the arm ordering was seeding-off 0.3484 vs seeding-on 0.3416 nDCG@5, but the paired SE is 0.0148, so the +0.02 acceptance bar sits inside the interval. A channel built with the same magnitude and sparsity but values shuffled at random scored 0.3415 +/- 0.0082 over 10 seeds, beating the best hand-designed replacement formula on 3 of 10 seeds.
- Three replacement formulas were designed independently and each refuted by an independent re-measurement. The defect is amplitude, not shape: the mean of `w_ent *` the largest `ent_score` a query actually produced was 0.0596 against a mean rank-5-to-6 score margin of 0.0118 — 5x on the means, and 24x to 108x on individual queries at full scale — so the term overrode the blend instead of informing it. No numerator or denominator repair survives that.
- With the cross-encoder on, which is the shipped CLI default, feeding the channel changed the top-5 on 3 of 30 queries and moved nothing at all on the six entity-named ones.

Consequences for anyone reading older records here:

- The weight rows no longer sum to 1.0 (WHY 0.90, WHEN 0.90, ENTITY 0.65, GENERAL 0.85). Those were always the real caps on the default path, because the deleted term contributed nothing there; the survivors keep their literal shipped values, so on that path no ranking and no score moves. Under `--expand` the term was live, so those runs DO reorder — that is the change, not a side effect. Rescaling to restore a cosmetic 1.0 would reorder nothing either way, since ranking is invariant to scaling one intent's row by a constant, but it would inflate every absolute score.
- `WEIGHTS_V2` rows are the historical four-slot candidates with the entity slot dropped and the other three untouched, so each arm still departs from production where it always did. What is gone is v2's ENTITY-entity claim, 0.35 -> 0.50, which was never testable: every sweep that reported it ran with `ent_score` identically 0.0, so every entity-weight conclusion in `results.csv` and `results_mmr.csv` was vacuous.
- Entities still reach recall twice over, so this is not a claim that entity data is useless: a candidate's keyword token set unions its content tokens with its entity-name tokens, and `entity` graph edges carry the highest edge weight of any intent under ENTITY (0.55). Dropping that keyword union measured -0.0137 nDCG@5 on this same 30-query instrument, which is inside the same noise band as the entity effect itself and so is not on its own a reason to keep it; what argues for keeping it is the token census - on 43.6% of rows the entity list contributes at least one token the content lacks.

Instrument caveat that bounds every number above: 65.9% of the (query, doc) pairs these arms return carry no relevance grade, the labels date from 2026-04-30, and 30 queries cannot resolve a 0.02 effect - about 130 would be needed. The labels and the nDCG scorer live under `experiments/eval/`, which is untracked, so a fresh clone can reproduce the churn columns but not the quality columns.

## MMR sweep record (2026-08-05, memman 0.19.0)

Sweep of `mmr_lambda` in {0.5, 0.7, 0.8, 0.9} x rerank {off, on} plus `baseline` and `rerank_voyage` controls; 12 queries, `--limit 10`, against a `/tmp` sandbox copy of the `search` store (2,299 rows, 100% embedded) on the 0.19.0 schema. Raw rows in `results_mmr.csv` (untracked artifact).

| config         | mean redundancy | vs baseline | jaccard vs rerank_voyage |
| -------------- | --------------- | ----------- | ------------------------ |
| baseline       | 0.655           | —           |                          |
| mmr_l50        | 0.504           | -0.151      |                          |
| mmr_l70        | 0.571           | -0.085      |                          |
| mmr_l80        | 0.599           | -0.057      |                          |
| mmr_l90        | 0.628           | -0.027      |                          |
| rerank_voyage  | 0.657           | +0.001      | 1.000                    |
| mmr_l50_rerank | 0.640           | -0.016      | 0.783 (top-3 same 9/12)  |
| mmr_l70_rerank | 0.655           | -0.000      | 0.916 (top-3 same 10/12) |
| mmr_l80_rerank | 0.653           | -0.002      | 0.972 (top-3 same 11/12) |
| mmr_l90_rerank | 0.657           | +0.001      | 1.000 (identical 12/12)  |

Conclusion — `MMR_LAMBDA = 1.0` (term disabled) is the measured value:

- Under the production default (rerank on), MMR contributes ~nothing at any lambda: byte-identical output at 0.9, and at the most aggressive 0.5 the final redundancy moves 1.6 points while the cross-encoder re-picks the same head anyway.
- Rerank-off gains are real (0.655 -> 0.504 at lambda 0.5) but rewrite most of the top-10 (jaccard 0.176 vs baseline) with no relevance labels to certify that the rewrite does not demote a fact's own decision rationale — exactly the pair memman's semantic edges link.
- The mechanism stays shipped and sweepable (`MMR_LAMBDA`/`MMR_POOL` in `search/recall.py`, `mmr_l*` configs here); revisit if a labeled-relevance eval lands or a rerank-off deployment materialises.

## MMR after-rerank placement sweep (2026-08-06, memman 0.20.0)

The spec's alternative placement — rerank the full top-100 shortlist first, then one-shot MMR over the reranked list (reranker relevance as the score term), then the limit slice — measured with the `mmr_after_l{NN}_rerank` configs; same store sandbox, 12 queries, `--limit 10`. Raw rows in `results_after.csv` (untracked artifact).

| config               | mean redundancy | vs rerank_voyage | jaccard vs rerank_voyage |
| -------------------- | --------------- | ---------------- | ------------------------ |
| rerank_voyage        | 0.657           | —                | 1.000                    |
| mmr_after_l50_rerank | 0.552           | -0.104           | 0.338 (top-3 same 0/12)  |
| mmr_after_l70_rerank | 0.597           | -0.059           | 0.549 (top-3 same 2/12)  |
| mmr_after_l80_rerank | 0.625           | -0.032           | 0.711 (top-3 same 4/12)  |
| mmr_after_l90_rerank | 0.644           | -0.013           | 0.854 (top-3 same 5/12)  |

Conclusion — the placement move does not change the shipped `MMR_LAMBDA = 1.0`:

- Downstream of the reranker the diversity term finally has leverage (unlike upstream, where the cross-encoder re-picks the same head regardless), but every redundancy point is bought by overriding the only relevance oracle in the pipeline: at lambda 0.5 the certified top-3 survives on 0/12 queries, and at lambda 0.9 the surviving 1.3-point gain is noise-level.
- Same verdict shape as the rerank-off arm above: real diversity, uncertifiable relevance cost. Ship it only when a labeled-relevance eval can prove the trade.
- The after-rerank arm also reranks all 100 shortlist docs instead of 10 and adds an O(n^2) cosine pass (~+200-400 ms observed on this store), a real hot-path cost for an unverifiable gain.
