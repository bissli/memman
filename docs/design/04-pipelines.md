# 4. Read & Write Pipelines

[< Back to Design Overview](../DESIGN.md)

---

## 4.1 Write pipeline: remember (deferred, two-tier)

`memman remember` appends one row to the queue in ~50 ms. A user-scope scheduler (systemd timer on Linux, launchd agent on macOS) invokes the hidden worker `memman scheduler drain --pending` every 60 s; the drain runs extraction, reconciliation, and enrichment out of band.

![Remember Pipeline](../diagrams/02-remember-pipeline.drawio.png)

### Tier 1: synchronous queue-append (host session)

1. `memman remember [--cat X --imp Y --entities a,b] "<text>"` validates input.
2. Insert one row into the deferred-write queue with `status='pending'`, priority, queued_at, and the raw text + hints. The queue is always `~/.memman/queue.db` (SQLite WAL) regardless of any store's backend choice — it is a process-global write buffer, not per-store state.
3. Return `{action: queued, queue_id: N, store: ...}` to the caller.

No LLM calls. No embeddings. No similarity scan. No edges. The host session never blocks.

Every write goes through the queue. When the scheduler is **stopped**, memman is recall-only and writes reject with a fixed error pointing at `memman scheduler start`.

### Tier 2: background worker (scheduler-driven)

`memman scheduler drain --pending` (hidden subcommand; only the trigger invokes it) runs under an environment-native trigger:

- **Linux host**: `systemctl --user` timer at `~/.config/systemd/user/memman-enrich.timer`, `Persistent=true` so sleep/off catch-up is automatic.
- **macOS host**: launchd agent at `~/Library/LaunchAgents/com.memman.enrich.plist` with `StartInterval=60`.
- **nanoclaw container** (no systemd / launchd): `memman scheduler serve --interval 60` runs as PID 1. Set `MEMMAN_SCHEDULER_KIND=serve`. The drain loop polls the state file every iteration so `scheduler stop` is observed within seconds; the loop then exits — in a PID-1 container that exits the container.

Per-blob processing inside `_process_queue_row`:

1. **Atomic claim** — `UPDATE queue SET claimed_at=..., attempts=attempts+1 WHERE id = (SELECT ... WHERE status='pending' ORDER BY priority DESC, queued_at ASC LIMIT 1) RETURNING ...`. The queue is SQLite WAL, so the claim is race-free under the WAL writer guarantee. Stale claims (>10 min) are reclaimable. Drains never overlap: an `fcntl.flock` on `~/.memman/drain.lock` gates `_drain_queue` regardless of which backend the store-under-drain routes to.
2. **Idempotency check** — if the target store already has any insight with `source='queue:<id>'`, skip and mark done (crash-recovery after partial commit).
3. **Quality gate** — regex-based `check_content_quality()` rejects transient patterns.
4. **LLM fact extraction** — decomposes into 1–5 atomic facts with category/importance/entities.
5. **Per-fact**: embed via the store's bound provider, keyword + cosine similarity scan, `reconcile_memories` → ADD/UPDATE/DELETE/NONE, insert/update, fast edges.
6. **Parallel enrichment + causal inference** (ThreadPoolExecutor, 2 workers).
7. **Re-embed** with enriched keywords; rebuild auto edges.
8. `mark_done(queue_id)` on success, or `mark_failed` (retry up to 5 times across stale-claim windows before status='failed').

Edge upserts and embed/LLM call sites no longer swallow exceptions; failures (constraint violation, network error, malformed payload) reach `mark_failed` and consume the retry budget. Best-effort cleanup (HTTP session resets, platform probes, pool teardown) keeps narrow typed catches at `logger.debug`.

### LLM routing

Both the session path (`memman recall` query expansion) and the scheduler path route through OpenRouter. They use three role slots:

- `MEMMAN_LLM_MODEL_FAST` — recall hot path and `doctor`'s connectivity probe.
- `MEMMAN_LLM_MODEL_SLOW_CANONICAL` — worker's canonical-content path (fact extraction, reconciliation).
- `MEMMAN_LLM_MODEL_SLOW_METADATA` — worker's derived-metadata path (enrichment summaries/keywords, causal-edge inference).

All three are populated by `memman install`, which queries OpenRouter's `/models` endpoint once per role and writes the resolved id to `~/.memman/env`. Runtime never queries the model inventory; it reads the persisted id and sends it through unchanged. Re-run `memman install` to bump to a current version when a new model family ships. Splitting the slow worker into `_CANONICAL` and `_METADATA` lets enrichment cost be tuned (e.g., a cheaper metadata model) independently of the load-bearing extraction prompt.

### Operational controls

| Command                                   | Effect                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| `memman scheduler queue list [--limit N]` | inspect pending/done/failed rows                                                           |
| `memman scheduler queue retry <id>`       | re-queue a failed row                                                                      |
| `memman scheduler queue purge --done`     | delete completed rows                                                                      |
| `memman scheduler status`                 | install state, interval, next run, queue depth                                             |
| `memman scheduler start`                  | activate the trigger (idempotent)                                                          |
| `memman scheduler stop`                   | deactivate the trigger; trigger files stay                                                 |
| `memman scheduler interval --seconds N`   | change cadence; min 60 s for systemd/launchd; serve mode accepts `>= 0` (`0` = continuous) |
| `memman scheduler trigger`                | run the drain now (rejects when stopped)                                                   |

`memman graph rebuild` re-enriches all already-stored insights through the full LLM pipeline (useful after model/prompt changes; rejects when the scheduler is stopped). Auto-created edges (semantic, entity, temporal) are recomputed on DB open when edge constants change — no operator command for that.

---

## 4.2 Read pipeline: smart recall

`memman recall` combines LLM query expansion, intent detection, multi-signal anchor selection, beam search graph traversal, and multi-factor re-ranking. Use `--basic` for SQL LIKE fallback.

![Smart Recall Pipeline](../diagrams/03-smart-recall-pipeline.drawio.png)

### Step 0: LLM query expansion (opt-in, off by default)

`expand_query(llm_client, query)` sends the raw query to the LLM and returns:

- **expanded_query**: original + synonyms and related terms
- **keywords**: extracted search keywords
- **entities**: entities mentioned or implied in the query
- **intent**: WHY / WHEN / ENTITY / GENERAL (can override regex detection)

Expansion runs only when the user passes `--expand`. By default the raw query is embedded directly. Expansion is gated because the LLM has no domain scope and can pull the candidate pool toward general-knowledge synonyms that recency-aware rerank (Step 4) then amplifies. Modern embedding models already capture most synonym intent; recency does the rest. See § 4.3.

### Step 1: Intent detection

Query intent is identified via regex (or LLM override from Step 0):

| Intent  | Trigger Patterns                                                                       |
| ------- | -------------------------------------------------------------------------------------- |
| WHY     | `why`, `reason`, `because`, `cause`, `motivation`, `rationale`                         |
| WHEN    | `when`, `time`, `date`, `before`, `after`, `during`, `timeline`, `history`, `sequence` |
| ENTITY  | `what is`, `who is`, `tell me about`, `describe`, `about`                              |
| GENERAL | None of the above match                                                                |

`--intent` manually overrides automatic detection.

### Step 2: Multi-signal anchor selection (RRF fusion)

Three signals run in parallel and fuse via Reciprocal Rank Fusion:

```
Signal 1: Keyword     → KeywordSearch(all_insights, query, top-30)
Signal 2: Vector      → CosineSimilarity(query_vec, all_embeddings, top-30)
Signal 3: Recency     → sort by created_at DESC, top-30

RRF Score = Σ  1 / (k + r)    (k = 60, r = 1-based rank)
                 for each signal
```

Each insight may rank differently across signals; RRF fusion produces a composite ranking that does not collapse when any one signal is noisy.

**Rationale.**

- **`ANCHOR_TOP_K = 30`**: per-signal anchor pool size. MAGMA Table 5 specifies 20; memman uses 30 to give beam search a richer starting frontier given the flat insight hierarchy (no episode/narrative super-nodes).
- **`RRF_K = 60`**: standard value from the original RRF paper (Cormack, Clarke & Büttcher, SIGIR 2009). MAP scores nearly flat from k=50–90, with k=60 validated across four TREC collections.
- **`VECTOR_SEARCH_MIN_SIM = 0.10`**: noise floor matching MAGMA's lower similarity threshold bound. Below 0.10, vector search hits add noise rather than signal.

### Step 3: Beam search graph traversal

From each anchor, beam search traverses the four graphs:

```
for each anchor:
    priority_queue = [(anchor, initial_score)]
    visited = {}

    while budget_remaining:
        node = pop(priority_queue)
        for edge in GetEdgesFrom(node):
            neighbor = edge.target
            structural_score = edge.weight × intent_weight[edge.type]
            semantic_score = cosine(vec_neighbor, vec_query)
            total = score_node + λ₁·structural + λ₂·semantic
            //  λ₁ = 1.0 (structural weight), λ₂ = 0.4 (semantic weight)

            if total > best_score[neighbor]:
                update(neighbor, total)
                push(priority_queue, neighbor)
```

Beam width, max depth, and max-visited budgets are intent-adaptive — see the per-intent tuning table in Step 4.

### Step 4: Multi-factor re-ranking

For all collected candidates, a four-dimensional score is computed and combined via weighted sum:

```
keyword_score  = token_intersection / query_token_count
entity_score   = matched_entities / max(1, query_entities_count)
similarity     = cosine(vec_candidate, vec_query)
graph_score    = (traversal_score - min) / (max - min)   // min-max normalization

final = w_kw·keyword + w_ent·entity + w_sim·similarity + w_gr·graph
```

**Per-intent tuning.** The Step 3 traversal budget and the Step 4 reranker weights both vary by intent. Left columns tune beam search; right columns tune the reranker:

| Intent  | Beam | Depth | MaxVis | KW   | Ent      | Sim      | Graph    |
| ------- | ---- | ----- | ------ | ---- | -------- | -------- | -------- |
| WHY     | 15   | 5     | 500    | 0.15 | 0.10     | **0.45** | **0.30** |
| WHEN    | 10   | 5     | 400    | 0.20 | 0.10     | **0.40** | **0.30** |
| ENTITY  | 10   | 4     | 400    | 0.20 | **0.35** | **0.35** | 0.10     |
| GENERAL | 10   | 4     | 500    | 0.25 | 0.15     | **0.45** | 0.15     |

**Rationale.**

- **`LAMBDA1 = 1.0`, `LAMBDA2 = 0.4`** (Step 3 traversal-score blend): `LAMBDA1` is from MAGMA Table 5 ("λ1 (Structure Coef.): 1.0 (Base)"); `LAMBDA2` falls within MAGMA's empirically tuned range (0.3–0.7), at the conservative end so structural signal is weighted 2.5× semantic.
- **Beam / Depth / MaxVis**: max depth 5 (WHY/WHEN) is from MAGMA Table 5. WHY gets beam width 15 (50% wider than the base 10) because causal chains typically span more hops. GENERAL gets `MaxVis=500` (matching WHY) because unknown intent should not restrict exploration. WHEN/ENTITY get 400 as a moderate budget — their primary edges (temporal/entity) form shorter chains.
- **KW / Ent / Sim / Graph**: extends MAGMA's intent-adaptive philosophy (which steers beam search via edge type weights) into the final reranking stage. MAGMA does not define a separate reranking stage — this is memman's extension.

Embeddings are Nd vectors from the store's bound provider (dim is provider-defined; current default is `voyage-3-lite`, 512-dim). The expanded query from Step 0 is embedded for vector search and reranking.

### Step 4b: Cross-encoder rerank

When the caller passes `--rerank` and the query has more than `MIN_RERANK_TOKENS` (default 2) whitespace tokens, the top `RERANK_SHORTLIST` (default 100) candidates from Step 4 are re-scored by the configured cross-encoder reranker (`MEMMAN_RERANK_PROVIDER`; current default `voyage` with model `rerank-2.5-lite`), and the rerank score replaces the multi-signal score for the final ordering. The skill files instruct LLM agents to always pass `--rerank` on natural-language queries.

Bi-encoder retrieval (Steps 1–4) embeds the query and each insight independently and ranks by cosine plus the four signals. A cross-encoder reads `(query, content)` together with full attention and outputs a relevance score directly, so it resolves cases where bi-encoder cosine misses the right answer despite low token overlap.

Failures (timeouts, non-200 responses) are caught and logged; the baseline ordering is returned unchanged with `meta.reranked = false`. The 1-2 token query gate skips rerank when there is too little query signal for the cross-encoder to use.

### Empirical evidence

Two evaluations led to shipping rerank behind `--rerank` and having `SKILL.md` always pass it. Phase 1 (12 queries × 1 store, no labels) ruled out cheap alternatives: bumping `ANCHOR_TOP_K`, retuning weights, and LLM query expansion none closed the gap. Phase 2 (90 queries × 3 stores, ~4500 graded relevance labels via Haiku 4.5 on a 0–3 scale) measured the lift against ground truth.

| Headline metric                                           | baseline | rerank           | Δ          |
| --------------------------------------------------------- | --------: | ----------------: | ----------: |
| nDCG@5 (combined)                                         | 0.648    | **0.788**        | **+0.140** |
| Recall@5                                                  | 0.573    | **0.695**        | +0.122     |
| MRR                                                       | 0.759    | **0.835**        | +0.076     |
| P@1                                                       | 0.711    | **0.789**        | +0.078     |
| Mean P@5 (fraction of top-5 with rel ≥ 2)                 | 0.476    | **0.556**        | +0.080     |
| Queries where rerank wins / ties / loses                  | —        | **56 / 12 / 22** | —          |
| Queries where rerank loses a rel=3 (directly-answers) doc | —        | **0**            | —          |

Rerank helps most where the bi-encoder is weakest: +0.40 nDCG@5 on the 22 weak-baseline queries vs +0.06 on the 44 already-strong ones. All three stores show the same shape. WHY/WHEN intents — predicted to regress — gained the most (+0.097 / +0.351) because their bi-encoder baselines were the weakest to begin with.

### Step 5: WHY post-processing — causal topological sort

If the intent is WHY, an additional topological sort using Kahn's algorithm arranges results along causal edges so that **causes come first, effects follow**.

### Step 5b: WHEN post-processing — chronological sort

If the intent is WHEN, results are re-sorted chronologically — **newest first** by `created_at`, with score as tiebreaker for equal timestamps.

### Signal breakdown

Each retrieval result includes signal details:

```json
{
  "insight": {
    "id": "...",
    "content": "...",
    "summary": "..."
  },
  "score": 0.73,
  "intent": "ENTITY",
  "via": "keyword",
  "signals": {
    "keyword": 0.85,
    "entity": 0.60,
    "similarity": 0.72,
    "graph": 0.45
  }
}
```

The `summary` field is the LLM-authored one-line gloss produced during enrichment (slow_metadata role). It is present only when (a) enrichment has run for the row and (b) the summary actually compresses the content (write-time gate at `len(summary) < len(insight.content) * 0.85`); rows that fail the gate emit no `summary` key. Calling LLMs see ~3.6× token compression with ~90% ranking-decision agreement vs full content.

The host LLM sees these signals and can apply its own judgment with full conversation context.

## 4.3 Model resilience

memman calls LLMs at write time (extraction, reconciliation, enrichment, causal inference) and embedding models on every vector. Prompts get edited, models get upgraded, providers get swapped. The design goal is detection and re-run, not bit-identical output across versions.

Two principles:

1. **Keep slow work off the hot path.** The write path defers LLM work to the scheduler drain (Tier 2 in 4.1). The read path is embedding-only at the bare-CLI level, with `--expand` (LLM query expansion) and `--rerank` (cross-encoder reranking) as explicit flags; LLM agents pass `--rerank` per `SKILL.md`. Where LLM judgment is unavoidable, the output is tagged with what produced it and re-runnable.
2. **Provenance + re-run beats deterministic-rule replacement.** Hard rules (length thresholds, importance clamps, similarity cutoffs) calcify with one model's behavior baked in. Provenance + re-run tracks what produced each row and re-derives when inputs change. Same precedent as the embed-fingerprint mechanism.

### Invalidation hooks

| Hook                                                             | Stored at | Detects                            | Operator action                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------- | --------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed_fingerprint`                                              | `meta`    | per-store embedder binding         | Each store's stored fingerprint binds the embedder used by recall, drain, snapshot, and graph rebuild. Change the binding via `memman embed swap` (online, resumable shadow-column backfill) or `memman embed reembed` (offline, scheduler-stopped). |
| `embed_swap_state` / `embed_swap_cursor` / `embed_swap_target_*` | `meta`    | in-flight swap progress            | Written by `embed swap`; **deleted** on cutover or `--abort`. `memman doctor`'s `no_stale_swap_meta` check warns if any key remains on an idle store.                                                                                                |
| `insights.prompt_version` + `insights.model_id`                  | per row   | system-prompt or slow-model change | `memman doctor` warns; remediate via `memman graph rebuild` or `UPDATE insights SET linked_at=NULL, enriched_at=NULL WHERE prompt_version='<old>' OR model_id='<old>';` then drain.                                                                  |
| `constants_hash`                                                 | `meta`    | edge-construction constants change | Auto-reindex on next open + warning.                                                                                                                                                                                                                 |
| `linked_at` / `enriched_at`                                      | per row   | per-row pipeline-stage completion  | `link_pending` drains naturally.                                                                                                                                                                                                                     |

Per-row provenance columns are preferred over global meta-key fingerprints because they expose the actual rebuild scope: how many rows came from which prompt or model. That distribution is what the operator needs to write a targeted hand-update SQL rather than rebuilding the whole store.

### What is NOT used

memman does not run multi-LLM consensus, calibrate against a target judgment distribution, or hold deterministic rules that override LLM output. Each adds permanent complexity that conflicts with future model improvements. Provenance + re-run keeps the implementation simple and lets future model upgrades be a deliberate operator action rather than a silent shift.
