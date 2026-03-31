# 5. Read & Write Pipelines

[< Back to Design Overview](../DESIGN.md)

---

## 5.1 Write Pipeline: Remember

`mnemon remember` is the core command for writing memories. It includes a built-in diff step that automatically detects duplicates and conflicts before storage. The write transaction executes atomically within a single SQLite transaction.

![Remember Pipeline](../diagrams/02-remember-pipeline.drawio.png)

### Detailed Flow

```
mnemon remember "Chose Qdrant as the vector database" \
  --cat decision --imp 5 --entities "Qdrant,Milvus"
```

**Step 1: Validate Input**
- Category must be one of the six types
- Importance 1-5
- Content must not exceed 8000 characters
- Up to 20 tags and 50 entities

**Rationale:** Content limit 8000 chars is a practical upper bound for a single insight — keeps token overlap computation fast and embedding generation within model input limits. Larger content should be decomposed into multiple insights. Max tags = 20 discourages tag abuse while remaining generous. Max entities = 50 accommodates automatic extraction (regex + dictionary) which can produce many matches.

**Step 1.5: Quality Gate (before embedding)**
- Scan content against transient patterns (AWS instance IDs, resource counts, verification/deployment receipts, state observations, line references)
- If 2+ patterns match: reject immediately with `action="rejected"`, log `quality-reject` to oplog
- If 0–1 patterns match: continue to embedding

**Step 2: Generate Embedding (outside transaction)**
- If Ollama is available: HTTP POST -> nomic-embed-text -> 768-dim float64 vector
- If unavailable: embedding = nil, falls back to token overlap downstream

**Step 2.5: Built-in Diff (outside transaction, read-only)**

Compute similarity against all active insights:
- **DUPLICATE** (sim > 0.90) → skip insert entirely, return `action="skipped"`
- **CONFLICT/UPDATE** (sim 0.55–0.90) → soft-delete old insight, insert new as replacement
- **ADD** (sim < 0.55) → normal insert

This step uses embedding cosine similarity when available, falling back to token overlap. The `--no-diff` flag disables this check. When embedding cosine similarity >= 0.7 and exceeds the token overlap score, cosine overrides — this allows embeddings to detect semantic overlap that token-level comparison misses (e.g., synonyms, paraphrases).

**Step 3: Atomic Transaction**

```
BEGIN TRANSACTION
  ⓪ Soft-delete replaced insight (if diff found CONFLICT/UPDATE)
  ① INSERT insight (UUID, content, category, importance, tags, entities, source)
  ② UPDATE embedding (if vector is available)
  ③ Graph Engine: fast_edges
     ├── CreateTemporalEdge    → backbone + 4h proximity
     ├── CreateEntityEdges     → regex + dictionary extraction → co-occurrence links
     └── CreateCausalEdges     → keywords + token overlap → auto causal edges
  ③b Deferred: consolidate_pending (semantic edges deferred to batch processing)
  ④ RefreshEffectiveImportance → update EI decay values
  ⑤ AutoPrune                 → soft-delete lowest EI when total > 1000
COMMIT
```

**Step 4: Candidate Output (post-transaction, read-only)**
- `FindSemanticCandidates`: Semantic candidates with cos >= 0.40 (`auto_linked` flag marks >= 0.70)
- `FindCausalCandidates`: Causal candidates in the 2-hop BFS neighborhood

**Step 5: JSON Output**

```json
{
  "id": "abc-123",
  "action": "added",
  "diff_suggestion": "ADD",
  "replaced_id": null,
  "edges_created": {"temporal": 2, "entity": 3, "causal": 1, "semantic": 1},
  "semantic_candidates": [
    {"id": "def-456", "content": "...", "cosine": 0.72, "auto_linked": false}
  ],
  "causal_candidates": [
    {"id": "ghi-789", "content": "...", "hop": 1, "suggested_sub_type": "causes"}
  ],
  "quality_warnings": [],
  "embedded": true,
  "effective_importance": 0.85,
  "auto_pruned": 0
}
```

The `action` field indicates what the built-in diff decided: `"added"` (new entry), `"replaced"` (conflict auto-replaced, `replaced_id` contains the old insight ID), or `"skipped"` (duplicate detected, no insert).

The `quality_warnings` field lists any transient content patterns detected (e.g., AWS instance IDs, deployment receipts, state observations). Content with **2 or more** quality warnings is **rejected** before embedding or diff computation — the response returns `action: "rejected"` with the warning list. Content with 0–1 warnings is stored normally (1 warning is advisory). This hard gate fires early in the pipeline to avoid wasting embedding and diff compute on transient content.

After receiving this output, the LLM can evaluate candidates and establish edges it considers appropriate via the `mnemon link` command.

---

## 5.2 Read Pipeline: Smart Recall

`mnemon recall` is Mnemon's core retrieval algorithm. Smart recall is the default mode for all queries. It combines intent detection, multi-signal anchor selection, Beam Search graph traversal, and multi-factor re-ranking to achieve intent-aware graph-enhanced retrieval. Use `--basic` for legacy SQL LIKE fallback.

![Smart Recall Pipeline](../diagrams/03-smart-recall-pipeline.drawio.png)

### Step 1: Intent Detection

Query intent is automatically identified via regex matching:

| Intent  | Trigger Patterns                                                                       |
| ------- | -------------------------------------------------------------------------------------- |
| WHY     | `why`, `reason`, `because`, `cause`, `motivation`, `rationale`                         |
| WHEN    | `when`, `time`, `date`, `before`, `after`, `during`, `timeline`, `history`, `sequence` |
| ENTITY  | `what is`, `who is`, `tell me about`, `describe`, `about`                              |
| GENERAL | None of the above match                                                                |

Supports the `--intent` flag to manually override automatic detection.

### Step 2: Multi-Signal Anchor Selection (RRF Fusion)

Multiple signals run in parallel and are merged via Reciprocal Rank Fusion:

```
Signal 1: Keyword     → KeywordSearch(all_insights, query, top-20)
Signal 2: Vector      → CosineSimilarity(query_vec, all_embeddings, top-20)
Signal 3: Recency     → sort by created_at DESC, top-20

RRF Score = Σ  1 / (k + rank_i + 1)    (k = 60)
                 for each signal
```

Each insight may rank differently across signals; RRF fusion produces a robust composite ranking.

**Rationale:**

- **`ANCHOR_TOP_K = 20`**: Directly from MAGMA Table 5 ("Vector Top-K: 20"). *(Cite: MAGMA, Jiang et al., arXiv 2601.03236, Table 5: Vector Top-K: 20)*
- **`RRF_K = 60`**: Standard value from the original RRF paper. Pilot experiments showed MAP scores nearly flat from k=50–90, with k=60 fixed early and validated across four TREC collections. The constant mitigates the impact of high rankings by outlier systems. *(Cite: Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009, Table 1: k=60 near-optimal across k=0–500)*
- **`VECTOR_SEARCH_MIN_SIM = 0.10`**: Noise floor matching MAGMA's lower similarity threshold bound. Below 0.10, vector search hits add noise rather than signal.

### Step 3: Beam Search Graph Traversal

Starting from each anchor, Beam Search is performed across the four graphs:

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

**Adaptive parameters**:

| Intent  | Beam Width | Max Depth | Max Visited |
| ------- | ---------- | --------- | ----------- |
| WHY     | 15         | 5         | 500         |
| WHEN    | 10         | 5         | 400         |
| ENTITY  | 10         | 4         | 400         |
| GENERAL | 10         | 4         | 500         |

WHY queries use a wider beam and deeper traversal because causal chains typically span multiple hops.

**Rationale:**

- **`LAMBDA1 = 1.0`**: Directly from MAGMA Table 5 ("λ1 (Structure Coef.): 1.0 (Base)").
- **`LAMBDA2 = 0.4`**: Falls within MAGMA's empirically tuned range ("λ2 (Semantic Coef.): 0.3–0.7"). 0.4 chosen as a conservative value — structural signal is weighted 2.5× semantic, prioritizing graph topology over embedding similarity during traversal. *(Cite: MAGMA, Jiang et al., arXiv 2601.03236, Table 5: λ1=1.0, λ2=0.3–0.7)*
- **Max depth 5** (WHY/WHEN): Directly from MAGMA Table 5 ("Max Depth: 5 hops"). WHY gets beam width 15 (50% wider than base 10) because causal chains typically span more hops — same reasoning as MAGMA's wider traversal for causal queries. GENERAL gets max_visited=500 (matching WHY) because unknown intent should not restrict exploration. WHEN/ENTITY get 400 as a moderate budget — their primary edges (temporal/entity) form shorter chains. *(Cite: MAGMA, Jiang et al., arXiv 2601.03236, Table 5: Max Depth: 5, Max Nodes: 200, scaled up for mnemon's smaller personal-use graphs)*

### Step 4: Multi-Factor Re-Ranking

For all collected candidates, a four-dimensional score is computed and combined via weighted sum:

```
keyword_score  = token_intersection / query_token_count
entity_score   = matched_entities / max(1, query_entities_count)
similarity     = cosine(vec_candidate, vec_query)
graph_score    = (traversal_score - min) / (max - min)   // min-max normalization

final = w_kw·keyword + w_ent·entity + w_sim·similarity + w_gr·graph
```

Weights vary by intent:

| Intent  | Keyword | Entity   | Similarity | Graph    |
| ------- | ------- | -------- | ---------- | -------- |
| WHY     | 0.15    | 0.10     | **0.45**   | **0.30** |
| WHEN    | 0.20    | 0.10     | **0.40**   | **0.30** |
| ENTITY  | 0.20    | **0.35** | **0.35**   | 0.10     |
| GENERAL | 0.25    | 0.15     | **0.45**   | 0.15     |

**Rationale:** These extend MAGMA's intent-adaptive philosophy (which steers beam search via edge type weights) into the final reranking stage. MAGMA does not define a separate reranking stage — this is mnemon's own extension.

- **WHY**: Similarity and graph traversal capture causal chains — the primary signals for "why" queries, weighted at 0.45 and 0.30 respectively.
- **WHEN**: Similarity (0.40) and graph (0.30) capture temporal ordering; keyword provides supporting context.
- **ENTITY**: Entity matching (0.35) and similarity (0.35) are co-primary; graph is de-emphasized (0.10) since entity edges are already captured in entity score.
- **GENERAL**: Similarity-weighted (0.45) with keyword (0.25) as secondary — no strong bias without clear intent, but semantic match is the default discriminator.

**No-embed fallback:** When embeddings are unavailable, the similarity weight redistributes to keyword and graph proportionally. For example, WHY becomes (0.25, 0.15, 0.0, 0.60) and WHEN becomes (0.30, 0.15, 0.0, 0.55) — keyword and graph share the redistributed similarity weight.

### Step 5: WHY Post-Processing — Causal Topological Sort

If the intent is WHY, an additional topological sort using Kahn's algorithm is performed: results are arranged along causal edges so that **causes come first, effects follow**.

### Step 5b: WHEN Post-Processing — Chronological Sort

If the intent is WHEN, results are re-sorted chronologically: **newest first** by `created_at`, with score as tiebreaker for equal timestamps.

### Signal Transparency

Each retrieval result includes a detailed signal breakdown:

```json
{
  "insight": {"id": "...", "content": "..."},
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

This is a unique innovation in Mnemon: **exposing the retrieval pipeline's internal signals to the host LLM**. Since the host LLM has the full conversation context, it can make better re-ranking judgments than any algorithm inside the pipeline.

---

## 5.3 Deduplication & Conflict Detection: Diff

![Diff & Dedup Pipeline](../diagrams/07-diff-dedup-pipeline.drawio.png)

Diff is **built into `remember`** — no separate call needed. When `mnemon remember` is invoked, it automatically runs a diff check before inserting.

When `remember` is called, the built-in diff runs before the transaction:

1. Compute similarity against all active insights (embedding cosine when available, token overlap as fallback)
2. Determine the action based on similarity thresholds:

| Similarity  | Action              | Behavior                                           |
| ----------- | ------------------- | -------------------------------------------------- |
| > 0.90      | **DUPLICATE**       | Skip insert entirely, return `action="skipped"`    |
| 0.55 ~ 0.90 | **CONFLICT/UPDATE** | Soft-delete old insight, insert new as replacement |
| < 0.55      | **ADD**             | Normal insert                                      |

**Rationale:**

- **`> 0.90` DUPLICATE**: Standard near-duplicate threshold in dedup literature. At 0.90+ similarity, content is functionally identical.
- **`0.55–0.90` CONFLICT/UPDATE**: Below 0.55, content is sufficiently distinct to coexist. Conflict detection is further refined by linguistic signals — antonym pairs and negation words in the content trigger CONFLICT classification within this range.
- **`< 0.55` ADD**: Content is distinct enough to stand as an independent insight.

The `--no-diff` flag disables this check for cases where the caller wants unconditional insertion.

### Typical Workflow

A single `remember` call handles everything:

```bash
# Single command — diff is automatic
mnemon remember "Chose PostgreSQL to replace SQLite as the primary database" \
  --cat decision --imp 5 --source agent
# → If conflict with existing "Chose SQLite as storage":
#   auto-replaces old insight, returns action="replaced", replaced_id="<old_id>"
# → If duplicate: returns action="skipped"
# → If new: returns action="added"
```
