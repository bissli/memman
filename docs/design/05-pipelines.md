# 5. Read & Write Pipelines

[< Back to Design Overview](../DESIGN.md)

---

## 5.1 Write Pipeline: Remember

`mnemon remember` is the core command for writing memories. It uses LLM fact extraction to decompose input into atomic facts, LLM reconciliation to classify each fact against existing memories, and Voyage AI embeddings for semantic search.

![Remember Pipeline](../diagrams/02-remember-pipeline.drawio.png)

### Two-Tier Pipeline

**Tier 1 (synchronous — user waits)**

1. **Quality gate** — `check_content_quality()` scans for transient patterns (AWS IDs, deployment receipts, state observations). 2+ matches → reject.
2. **LLM fact extraction** — `extract_facts(llm_client, content)` decomposes input into 1-5 atomic facts with category, importance, and entities. Trivial content (greetings, filler) returns `[]` → skip.
3. **Per-fact processing**:
   a. Embed fact text via Voyage AI (512-dim).
   b. Find similar existing insights: keyword search (top 5) + cosine scan (threshold 0.5).
   c. `reconcile_memories(llm_client, [fact], similar)` → action:
      - **ADD**: new content, insert as new insight
      - **UPDATE**: refines existing, insert and link to original
      - **DELETE**: contradicts existing, soft-delete target
      - **NONE**: already captured, skip
   d. Create edges: `fast_edges()` (temporal + entity) + `create_semantic_edges()`.
   e. `refresh_effective_importance()`, `auto_prune()`.
4. **JSON output** — `{facts: [...], quality_warnings, llm_calls, link_pending}`.

**Tier 2 (async subprocess — `graph link` → `link_pending()`)**

Spawned by `_trigger_background_link()` as a detached subprocess.

1. **LLM enrichment** — `enrich_with_llm()` extracts keywords, summary, semantic facts, and additional entities.
2. **Re-embedding** — rebuilds embedding from enriched text (content + keywords).
3. **LLM causal inference** — `infer_llm_causal_edges()` uses 2-hop BFS + recent insights as candidates.
4. **Edge rebuild** — deletes old auto entity/semantic/causal edges, re-creates from enriched data.
5. **Stamps** — `linked_at` and `enriched_at`.

The `--no-reconcile` flag skips LLM reconciliation for direct insert.

---

## 5.2 Read Pipeline: Smart Recall

`mnemon recall` is Mnemon's core retrieval algorithm. Smart recall is the default mode for all queries. It combines LLM query expansion, intent detection, multi-signal anchor selection, Beam Search graph traversal, and multi-factor re-ranking. Use `--basic` for SQL LIKE fallback.

![Smart Recall Pipeline](../diagrams/03-smart-recall-pipeline.drawio.png)

### Step 0: LLM Query Expansion

`expand_query(llm_client, query)` sends the raw query to the LLM, which returns:
- **expanded_query**: original + synonyms and related terms
- **keywords**: extracted search keywords
- **entities**: entities mentioned or implied in the query
- **intent**: WHY / WHEN / ENTITY / GENERAL (can override regex detection)

The expanded query and extracted entities feed into anchor selection.

### Step 1: Intent Detection

Query intent is identified via regex matching (or LLM override from Step 0):

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

Embeddings are Voyage AI 512-dim vectors. The expanded query from Step 0 is embedded for vector search and reranking.

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

