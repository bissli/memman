# 3. Graph Model

[< Back to Design Overview](../DESIGN.md)

---

MAGMA's core idea: a single edge type (such as pure vector similarity) is insufficient to capture the multidimensional relationships between memories. Different query intents require different relational perspectives — asking "why" requires causal chains, asking "when" requires timelines, asking "about X" requires entity associations.

MemMan implements four graphs, each capturing one dimension of relationships:

![MAGMA Four-Graph Model](../diagrams/04-magma-four-graph.drawio.png)

## 3.1 Temporal Graph

**Purpose**: Capture the chronological order of memories.

**Automatically created edges**:

- **Backbone**: New insight → most recent insight from the same source (bidirectional)
  - Ensures memories from each source (user/agent) form a continuous timeline
  - Metadata: `{"sub_type": "backbone", "direction": "precedes"|"succeeds"}`
- **Proximity**: New insight <-> insights within a 4-hour window (bidirectional)
  - Weight formula: `w = 1 / (1 + hours_diff)`
  - Up to 10 proximity edges
  - Metadata: `{"sub_type": "proximity", "hours_diff": "2.34"}`

```
Insight A (2h ago) ←── backbone ──→ Insight B (1h ago) ←── backbone ──→ Insight C (now)
     ↑                                     ↑
     └──────── proximity (w=0.33) ─────────┘
```

**Constants:**

- **`TEMPORAL_WINDOW_HOURS = 4`**: A focused session window. Memories created within the same few hours are likely contextually related.
- **`MAX_PROXIMITY_EDGES = 10`**: Limits fan-out per insert.

## 3.2 Entity Graph

**Purpose**: Link insights that mention the same entities.

**Entity extraction**: LLM-based via `extract_facts()` (sequential phase) and `enrich_with_llm()` (parallel enrichment via ThreadPoolExecutor). The LLM extracts entities as part of fact decomposition. User-provided entities via `--entities` flag are merged with LLM-extracted ones.

**Automatically created edges**: New insight <-> up to 5 existing insights per shared entity (bidirectional). Edge weight is computed via IDF: rare entities produce high-weight edges; common entities produce low-weight edges. When the store has ≤5 insights, all entity edges use weight 1.0 (IDF is not meaningful at that scale).

```
                   ┌─── "Qdrant" ───┐
                   │                │
Insight A ←── entity ──→ Insight B ←── entity ──→ Insight C
("Chose Qdrant")         ("Qdrant perf test")     ("Qdrant deployment config")
```

**Metadata**: `{"entity": "Qdrant"}`

**Constants:**

- **`MAX_ENTITY_LINKS = 5` per entity**: Caps edge creation per shared entity to prevent popular entities from creating O(n) edges on every insert.
- **`MAX_TOTAL_ENTITY_EDGES = 50`**: Hard cap across all entities per insert.
- **IDF weighting**: `log(N/df) / log(N)` (floored at 0.1), where N = total active insights and df = count of insights containing the entity. Disabled when N ≤ 5.

## 3.3 Causal Graph

**Purpose**: Capture cause-effect relationships and decision rationale.

**LLM-only inference** (parallel enrichment via ThreadPoolExecutor, `infer_llm_causal_edges()`):

1. Candidate discovery: 2-hop BFS neighbors (max 10 nodes) + recent insights (up to 20), filtered by token overlap ≥ 15%, capped at `MAX_CAUSAL_CANDIDATES = 10` before sending to LLM
2. LLM evaluates candidates and returns edges with confidence scores
3. Edges below `LLM_CONFIDENCE_FLOOR` (0.75) are rejected
4. Sub-types: `causes` (direct cause), `enables` (enabling condition), `prevents` (preventing factor)

```
Insight A ──── causal ────→ Insight B
("Team lacks Redis exp.")   ("Chose SQLite as storage")
  sub_type: "causes"
  weight: 0.80 (LLM confidence)
```

**Metadata**: `{"created_by": "llm", "confidence": 0.80, "rationale": "...", "sub_type": "causes"}`

**Constants:**

- **`MIN_CAUSAL_OVERLAP = 0.15`**: Filters BFS candidates — below this, shared tokens are incidental.
- **`MAX_CAUSAL_CANDIDATES = 10`**: Caps candidates sent to LLM.
- **`LLM_BFS_NEIGHBORS = 10`**: Max nodes in 2-hop BFS.
- **`LLM_RECENT_COUNT = 20`**: Recent insights added to candidate pool.
- **`LLM_CONFIDENCE_FLOOR = 0.75`**: High bar for accepting inferred edges.

## 3.4 Semantic Graph

**Purpose**: Connect semantically similar insights based on embedding distance.

**Auto-link**: Cosine similarity ≥ 0.62 (`AUTO_SEMANTIC_THRESHOLD`), top 3 per insight (`MAX_AUTO_SEMANTIC_EDGES`). Bidirectional edges created automatically.

Embeddings are Voyage AI 512-dim vectors. Semantic edges are created initially from the raw embedding, then rebuilt after enrichment when keywords are appended and the embedding is recomputed.

**Metadata**: `{"created_by": "auto", "cosine": "0.8234"}`

**Constants:**

- **`AUTO_SEMANTIC_THRESHOLD = 0.62`**: Calibrated for Voyage `voyage-3-lite` 512-dim embeddings. Empirically verified: all pairs above 0.62 are genuine topical links; noise floor begins around 0.50.
- **`MAX_AUTO_SEMANTIC_EDGES = 3`**: Limits edges per insert to prevent over-linking on dense topics.

> **If the embedding model changes, this threshold must be recalibrated.** Different models (and different dimensionalities) produce different similarity distributions. Compute all pairwise cosine similarities, inspect quality at each band, and pick the cutoff where noise begins. There is no reliable formula — empirical calibration on actual data is required.

## 3.5 Intent-Adaptive Weighting

Different query intents activate different graph traversal weights:

| Intent      | Causal   | Temporal | Entity   | Semantic |
| ----------- | -------- | -------- | -------- | -------- |
| **WHY**     | **0.70** | 0.20     | 0.05     | 0.05     |
| **WHEN**    | 0.15     | **0.65** | 0.10     | 0.10     |
| **ENTITY**  | 0.10     | 0.05     | **0.55** | 0.30     |
| **GENERAL** | 0.25     | 0.25     | 0.25     | 0.25     |

When asking "why was SQLite chosen," causal edge weight is highest — the system traces decision rationale along causal chains. When asking for "memories related to React," entity edge weight is highest.

Weight distributions are inspired by MAGMA's intent-adaptive traversal (§3.3) and the weight ranges in Table 5, normalized to sum to 1.0. MAGMA Table 5 provides per-edge-type ranges (e.g., `w_causal`: 3.0–5.0, `w_phrase`/semantic: 2.5–5.0) but not per-intent distributions; MemMan interpolates specific per-intent values from these ranges and the paper's qualitative guidance (e.g., "Why queries trigger a bias for Causal edges"). The relative ordering is preserved: causal dominates WHY, temporal dominates WHEN, entity dominates ENTITY. GENERAL uses uniform 0.25 as the unbiased baseline.
