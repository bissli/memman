# 3. Graph Model

[< Back to Design Overview](../DESIGN.md)

---

MAGMA's argument: a single edge type (e.g., vector similarity) cannot capture every relationship between memories. WHY asks for causal chains, WHEN for timelines, ABOUT-X for entity associations. memman implements four graphs, each capturing one relational dimension.

![MAGMA Four-Graph Model](../diagrams/04-magma-four-graph.drawio.png)

## 3.1 Edge type reference

| Edge Type | Purpose                                                      | How edges are created                                                                                                                                                                                  | Weight                                                         | Key constants                                                                                                                               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Temporal  | Chronological ordering and session-window co-occurrence      | Auto. Two sub-types: a `backbone` edge from each new insight to the most recent insight from the same source, and bidirectional `proximity` edges to insights within a `TEMPORAL_WINDOW_HOURS` window. | proximity: `1 / (1 + hours_diff)`; backbone: `1.0`             | `TEMPORAL_WINDOW_HOURS = 4`, `MAX_PROXIMITY_EDGES = 5`                                                                                      |
| Entity    | Link insights that mention the same entity                   | Auto. Bidirectional edges to insights sharing each LLM-extracted (or `--entities` provided) entity. Edge weight is IDF-weighted so rare entities produce stronger links.                               | `log(N/df) / log(N)` floored at 0.1; `1.0` when N ≤ 5          | `MAX_ENTITY_LINKS = 5` per entity, `MAX_TOTAL_ENTITY_EDGES = 50` per insert                                                                 |
| Causal    | Capture cause-effect relationships and decision rationale    | LLM-inferred during enrichment from a bounded candidate pool (2-hop BFS + recent insights, ≥15% token overlap). Sub-types `causes`/`enables`/`prevents` come from the LLM.                             | LLM confidence (0–1); rejected if below `LLM_CONFIDENCE_FLOOR` | `MIN_CAUSAL_OVERLAP = 0.15`, `MAX_CAUSAL_CANDIDATES = 10`, `LLM_BFS_NEIGHBORS = 10`, `LLM_RECENT_COUNT = 20`, `LLM_CONFIDENCE_FLOOR = 0.75` |
| Semantic  | Connect semantically similar insights via embedding distance | Auto. Bidirectional edges to the top `MAX_AUTO_SEMANTIC_EDGES` insights with cosine similarity ≥ `AUTO_SEMANTIC_THRESHOLD`. Recomputed after enrichment when keywords are appended to the embedding.   | cosine similarity                                              | `AUTO_SEMANTIC_THRESHOLD = 0.62`, `MAX_AUTO_SEMANTIC_EDGES = 3`                                                                             |

Edge metadata is JSON:

- temporal: `{"sub_type": "proximity", "hours_diff": "2.34"}`
- entity:   `{"entity": "Qdrant"}`
- causal:   `{"created_by": "llm", "confidence": 0.80, "sub_type": "causes", "rationale": "..."}`
- semantic: `{"created_by": "auto", "cosine": "0.8234"}`

> **Threshold recalibration.** `AUTO_SEMANTIC_THRESHOLD = 0.62` was calibrated against the current default embedding model (`voyage-3-lite`, 512-dim) — pairs above 0.62 are genuine topical links; noise begins ~0.50. Different embedding models produce different similarity distributions. When switching providers, compute all pairwise cosines, inspect quality bands, and pick the cutoff where noise begins. There is no reliable formula.

> **Causal candidate-pool rationale.** The 2-hop BFS + recent-insights union keeps the LLM's input bounded while surfacing both topologically near (recently linked) and temporally near candidates. The 15% token overlap floor and `LLM_CONFIDENCE_FLOOR = 0.75` together reject incidental token overlap and uncertain LLM judgments.

## 3.2 Intent-adaptive weighting

Different query intents activate different graph-traversal weights:

| Intent      | Causal   | Temporal | Entity   | Semantic |
| ----------- | -------- | -------- | -------- | -------- |
| **WHY**     | **0.70** | 0.20     | 0.05     | 0.05     |
| **WHEN**    | 0.15     | **0.65** | 0.10     | 0.10     |
| **ENTITY**  | 0.10     | 0.05     | **0.55** | 0.30     |
| **GENERAL** | 0.25     | 0.25     | 0.25     | 0.25     |

"Why was SQLite chosen?" puts causal weight highest — the system traces decision rationale along causal chains. "Memories related to React" puts entity weight highest.

Distributions are inspired by MAGMA's intent-adaptive traversal (§3.3) and the weight ranges in Table 5, normalized to sum to 1.0. MAGMA Table 5 gives per-edge-type ranges (e.g., `w_causal` 3.0–5.0, `w_phrase`/semantic 2.5–5.0) but not per-intent distributions; memman interpolates per-intent values from those ranges and the paper's qualitative guidance ("Why queries trigger a bias for Causal edges"). The relative ordering is preserved — causal dominates WHY, temporal dominates WHEN, entity dominates ENTITY. GENERAL uses uniform 0.25 as an unbiased baseline.
