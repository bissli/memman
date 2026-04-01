# 4. Graph Model & Theory

[< Back to Design Overview](../DESIGN.md)

---

MAGMA provides the specific data structure for the external environment that the LLM orchestrates. The core idea of the MAGMA paper is: **a single edge type (such as pure vector similarity) is insufficient to capture the multidimensional relationships between memories.** Different query intents require different relational perspectives — asking "why" requires causal chains, asking "when" requires timelines, asking "about X" requires entity associations.

Mnemon implements four graphs, each capturing one dimension of relationships:

![MAGMA Four-Graph Model](../diagrams/04-magma-four-graph.drawio.png)

## 4.1 Temporal Graph

**Purpose**: Capture the chronological order of memories, building a temporal skeleton of the knowledge flow.

**Automatically created edges**:

- **Backbone**: New insight → most recent insight from the same source (bidirectional)
  - Ensures memories from each source (user/agent) form a continuous timeline
- **Proximity**: New insight <-> insights within a 4-hour window (bidirectional)
  - Weight formula: `w = 1 / (1 + hours_diff)`
  - Up to 10 proximity edges

```
Insight A (2h ago) ←── backbone ──→ Insight B (1h ago) ←── backbone ──→ Insight C (now)
     ↑                                     ↑
     └──────── proximity (w=0.33) ─────────┘
```

**Metadata**: `{"sub_type": "backbone"|"proximity", "hours_diff": "2.34"}`

**Rationale:**

- **`TEMPORAL_WINDOW_HOURS = 4`**: A focused session window. Memories created within the same few hours are likely contextually related.
- **`MAX_PROXIMITY_EDGES = 10`**: Limits fan-out per insert. With ~10 insights/day as typical usage, this connects to most same-day memories without excessive edge density.

## 4.2 Entity Graph

**Purpose**: Link insights that mention the same entities.

**Entity extraction**: LLM-based via `extract_facts()` (Tier 1) and `enrich_with_llm()` (Tier 2). The LLM extracts entities as part of fact decomposition — no regex patterns or dictionaries. User-provided entities via `--entities` flag are merged with LLM-extracted ones.

**Automatically created edges**: New insight <-> up to 5 existing insights per shared entity (bidirectional). Edge weight is computed via IDF: rare entities (appearing in few insights) produce high-weight edges; common entities produce low-weight edges. When the store has ≤5 insights, all entity edges use weight 1.0 (IDF is not meaningful at that scale).

```
                   ┌─── "Qdrant" ───┐
                   │                │
Insight A ←── entity ──→ Insight B ←── entity ──→ Insight C
("Chose Qdrant")         ("Qdrant perf test")     ("Qdrant deployment config")
```

**Metadata**: `{"entity": "Qdrant"}`

**Rationale:**

- **`MAX_ENTITY_LINKS = 5` per entity**: Caps edge creation per shared entity to prevent popular entities from creating O(n) edges on every insert.
- **`MAX_TOTAL_ENTITY_EDGES = 50`**: Hard cap across all entities per insert.
- **IDF weighting**: Entity edge weight uses `log(N/df) / log(N)` (floored at 0.1), where N = total active insights and df = count of insights containing the entity. Common entities create weak edges; rare entities create strong edges. IDF is disabled when N ≤ 5.

## 4.3 Causal Graph

**Purpose**: Capture the reasons behind decisions and cause-effect relationships.

**LLM-only inference** (Tier 2, via `infer_llm_causal_edges()`):
1. Candidate discovery: 2-hop BFS neighbors + recent insights (up to 20), filtered by token overlap >= 15%
2. LLM evaluates candidates and returns edges with confidence scores
3. Edges below `LLM_CONFIDENCE_FLOOR` (0.75) are rejected
4. Sub-types: `causes` (direct cause), `enables` (enabling condition), `prevents` (preventing factor)

```
Insight A ──── causal ────→ Insight B
("Team lacks Redis exp.")   ("Chose SQLite as storage")
  sub_type: "causes"
  weight: 0.80 (LLM confidence)
```

**Rationale:**

- **`MIN_CAUSAL_OVERLAP = 0.15`**: Filters BFS candidates — below this, shared tokens are incidental.
- **`MAX_CAUSAL_CANDIDATES = 10`**: Caps candidates sent to LLM.
- **`LLM_CONFIDENCE_FLOOR = 0.75`**: High bar for accepting inferred edges.

## 4.4 Semantic Graph

**Purpose**: Connect semantically similar insights based on meaning.

**Auto-link**: Cosine similarity >= 0.70 (`AUTO_SEMANTIC_THRESHOLD`), top 3 per insight (`MAX_AUTO_SEMANTIC_EDGES`). Bidirectional edges created automatically.

Embeddings are Voyage AI 512-dim vectors. Both Tier 1 (initial insert) and Tier 2 (post-enrichment re-embed) create semantic edges.

**Rationale:**

- **`AUTO_SEMANTIC_THRESHOLD = 0.70`**: High-confidence cutoff. With Voyage `voyage-3-lite`, 0.70 cosine represents strong semantic overlap.
- **`MAX_AUTO_SEMANTIC_EDGES = 3`**: Limits edges per insert to prevent over-linking on dense topics.

```
Insight A ←── semantic (auto, cos=0.92) ──→ Insight B

Insight C ←── semantic (LLM review) ──→ Insight D
                cos=0.65, manually linked after LLM judged "related"
```

## 4.5 Four-Graph Synergy: Intent-Adaptive Weighting

Different query intents activate different graph traversal weights:

| Intent      | Causal   | Temporal | Entity   | Semantic |
| ----------- | -------- | -------- | -------- | -------- |
| **WHY**     | **0.70** | 0.20     | 0.05     | 0.05     |
| **WHEN**    | 0.15     | **0.65** | 0.10     | 0.10     |
| **ENTITY**  | 0.10     | 0.05     | **0.55** | 0.30     |
| **GENERAL** | 0.25     | 0.25     | 0.25     | 0.25     |

When asking "why was SQLite chosen," the causal edge weight is highest, so the system traces decision rationale along causal chains. When asking for "memories related to React," the entity edge weight is highest, so the system finds all insights mentioning React.

**Rationale:** Weight distributions follow MAGMA's unnormalized weight ranges (Table 5: `w_causal: 3.0–5.0`, `w_temporal: 0.5–4.0`, `w_entity: 2.5–6.0`, `w_phrase: 2.5–5.0`), normalized to sum to 1.0. The relative ordering is preserved: causal dominates WHY, temporal dominates WHEN, entity dominates ENTITY. GENERAL uses uniform 0.25 as the unbiased baseline. *(Cite: MAGMA, Jiang et al., arXiv 2601.03236, Table 5: intent-specific edge type weights)*

---

## 4.6 Graph Memory Theory

The following sections establish why graph databases are well-suited for agent memory, and why `remember / link / recall` constitutes a universal protocol for agent memory systems.

### The Three-Step Paradigm: Extract → Candidate → Associate

Graph construction engines universally decompose into three steps:

| Step          | Purpose                               | mnemon Implementation                       |
| ------------- | ------------------------------------- | ------------------------------------------- |
| **Extract**   | Parse raw input into structured units | `remember` → nodes + entities               |
| **Candidate** | Find potential connections            | `semantic_candidates` / `causal_candidates` |
| **Associate** | Establish typed, weighted edges       | `link` → 4 edge types                       |

### Spectrum Across Database Types

The three-step model is a spectrum — the more semantically rich the data model, the more complete all three steps are:

| Database Type  | Extract           | Candidate           | Associate                                |
| -------------- | ----------------- | ------------------- | ---------------------------------------- |
| **Graph**      | Full              | Full                | Full (multi-type edges)                  |
| **Relational** | Schema mapping    | PK/unique dedup     | Foreign keys (fixed at DDL time)         |
| **Document**   | Structure mapping | _id dedup           | Nested refs (lose global traversability) |
| **Vector**     | Text → embedding  | ANN dedup           | Metadata only (single relation type)     |
| **KV**         | Key:value         | Key existence check | _(nearly none)_                          |

### Read-Write Symmetry (Unique to Graphs)

On graph databases, the read and write paths mirror each other using the same three-step model:

```
Write:  Extract → Candidate → Associate     (text → graph)
Read:   Extract → Candidate → Associate     (graph → text)
        (parse    (retrieve)   (traverse)
         query)
```

|               | Write (Construction)                     | Read (Query)                        |
| ------------- | ---------------------------------------- | ----------------------------------- |
| **Extract**   | Text → entities/facts                    | Question → intent/keywords          |
| **Candidate** | Find potential related nodes             | Find potential matching nodes       |
| **Associate** | Create edges (persist)                   | Traverse edges (rank & return)      |
| **Reason**    | _(optional: LLM judges whether to link)_ | LLM synthesizes results into answer |

This symmetry does NOT hold for other database types — relational write is schema mapping while read is join planning; the two share no cognitive model.

**Implication**: An LLM needs to master only one cognitive pattern to handle both graph reads and writes.

### From the LLM Perspective: Query → Reason

Regardless of the underlying database, LLM interactions on the read side collapse to two steps:

```
Natural language → [Query (tool call)] → Structured results → [Reason] → Natural language answer
```

This is the RAG paradigm applied to any data store. The variation lies in the translation layer complexity:

- **Text-to-SQL**: must understand schema
- **Text-to-Cypher**: must understand graph structure
- **Text-to-Vector**: encode only, near-zero translation

### Other Storage Types as Degenerate Graphs

| Storage Type   | What's Lost Compared to Graph                                     |
| -------------- | ----------------------------------------------------------------- |
| **KV**         | Isolated nodes, zero edges                                        |
| **Relational** | Edges compressed to foreign keys, types fixed in schema           |
| **Document**   | Edges inlined as nesting, global traversability lost              |
| **Vector**     | All edges are a single type (similarity), no semantic distinction |

A vector database can answer "what is **similar** to what" but cannot answer "what **caused** what" or "what **belongs to** what". Graphs can.

### remember / link / recall as Universal Algebra

The three-step paradigm (Extract → Candidate → Associate) maps directly to three primitive operations: **remember**, **link**, **recall**. This is not an implementation detail of mnemon — it is the minimal complete interface for any agent memory system.

```
Any memory system = remember(write) + link(associate) + recall(retrieve)
```

### Cross-System Validation

| System           | remember                | link                                             | recall                            |
| ---------------- | ----------------------- | ------------------------------------------------ | --------------------------------- |
| **mnemon**       | Explicit three-step     | Explicit 4 edge types                            | Graph traversal                   |
| **OpenViking**   | File write to viking:// | Directory placement (implicit, containment only) | Path navigation + semantic search |
| **mem0**         | add()                   | Auto dedup/merge                                 | search()                          |
| **Letta/MemGPT** | insert()                | Tiered storage (core/recall/archival)            | query()                           |
| **Native RAG**   | embed + upsert          | _(none)_                                         | ANN search                        |

Every system implements all three operations. The differences lie in:

1. **Explicitness of link**: From mnemon's explicit multi-type edges to RAG's complete absence
2. **Timing of link**: Pre-computed at write time (mnemon) vs inferred by LLM at query time (OpenViking)
3. **Signal dimensions of recall**: Multi-signal weighted (mnemon) vs single-signal (vector/path)

### OpenViking: link Folded into remember

OpenViking adopts a file system paradigm — memories, resources, and skills are organized as directories under the `viking://` protocol with L0/L1/L2 tiered context loading. Its `link` step has not disappeared but is **folded into `remember`**: choosing which directory to place a file in IS the linking decision, reduced to a single classification problem (containment edge only).

This represents an explicit architectural trade-off: push association complexity from storage time to inference time, relying on LLM's reasoning capability within the context window. This works when memory volume fits within context limits, but loses advantage as memories scale beyond what the LLM can process in a single pass.

### Degeneracy Spectrum

The three primitives form a spectrum of degeneracy:

```
mnemon          fully explicit remember + link + recall
OpenViking      link folded into remember (implicit containment)
mem0            link automated (dedup/merge heuristics)
Letta/MemGPT    link reduced to tier placement
Native RAG      link absent entirely
```

The more degenerate the `link` operation, the more burden falls on the LLM at recall time to infer associations that were never stored.

See [Design Philosophy](02-philosophy.md#23-memory-gateway-protocol-not-database) for the protocol gap analysis and protocol primitives.
