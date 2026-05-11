# 1. Background

[< Back to Design Overview](../DESIGN.md)

---

## 1.1 The amnesia problem

LLM agents lose context three ways:

- **Context compression loss**: after compaction, prior decisions and context disappear from the active window.
- **Cross-session forgetting**: each new session starts from scratch.
- **Long-session decay**: once the context window fills, early information falls out of attention range.

The result: users restate preferences, re-explain project context, and re-derive conclusions they already reached.

## 1.2 Scope

Persist decisions, preferences, and project context across sessions. memman is a CLI, not a library. Any LLM CLI that can shell out can call it.

---

## 1.3 LLM-Supervised pattern

Mem0 and the MAGMA reference implementation embed an LLM inside the memory pipeline for extraction, conflict detection, and causal reasoning. Call this the LLM-Embedded pattern. memman adopts a different one:

| Pattern            | Where is the LLM               | What does the LLM do                                                 | Representative        |
| ------------------ | ------------------------------ | -------------------------------------------------------------------- | --------------------- |
| **LLM-Embedded**   | Inside the pipeline            | Executor (extraction, classification, reasoning)                     | Mem0, MAGMA           |
| **File Injection** | Reads file at session start    | None — static file loaded into context window                        | Claude Code CLAUDE.md |
| **MCP Server**     | Tool provider via MCP protocol | Exposes memory operations as MCP tools for the host LLM              | MemCP                 |
| **LLM-Supervised** | Outside the pipeline           | Supervisor (reviews candidates, makes judgments, decides trade-offs) | memman                |

Responsibilities split into two tiers:

| Tier         | Role                      | Handles                                                                            |
| ------------ | ------------------------- | ---------------------------------------------------------------------------------- |
| **Binary**   | Deterministic computation | Storage, graph indexing, keyword search, vector math, decay formulas, auto-pruning |
| **Host LLM** | High-level judgment       | Decides what to remember, when to recall, which links to create, what to forget    |

The same binary + skill works across Claude Code, Cursor, or any LLM CLI. Swapping the host LLM requires no changes to the binary.

## 1.4 Theoretical foundations

memman draws on two directly implemented papers.

**MAGMA: four-graph memory architecture.**
[MAGMA](https://arxiv.org/abs/2601.03236) (Jiang et al., 2025) provides the data model and retrieval algorithms. Its central claim: a single edge type (e.g., vector similarity) is insufficient because different query intents demand different relational perspectives. memman inherits MAGMA's four-graph architecture (temporal, entity, causal, semantic) and intent-adaptive retrieval, and adopts the hyperparameter values from Table 5: anchor top-K (20), RRF constant (60), structural/semantic coefficients (λ1=1.0, λ2=0.3–0.7), max traversal depth (5), similarity threshold range (0.10–0.30). These values are documented inline in [Pipelines](04-pipelines.md) and [Graph Model](03-graph-model.md).

**RRF: Reciprocal Rank Fusion.**
The [RRF paper](https://dl.acm.org/doi/10.1145/1571941.1572114) (Cormack, Clarke & Buttcher, SIGIR 2009) provides the multi-signal fusion algorithm used in anchor selection. memman uses the exact `1/(k + rank)` formula with k=60, fusing keyword, vector, and recency signals into one composite ranking.

**Engineering choices.**
The pipeline uses three LLM role slots (recall fast path, worker canonical, worker metadata); see [§ LLM routing](04-pipelines.md#llm-routing) for the model assignments and cost-tuning rationale. The write path uses LLM reconciliation (ADD/UPDATE/DELETE/NONE) instead of threshold-based comparison. The lifecycle is hook-driven: remember → reconcile → enrich → auto-prune.

Where MAGMA's reference implementation is a Python library with in-memory NetworkX graphs, memman persists everything in SQLite (or Postgres + pgvector) with a complete write-back lifecycle and exposes the system through CLI commands — auditable, portable, sandboxed.

![LLM-Supervised Architecture](../diagrams/05-llm-supervised.drawio.png)

![System Architecture](../diagrams/01-system-architecture.drawio.png)

---

## 1.5 Design decisions & trade-offs

### Why LLM-Supervised instead of an embedded LLM?

| Dimension          | LLM-Embedded (Mem0, etc.) | LLM-Supervised (memman)                                                                                  |
| ------------------ | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| LLM capability     | One model for everything  | Host LLM + Haiku/Sonnet split for pipeline                                                               |
| Pipeline LLM       | One model for everything  | Haiku for recall fast path; Sonnet for extraction/reconciliation; Sonnet for enrichment/causal inference |
| Network dependency | Required                  | Required (OpenRouter + Voyage APIs)                                                                      |
| Swappability       | API-bound                 | Any LLM CLI                                                                                              |

### Why SQLite WAL instead of an embedded graph database?

- **Single-file deployment**: one `.db` file per store, easy to manage and back up.
- **ACID transactions**: atomicity for the remember pipeline.
- **WAL concurrency**: simultaneous hook reads and CLI writes.
- **Zero external dependencies**: no Redis/Neo4j/Qdrant required.
- **Store isolation**: named stores (`~/.memman/data/<name>/memman.db`) give data isolation via the `MEMMAN_STORE` env var.

### Why beam search instead of full BFS?

- **Budget control**: MaxVisited prevents graph explosion.
- **Intent-adaptive**: different intents use different beam widths and depths.
- **Quality assurance**: only the highest-scoring candidates survive each level.

### Why soft delete?

- Preserves audit trail.
- Supports undo (recovering accidental deletions).
- Simplifies cascade cleanup.
- Query consistency (`WHERE deleted_at IS NULL`).

### Key deviations from the MAGMA paper

| Aspect            | MAGMA Paper                                        | memman Implementation                                                                                                                                                                       |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transition Score  | `exp(λ1·φ + λ2·sim)` (Eq. 5) — exponential wrapper | Linear `λ1·structural + λ2·semantic` — better discrimination (`exp` compresses score ratios; irrelevant edges with zero structural+semantic contribution score `exp(0) = 1.0` instead of 0) |
| Depth Decay       | `score_v = score_u · γ + s_uv` (Alg. 1) — decay γ  | No decay — accumulative scoring. Mitigated by bounded depth (4-5), beam pruning, and min-max normalized multi-factor re-ranking. γ is unspecified in the paper                              |
| RRF Weighting     | `w_keyword (Fusion): 2.0-5.0` (Table 5)            | Standard unweighted RRF — all three signals (keyword, vector, recency) contribute equally via `1/(k + rank)`                                                                                |
| Traversal Budget  | `Max Nodes: 200` (Table 5)                         | 400-500 (intent-dependent). Flat node hierarchy (insights only, no episodes/narratives) requires a larger budget for equivalent coverage                                                    |
| Intent Types      | {Why, When, Entity} — 3 types                      | Adds GENERAL (uniform 0.25 weights) as a 4th intent for queries that don't match specific patterns                                                                                          |
| Entity Extraction | LLM-driven full pipeline                           | LLM-based: `extract_facts` runs against `slow_canonical` (Sonnet); `enrich_with_llm` runs against `slow_metadata` (Sonnet, separately tunable)                                              |
| Causal Reasoning  | Embedded prompt chain                              | LLM causal inference (`infer_llm_causal_edges`) via ThreadPoolExecutor                                                                                                                      |
| Deduplication     | Not addressed                                      | LLM reconciliation (ADD/UPDATE/DELETE/NONE)                                                                                                                                                 |
| Node Types        | EVENT, EPISODE, SESSION, NARRATIVE                 | Insight only (flat)                                                                                                                                                                         |
| Storage           | NetworkX (in-memory)                               | SQLite (persistent)                                                                                                                                                                         |
| Embeddings        | FAISS + OpenAI                                     | Voyage AI (voyage-3-lite, 512-dim)                                                                                                                                                          |
| Quality Review    | Slow-path LLM refinement (Alg. 3)                  | Pattern-based quality warnings + `memman insights review`                                                                                                                                   |
| Deployment        | Python library                                     | Python package (CLI)                                                                                                                                                                        |

## 1.6 Pluggability

Any LLM CLI interacts with memman through the CLI protocol (agent-side pluggability). On the storage side, the engine sits behind a backend Protocol so the same beam-search / RRF / lifecycle code runs over either of two ACID-aware backends.

### Storage-side pluggability

| Backend  | Install                          | Topology                                                              | Vector storage                               |
| -------- | -------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- |
| SQLite   | default                          | One `memman.db` per store under `~/.memman/data/<store>/`             | float64 BLOB in `insights.embedding`         |
| Postgres | `pip install 'memman[postgres]'` | One Postgres schema per store (`store_<name>`); shared `queue` schema | `pgvector` `vector(N)` (float32; HNSW index) |

Backend selection is per-store. `MEMMAN_BACKEND_<store>` (with `MEMMAN_DEFAULT_BACKEND` as fallback) chooses sqlite or postgres for that store; `MEMMAN_POSTGRES_DSN_<store>` (with `MEMMAN_DEFAULT_POSTGRES_DSN` as fallback) provides the connection string. Different stores under one `~/.memman/data/` can sit on different backends — a `work` store on Postgres can coexist with a `default` store on SQLite.

The graph-aware abstraction landed at the storage Protocol layer (`src/memman/store/backend.py`): node/edge access, drain-lock primitives, and queue verbs are virtualized so beam search and RRF fusion remain shared. SQLite-specific concerns (`PRAGMA`, `WAL`, BLOB serialization) and Postgres-specific concerns (pgvector adapters, schema-per-store, advisory locks) stay inside their respective implementations. Each backend has a single baseline schema (`_BASELINE_SCHEMA` / `PG_BASELINE_SCHEMA`); there is no in-place migration ladder.

`memman migrate <store>` is the operator path between backends. See [USAGE.md](../USAGE.md#migrating-between-sqlite-and-postgres) for the workflow and [CONTRIBUTING.md](../../CONTRIBUTING.md#migrating-between-sqlite-and-postgres) for the implementation outline.
