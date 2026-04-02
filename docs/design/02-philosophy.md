# 2. Design Philosophy

[< Back to Design Overview](../DESIGN.md)

---

## 2.1 LLM-Supervised Pattern

Traditional LLM memory systems (such as Mem0 and the original MAGMA implementation) embed a small LLM inside the pipeline to handle memory operations — entity extraction, conflict detection, causal reasoning. This is the **LLM-Embedded** pattern.

MemMan adopts the **LLM-Supervised** pattern:

| Pattern            | Where is the LLM               | What does the LLM do                                                 | Representative        |
| ------------------ | ------------------------------ | -------------------------------------------------------------------- | --------------------- |
| **LLM-Embedded**   | Inside the pipeline            | Executor (extraction, classification, reasoning)                     | Mem0, MAGMA           |
| **File Injection** | Reads file at session start    | None — static file loaded into context window                        | Claude Code CLAUDE.md |
| **MCP Server**     | Tool provider via MCP protocol | Exposes memory operations as MCP tools for the host LLM              | MemCP                 |
| **LLM-Supervised** | Outside the pipeline           | Supervisor (reviews candidates, makes judgments, decides trade-offs) | MemMan                |

Responsibilities split into two tiers:

| Tier         | Role                      | Handles                                                                            |
| ------------ | ------------------------- | ---------------------------------------------------------------------------------- |
| **Binary**   | Deterministic computation | Storage, graph indexing, keyword search, vector math, decay formulas, auto-pruning |
| **Host LLM** | High-level judgment       | Decides what to remember, when to recall, which links to create, what to forget    |

Haiku handles pipeline intelligence (fact extraction, reconciliation, enrichment, causal inference, query expansion). The host LLM decides *when* and *what* to remember — it does not execute pipeline steps directly.

The same binary + skill works across Claude Code, Cursor, or any LLM CLI. Swapping the host LLM requires no changes to the binary.

## 2.2 Theoretical Foundations

MemMan's design draws on two directly implemented papers.

**MAGMA: Four-Graph Memory Architecture**

The [MAGMA](https://arxiv.org/abs/2601.03236) paper (Jiang et al., 2025) provides the data model and retrieval algorithms. Its key contribution: a single edge type (e.g., vector similarity) is insufficient for memory — different query intents require different relational perspectives. MAGMA's four-graph architecture (temporal, entity, causal, semantic) with intent-adaptive retrieval and multi-signal fusion gives MemMan its graph model and recall pipeline.

MAGMA also provides specific hyperparameter values adopted by MemMan. See Table 5 of the MAGMA paper for: anchor top-K (20), RRF constant (60), structural/semantic coefficients (λ1=1.0, λ2=0.3–0.7), max traversal depth (5), and similarity threshold range (0.10–0.30). These values are documented inline in [Pipelines](05-pipelines.md) and [Graph Model](04-graph-model.md).

**RRF: Reciprocal Rank Fusion**

The [RRF paper](https://dl.acm.org/doi/10.1145/1571941.1572114) (Cormack, Clarke & Buttcher, SIGIR 2009) provides the multi-signal fusion algorithm used in recall anchor selection. MemMan uses the exact `1/(k + rank)` formula with k=60, fusing keyword, vector, and recency signals into a composite anchor ranking.

**MemMan's Engineering Choices**

MemMan uses Haiku for fact extraction, reconciliation, enrichment, causal inference, and query expansion. The host LLM handles higher-level judgment (what to remember, when to recall). The write path uses LLM reconciliation (ADD/UPDATE/DELETE/NONE) instead of threshold-based comparison. The lifecycle is hook-driven: remember → reconcile → link → gc.

Where MAGMA's reference implementation is a Python library with in-memory NetworkX graphs, MemMan persists everything in SQLite with a complete write-back lifecycle. CLI commands as the interface — constrained, but auditable, portable, and sandboxed.

![LLM-Supervised Architecture](../diagrams/05-llm-supervised.drawio.png)

![System Architecture](../diagrams/01-system-architecture.drawio.png)
