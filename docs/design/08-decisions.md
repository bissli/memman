# 8. Design Decisions & Future Direction

[< Back to Design Overview](../DESIGN.md)

---

## 8.1 Design Decisions & Trade-offs

### Why LLM-Supervised Instead of an Embedded LLM?

| Dimension          | LLM-Embedded (Mem0, etc.) | LLM-Supervised (Mnemon)                                                       |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------- |
| LLM Capability     | Same model for everything | Host LLM + Haiku for pipeline                                                 |
| Pipeline LLM       | Same model for everything | Haiku for extraction, reconciliation, expansion, enrichment, causal inference |
| Network Dependency | Required                  | Required (Anthropic + Voyage APIs)                                            |
| Swappability       | API-bound                 | Any LLM CLI                                                                   |

### Why SQLite WAL Instead of an Embedded Graph Database?

- **Single-file deployment**: one `.db` file per store — easy to manage and backup
- **ACID transactions**: Atomicity guarantee for the remember pipeline
- **WAL concurrency**: Supports simultaneous hook reads and CLI writes
- **Zero external dependencies**: No Redis/Neo4j/Qdrant required
- **Store isolation**: Named stores (`~/.mnemon/data/<name>/mnemon.db`) provide lightweight data isolation via `MNEMON_STORE` env var

### Why Beam Search Instead of Full BFS?

- **Budget control**: MaxVisited parameter prevents graph explosion
- **Intent-adaptive**: Different intents use different beam widths and depths
- **Quality assurance**: Only the highest-scoring candidates are retained at each level, similar to pruning

### Why Soft Delete?

- Preserves audit trail
- Supports "undo" (recovering accidental deletions)
- Simplifies cascade cleanup
- Query consistency (`WHERE deleted_at IS NULL`)

### Key Deviations from the MAGMA Paper

| Aspect            | MAGMA Paper                        | Mnemon Implementation                                                  |
| ----------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| Entity Extraction | LLM-driven full pipeline           | LLM-based via Haiku (`extract_facts` + `enrich_with_llm`)              |
| Causal Reasoning  | Embedded prompt chain              | LLM causal inference (`infer_llm_causal_edges`) via ThreadPoolExecutor |
| Deduplication     | Not addressed                      | LLM reconciliation (ADD/UPDATE/DELETE/NONE)                            |
| Node Types        | EVENT, EPISODE, SESSION, NARRATIVE | Insight only (flat)                                                    |
| Storage           | NetworkX (in-memory)               | SQLite (persistent)                                                    |
| Embeddings        | FAISS + OpenAI                     | Voyage AI (voyage-3-lite, 512-dim)                                     |
| Quality Review    | Slow-path LLM refinement (Alg. 3)  | Pattern-based quality warnings + async gc --review                     |
| Deployment        | Python library                     | Python package (CLI)                                                   |

Mnemon retains MAGMA's **architectural skeleton** (four-graph separation, intent-adaptive retrieval, multi-signal fusion) while using Haiku for pipeline intelligence (fact extraction, reconciliation, query expansion, enrichment, causal inference) and the host LLM for high-level judgment.

---

## 8.2 Future Direction

Any LLM CLI can interact with Mnemon through the CLI protocol today (agent-side pluggability). The remaining work is on the storage side.

### Storage-Side Pluggability

The storage engine is currently tightly built on SQLite — graph traversal, EI decay, and atomic transactions all depend on SQLite-specific features (WAL, single-file deployment, in-process access). This is the right choice for the current goal of zero-dependency single-package distribution, but it means the storage backend is not yet swappable.

Abstracting the storage interface — so the protocol layer can sit on top of PostgreSQL, a dedicated graph database, or a remote service — is the next architectural milestone.

The key challenge is defining the right abstraction boundary: too high and you lose the storage engine's graph-aware optimizations; too low and every backend must reimplement beam search and RRF fusion.
