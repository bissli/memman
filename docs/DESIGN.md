# MemMan — Design & Architecture

MemMan is a persistent memory system designed for LLM agents. It adopts the **LLM-Supervised** pattern: the host LLM acts as external orchestrator of a standalone memory binary through symbolic CLI interfaces, while Haiku handles pipeline intelligence (fact extraction, reconciliation, query expansion). Memory is organized as a four-graph knowledge structure with temporal, entity, causal, and semantic edges. Implemented as a Python package + SQLite, requiring Anthropic and Voyage AI API keys.

---

## Table of Contents

### [1. Vision & Problem](design/01-vision.md)

Why MemMan exists — the amnesia problem in LLM agents and structural limitations of traditional approaches.

### [2. Design Philosophy](design/02-philosophy.md)

The LLM-Supervised pattern, theoretical foundations from MAGMA and RRF.

### [3. Core Concepts & Architecture](design/03-concepts.md)

The Insight/Edge data model, database schema (SQLite WAL), system architecture (CLI layer → engine → storage), code structure, and store isolation via named stores.

### [4. Graph Model](design/04-graph-model.md)

MAGMA four-graph model (temporal, entity, causal, semantic) with creation logic, thresholds, and metadata for each edge type.

### [5. Read & Write Pipelines](design/05-pipelines.md)

The single-tier synchronous write pipeline (`remember` with LLM fact extraction, reconciliation, and parallel enrichment/causal inference via ThreadPoolExecutor), read pipeline (LLM query expansion, RRF anchor fusion, beam search traversal, multi-factor re-ranking).

### [6. Lifecycle & Embedding](design/06-lifecycle.md)

Effective Importance (EI) decay formula, immunity rules, auto-pruning, GC commands, and Voyage AI embedding support.

### [7. LLM CLI Integration](design/07-integration.md)

Lifecycle hooks (Prime, Remind, Nudge, Compact, Recall), skill file, behavioral guide, automated setup via `memman setup`, sub-agent delegation pattern, and adaptation to other LLM CLIs.

### [8. Design Decisions & Future Direction](design/08-decisions.md)

Key trade-offs (LLM-Supervised vs embedded, SQLite WAL vs graph DB, beam search vs BFS, soft delete), deviations from the MAGMA paper, and storage-side pluggability roadmap.
