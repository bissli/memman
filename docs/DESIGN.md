# MemMan — Design & Architecture

MemMan is a persistent memory system designed for LLM agents. It adopts the **LLM-Supervised** pattern: the host LLM acts as external orchestrator of a standalone memory binary through symbolic CLI interfaces, while Haiku (routed through OpenRouter) handles pipeline intelligence (fact extraction, reconciliation, query expansion). Memory is organized as a four-graph knowledge structure with temporal, entity, causal, and semantic edges. Writes are deferred to a scheduler-driven background worker so the host session never blocks on LLM calls. Implemented as a Python package + SQLite, requiring OpenRouter and Voyage AI API keys.

---

## Table of Contents

### [1. Background](design/01-background.md)

The amnesia problem MemMan addresses, the LLM-Supervised pattern, theoretical foundations from MAGMA and RRF, and key design trade-offs (LLM-Supervised vs embedded, SQLite WAL vs graph DB, beam search vs BFS, soft delete) including deviations from the MAGMA paper and the storage-side pluggability roadmap.

### [2. Core Concepts & Architecture](design/02-concepts.md)

The Insight/Edge data model, database schema (SQLite WAL), system architecture (CLI layer → engine → storage), code structure, and store isolation via named stores.

### [3. Graph Model](design/03-graph-model.md)

MAGMA four-graph model (temporal, entity, causal, semantic) with creation logic, thresholds, and metadata for each edge type.

### [4. Read & Write Pipelines](design/04-pipelines.md)

The two-tier deferred write pipeline (`remember` is a queue-append; a scheduler-driven `scheduler drain --pending` worker runs fact extraction, reconciliation, parallel enrichment/causal inference out of band). Scheduler installs per platform (systemd on Linux, launchd on macOS) and routes LLM calls through OpenRouter. Read pipeline (LLM query expansion, RRF anchor fusion, beam search traversal, multi-factor re-ranking).

### [5. Lifecycle & Embedding](design/05-lifecycle.md)

Effective Importance (EI) decay formula, immunity rules, auto-pruning, GC commands, and Voyage AI embedding support.

### [6. LLM CLI Integration](design/06-integration.md)

Six lifecycle hooks (Prime, Remind, Nudge, Compact, Recall, ExitPlan), skill file, behavioral guide, automated setup via `memman install`. The host agent calls `memman remember` directly via Bash — no sub-agent delegation — because the binary is a fast queue-append. Supported targets: claude-code, openclaw, nanoclaw.
