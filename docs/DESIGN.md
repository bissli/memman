# Mnemon — Design & Architecture

> **Mnemon** (/ˈniːmɒn/), from Ancient Greek μνήμων (mnemon), formed by μνάομαι ("to remember") and the agent suffix -μων, meaning "one who remembers, a person of good memory." Homer uses "καὶ γὰρ μνήμων εἰμί" ("I remember it well") in the *Odyssey* to describe this quality. In the city-states of Ancient Greece, Mnemones were officials dedicated to record-keeping, serving as witnesses and archivists in property transactions and legal proceedings — institutional memory carriers during the transition from oral tradition to written records.
>
> The word shares its root with Mnemosyne (Μνημοσύνη), the goddess of memory — from her union with Zeus the nine Muses were born, symbolizing memory as the wellspring of all knowledge and creativity.

Mnemon is a persistent memory system designed for LLM agents. It adopts the **LLM-Supervised** pattern: the host LLM acts as external orchestrator of a standalone memory binary through symbolic CLI interfaces, while Haiku handles pipeline intelligence (fact extraction, reconciliation, query expansion). Memory is organized as a four-graph knowledge structure with temporal, entity, causal, and semantic edges. Implemented as a Python package + SQLite, requiring Anthropic and Voyage AI API keys.

---

## Table of Contents

### [1. Vision & Problem](design/01-vision.md)

Why Mnemon exists — the amnesia problem in LLM agents and structural bottlenecks of traditional approaches.

### [2. Design Philosophy](design/02-philosophy.md)

The LLM-Supervised pattern, Organs vs Textbooks metaphor, Memory Gateway protocol (the MCP analogy for LLM↔DB interaction), key design insights, and theoretical foundations from MAGMA and RRF.

### [3. Core Concepts & Architecture](design/03-concepts.md)

The Insight/Edge data model, database schema (SQLite WAL), system architecture (CLI layer → engine → storage), code structure, and store isolation via named stores.

### [4. Graph Model & Structural Theory](design/04-graph-model.md)

MAGMA four-graph model (temporal, entity, causal, semantic), the Extract→Candidate→Associate paradigm, read-write symmetry, and `remember/link/recall` as universal algebra.

### [5. Read & Write Pipelines](design/05-pipelines.md)

The two-tier write pipeline (`remember` with LLM fact extraction and reconciliation), read pipeline (LLM query expansion, RRF anchor fusion, Beam Search traversal, multi-factor re-ranking).

### [6. Lifecycle & Embedding](design/06-lifecycle.md)

Effective Importance (EI) decay formula, immunity rules, auto-pruning, GC commands, and Voyage AI embedding support.

### [7. LLM CLI Integration](design/07-integration.md)

Lifecycle hooks (Prime, Remind, Nudge, Compact, Recall), skill file, behavioral guide, automated setup via `mnemon setup`, sub-agent delegation pattern, and adaptation to other LLM CLIs.

### [8. Design Decisions & Future Direction](design/08-decisions.md)

Key trade-offs (LLM-Supervised vs embedded, SQLite WAL vs graph DB, Beam Search vs BFS, soft delete), deviations from the MAGMA paper, storage-side pluggability roadmap, and the vision toward a memory gateway.
