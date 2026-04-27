# 2. Core Concepts & Architecture

[< Back to Design Overview](../DESIGN.md)

---

![Insight & Edge Data Model](../diagrams/08-insight-edge-datamodel.drawio.png)

## 2.1 Insight (Memory Node)

An Insight is the fundamental memory unit in MemMan. Each insight represents an independent piece of knowledge:

```
┌──────────────────────────────────────────────┐
│ Insight                                      │
├──────────────────────────────────────────────┤
│ id         : UUID                            │
│ content    : "Chose Qdrant over Milvus..."   │
│ category   : decision                        │
│ importance : 5  (1-5)                        │
│ tags       : ["vector-db", "architecture"]   │
│ entities   : ["Qdrant", "Milvus"]            │
│ source     : "user"                          │
│ access_count        : 3                      │
│ effective_importance : 0.85                  │
│ created_at : 2026-02-18T10:00:00Z            │
└──────────────────────────────────────────────┘
```

**Categories** are divided into six types that help distinguish the nature of a memory:

| Category     | Meaning                          | Example                                             |
| ------------ | -------------------------------- | --------------------------------------------------- |
| `preference` | User preference                  | "Prefers communicating in Chinese"                  |
| `decision`   | Architectural/technical decision | "Chose SQLite over PostgreSQL"                      |
| `fact`       | Objective fact                   | "API rate limit is 100 req/s"                       |
| `insight`    | Reasoning conclusion             | "Beam search is more suitable than full BFS for..." |
| `context`    | Project context                  | "Phase 3 completed, 118 tests passing"              |
| `general`    | General                          | Content that doesn't fit the above categories       |

**Importance** ranges from 2 to 5 and affects retrieval ranking and lifecycle. The CLI accepts `--imp 1` but the LLM extraction pipeline floors it at 2 — `1` is reserved for raw `--no-reconcile` writes only:

- **5**: Critical decision, never automatically cleaned up
- **4**: Important fact, immune to auto-pruning
- **3**: Standard memory (default `--imp`)
- **2**: Low priority / passing mention (effective floor for extracted facts)

## 2.2 Edge (Relationship)

An Edge connects two insights, representing their relationship. Each edge contains:

```
┌────────────────────────────────────────────┐
│ Edge                                       │
├────────────────────────────────────────────┤
│ source_id  : UUID  ──→  target_id : UUID   │
│ edge_type  : temporal | semantic |         │
│              causal   | entity             │
│ weight     : 0.0 ~ 1.0                     │
│ metadata   : {"sub_type": "backbone", ...} │
└────────────────────────────────────────────┘
```

The four edge types form the foundation of the MAGMA four-graph model, detailed in [Graph Model & Theory](03-graph-model.md).

## 2.3 Database Schema

Each named store has its own SQLite file under `~/.memman/data/<store>/memman.db`, using WAL mode to support concurrent reads. The default store is `default`; additional stores can be created for data isolation (see [Store Management](../USAGE.md#store-management)).

```sql
-- Memory nodes
insights (
  id, content, category, importance,
  tags, entities, source,
  embedding,                                    -- embedding vector (BLOB)
  keywords, summary, semantic_facts,            -- LLM enrichment columns
  access_count, last_accessed_at,
  effective_importance,                         -- Decayed effective importance
  linked_at, enriched_at,                       -- Pipeline progress timestamps
  prompt_version, model_id, embedding_model,    -- Provenance for re-enrichment
  created_at, updated_at, deleted_at
)

-- Relationship edges (composite primary key)
edges (
  source_id, target_id, edge_type,  -- PK
  weight, metadata, created_at
)

-- Operation log (audit trail, queryable with --since/--stats)
oplog (
  id, operation, insight_id, detail, created_at
)

-- Key/value metadata (e.g., embed/graph constants fingerprints)
meta (
  key, value
)
```

**Provenance columns** (`prompt_version`, `model_id`, `embedding_model`)
record which LLM and embedding model produced each insight. They power
`memman embed reembed` and `memman graph rebuild` decisions when models
or prompts change.

**Insight dataclass vs DB schema.** The `Insight` dataclass in
`src/memman/model.py` is a subset of the DB schema — it holds the
identity, content, category, importance, tags, entities, source,
timestamps, access bookkeeping, effective_importance, and provenance
columns. The enrichment payload (`embedding`, `keywords`, `summary`,
`semantic_facts`, `linked_at`, `enriched_at`) lives in the DB only and
is read/written through SQL helpers, not via the dataclass.

---

## 2.4 System Architecture

MemMan's architecture is divided into five layers:

```
┌──────────────────────────────────────────────────────────────┐
│  Integration Layer    Hook / Skill / Guide                   │
├──────────────────────────────────────────────────────────────┤
│  CLI Layer            remember · recall · replace · forget   │
│                       prime · status · doctor · install      │
│                       graph · scheduler · insights · store   │
│                       embed · log · config                   │
├──────────────────────────────────────────────────────────────┤
│  Pipeline             pipeline/ (remember, drain worker)     │
├──────────────────────────────────────────────────────────────┤
│  Core Engine          search/ (recall, intent, keyword,      │
│                                quality)                      │
│                       graph/  (temporal, entity, causal,     │
│                                semantic, engine, bfs,        │
│                                enrichment)                   │
│                       embed/  (voyage, openai_compat,        │
│                                ollama, vector)               │
│                       llm/    (client, extract,              │
│                                openrouter_client)            │
├──────────────────────────────────────────────────────────────┤
│  Storage Layer        store/   (db, node, edge, oplog,       │
│                                snapshot)                     │
│                       queue.py (deferred-write queue)        │
├──────────────────────────────────────────────────────────────┤
│  External             OpenRouter (LLM, ZDR-routed Haiku)     │
│                       Voyage AI (embeddings, default)        │
└──────────────────────────────────────────────────────────────┘
```


**Project code structure:**

```
memman/
├── src/memman/
│   ├── __init__.py
│   ├── cli.py               # Click CLI (all commands)
│   ├── model.py              # Data structures (Insight, Edge)
│   ├── store/                # SQLite persistence
│   ├── graph/                # MAGMA four-graph implementation
│   ├── search/               # Retrieval algorithms
│   ├── embed/                # Embedding support (Voyage AI)
│   ├── llm/                  # LLM client + extraction/reconciliation
│   └── setup/                # LLM CLI integration setup
├── tests/
├── pyproject.toml            # Poetry package config
└── Makefile
```

## 2.5 Data Directory Layout

```
~/.memman/
├── active                        # Current default store name (plain text)
├── env                           # Mode-600 API-key exports for the scheduler
├── queue.db                      # Deferred-write queue (SQLite)
├── cache/                        # LLM response cache
├── compact/                      # Session-compact flag files
├── logs/                         # Scheduler stdout/stderr
│   ├── enrich.log
│   └── enrich.err
└── data/                         # Each store has its own isolated directory
    ├── default/
    │   └── memman.db             # SQLite database (WAL mode)
    ├── work/
    │   └── memman.db
    └── <name>/
        └── memman.db
```

**Isolation boundary**: Each store contains an independent `memman.db` — insights, edges, and oplog are fully isolated. Shipped assets (`guide.md`, `SKILL.md`) live inside the installed package and are read via `importlib.resources`; nothing memman deploys lives under `~/.memman/`. `~/.memman/` is strictly user state: memory data, API keys, caches, logs, queued work.

## 2.6 Store Isolation

MemMan supports named stores for lightweight data isolation between different agents, projects, or scenarios.

**Why named stores instead of just `--data-dir`?**

`--data-dir` overrides the entire base directory — a blunt instrument that requires the caller to manage full paths. Named stores provide semantic clarity (`MEMMAN_STORE=work` vs `--data-dir ~/.memman-work`) and work naturally with environment variables, which are the standard isolation mechanism for concurrent processes.

**Resolution priority** (highest to lowest):

```
--store flag  >  MEMMAN_STORE env  >  ~/.memman/active file  >  "default"
```

This layered design serves different scenarios:

| Mechanism          | Scenario                                                      |
| ------------------ | ------------------------------------------------------------- |
| `--store` flag     | One-off CLI override, scripting                               |
| `MEMMAN_STORE` env | Per-process isolation — different agents use different stores |
| `active` file      | Persistent user preference — `memman store use work`          |
| `"default"`        | Zero-config — works out of the box                            |

**Design principle — lightweight and bounded**: Store isolation addresses a necessary data separation concern without growing into a multi-tenant system. There are no access controls, no cross-store queries, no store metadata beyond the name. This keeps the feature bounded — MemMan is a memory daemon, not a knowledge base platform.
