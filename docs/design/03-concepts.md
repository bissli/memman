# 3. Core Concepts & Architecture

[< Back to Design Overview](../DESIGN.md)

---

![Insight & Edge Data Model](../diagrams/09-insight-edge-datamodel.drawio.png)

## 3.1 Insight (Memory Node)

An Insight is the fundamental memory unit in MemMan. Each insight represents an independent piece of knowledge:

```
┌──────────────────────────────────────────────┐
│ Insight                                      │
├─────────────────────────────────────────────┤
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

**Importance** ranges from 1 to 5 and affects retrieval ranking and lifecycle:

- **5**: Critical decision, never automatically cleaned up
- **4**: Important fact, immune to auto-pruning
- **3**: Standard memory
- **2**: Low priority
- **1**: Temporary information, first to be cleaned up

## 3.2 Edge (Relationship)

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

The four edge types form the foundation of the MAGMA four-graph model, detailed in [Graph Model & Theory](04-graph-model.md).

## 3.3 Database Schema

Each named store has its own SQLite file under `~/.memman/data/<store>/memman.db`, using WAL mode to support concurrent reads. The default store is `default`; additional stores can be created for data isolation (see [Store Management](../USAGE.md#store-management)).

```sql
-- Memory nodes
insights (
  id, content, category, importance,
  tags, entities, source,
  embedding,                    -- Voyage AI 512-dim vector (BLOB)
  keywords, summary, semantic_facts,  -- LLM enrichment columns
  access_count, last_accessed_at,
  effective_importance,          -- Decayed effective importance
  linked_at, enriched_at,        -- Pipeline progress timestamps
  created_at, updated_at, deleted_at
)

-- Relationship edges (composite primary key)
edges (
  source_id, target_id, edge_type,  -- PK
  weight, metadata, created_at
)

-- Operation log (audit trail, queryable with --since/--group-by/--stats)
oplog (
  id, operation, insight_id, detail, created_at
)
```

---

## 3.4 System Architecture

MemMan's architecture is divided into five layers:

```
┌──────────────────────────────────────────────────────────────┐
│  Integration Layer    Hook / Skill / Guide                   │
├─────────────────────────────────────────────────────────────┤
│  CLI Layer            remember, recall, search, link, gc ... │
├─────────────────────────────────────────────────────────────┤
│  Core Engine          search/ (recall, intent, keyword)      │
│                       graph/  (temporal, entity, causal,     │
│                                semantic)                     │
│                       embed/  (voyage, vector)               │
│                       llm/   (client, extract)               │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer        store/  (db, node, edge, oplog)        │
├─────────────────────────────────────────────────────────────┤
│  External             Anthropic API (Haiku)                  │
│                       Voyage AI (embeddings)                 │
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
├── scripts/
│   ├── e2e_test.sh
│   └── visualize_graph.py
├── pyproject.toml            # Poetry package config
└── Makefile
```

## 3.5 Data Directory Layout

```
~/.memman/
├── active                        # Current default store name (plain text)
├── prompt/                       # Shared across all stores
│   ├── guide.md                  # Behavioral guide (recall/remember rules)
│   └── skill.md                  # Skill definition (command reference)
└── data/                         # Each store has its own isolated directory
    ├── default/
    │   └── memman.db                # SQLite database (WAL mode)
    ├── work/
    │   └── memman.db
    └── <name>/
        └── memman.db
```

**Isolation boundary**: Each store contains an independent `memman.db` — insights, edges, and oplog are fully isolated. Prompt files (`guide.md`, `skill.md`) are shared — behavioral rules are universal, memory data is private.

## 3.6 Store Isolation

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
| `active` file      | Persistent user preference — `memman store set work`          |
| `"default"`        | Zero-config — works out of the box                            |

**Automatic migration**: When the new `data/` directory doesn't exist but a legacy `~/.memman/memman.db` does, memman automatically moves it to `data/default/memman.db`. Users upgrading from older versions experience a seamless transition.

**Design principle — lightweight and bounded**: Store isolation addresses a necessary data separation concern without growing into a multi-tenant system. There are no access controls, no cross-store queries, no store metadata beyond the name. This keeps the feature bounded — MemMan is a memory daemon, not a knowledge base platform.
