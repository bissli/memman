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

Each named store is physically isolated via the active storage backend (`MEMMAN_BACKEND`):

- **SQLite (default)** — one `~/.memman/data/<store>/memman.db` file per store, in WAL mode (concurrent reads + serial writer). Schema source of truth: `_BASELINE_SCHEMA` in `src/memman/store/db.py`.
- **Postgres** — one Postgres schema per store (`store_<name>`) sharing one database; `pgvector` provides the `vector(N)` column type. Schema source of truth: `_PG_BASELINE_SCHEMA` plus an additive forward-only `_PG_MIGRATIONS` ladder in `src/memman/store/postgres.py`. The backend is enabled with the `memman[postgres]` install extra.

Switching backends is all-or-nothing across every store under `~/.memman/data/`; per-store backend choice is not supported. See [Migrating from SQLite to Postgres](../USAGE.md#migrating-from-sqlite-to-postgres) for the operator workflow.

The logical column layout below is shared between backends; the type translations are SQLite `TEXT`/`BLOB` ↔ Postgres `TIMESTAMPTZ`/`JSONB`/`vector(N)`.

```sql
-- Memory nodes
insights (
  id, content, category, importance,
  entities, source,
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
identity, content, category, importance, entities, source,
timestamps, access bookkeeping, effective_importance, and provenance
columns. The enrichment payload (`embedding`, `keywords`, `summary`,
`semantic_facts`, `linked_at`, `enriched_at`) lives in the DB only and
is read/written through SQL helpers, not via the dataclass.

---

## 2.4 System Architecture

MemMan's architecture is divided into five layers:

```
┌───────────────────────────────────────────────────────────────┐
│  Integration Layer    Hook / Skill / Guide                    │
├──────────────────────────────────────────────────────────────┤
│  CLI Layer            remember · recall · replace · forget    │
│                       prime · status · doctor · install       │
│                       graph · scheduler · insights · store    │
│                       embed · log · config                    │
├──────────────────────────────────────────────────────────────┤
│  Pipeline             pipeline/ (remember, drain worker)      │
├──────────────────────────────────────────────────────────────┤
│  Core Engine          search/ (recall, intent, keyword,       │
│                                quality)                       │
│                       graph/  (temporal, entity, causal,      │
│                                semantic, engine, bfs,         │
│                                enrichment)                    │
│                       embed/  (voyage, openai_compat,         │
│                                openrouter, ollama, vector)    │
│                       llm/    (client, extract,               │
│                                openrouter_client,             │
│                                openrouter_models)             │
├──────────────────────────────────────────────────────────────┤
│  Storage Layer        store/   (backend, factory, db, node,   │
│                                edge, oplog, model, snapshot,  │
│                                sqlite, postgres)              │
│                       queue.py (deferred-write queue)         │
│                       migrate.py (SQLite -> Postgres copy)    │
├──────────────────────────────────────────────────────────────┤
│  External             OpenRouter (LLM, Anthropic Haiku/Sonnet)│
│                       Voyage AI (embeddings, default)         │
│                       Postgres + pgvector (optional backend)  │
└───────────────────────────────────────────────────────────────┘
```


**Project code structure:**

```
memman/
├── src/memman/
│   ├── __init__.py
│   ├── cli.py                # Click CLI (all commands)
│   ├── config.py             # Env-file resolver (INSTALLABLE_KEYS)
│   ├── doctor.py             # Health checks (memman doctor)
│   ├── drain_lock.py         # Cross-process drain.lock
│   ├── maintenance.py        # GC, auto-prune, EI recompute
│   ├── migrate.py            # SQLite -> Postgres migration
│   ├── queue.py              # Deferred-write queue
│   ├── trace.py              # JSONL debug tracing
│   ├── pipeline/             # remember (drain worker)
│   ├── store/                # Storage backends (sqlite, postgres)
│   ├── graph/                # MAGMA four-graph implementation
│   ├── search/               # Retrieval algorithms
│   ├── embed/                # Pluggable embedding providers
│   ├── rerank/               # Cross-encoder rerank (Voyage)
│   ├── llm/                  # LLM client + extraction/reconciliation
│   └── setup/                # LLM CLI integration + install wizard
├── scripts/
│   └── import_sqlite_to_postgres.py   # Streaming reader used by migrate
├── tests/
├── pyproject.toml            # Poetry package config (memman[postgres] extra)
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

**Isolation boundary**: Each store is fully independent — insights, edges, and oplog do not cross stores. On SQLite this is one `memman.db` per store; on Postgres it is one `store_<name>` schema per store inside one shared database. Shipped assets (`guide.md`, `SKILL.md`) live inside the installed package and are read via `importlib.resources`; nothing memman deploys lives under `~/.memman/`. `~/.memman/` is strictly user state: memory data, API keys, caches, logs, queued work.

When `MEMMAN_BACKEND=postgres`, `~/.memman/queue.db` and the per-store `memman.db` files are not used at runtime — the queue lives in the shared `queue` Postgres schema and store rows live in `store_<name>`. The SQLite files remain on disk after `memman migrate` as a durable fallback; the operator removes them manually after verifying the new backend with `memman doctor`.

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
