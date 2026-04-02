# Mnemon — Usage & Reference

> You don't run mnemon commands yourself — the agent does, driven by hooks and guided by the skill file. This document is a reference for understanding what the agent can do, for debugging, and for advanced manual operation.

---

## Global Flags

These flags are available on every command:

| Flag                | Default     | Description                                                   |
| ------------------- | ----------- | ------------------------------------------------------------- |
| `--store <name>`    | (auto)      | Named memory store (overrides `MNEMON_STORE` and active file) |
| `--data-dir <path>` | `~/.mnemon` | Base data directory                                           |
| `--readonly`        | `false`     | Open database in read-only mode                               |
| `--version`         |             | Print version and exit                                        |

---

## Setup

Deploy mnemon into LLM CLI environments. This is the first command to run after installation.

```bash
# Interactive: detect environments and install (project-local)
mnemon setup

# User-wide install (all projects)
mnemon setup --global

# Non-interactive: specific target only
mnemon setup --target claude-code
mnemon setup --target openclaw

# Auto-confirm all prompts (CI-friendly)
mnemon setup --yes

# Remove mnemon integrations
mnemon setup --eject
mnemon setup --eject --target claude-code
```

| Flag              | Default       | Description                                                                      |
| ----------------- | ------------- | -------------------------------------------------------------------------------- |
| `--global`        | `false`       | Install to user-wide config (`~/.claude/`) instead of project-local (`.claude/`) |
| `--target <name>` | (auto-detect) | Target environment: `claude-code` or `openclaw`                                  |
| `--eject`         | `false`       | Remove mnemon integrations                                                       |
| `--yes`           | `false`       | Auto-confirm all prompts                                                         |

---

## CLI Commands

### Core

```bash
# Remember — store a new insight (LLM reconciliation: duplicates skipped, conflicts resolved)
mnemon remember "Chose Qdrant over Milvus for vector search" \
  --cat decision --imp 5 --entities "Qdrant,Milvus" --tags "architecture,search" --source agent

# Skip LLM reconciliation (direct insert)
mnemon remember "Raw note" --no-reconcile

# Recall — intent-aware graph-enhanced retrieval (default)
mnemon recall "vector database" --limit 10

# Recall with explicit intent override
mnemon recall "why did we choose Qdrant" --intent WHY

# Recall with category/source filter
mnemon recall "auth" --cat decision --source agent

# Simple SQL LIKE matching (faster, no graph traversal)
mnemon recall "auth" --basic

# Search — token-scored keyword search
mnemon search "authentication" --limit 10

# Replace — deterministic replacement by ID (inherits metadata from original)
mnemon replace <id> "Updated content" --cat decision --imp 5

# Forget — soft-delete an insight
mnemon forget <id>
```

**Remember flags:**

| Flag             | Default   | Description                                                                 |
| ---------------- | --------- | --------------------------------------------------------------------------- |
| `--cat`          | `general` | Category: `preference`, `decision`, `fact`, `insight`, `context`, `general` |
| `--imp`          | `3`       | Importance: 1–5                                                             |
| `--tags`         |           | Comma-separated tags                                                        |
| `--entities`     |           | Comma-separated entities (merged with LLM-extracted)                        |
| `--source`       | `user`    | Source: `user`, `agent`, `external`                                         |
| `--no-reconcile` | `false`   | Skip LLM reconciliation (direct insert)                                     |

**Recall flags:**

| Flag       | Default       | Description                                          |
| ---------- | ------------- | ---------------------------------------------------- |
| `--limit`  | `10`          | Max results                                          |
| `--intent` | (auto-detect) | Override intent: `WHY`, `WHEN`, `ENTITY`, `GENERAL`  |
| `--cat`    |               | Filter by category                                   |
| `--source` |               | Filter by source                                     |
| `--basic`  | `false`       | Use simple SQL LIKE matching instead of smart recall |

### Graph Operations

```bash
# Link — create a typed edge
mnemon link <source_id> <target_id> --type semantic --weight 0.85
mnemon link <source_id> <target_id> --type causal --weight 0.8 \
  --meta '{"sub_type":"causes","reason":"..."}'

# Related — BFS traversal from an insight
mnemon related <id> --edge causal --depth 2

# Reindex — regenerate auto-computed edges (triggered automatically on constants change)
mnemon graph reindex              # live reindex
mnemon graph reindex --dry-run    # preview changes without modifying DB

# Rebuild — full LLM re-enrichment + re-embed + edge rebuild
mnemon graph rebuild              # process all insights
mnemon graph rebuild --dry-run    # preview count without modifying DB
```

Auto-reindex fires transparently when `open_db()` detects graph constants (thresholds, weights) have changed. Manual reindex is available for debugging or forcing edge regeneration. Use `--dry-run` to preview what would change.

Rebuild re-enriches all insights through the full LLM pipeline (enrichment, re-embedding, causal inference, edge recreation). Processes in batches of 20. Returns `{"processed": N, "remaining": 0}`.

### Lifecycle Management

```bash
# GC — view low-retention candidates
mnemon gc --threshold 0.5 --limit 20

# GC keep — boost an insight's retention
mnemon gc --keep <id>

# GC review — scan stored insights for content quality issues
mnemon gc --review
```

### Store Management

Mnemon supports named stores for data isolation. Each store has its own independent database.

```bash
# List all stores (* marks the active one)
mnemon store list

# Create a new store
mnemon store create work

# Switch the default active store
mnemon store set work

# Remove a store (cannot remove the active store)
mnemon store remove old-project
```

**Store resolution priority** (highest to lowest):

1. `--store <name>` CLI flag
2. `MNEMON_STORE` environment variable
3. `~/.mnemon/active` file
4. Falls back to `"default"`

Different agents or processes can use different stores via the `MNEMON_STORE` environment variable — no global state contention. Legacy databases (`~/.mnemon/mnemon.db`) are automatically migrated to `~/.mnemon/data/default/` on first run.

### Observability

```bash
mnemon status                                       # memory statistics
mnemon doctor                                       # run health checks on the database
mnemon log                                          # operation log (default: last 20)
mnemon log --limit 50                               # show more entries
mnemon log --since 7d                               # entries from last 7 days
mnemon log --since 24h                              # entries from last 24 hours
mnemon log --since 7d --group-by operation --stats  # grouped counts + never-accessed
```

### Visualization

Export the knowledge graph for visual exploration:

```bash
# DOT format — render with Graphviz (brew install graphviz)
mnemon viz --format dot -o graph.dot
dot -Tpng graph.dot -o graph.png

# Interactive HTML — open directly in the browser (vis.js, no install needed)
mnemon viz --format html -o graph.html
open graph.html
```

Nodes are colored by category (decision, fact, insight, preference, context); edges are colored by type (temporal, semantic, causal, entity).

---

## Configuration

| Variable              | Default                         | Description                                           |
| --------------------- | ------------------------------- | ----------------------------------------------------- |
| `MNEMON_DATA_DIR`     | `~/.mnemon`                     | Base data directory                                   |
| `MNEMON_STORE`        | `default`                       | Active named store                                    |
| `ANTHROPIC_API_KEY`   | —                               | Required: LLM fact extraction, reconciliation, recall |
| `VOYAGE_API_KEY`      | —                               | Required: Voyage AI embeddings (512-dim)              |
| `MNEMON_LLM_ENDPOINT` | `https://api.anthropic.com`     | Override LLM endpoint                                 |
| `MNEMON_LLM_API_KEY`  | falls back to ANTHROPIC_API_KEY | Override API key for LLM endpoint                     |
| `MNEMON_LLM_MODEL`    | `claude-haiku-4-5-20251001`     | Model for LLM inference                               |

---

## Architecture

### Write Pipeline (Single-Tier Synchronous)

`remember` uses a single synchronous pipeline:

1. **Sequential** — Quality check, LLM fact extraction, Voyage embedding,
   LLM reconciliation (ADD/UPDATE/DELETE/NONE), insert, fast edges
   (temporal + entity), semantic edges, EI refresh, auto-prune.
2. **Parallel (ThreadPoolExecutor)** — LLM enrichment (keywords, summary,
   entities) + LLM causal edge inference run concurrently.
3. **Sequential finalization** — Write enrichment results, re-embed with
   keywords, rebuild entity/semantic/causal edges, stamp `linked_at`.
   Returns JSON with `facts` array.

### Recall Pipeline

1. **LLM query expansion** — synonyms, entity extraction, intent detection
2. **RRF anchor selection** — keyword + vector + recency fused with K=60
3. **Beam search** — intent-weighted graph traversal from anchors
4. **4-signal rerank** — keyword, entity, similarity, graph (intent-weighted)
5. **Post-sort** — causal topological (WHY), chronological (WHEN), score (default)

Inspired by [MAGMA](https://arxiv.org/abs/2601.03236) four-graph model. See [Design & Architecture](DESIGN.md) for the full deep dive.
