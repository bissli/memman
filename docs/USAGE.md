# MemMan — Usage & Reference

> You don't run memman commands yourself — the agent does, driven by hooks and guided by the skill file. This document is a reference for understanding what the agent can do, for debugging, and for advanced manual operation.

---

## Global Flags

These flags are available on every command:

| Flag                | Default     | Description                                                   |
| ------------------- | ----------- | ------------------------------------------------------------- |
| `--store <name>`    | (auto)      | Named memory store (overrides `MEMMAN_STORE` and active file) |
| `--data-dir <path>` | `~/.memman` | Base data directory                                           |
| `--readonly`        | `false`     | Open database in read-only mode                               |
| `--version`         |             | Print version and exit                                        |

---

## Install / Uninstall

Deploy memman into LLM CLI environments. Run after `pipx install memman` (or `pipx install -e .` for development).

```bash
# Interactive: detect environments and install
memman install

# Non-interactive: specific target only
memman install --target claude-code
memman install --target openclaw

# Remove memman integrations
memman uninstall
memman uninstall --target claude-code
```

| Command            | `--target <name>` | Effect                                                                                                                              |
| ------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `memman install`   | (auto-detect)     | Deploy hook and skill symlinks, register in settings.json, install the scheduler unit, write `~/.memman/prompt/guide.local.md` stub |
| `memman install`   | `claude-code`     | Install into `~/.claude/` only                                                                                                      |
| `memman install`   | `openclaw`        | Install into `~/.openclaw/` only                                                                                                    |
| `memman install`   | `nanoclaw`        | Install into `~/.nanoclaw/` only                                                                                                    |
| `memman uninstall` | (auto-detect)     | Remove hooks, skill, settings.json entries, and scheduler unit. Never touches `~/.memman/`                                          |
| `memman uninstall` | `<name>`          | Remove memman from that environment only                                                                                            |

### Live-read commands

| Command        | What it prints                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `memman guide` | Shipped `guide.md` plus `~/.memman/prompt/guide.local.md` (appended)                              |
| `memman skill` | Shipped `SKILL.md`                                                                                |
| `memman prime` | Reads SessionStart JSON on stdin; emits status + compact-recall hint + guide (called by prime.sh) |

`guide.local.md` is the supported customization point. Created with a stub on first `memman install` and never overwritten.

---

## CLI Commands

### Core

```bash
# Remember — store a new insight (LLM reconciliation: duplicates skipped, conflicts resolved)
memman remember "Chose Qdrant over Milvus for vector search" \
  --cat decision --imp 5 --entities "Qdrant,Milvus" --tags "architecture,search" --source agent

# Skip LLM reconciliation (direct insert)
memman remember "Raw note" --no-reconcile

# Recall — intent-aware graph-enhanced retrieval (default)
memman recall "vector database" --limit 10

# Recall with explicit intent override
memman recall "why did we choose Qdrant" --intent WHY

# Recall with category/source filter
memman recall "auth" --cat decision --source agent

# Simple SQL LIKE matching (faster, no graph traversal)
memman recall "auth" --basic

# Search — token-scored keyword search
memman search "authentication" --limit 10

# Replace — deterministic replacement by ID (inherits metadata from original)
memman replace <id> "Updated content" --cat decision --imp 5

# Forget — soft-delete an insight
memman forget <id>
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
memman link <source_id> <target_id> --type semantic --weight 0.85
memman link <source_id> <target_id> --type causal --weight 0.8 \
  --meta '{"sub_type":"causes","reason":"..."}'

# Related — BFS traversal from an insight
memman related <id> --edge causal --depth 2

# Reindex — regenerate auto-computed edges (triggered automatically on constants change)
memman graph reindex              # live reindex
memman graph reindex --dry-run    # preview changes without modifying DB

# Rebuild — full LLM re-enrichment + re-embed + edge rebuild
memman graph rebuild              # process all insights
memman graph rebuild --dry-run    # preview count without modifying DB
```

Auto-reindex fires transparently when `open_db()` detects graph constants (thresholds, weights) have changed. Manual reindex is available for debugging or forcing edge regeneration. Use `--dry-run` to preview what would change.

Rebuild re-enriches all insights through the full LLM pipeline (enrichment, re-embedding, causal inference, edge recreation). Processes in batches of 20. Returns `{"processed": N, "remaining": 0}`.

### Lifecycle Management

```bash
# GC — view low-retention candidates
memman gc --threshold 0.5 --limit 20

# GC keep — boost an insight's retention
memman gc --keep <id>

# GC review — scan stored insights for content quality issues
memman gc --review
```

### Store Management

MemMan supports named stores for data isolation. Each store has its own independent database.

```bash
# List all stores (* marks the active one)
memman store list

# Create a new store
memman store create work

# Switch the default active store
memman store set work

# Remove a store (cannot remove the active store)
memman store remove old-project
```

**Store resolution priority** (highest to lowest):

1. `--store <name>` CLI flag
2. `MEMMAN_STORE` environment variable
3. `~/.memman/active` file
4. Falls back to `"default"`

Different agents or processes can use different stores via the `MEMMAN_STORE` environment variable — no global state contention. Legacy databases (`~/.memman/memman.db`) are automatically migrated to `~/.memman/data/default/` on first run.

### Observability

```bash
memman status                                       # memory statistics
memman doctor                                       # run health checks on the database
memman log                                          # operation log (default: last 20)
memman log --limit 50                               # show more entries
memman log --since 7d                               # entries from last 7 days
memman log --since 24h                              # entries from last 24 hours
memman log --since 7d --group-by operation --stats  # grouped counts + never-accessed
```

### Visualization

Export the knowledge graph for visual exploration:

```bash
# DOT format — render with Graphviz (brew install graphviz)
memman viz --format dot -o graph.dot
dot -Tpng graph.dot -o graph.png

# Interactive HTML — open directly in the browser (vis.js, no install needed)
memman viz --format html -o graph.html
open graph.html
```

Nodes are colored by category (decision, fact, insight, preference, context); edges are colored by type (temporal, semantic, causal, entity).

---

## Configuration

| Variable              | Default                           | Description                                                                     |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`     | `~/.memman`                       | Base data directory                                                             |
| `MEMMAN_STORE`        | `default`                         | Active named store                                                              |
| `OPENROUTER_API_KEY`  | —                                 | Required: enrichment worker (fact extraction, reconciliation, causal inference) |
| `VOYAGE_API_KEY`      | —                                 | Required: Voyage AI embeddings (512-dim)                                        |
| `ANTHROPIC_API_KEY`   | —                                 | Optional: session-path query expansion; degrades gracefully when unset          |
| `MEMMAN_LLM_ENDPOINT` | `https://api.anthropic.com`       | Override LLM endpoint for the optional session-path client                      |
| `MEMMAN_LLM_API_KEY`  | falls back to `ANTHROPIC_API_KEY` | Override API key for the session-path LLM endpoint                              |
| `MEMMAN_LLM_MODEL`    | `claude-haiku-4-5-20251001`       | Model for LLM inference                                                         |

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
