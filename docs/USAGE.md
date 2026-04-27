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

| Command            | `--target <name>` | Effect                                                                                                                                                                                                                                                  |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `memman install`   | (auto-detect)     | Deploy hook and skill symlinks, register in settings.json, install the scheduler unit, create `~/.memman/logs/` for scheduler output                                                                                                                    |
| `memman install`   | `claude-code`     | Install into `~/.claude/` only                                                                                                                                                                                                                          |
| `memman install`   | `openclaw`        | Install into `~/.openclaw/` only                                                                                                                                                                                                                        |
| `memman install`   | `nanoclaw`        | Install into `~/.nanoclaw/` only                                                                                                                                                                                                                        |
| `memman uninstall` | (auto-detect)     | Remove hooks, skill, settings.json entries, and scheduler unit. Strips secret keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`) from `~/.memman/env` but keeps non-secret settings; memory store, queue, and logs untouched. |
| `memman uninstall` | `<name>`          | Remove memman from that environment only                                                                                                                                                                                                                |

### Live-read commands

| Command        | What it prints                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------- |
| `memman guide` | Shipped `guide.md` (hidden, called by openclaw bootstrap; humans read `guide.md` from the package) |
| `memman prime` | Reads SessionStart JSON on stdin; emits status + compact-recall hint + guide (called by prime.sh)  |

`memman log worker [--errors] [--lines N]` tails `~/.memman/logs/enrich.{log,err}` — the enrichment worker's output. Use `--errors` for stderr / Python tracebacks.

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

# Simple SQL LIKE matching (faster, no graph traversal, no LLM expansion)
memman recall "auth" --basic

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
memman graph link <source_id> <target_id> --type semantic --weight 0.85
memman graph link <source_id> <target_id> --type causal --weight 0.8 \
  --meta '{"sub_type":"causes","reason":"..."}'

# Related — BFS traversal from an insight
memman graph related <id> --edge causal --depth 2

# Rebuild — full LLM re-enrichment + re-embed + edge rebuild
memman graph rebuild              # process all insights
memman graph rebuild --dry-run    # preview count without modifying DB
```

Auto-reindex of computed edges (semantic, entity, temporal) fires transparently when `open_db()` detects graph constants have changed — there is no operator command for it.

Rebuild re-enriches all insights through the full LLM pipeline (enrichment, re-embedding, causal inference, edge recreation). Processes in batches of 20. Returns `{"processed": N, "remaining": 0}`. Rejected when the scheduler is stopped.

### Insights Lifecycle

```bash
# Read a single insight by ID (full content + metadata)
memman insights show <id>

# List low-retention candidates (read-only — does NOT delete)
memman insights candidates --threshold 0.5 --limit 20

# Boost retention of a specific insight (immune from candidates list)
memman insights protect <id>

# Scan stored insights for content quality issues
memman insights review
```

To actually delete an insight, use `memman forget <id>`.

### Embedding Operations

```bash
# Show current embedding provider, model, and per-store fingerprint
memman embed status

# Re-embed all insights with the current provider (e.g., after switching
# from Voyage to OpenAI-compatible). Rejected when scheduler is stopped.
memman embed reembed
memman embed reembed --dry-run    # preview count without modifying DB
```

Switching embedding providers (`MEMMAN_EMBED_PROVIDER`) requires an
explicit `embed reembed` step. The per-store fingerprint detects
provider/model drift and surfaces it in `embed status`.

### Store Management

MemMan supports named stores for data isolation. Each store has its own independent database.

```bash
# List all stores (* marks the active one)
memman store list

# Create a new store
memman store create work

# Switch the default active store
memman store use work

# Remove a store (cannot remove the active store)
memman store remove old-project
```

**Store resolution priority** (highest to lowest):

1. `--store <name>` CLI flag
2. `MEMMAN_STORE` environment variable
3. `~/.memman/active` file
4. Falls back to `"default"`

Different agents or processes can use different stores via the `MEMMAN_STORE` environment variable — no global state contention.

### Observability

```bash
memman status                                       # memory statistics
memman doctor                                       # health checks (sqlite, queue, keys, scheduler, env_completeness)
memman config show                                  # effective configuration (env + on-disk)

memman log list                                     # operation audit log (default JSON, last 20)
memman log list --limit 50                          # show more entries
memman log list --since 7d                          # entries from last 7 days
memman log list --since 7d --stats                  # grouped counts + never-accessed
memman log list --text                              # human-readable text table

memman log worker [--errors] [--lines N]            # tail worker output (~/.memman/logs/enrich.{log,err})
```

---

## Configuration

memman resolves user-config vars in two layers: `os.environ` first, then `<MEMMAN_DATA_DIR>/env` (the env file written by `memman install`, mode 0600). There is no code-default fallback at runtime — the defaults below live in `config.INSTALL_DEFAULTS` and are written to the env file at install time. See [CONTRIBUTING.md](../CONTRIBUTING.md#configuration) for the full design. Process-control vars (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`) bypass the file and read directly from `os.environ`.

| Variable                       | Install-time default                              | Description                                                                              |
| ------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`              | `~/.memman`                                       | Base data directory (process-control; not persisted).                                    |
| `MEMMAN_STORE`                 | `default`                                         | Active named store (process-control; not persisted).                                     |
| `OPENROUTER_API_KEY`           | —                                                 | Required at install: LLM inference (fact extraction, reconciliation, causal, expansion). |
| `VOYAGE_API_KEY`               | —                                                 | Required at install: Voyage AI embeddings (512-dim).                                     |
| `MEMMAN_LLM_PROVIDER`          | `openrouter`                                      | Registered LLM provider name (see `memman.llm.client.PROVIDERS`).                        |
| `MEMMAN_OPENROUTER_ENDPOINT`   | `https://openrouter.ai/api/v1`                    | Endpoint for the OpenRouter client.                                                      |
| `MEMMAN_LLM_MODEL_FAST`        | resolved at install (haiku family via `/models`)  | Recall hot path model id (query expansion, doctor probe).                                |
| `MEMMAN_LLM_MODEL_SLOW`        | resolved at install (sonnet family via `/models`) | Scheduler worker model id (extraction, reconciliation, enrichment, causal).              |
| `MEMMAN_EMBED_PROVIDER`        | `voyage`                                          | Embedding provider: `voyage`, `openai_compat`, `ollama`.                                 |
| `MEMMAN_OPENAI_EMBED_API_KEY`  | —                                                 | API key for `openai_compat` provider.                                                    |
| `MEMMAN_OPENAI_EMBED_ENDPOINT` | `https://api.openai.com`                          | Endpoint URL for `openai_compat` provider.                                               |
| `MEMMAN_OPENAI_EMBED_MODEL`    | `text-embedding-3-small`                          | Model id for `openai_compat` provider.                                                   |
| `MEMMAN_OLLAMA_HOST`           | `http://localhost:11434`                          | Host URL for `ollama` provider.                                                          |
| `MEMMAN_OLLAMA_EMBED_MODEL`    | `nomic-embed-text`                                | Model id for `ollama` provider.                                                          |
| `MEMMAN_DEBUG`                 | (unset)                                           | Truthy value enables JSONL tracing to `~/.memman/logs/debug.log`.                        |
| `MEMMAN_WORKER`                | (unset)                                           | `1` inside the scheduler-triggered worker; enables the rotating log.                     |
| `MEMMAN_LOG_LEVEL`             | `WARNING`                                         | Logger level when neither `--verbose` nor `--debug` is passed.                           |

---

## Architecture

### Write Pipeline (Deferred, Two-Tier)

`memman remember` is a fast queue-append (~50 ms) on the host session — no
LLM calls, no embeddings, no edges. The full pipeline runs out of band in
a scheduler-driven worker:

1. **Tier 1 (host session)** — append a row to `~/.memman/queue.db` with
   `status='pending'`, the raw text, and any `--cat`/`--imp`/`--entities`
   hints. Returns `{action: queued, queue_id, store}`.
2. **Tier 2 (worker)** — systemd timer (Linux), launchd agent (macOS), or
   `memman scheduler serve` PID 1 (containers) invokes
   `memman scheduler drain --pending` every 60 s under an `flock` on
   `~/.memman/drain.lock`. For each queued row: quality gate → LLM fact
   extraction → per-fact embed (Voyage) + similarity scan + LLM
   reconciliation (ADD/UPDATE/DELETE/NONE) → insert/update → fast edges
   (temporal + entity + semantic) → parallel enrichment + LLM causal
   inference → re-embed → rebuild auto edges → mark done.

The host session never blocks on the network. Newly stored memories
become recallable on the next drain tick (default 60 s).

### Recall Pipeline

1. **LLM query expansion** — synonyms, entity extraction, intent detection
2. **RRF anchor selection** — keyword + vector + recency fused with K=60
3. **Beam search** — intent-weighted graph traversal from anchors
4. **4-signal rerank** — keyword, entity, similarity, graph (intent-weighted)
5. **Post-sort** — causal topological (WHY), chronological (WHEN), score (default)

Inspired by [MAGMA](https://arxiv.org/abs/2601.03236) four-graph model. See [Design & Architecture](DESIGN.md) for the full deep dive.
