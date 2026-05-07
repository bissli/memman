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
| `--verbose` / `-v`  | `false`     | INFO-level logging to stderr                                  |
| `--debug`           | `false`     | DEBUG-level logging to stderr (overrides `--verbose`)         |
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
  --cat decision --imp 5 --entities "Qdrant,Milvus" --source agent

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
| `--entities`     |           | Comma-separated entities (merged with LLM-extracted)                        |
| `--source`       | `user`    | Source: `user`, `agent`, `external`                                         |
| `--no-reconcile` | `false`   | Skip LLM reconciliation (direct insert)                                     |

**Recall flags:**

| Flag       | Default       | Description                                                                                       |
| ---------- | ------------- | ------------------------------------------------------------------------------------------------- |
| `--limit`  | `10`          | Max results                                                                                       |
| `--intent` | (auto-detect) | Override intent: `WHY`, `WHEN`, `ENTITY`, `GENERAL`                                               |
| `--cat`    |               | Filter by category                                                                                |
| `--source` |               | Filter by source                                                                                  |
| `--basic`  | `false`       | Use simple SQL LIKE matching instead of smart recall                                              |
| `--expand` | `false`       | Opt-in LLM query expansion (synonyms + entity hints)                                              |
| `--rerank` | `false`       | Cross-encoder rerank stage (Voyage `rerank-2.5-lite` by default; auto-skips on 1-2 token queries) |

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

# Online provider/model swap (resumable shadow-column backfill, atomic cutover)
memman embed swap --to voyage-3-large
memman embed swap --to text-embedding-3-small --provider openai
memman embed swap --resume                     # continue an in-flight swap
memman embed swap --abort                      # discard an in-flight swap

# Offline full re-embed under the current provider (rejected when scheduler is running)
memman embed reembed
memman embed reembed --dry-run                 # preview count without modifying DB
```

Two switching paths:

- **`embed swap`** is the online path. It populates `embedding_pending` (shadow column on SQLite, side column on Postgres) under the active provider while the existing column keeps serving recall, then commits an atomic cutover transaction. State machine: `backfilling → cutover → done`. Resumable via `--resume`; abortable via `--abort`. Per-store; the in-flight target is recorded in `meta.embed_swap_*` keys (deleted on completion).
- **`embed reembed`** is the offline path: every store is rewritten in place with the current `MEMMAN_EMBED_PROVIDER`. Requires the scheduler to be **stopped** (`memman scheduler stop`) so a drain cannot race the rewrite.

The per-store fingerprint (`meta.embed_fingerprint`) detects provider/model drift and surfaces it in `embed status`.

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

#### Migrating from SQLite to Postgres

`memman migrate` copies a store's data from SQLite into the configured Postgres backend. The SQLite source is preserved (copy-only, never modified), so it remains a durable fallback until you choose to remove it.

The command echoes a plan (source paths, redacted destination DSN, per-store target schema state — `ABSENT` / `EMPTY` / `POPULATED`) and prompts for confirmation. Stores whose target schema already exists are dropped and recreated. On success the command writes `MEMMAN_BACKEND_<store>=postgres` to the env file so the next drain routes that specific store to Postgres; other stores are unaffected.

```bash
# Dry-run: print the plan only, no writes, no prompt
memman migrate --store work --dry-run

# Interactive: print plan, prompt, then migrate + flip backend
memman migrate --store work

# Non-interactive (CI / scripts): skip the prompt
memman migrate --all --yes
```

The command holds the shared `drain.lock` for its duration so a scheduler-fired drain cannot race the SQLite reader. To verify the cutover, run `memman doctor`. To revert per-store, use `memman config set MEMMAN_BACKEND_<store> sqlite` (or unset the key to fall back to `MEMMAN_DEFAULT_BACKEND`).

Reverse migration (Postgres → SQLite) is not implemented; restore from the preserved SQLite source if needed.

### Observability

```bash
memman status                                       # memory statistics
memman doctor                                       # health checks (integrity, schema, enrichment, embeddings, fingerprint, queue, scheduler, drain heartbeat, env, no_stale_swap_meta, provenance_drift)
memman doctor --text                                # human-readable colored table
memman config show                                  # effective configuration (env + on-disk)

memman log list                                     # operation audit log (default JSON, last 20)
memman log list --limit 50                          # show more entries
memman log list --since 7d                          # entries from last 7 days
memman log list --since 7d --stats                  # grouped counts + never-accessed
memman log list --text                              # human-readable text table

memman log worker [--errors] [--lines N]            # tail worker output (~/.memman/logs/enrich.{log,err})
```

### Scheduler

```bash
memman scheduler status                  # platform, interval, state, next run, last heartbeat
memman scheduler start                   # flip persistent state to STARTED (resume drains + writes)
memman scheduler stop                    # flip persistent state to STOPPED (pause drains + reject writes)
memman scheduler trigger                 # run a drain now (systemd/launchd; not applicable in serve mode)
memman scheduler interval --seconds N    # change cadence (60s minimum on systemd/launchd)
memman scheduler install                 # install the scheduler unit (idempotent)
memman scheduler uninstall               # remove the scheduler unit; preserves persistent state
memman scheduler serve --interval N      # long-running drain loop (used as PID 1 in containers)
memman scheduler debug on|off|status     # toggle the verbose worker trace log

memman scheduler queue list [--limit N]  # peek pending rows
memman scheduler queue failed [--limit N]# rows in 'failed' state
memman scheduler queue show <row_id>     # full payload + trace events for one row
memman scheduler queue retry <row_id>    # requeue a single failed row
memman scheduler queue purge --done      # delete rows where status='done'
```

When the scheduler is **stopped**, memman is recall-only: every write exits 1 with `Scheduler is stopped; cannot <verb>`. The `serve` loop polls the state file every iteration, so pause is observed within seconds even mid-drain.

---

## Configuration

memman resolves user-config vars from a single canonical source at runtime: `<MEMMAN_DATA_DIR>/env` (the env file written by `memman install`, mode 0600). Shell environment variables are NOT consulted at runtime for installable settings, so a stale shell export cannot silently override values committed via `memman install`. `memman install` itself performs a one-time pull from the current shell into the env file (precedence: existing file value > wizard prompt in TTY mode > `os.environ` > OpenRouter resolver > `INSTALL_DEFAULTS`); existing file values are sticky and reinstall never lets a shell export override them. There is no code-default fallback at runtime; the defaults below live in `config.INSTALL_DEFAULTS` and are written to the env file at install time. See [CONTRIBUTING.md](../CONTRIBUTING.md#configuration) for the full design. Process-control vars (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`) are never persisted to the file; they are read directly from `os.environ` by the components that own them.

### Install wizard

Run `memman install` in a TTY to get the interactive wizard. It prompts (with masked input) for `OPENROUTER_API_KEY` / `VOYAGE_API_KEY` when both are missing from the env file and the shell, eliminating the `export X=...; export Y=...` ceremony for first-time installs. It also offers a backend selector (sqlite/postgres) when the `memman[postgres]` extra is installed; the wizard probes the DSN, verifies the `pgvector` extension is present, and (for non-localhost DSNs) emits a hint about PgBouncer transaction pooling. Headless installs and CI bypass the wizard via flags:

- `--backend [sqlite|postgres]` -- explicit backend choice; required in non-interactive mode if you want anything other than `sqlite`.
- `--pg-dsn URL` -- Postgres DSN; required with `--backend postgres` in non-interactive mode. The DSN may omit the password to use `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD` (psycopg3 honors all three).
- `--no-wizard` -- disables prompts even in a TTY; flags + defaults only.

If you pass an `INSTALLABLE_KEYS` flag whose value conflicts with the env file, install exits with a clear message pointing at `memman config set` -- the explicit override path. Use `memman config set KEY VALUE` to change a setting after the initial install (e.g., `memman config set MEMMAN_DEFAULT_BACKEND postgres` to change the fallback default, or `memman config set MEMMAN_BACKEND_<store> postgres` to route a specific store).

**Backend selection.** memman routes each store through a backend chosen by an env-file lookup:

1. `MEMMAN_BACKEND_<store>` — explicit per-store override (e.g., `MEMMAN_BACKEND_work=postgres`).
2. `MEMMAN_DEFAULT_BACKEND` — fallback when no per-store key is set (default `sqlite`).

`memman migrate <store>` writes `MEMMAN_BACKEND_<store>=postgres` so a single store can move to Postgres while others stay on SQLite. Use `memman config set MEMMAN_DEFAULT_BACKEND postgres` only when you want every newly-created store to default to Postgres.

The deferred-write queue is always SQLite at `<data_dir>/queue.db`; the Postgres backend stores per-store data in `store_<name>` schemas, each carrying its own `worker_runs` heartbeat table.

**Postgres DSN syntax.** Standard PostgreSQL libpq URI per psycopg3:

```
postgresql://[user[:password]@][host][:port]/[dbname][?param=value&...]
```

Worked examples (replace `<store>` with the store name, or use `MEMMAN_DEFAULT_PG_DSN` for the fallback):

```bash
# Local dev (defaults: localhost:5432, no password)
memman config set MEMMAN_PG_DSN_default 'postgresql://memman@localhost/memman'

# Inline credentials (URL-encode special chars in the password: ':' -> %3A, '@' -> %40, '/' -> %2F)
memman config set MEMMAN_PG_DSN_work 'postgresql://memman:s3cret@db.internal:5432/memman'

# Passwordless DSN sourcing the password from ~/.pgpass (recommended on shared hosts)
memman config set MEMMAN_DEFAULT_PG_DSN 'postgresql://memman@db.internal:5432/memman'

# Production: TLS-required, custom application_name (any libpq parameter is accepted)
memman config set MEMMAN_PG_DSN_default 'postgresql://memman@db.internal:5432/memman?sslmode=require&application_name=memman'
```

> **Security note.** `MEMMAN_DEFAULT_PG_DSN` and any `MEMMAN_PG_DSN_<store>` are stored plaintext in `~/.memman/env` at mode 0600. Root and any process running as your user can read them. For shared hosts, prefer `~/.pgpass` (mode 0600) and pass a passwordless DSN -- psycopg3 will source the password from `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD` automatically.

| Variable                          | Install-time default                              | Description                                                                                                                                          |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`                 | `~/.memman`                                       | Base data directory (process-control; not persisted).                                                                                                |
| `MEMMAN_STORE`                    | `default`                                         | Active named store (process-control; not persisted).                                                                                                 |
| `OPENROUTER_API_KEY`              | —                                                 | Required at install: LLM inference (fact extraction, reconciliation, causal, expansion).                                                             |
| `VOYAGE_API_KEY`                  | —                                                 | Required at install: Voyage AI embeddings (512-dim).                                                                                                 |
| `MEMMAN_LLM_PROVIDER`             | `openrouter`                                      | Registered LLM provider name (see `memman.llm.client.PROVIDERS`).                                                                                    |
| `MEMMAN_OPENROUTER_ENDPOINT`      | `https://openrouter.ai/api/v1`                    | Endpoint for the OpenRouter client.                                                                                                                  |
| `MEMMAN_LLM_MODEL_FAST`           | resolved at install (haiku family via `/models`)  | Recall hot path model id (query expansion, doctor probe).                                                                                            |
| `MEMMAN_LLM_MODEL_SLOW_CANONICAL` | resolved at install (sonnet family via `/models`) | Worker model for canonical content (fact extraction, reconciliation).                                                                                |
| `MEMMAN_LLM_MODEL_SLOW_METADATA`  | resolved at install (sonnet family via `/models`) | Worker model for derived metadata (enrichment summaries/keywords, causal-edge inference).                                                            |
| `MEMMAN_EMBED_PROVIDER`           | `voyage`                                          | Embedding provider: `voyage`, `openai`, `openrouter`, `ollama`.                                                                                      |
| `MEMMAN_RERANK_PROVIDER`          | `voyage`                                          | Cross-encoder rerank provider used when callers pass `memman recall --rerank`.                                                                       |
| `MEMMAN_VOYAGE_RERANK_MODEL`      | `rerank-2.5-lite`                                 | Voyage rerank model id.                                                                                                                              |
| `MEMMAN_OPENAI_EMBED_API_KEY`     | —                                                 | API key for `openai` provider.                                                                                                                       |
| `MEMMAN_OPENAI_EMBED_ENDPOINT`    | `https://api.openai.com`                          | Endpoint URL for `openai` provider.                                                                                                                  |
| `MEMMAN_OPENAI_EMBED_MODEL`       | `text-embedding-3-small`                          | Model id for `openai` provider.                                                                                                                      |
| `MEMMAN_OPENROUTER_EMBED_MODEL`   | `baai/bge-m3`                                     | Model id for `openrouter` embed provider; reuses `OPENROUTER_API_KEY` + `MEMMAN_OPENROUTER_ENDPOINT`.                                                |
| `MEMMAN_OLLAMA_HOST`              | `http://localhost:11434`                          | Host URL for `ollama` provider.                                                                                                                      |
| `MEMMAN_OLLAMA_EMBED_MODEL`       | `nomic-embed-text`                                | Model id for `ollama` provider.                                                                                                                      |
| `MEMMAN_DEBUG`                    | (unset)                                           | Truthy value enables JSONL tracing to `~/.memman/logs/debug.log`.                                                                                    |
| `MEMMAN_WORKER`                   | (unset)                                           | `1` inside the scheduler-triggered worker; enables the rotating log.                                                                                 |
| `MEMMAN_LOG_LEVEL`                | `WARNING`                                         | Logger level when neither `--verbose` nor `--debug` is passed.                                                                                       |
| `MEMMAN_DEFAULT_BACKEND`          | `sqlite`                                          | Fallback storage backend for stores without an explicit per-store override (`sqlite` or `postgres`). Postgres requires the `memman[postgres]` extra. |
| `MEMMAN_DEFAULT_PG_DSN`           | —                                                 | Fallback Postgres DSN (`postgresql://...`) when no `MEMMAN_PG_DSN_<store>` is set. Secret. Stripped on `memman uninstall`.                           |
| `MEMMAN_BACKEND_<store>`          | (unset)                                           | Per-store backend override (e.g., `MEMMAN_BACKEND_work=postgres`). Written by `memman migrate <store>`; not seeded by `install`.                     |
| `MEMMAN_PG_DSN_<store>`           | —                                                 | Per-store Postgres DSN override. Secret. Stripped on `memman uninstall`.                                                                             |
| `MEMMAN_REINDEX_TIMEOUT`          | `180`                                             | Seconds Postgres reindex (HNSW) is allowed to run before `statement_timeout` aborts; reraised idempotently next call.                                |
| `MEMMAN_EMBED_SWAP_BATCH_SIZE`    | `200`                                             | Rows per backfill batch in `memman embed swap`.                                                                                                      |
| `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` | `0` (unlimited)                                   | Seconds Postgres `CREATE INDEX CONCURRENTLY` may run during cutover; `0` disables `statement_timeout`.                                               |

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
