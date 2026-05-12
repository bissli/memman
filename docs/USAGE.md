# memman — Usage & Reference

## Global flags

Available on every command:

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

| Command            | `--target <name>` | Effect                                                                                                                                                                                                                                                                                      |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `memman install`   | (auto-detect)     | Deploy hook and skill symlinks, register in settings.json, install the scheduler unit, create `~/.memman/logs/` for scheduler output                                                                                                                                                        |
| `memman install`   | `claude-code`     | Install into `~/.claude/` only                                                                                                                                                                                                                                                              |
| `memman install`   | `openclaw`        | Install into `~/.openclaw/` only                                                                                                                                                                                                                                                            |
| `memman install`   | `nanoclaw`        | Install into `~/.nanoclaw/` only                                                                                                                                                                                                                                                            |
| `memman uninstall` | (auto-detect)     | Remove hooks, skill, settings.json entries, and scheduler unit. Strips secret keys (`MEMMAN_LLM_API_KEY`, `MEMMAN_OPENROUTER_API_KEY`, `MEMMAN_VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`) from `~/.memman/env` but keeps non-secret settings; memory store, queue, and logs untouched. |
| `memman uninstall` | `<name>`          | Remove memman from that environment only                                                                                                                                                                                                                                                    |

Two live-read commands (called by hooks, not by hand):

| Command        | What it prints                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------- |
| `memman guide` | Shipped `guide.md` (hidden, called by openclaw bootstrap; humans read `guide.md` from the package) |
| `memman prime` | Reads SessionStart JSON on stdin; emits status + compact-recall hint + guide (called by prime.sh)  |

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

| Flag       | Default       | Description                                          |
| ---------- | ------------- | ---------------------------------------------------- |
| `--limit`  | `10`          | Max results                                          |
| `--intent` | (auto-detect) | Override intent: `WHY`, `WHEN`, `ENTITY`, `GENERAL`  |
| `--cat`    |               | Filter by category                                   |
| `--source` |               | Filter by source                                     |
| `--basic`  | `false`       | Use simple SQL LIKE matching instead of smart recall |
| `--expand` | `false`       | Opt-in LLM query expansion (synonyms + entity hints) |

The cross-encoder rerank stage is on by default and auto-skips on 1-2 token
queries. Provider is set via `MEMMAN_RERANK_PROVIDER` (default `voyage` /
`rerank-2.5-lite`). Toggle per-store with
`memman config set MEMMAN_RERANK_ENABLED_<store> false` or globally with
`memman config set MEMMAN_RERANK_ENABLED false`.

### Graph operations

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
memman graph rebuild --stale-only # re-enrich only rows whose prompt_version
                                  # or model_id no longer matches active config
```

Auto-reindex of computed edges (semantic, entity, temporal) fires on `open_db()` when graph constants have changed; no operator command for it.

`graph rebuild` re-enriches all insights through the full LLM pipeline (enrichment, re-embedding, causal inference, edge recreation). Processes in batches of 20. Returns `{"processed": N, "remaining": 0}`. Rejected when the scheduler is stopped.

`--stale-only` is the targeted variant: it only touches rows whose persisted `prompt_version` or `model_id` no longer matches the active config. Cross-backend (works on Postgres, unlike wholesale `graph rebuild` which remains SQLite-only). Shares the `'rebuild'` advisory lock so it cannot race a wholesale rebuild. NULL-provenance rows are not swept; they need a separate backfill.

### Insights lifecycle

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

To delete an insight, use `memman forget <id>`.

### Embedding operations

```bash
# Show this store's bound fingerprint and whether its provider's
# credentials are available in this process
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
- **`embed reembed`** is the offline path: every store is rewritten in place with the current `MEMMAN_EMBED_PROVIDER`. Requires the scheduler to be **stopped** (`memman scheduler stop`).

**Per-store embedder sovereignty.** Each store's `meta.embed_fingerprint` is the runtime authority over its embedder. Recall, drain, graph rebuild, and snapshot writes all bind the embedder from the store's fingerprint, not from `MEMMAN_EMBED_PROVIDER`. One process can sequentially open two stores fingerprinted to different providers without env mutation — e.g., `MEMMAN_EMBED_PROVIDER=voyage memman --store openai_store recall ...` succeeds against an OpenAI-fingerprinted store. Switching a store's embedder is explicit (`embed swap` or `embed reembed`); there is no silent migration. Implementation details: [05-lifecycle.md § 5.5](design/05-lifecycle.md#55-embedding-support).

### Store management

memman supports named stores for data isolation. Each store has its own database.

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

**Per-directory automatic switching.** `MEMMAN_STORE` is read from `os.environ`, so any tool that scopes env vars to a working directory will flip the active store on `cd`. Four mechanisms:

| Mechanism                | Setup                                               | Scope                                                       |
| ------------------------ | --------------------------------------------------- | ----------------------------------------------------------- |
| `direnv` (recommended)   | `.envrc` in the project: `export MEMMAN_STORE=work` | Every shell, agent, and subprocess started in the directory |
| `--store <name>` flag    | Pass `--store work` on every invocation             | One command; explicit, survives a missing env               |
| Project `CLAUDE.md` rule | Instruct the agent to pass `--store work`           | Claude Code sessions only; not honored by terminal callers  |
| `memman store use work`  | Set the global `~/.memman/active` file              | Persistent and global; last `use` wins everywhere           |

Do not set `MEMMAN_DATA_DIR` per directory. The scheduler unit is installed once against `~/.memman/queue.db`; a per-directory data dir creates an isolated queue that the host scheduler never drains. Use a named store instead and let the worker dispatch per row.

#### Migrating between SQLite and Postgres

`memman migrate` is symmetric: `--to postgres` (default) copies a store from SQLite into Postgres; `--to sqlite` copies it back. Both directions hold the shared `drain.lock` so a scheduler-fired drain cannot race.

| Direction       | Source                                                       | Destination                 | Backend flag flipped to           |
| --------------- | ------------------------------------------------------------ | --------------------------- | --------------------------------- |
| `--to postgres` | SQLite store (preserved)                                     | `store_<name>` schema in PG | `MEMMAN_BACKEND_<store>=postgres` |
| `--to sqlite`   | Postgres `store_<name>` (dumped to `archive/`, then dropped) | Fresh SQLite store          | `MEMMAN_BACKEND_<store>=sqlite`   |

The command echoes a plan (source paths, redacted destination DSN, per-store target schema state — `ABSENT` / `EMPTY` / `POPULATED`) and prompts for confirmation. Stores already on the target backend emit a warning and are skipped (idempotent). `--dry-run` is supported only with `--to postgres`.

```bash
# Forward (default): SQLite -> Postgres, dry-run plan only
memman migrate --store work --dry-run

# Forward (default): SQLite -> Postgres, interactive
memman migrate --store work

# Reverse: Postgres -> SQLite (no --dry-run); preserves a dump under archive/
memman migrate --store work --to sqlite

# Non-interactive (CI / scripts): skip the prompt
memman migrate --all --yes
```

To revert a single store without re-migrating data, set the backend flag directly: `memman config set MEMMAN_BACKEND_<store> sqlite` (or unset the key to fall back to `MEMMAN_DEFAULT_BACKEND`). To verify the cutover, run `memman doctor`.

### Observability

```bash
memman status                                       # memory statistics; JSON includes stale_insights count
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
memman scheduler status [--text]         # platform, interval, state, next run, last heartbeat (default JSON)
memman scheduler start [--text]          # flip persistent state to STARTED (resume drains + writes)
memman scheduler stop [--text]           # flip persistent state to STOPPED (pause drains + reject writes)
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
memman scheduler queue retry --all-stale # requeue every row currently in status='stale'
memman scheduler queue purge --done      # delete rows where status='done'
memman scheduler queue purge --stale     # delete rows where status='stale'
```

A stale row is a pending entry claimed more than `STALE_CLAIM_SECONDS` ago (default 600 s), usually from a mid-drain worker crash. The post-drain maintenance pass auto-recovers via `queue.retry_stale` alongside `purge_done` and `purge_worker_runs`; the explicit verbs exist for incident response.

When the scheduler is stopped, memman is recall-only: every write exits 1 with `Scheduler is stopped; cannot <verb>`. The `serve` loop polls the state file every iteration, so pause is observed within seconds even mid-drain.

---

## Configuration

memman reads config at runtime from one source: `<MEMMAN_DATA_DIR>/env`, a `KEY=VALUE` file at mode 0600 (default `~/.memman/env`). Shell environment variables are not consulted at runtime for installable settings, so a stale shell export cannot override a committed value.

`memman install` performs a one-time pull from the current shell into the env file. Precedence per key: existing file value > wizard prompt (TTY only) > `os.environ` > OpenRouter `/models` resolver (FAST/SLOW only) > `INSTALL_DEFAULTS`. Existing file values are sticky; reinstall never lets a shell export override them.

`memman config set KEY VALUE` is the override path. Use it after install to change a backend, rotate an API key, or update a DSN. Conflicts between an `INSTALLABLE_KEYS` flag and an existing env-file value are rejected with the exact `memman config set ...` command to run.

Process-control variables (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`) are not persisted to the file; they are read directly from `os.environ` by the components that own them.

The full variable list lives in [CONTRIBUTING.md § Variable reference](../CONTRIBUTING.md#variable-reference).

### Install wizard

Run `memman install` in a TTY to get the interactive wizard. It first prompts for the LLM endpoint URL (default `https://openrouter.ai/api/v1`); for OpenRouter endpoints it auto-resolves the three role model slugs (`MEMMAN_LLM_MODEL_FAST` / `_SLOW_CANONICAL` / `_SLOW_METADATA`) against `/v1/models`, for any other endpoint it prompts for each slug interactively. It then prompts (with masked input) for `MEMMAN_LLM_API_KEY` (required for non-loopback endpoints; loopback endpoints like Ollama may leave it blank) and `MEMMAN_VOYAGE_API_KEY` (required when `MEMMAN_EMBED_PROVIDER=voyage`, the default). It also offers a backend selector (sqlite/postgres) when the `memman[postgres]` extra is installed; the wizard probes the DSN, verifies the `pgvector` extension, and (for non-localhost DSNs) emits a hint about PgBouncer transaction pooling. Headless installs bypass the wizard:

- `--backend [sqlite|postgres]` — explicit backend choice; required in non-interactive mode if you want anything other than sqlite.
- `--pg-dsn URL` — Postgres DSN; required with `--backend postgres` in non-interactive mode. The DSN may omit the password to use `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD`.
- `--no-wizard` — disables prompts even in a TTY; flags + defaults only.

### Backend selection

memman routes each store through a backend chosen by env-file lookup:

1. `MEMMAN_BACKEND_<store>` — explicit per-store override (e.g., `MEMMAN_BACKEND_work=postgres`).
2. `MEMMAN_DEFAULT_BACKEND` — fallback when no per-store key is set (default `sqlite`).

`memman migrate <store>` writes `MEMMAN_BACKEND_<store>=postgres` so a single store can move to Postgres while others stay on SQLite. Use `memman config set MEMMAN_DEFAULT_BACKEND postgres` only when you want every newly-created store to default to Postgres.

The deferred-write queue is always SQLite at `<data_dir>/queue.db`. The Postgres backend stores per-store data in `store_<name>` schemas, each with its own `worker_runs` heartbeat table.

### Postgres DSN

Standard PostgreSQL libpq URI per psycopg3: `postgresql://[user[:password]@][host][:port]/[dbname][?param=value&...]`.

`memman config set-pg-dsn` walks you through host / port / user / password (masked) / dbname and writes the URI for you (URL-encoding special characters). Pass `--default` for `MEMMAN_DEFAULT_POSTGRES_DSN` or `--store NAME` for `MEMMAN_POSTGRES_DSN_<store>`:

```bash
memman config set-pg-dsn --default       # writes MEMMAN_DEFAULT_POSTGRES_DSN
memman config set-pg-dsn --store work    # writes MEMMAN_POSTGRES_DSN_work
```

Leave the password prompt empty to produce a passwordless DSN that defers to `~/.pgpass` (recommended on shared hosts). The command does not probe connectivity — verify with `memman doctor` or `memman migrate --dry-run`.

| Scenario     | DSN                                                      | Notes                                                  |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------ |
| local dev    | `postgresql://memman@localhost/memman`                   | no password                                            |
| inline creds | `postgresql://memman:s3cret@db.internal:5432/memman`     | URL-encode `: @ /` in the password                     |
| `~/.pgpass`  | `postgresql://memman@db.internal:5432/memman`            | passwordless URL, recommended                          |
| TLS-required | `postgresql://memman@db.internal/memman?sslmode=require` | + any libpq parameter (e.g. `application_name=memman`) |

> **Security.** `MEMMAN_DEFAULT_POSTGRES_DSN` and any `MEMMAN_POSTGRES_DSN_<store>` are stored plaintext in `~/.memman/env` at mode 0600. Root and any process running as your user can read them. For shared hosts, prefer `~/.pgpass` (mode 0600) and a passwordless DSN — psycopg3 sources the password from `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD` automatically.

### Runtime tunables

The variables below are not installable — they are read from the env file on demand by the components that own them, with no install-time seeding:

| Variable                          | Default         | Description                                                                                                           |
| --------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_REINDEX_TIMEOUT`          | `180`           | Seconds Postgres reindex (HNSW) is allowed to run before `statement_timeout` aborts; reraised idempotently next call. |
| `MEMMAN_EMBED_SWAP_BATCH_SIZE`    | `200`           | Rows per backfill batch in `memman embed swap`.                                                                       |
| `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` | `0` (unlimited) | Seconds Postgres `CREATE INDEX CONCURRENTLY` may run during cutover; `0` disables `statement_timeout`.                |

---

## Architecture

### Write pipeline (deferred, two-tier)

`memman remember` appends one row to the queue in ~50 ms on the host session — no LLM calls, no embeddings, no edges. The full pipeline runs out of band:

1. **Tier 1 (host)** — append a row to `~/.memman/queue.db` with `status='pending'`, the raw text, and any `--cat`/`--imp`/`--entities` hints. Returns `{action: queued, queue_id, store}`.
2. **Tier 2 (worker)** — systemd timer (Linux), launchd agent (macOS), or `memman scheduler serve` PID 1 (containers) invokes `memman scheduler drain --pending` every 60 s under an `flock` on `~/.memman/drain.lock`. Per row: quality gate → LLM fact extraction → per-fact embed + similarity scan + LLM reconciliation (ADD/UPDATE/DELETE/NONE) → insert/update → fast edges (temporal + entity + semantic) → parallel enrichment + LLM causal inference → re-embed → rebuild auto edges → mark done.

The host session never blocks on the network. Newly stored memories become recallable on the next drain tick (default 60 s).

### Recall pipeline

1. **LLM query expansion** (opt-in via `--expand`) — synonyms, entity extraction, intent detection.
2. **RRF anchor selection** — keyword + vector + recency fused with K=60.
3. **Beam search** — intent-weighted graph traversal from anchors.
4. **4-signal rerank** — keyword, entity, similarity, graph (intent-weighted).
5. **Cross-encoder rerank** (on by default; toggle per-store via `MEMMAN_RERANK_ENABLED_<store>`) — the configured reranker (default `voyage` / `rerank-2.5-lite`) re-scores the top 100 candidates; replaces the multi-signal score for the final ordering. Auto-skips on 1-2 token queries.
6. **Post-sort** — causal topological (WHY), chronological (WHEN), score (default).

Inspired by [MAGMA](https://arxiv.org/abs/2601.03236). See [Design & Architecture](DESIGN.md) for the full deep dive.
