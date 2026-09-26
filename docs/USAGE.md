# memman - Usage & Reference

## Global flags

Available on every command:

| Flag                | Default     | Description                                                   |
| ------------------- | ----------- | ------------------------------------------------------------- |
| `--store <name>`    | (auto)      | Named memory store (overrides `MEMMAN_STORE` and active file) |
| `--data-dir <path>` | `~/.memman` | Base data directory                                           |
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

# Remove memman integrations
memman uninstall
memman uninstall --target claude-code
```

| Command            | `--target <name>` | Effect                                                                                                                                                                                                                                                                                      |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `memman install`   | (auto-detect)     | Deploy hook and skill symlinks, register in settings.json, install the scheduler unit, create `~/.memman/logs/` for scheduler output                                                                                                                                                        |
| `memman install`   | `claude-code`     | Install into `~/.claude/` even when Claude Code is not detected                                                                                                                                                                                                                             |
| `memman uninstall` | (auto-detect)     | Remove hooks, skill, settings.json entries, and scheduler unit. Strips secret keys (`MEMMAN_LLM_API_KEY`, `MEMMAN_OPENROUTER_API_KEY`, `MEMMAN_VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`) from `~/.memman/env` but keeps non-secret settings; memory store, queue, and logs untouched. |
| `memman uninstall` | `claude-code`     | Remove the Claude Code integration even when Claude Code is not detected                                                                                                                                                                                                                    |

One live-read command (called by the SessionStart hook, not by hand):

| Command        | What it prints                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `memman prime` | Reads SessionStart JSON on stdin; emits status + compact-recall hint + guide (called by prime.sh) |

---

## CLI Commands

### Core

```bash
# Remember - store a new insight
memman remember "Chose Qdrant over Milvus for vector search" \
  --cat decision --imp 5 --entity Qdrant --entity Milvus --source agent

# Recall - smart retrieval (default), one line per row
memman recall "vector database" --limit 10

# Recall with category/source filter (fills to --limit: the filter
# runs inside the anchor scans, not as a post-cut)
memman recall "auth" --cat decision --source agent

# Simple SQL LIKE matching (faster, skips ranking)
memman recall "auth" --basic

# Replace - deterministic replacement by ID (inherits metadata from
# original); the replaced row is superseded, never deleted
memman replace <id> "Updated content" --cat decision --imp 5

# Link two insights that both already exist as predecessor and successor
memman supersede <old_id> <new_id>

# Reverse a supersession once the successor has been forgotten
memman unsupersede <old_id>

# Forget - soft-delete an insight
memman forget <id>
```

Every command that takes an insight id also accepts any unambiguous
prefix of it, such as the eight-character ids memman prints; an
ambiguous prefix is refused with the number of rows it matches.

**Remember flags:**

| Flag       | Default | Description                                                                                                           |
| ---------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `--cat`    | `fact`  | Category: `preference`, `decision`, `fact`, `insight`, `context`                                                      |
| `--imp`    | `3`     | Importance 1-5, a sort key stored as passed                                                                           |
| `--entity` |         | Entity name (repeatable); the row stores exactly these names                                                          |
| `--source` | `user`  | Source: `user` (default), `agent`, or a locator for imported material; stored verbatim; recall filters by exact match |

**Recall flags:**

| Flag       | Default | Description                                                                                                   |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `--limit`  | `20`    | Max results                                                                                                   |
| `--cat`    |         | Filter by category                                                                                            |
| `--source` |         | Filter by source                                                                                              |
| `--basic`  | `false` | Use simple SQL LIKE matching instead of smart recall; returns before ranking, so each line carries no `score` |

The cross-encoder rerank stage is on by default and auto-skips on 1-2 token
queries. Provider is selected via `MEMMAN_RERANK_PROVIDER` (any registered
rerank provider; ships defaulted to `voyage` / `rerank-3-lite`). Toggle
per-store with `memman config set MEMMAN_RERANK_ENABLED_<store> false` or
globally with `memman config set MEMMAN_RERANK_ENABLED false`.

**Telling a weak result set from a strong one.** Smart recall returns
rows even when nothing matches: a recency channel seeds the newest
insights as anchors regardless, so a query that matches
nothing still comes back full. A full page is therefore not evidence
that anything on it is relevant, and a page that looks thin usually is
not: a store nearly always holds something bearing on a query drawn
from the same work. An empty page means the store itself is empty,
not that the query failed.

There is no flag for this, deliberately. Every printed row carries its
own `score`, and that is what a caller judges on - compared WITHIN one
page, never against a fixed number, because the scale belongs to
whichever reranker is configured and changes when the model does. A
boolean computed from a threshold would freeze one model's scale into
the output.

Rows come back in relevance order at every `--limit`, so the first `n`
rows of a page of `m` are exactly what a page of `n` returns. Nothing
re-sorts after the limit cut. Each row's own `created_at` field is
always printed, so a chronological view is a caller's own re-sort of
the page, not something recall does for it.

If a query returns nothing that bears on it, the likeliest cause is
vocabulary: re-ask in the store's own words before concluding the
store does not hold it.

### Graph operations

```bash
# Rebuild - full LLM re-enrichment + re-embed
memman graph rebuild              # process all insights
memman graph rebuild --dry-run    # preview count without modifying DB
memman graph rebuild --stale-only # re-enrich only rows whose prompt_version
                                  # no longer matches the active enrichment key
```

`graph rebuild` re-enriches all insights through the full LLM pipeline (enrichment, re-embedding). Processes in batches of 20. Returns `{"processed": N, "remaining": 0}`. Rejected when the scheduler is stopped.

`--stale-only` is the targeted variant: it only touches rows whose persisted `prompt_version` no longer matches `compute_prompt_version()` -- the enrichment prompt and the `slow` model, which is exactly the set this command replays. Cross-backend (works on Postgres, unlike wholesale `graph rebuild` which remains SQLite-only). Shares the `'rebuild'` advisory lock so it cannot race a wholesale rebuild. NULL-provenance rows are not swept; they need a separate backfill.

### Insights lifecycle

```bash
# Read a single insight by ID (full content + metadata; a superseded
# row shows its successor under `superseded_by`)
memman insights show <id>

# Walk the supersession chain through an id, oldest first
memman insights show <id> --history

# Resolve a write to the insights it produced (key from remember/replace)
memman insights by-queue <queue_uuid>

# Scan stored insights for content quality issues
memman insights review
```

To delete an insight, use `memman forget <id>`. A `replace` or
`memman supersede` never deletes: the corrected row is superseded,
keeps its content, and leaves recall and every listing. Nothing
deletes on its own: the store is uncapped and carries
no retention score, so a stored insight persists until an operator
removes it.

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

**Per-store embedder sovereignty.** Each store's `meta.embed_fingerprint` is the runtime authority over its embedder. Recall, drain, and graph rebuild all bind the embedder from the store's fingerprint, not from `MEMMAN_EMBED_PROVIDER`. One process can sequentially open two stores fingerprinted to different providers without env mutation - e.g., `MEMMAN_EMBED_PROVIDER=voyage memman --store openai_store recall ...` succeeds against an OpenAI-fingerprinted store. Switching a store's embedder is explicit (`embed swap` or `embed reembed`); there is no silent migration. Implementation details: [05-lifecycle.md § 5.3](design/05-lifecycle.md#53-embedding-support).

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

The command echoes a plan (source paths, redacted destination DSN, per-store target schema state - `ABSENT` / `EMPTY` / `POPULATED`) and prompts for confirmation. Stores already on the target backend emit a warning and are skipped (idempotent). `--dry-run` is supported only with `--to postgres`.

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
memman doctor                                       # health checks (integrity, schema, partial_index_predicates, enrichment, embeddings, fingerprint, supersession_integrity, queue, scheduler, drain heartbeat, env, no_stale_swap_meta, provenance_drift)
memman doctor --text                                # human-readable colored table
memman config show                                  # effective configuration (env + on-disk)

memman log list                                     # operation audit log (default JSON, last 20)
memman log list --limit 50                          # show more entries
memman log list --since 7d                          # entries from last 7 days
memman log list --since 7d --stats                  # grouped counts by operation
memman log list --text                              # human-readable text table

memman log worker [--errors] [--lines N]            # tail worker stdout/stderr (~/.memman/logs/enrich.{log,err})
memman log worker --stack [--lines N]               # tail the rotated log + backups (<data-dir>/logs/memman.log); excludes --errors
```

### Scheduler

```bash
memman scheduler status [--text]         # platform, interval, state, next run, last heartbeat, log paths (default JSON)
memman scheduler start [--text]          # flip persistent state to STARTED (resume drains + writes)
memman scheduler stop [--text]           # flip persistent state to STOPPED (pause drains + reject writes)
memman scheduler trigger                 # dispatch a drain, do not wait for it (systemd/launchd; not applicable in serve mode)
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

A row that exhausts its retries parks at `status='failed'` with its text intact. `memman scheduler queue retry <row_id>` requeues it; nothing deletes it automatically.

A stale row is a pending entry claimed more than `STALE_CLAIM_SECONDS` ago (default 600 s), usually from a mid-drain worker crash. The post-drain maintenance pass auto-recovers via `queue.retry_stale` alongside `purge_done` and `purge_worker_runs`; the explicit verbs exist for incident response.

When the scheduler is stopped, memman is recall-only: every write exits 1 with `Scheduler is stopped; cannot <verb>`. The `serve` loop polls the state file every iteration, so pause is observed within seconds even mid-drain.

### Backup

`memman backup` snapshots every store to an **external, durable directory** (e.g. a Dropbox path) on a cron schedule, rotates old bundles, and can rebuild a working store after total loss of `~/.memman/`. Bundles are written **only** to the target directory, never into `~/.memman/` (which is per-host and disposable, so an in-place archive dies with it). Snapshots are online and non-disruptive - the enrichment worker keeps draining (SQLite via the `sqlite3` online-backup API, Postgres via `pg_dump -Fc`), so no scheduler stop is needed.

```bash
memman backup run [TARGET]                        # build one bundle now (TARGET or MEMMAN_BACKUP_TARGET)
memman backup schedule '<cron>' TARGET [--keep N] # install a scheduled backup (cron -> native scheduler)
memman backup unschedule                          # remove the scheduled backup trigger (keeps env config)
memman backup list [TARGET]                        # list bundles at TARGET (read from sidecar manifests)
memman backup status                               # cron, target, keep, last fire, next run, latest bundle
memman backup restore BUNDLE [--yes]               # rebuild stores + non-secret config from a bundle
```

The cron string is a 5-field expression (`min hour dom month dow`, interpreted in local time) and is translated to the host's native scheduler at install time: systemd `OnCalendar=` (+`Persistent=true` for sleep/power-off catch-up), launchd `StartCalendarInterval`, or an in-process matcher in `serve` mode. The target directory is created if it does not exist. Retention keeps the newest `MEMMAN_BACKUP_KEEP` bundles (default 7).

Each bundle is one atomic `.tar.gz` plus an uncompressed sidecar `<bundle>.manifest.json` (for cheap `list`). A failing store is recorded with `status="failed"` in the manifest and never aborts the whole bundle. The global write queue (`queue.db`) is snapshotted too - and before the store DBs - so a `remember` that has not yet drained into its store is never lost: it rides in the bundle and drains on the restored host (the manifest records `queue_pending`). In `serve` mode the loop also drains the queue to empty before snapshotting, so the bundle is settled when possible.

**Secrets are excluded.** API keys, the default Postgres DSN, and every per-store `MEMMAN_POSTGRES_DSN_<store>` are stripped from the bundle's `env.nonsecret` member; per-store backend selection and model/provider/threshold knobs are kept. On `restore`, the non-secret config is merged first (so per-store backend routing is in place), each store is written by its manifest `backend` (SQLite file copy, or `pg_restore` resolving the DSN on the target host), and the active-store pointer is restored. `restore` holds the shared `drain.lock` and reports `secret_keys_needed` (re-enter these on the host), `pg_restore_skipped` (postgres stores with no DSN configured here), `embed_mismatch` (stores whose fingerprint differs from the restored embedding config), and any `failed` stores.

```bash
# A daily 03:00 backup to a Dropbox archive
memman backup schedule '0 3 * * *' ~/Dropbox/code/archive/
memman backup status                               # confirm next run + installed timer

# Restore after losing ~/.memman (re-enter secrets afterward, per secret_keys_needed)
memman backup restore ~/Dropbox/code/archive/memman-backup-<host>-<stamp>.tar.gz --yes
```

`memman uninstall` tears down the backup timer/agent alongside the enrichment scheduler; the `MEMMAN_BACKUP_*` env keys are kept so a later `memman backup schedule` resurrects the configuration.

---

## Configuration

memman reads config at runtime from one source: `<MEMMAN_DATA_DIR>/env`, a `KEY=VALUE` file at mode 0600 (default `~/.memman/env`). Shell environment variables are not consulted at runtime for installable settings, so a stale shell export cannot override a committed value.

`memman install` performs a one-time pull from the current shell into the env file. Precedence per key: existing file value > wizard prompt (TTY only) > `os.environ` > OpenRouter `/models` resolver (SLOW only) > `INSTALL_DEFAULTS`. Existing file values are sticky; reinstall never lets a shell export override them.

`memman config set KEY VALUE` is the override path. Use it after install to change a backend, rotate an API key, or update a DSN. Conflicts between an `INSTALLABLE_KEYS` flag and an existing env-file value are rejected with the exact `memman config set ...` command to run.

Process-control variables (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`, `MEMMAN_AUTHOR`) are not persisted to the file; they are read directly from `os.environ` by the components that own them. `MEMMAN_AUTHOR` names the person or agent issuing the write; when unset, memman falls back to `getpass.getuser()`. It is stamped on the queue row at enqueue time so the scheduler subprocess, which runs without directory environment, carries the correct author into the stored insight. `remember` and `replace` refuse content whose first word matches the resolved author (case-insensitive, word-boundary) - the author field already records who wrote it. The same two commands refuse content that names a line number: a locator after a source, config or doc extension (`scripts/auth.py:88`, `config.yaml:12`, `app.py-1233`, `cli.py ~1190`), the phrase `line N` or `lines N`, or a bare `:N` or `L123` after a space, `(`, `,` or `;` (`at :1774`, `emsx.py L419`). The refusal quotes the locator and asks for the file and the function or symbol instead, since a line number goes stale on the next edit. A host port (`localhost:6379`, `db.example.com:5432`), an image tag (`python:3.11`), a clock time (`14:18`), `code:404`, a slice (`[:80]`) and `DISPLAY=:99` pass unrefused. A port written without its host (`:9222`) is refused, and `localhost:9222` passes. A memory is one thought written as one paragraph that opens on its subject, so the two commands also refuse content that spans several lines, and content that opens with a label of at most three words before a colon and a space (`Fix:`, `AWS gotcha:`, `User decision 2026-09-17:`). A longer run before the colon passes, since it is as often a sentence ("The rule is simple:") as a label.

The full variable list lives in [CONTRIBUTING.md § Variable reference](../CONTRIBUTING.md#variable-reference).

### Install wizard

Run `memman install` in a TTY to get the interactive wizard. It prompts for the LLM endpoint URL (any OpenAI-compatible endpoint; ships defaulted to `https://openrouter.ai/api/v1`); for OpenRouter endpoints it auto-resolves the `slow` role's model slug (`MEMMAN_LLM_MODEL`) against `/v1/models`, for any other endpoint it prompts for the slug interactively. It then prompts (masked input) for `MEMMAN_LLM_API_KEY` (required for non-loopback endpoints; loopback endpoints like Ollama may leave it blank), then for the embedding provider (any registered provider; ships defaulted to `voyage`) and the matching key for that provider (e.g. `MEMMAN_VOYAGE_API_KEY` for voyage, `MEMMAN_OPENAI_EMBED_API_KEY` for openai; openrouter reuses the LLM key). It also offers a backend selector (sqlite/postgres) when the `memman[postgres]` extra is installed; the wizard probes the DSN, verifies the `pgvector` extension, and (for non-localhost DSNs) emits a hint about PgBouncer transaction pooling. Headless installs bypass the wizard:

- `--backend [sqlite|postgres]` - explicit backend choice; required in non-interactive mode if you want anything other than sqlite.
- `--pg-dsn URL` - Postgres DSN; required with `--backend postgres` in non-interactive mode. The DSN may omit the password to use `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD`.
- `--no-wizard` - disables prompts even in a TTY; flags + defaults only.

### Backend selection

memman routes each store through a backend chosen by env-file lookup:

1. `MEMMAN_BACKEND_<store>` - explicit per-store override (e.g., `MEMMAN_BACKEND_work=postgres`).
2. `MEMMAN_DEFAULT_BACKEND` - fallback when no per-store key is set (default `sqlite`).

`memman migrate <store>` writes `MEMMAN_BACKEND_<store>=postgres` so a single store can move to Postgres while others stay on SQLite. Use `memman config set MEMMAN_DEFAULT_BACKEND postgres` only when you want every newly-created store to default to Postgres.

The deferred-write queue is always SQLite at `<data_dir>/queue.db`. The Postgres backend stores per-store data in `store_<name>` schemas, each with its own `worker_runs` heartbeat table.

### Postgres DSN

Standard PostgreSQL libpq URI per psycopg3: `postgresql://[user[:password]@][host][:port]/[dbname][?param=value&...]`.

`memman config set-pg-dsn` walks you through host / port / user / password (masked) / dbname and writes the URI for you (URL-encoding special characters). Pass `--default` for `MEMMAN_DEFAULT_POSTGRES_DSN` or `--store NAME` for `MEMMAN_POSTGRES_DSN_<store>`:

```bash
memman config set-pg-dsn --default       # writes MEMMAN_DEFAULT_POSTGRES_DSN
memman config set-pg-dsn --store work    # writes MEMMAN_POSTGRES_DSN_work
```

Leave the password prompt empty to produce a passwordless DSN that defers to `~/.pgpass` (recommended on shared hosts). The command does not probe connectivity - verify with `memman doctor` or `memman migrate --dry-run`.

| Scenario     | DSN                                                      | Notes                                                  |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------ |
| local dev    | `postgresql://memman@localhost/memman`                   | no password                                            |
| inline creds | `postgresql://memman:s3cret@db.internal:5432/memman`     | URL-encode `: @ /` in the password                     |
| `~/.pgpass`  | `postgresql://memman@db.internal:5432/memman`            | passwordless URL, recommended                          |
| TLS-required | `postgresql://memman@db.internal/memman?sslmode=require` | + any libpq parameter (e.g. `application_name=memman`) |

> **Security.** `MEMMAN_DEFAULT_POSTGRES_DSN` and any `MEMMAN_POSTGRES_DSN_<store>` are stored plaintext in `~/.memman/env` at mode 0600. Root and any process running as your user can read them. For shared hosts, prefer `~/.pgpass` (mode 0600) and a passwordless DSN - psycopg3 sources the password from `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD` automatically.

### Runtime tunables

The variables below are not installable - they are read from the env file on demand by the components that own them, with no install-time seeding:

| Variable                          | Default         | Description                                                                                                           |
| --------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_REINDEX_TIMEOUT`          | `180`           | Seconds Postgres reindex (HNSW) is allowed to run before `statement_timeout` aborts; reraised idempotently next call. |
| `MEMMAN_EMBED_SWAP_BATCH_SIZE`    | `200`           | Rows per backfill batch in `memman embed swap`.                                                                       |
| `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` | `0` (unlimited) | Seconds Postgres `CREATE INDEX CONCURRENTLY` may run during cutover; `0` disables `statement_timeout`.                |

---

## Architecture

### Write pipeline (deferred, two-tier)

`memman remember` appends one row to the queue in ~50 ms on the host session - no LLM calls, no embeddings. The full pipeline runs out of band:

1. **Tier 1 (host)** - append a row to `~/.memman/queue.db` with `status='pending'`, the raw text, and any `--cat`/`--imp`/`--entity` hints. Returns `{action: queued, queue_id, queue_uuid, store}`. The `queue_uuid` is the join key: it is stamped on every insight this write produces and outlives the queue row, which `purge_done` drops about a minute after the drain.
2. **Tier 2 (worker)** - systemd timer (Linux), launchd agent (macOS), or `memman scheduler serve` PID 1 (containers) invokes `memman scheduler drain --timeout 60` every 60 s under an `flock` on `~/.memman/drain.lock`. Per row: quality gate → enrichment → embed (keyword-enriched text, or content alone) → add, or replace the row `replace <id>` names → mark done.

The host session never blocks on the network. Newly stored memories become recallable on the next drain tick (default 60 s).

### Recall pipeline

1. **RRF anchor selection** - keyword + vector + recency fused with K=60.
2. **3-signal blend** - keyword, similarity, and the fused anchor score. A stored entity name reaches the keyword signal because a candidate's token set unions its content tokens with its entity-name tokens.
3. **Cross-encoder rerank** (on by default; toggle per-store via `MEMMAN_RERANK_ENABLED_<store>`) - the configured reranker (default `voyage` / `rerank-3-lite`) re-scores the top 100 candidates; replaces the multi-signal score for the final ordering. Auto-skips on 1-2 token queries.
4. **Ordering** - nothing re-sorts after the limit cut: rows come back in relevance order.

See [Design & Architecture](DESIGN.md) for the full deep dive.
