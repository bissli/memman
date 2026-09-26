# memman usage and reference

This page lists every memman command and setting. [DESIGN.md](DESIGN.md) explains how the parts work.

## Global flags

A global flag goes before the subcommand: `memman --store work recall "retry cap"`. The `--store` option of `memman migrate` and `memman config set-pg-dsn` is a separate subcommand option with its own meaning.

| Flag                | Default     | Description                                                                                                                                                                                           |
| ------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--store <name>`    | none        | Store to use. Takes precedence over `MEMMAN_STORE` and the active-store file (see [Store management](#store-management)).                                                                             |
| `--data-dir <path>` | `~/.memman` | Data directory for the stores, the queue, and `memman.log`. Falls back to `MEMMAN_DATA_DIR`, then `~/.memman`. Settings still come from `$MEMMAN_DATA_DIR/env` (see [Configuration](#configuration)). |
| `--verbose` / `-v`  | off         | INFO-level logging to stderr.                                                                                                                                                                         |
| `--debug`           | off         | DEBUG-level logging to stderr. Takes precedence over `--verbose`.                                                                                                                                     |
| `--version`         |             | Print the version and exit.                                                                                                                                                                           |

Without `--verbose` or `--debug`, the stderr level is `MEMMAN_LOG_LEVEL` (default `WARNING`).

---

## Install and uninstall

`memman install` runs after `pipx install memman`. The [README](../README.md#install) covers the package install and provider keys.

```bash
memman install                        # interactive wizard in a terminal
memman install --target claude-code   # install into ~/.claude even when Claude Code is not detected
memman install --no-wizard --backend postgres --pg-dsn postgresql://memman@localhost/memman
memman uninstall
memman uninstall --target claude-code
```

**Install flags:**

| Flag                    | Effect                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--target claude-code`  | Install into `~/.claude`, whether or not Claude Code is detected.                                                                                                         |
| `--backend NAME`        | Default storage backend, `sqlite` or `postgres`. Skips the backend prompt.                                                                                                |
| `--pg-dsn URL`          | Postgres DSN. Install connects, checks for `pgvector`, and stops when either fails. Required with `--backend postgres` when no prompt runs and the env file holds no DSN. |
| `--llm-endpoint URL`    | LLM endpoint URL. Skips the endpoint prompt. Must start with `http://` or `https://`.                                                                                     |
| `--embed-provider NAME` | Embedding provider: `voyage`, `openai`, or `openrouter`. Skips the provider prompt.                                                                                       |
| `--no-wizard`           | Skip all prompts, even in a terminal. Flags, the shell, and defaults supply every value.                                                                                  |

**What install does, in order:**

1. Refuses a flag whose value differs from the env file, and prints the `memman config set` command that changes it.
2. Runs the install wizard (below) and writes its answers to the env file.
3. Checks the host and the required keys, and works out every missing setting (see [Configuration](#configuration)).
4. When Claude Code is detected (a `claude` binary on `PATH` or a `~/.claude` directory), or with `--target claude-code`:
   - creates `~/.memman/logs/` at mode 0700,
   - symlinks the skill to `~/.claude/skills/memman/SKILL.md` and the five hook scripts into `~/.claude/hooks/memman/`,
   - registers the hooks in `~/.claude/settings.json`,
   - adds a `permissions.allow` entry for each memman command the agent may call. In a terminal and without `--no-wizard`, install lists the entries and asks first,
   - creates the `default` store when it is a SQLite store that does not exist yet.
5. Writes those settings to the env file and installs the scheduler unit: a systemd timer on Linux or a launchd agent on macOS.
6. On an OpenRouter endpoint, checks that a zero-data-retention endpoint on a vendor in `MEMMAN_LLM_PROVIDER_ONLY` serves `MEMMAN_LLM_MODEL`, and that OpenRouter lists no retirement date for it. The drain repeats the check once a day. A catalog outage prints an error, and the install still finishes.

Without Claude Code and without `--target`, install sets up the scheduler only. Install needs systemd or launchd. On a host with neither, `MEMMAN_SCHEDULER_KIND=serve` in the environment selects serve mode, where a `memman scheduler serve` process drains the queue.

A new Claude Code session picks up the hooks. [Chapter 5](design/05-integration.md) describes the hooks, the guide, and the skill. The SessionStart hook runs the hidden `memman prime`, which prints the status line, any model notice, a reminder to recall after a compaction, and the guide.

### Install wizard

The wizard runs only in a terminal and only without `--no-wizard`. Each step prompts only when no flag supplies the value and the env file lacks it. The key and model steps also stay silent when the shell exports the `MEMMAN_` variable. Key prompts hide the input.

1. **LLM endpoint.** Any OpenAI-compatible URL. The default is `https://openrouter.ai/api/v1`.
2. **Embedding provider.** `voyage` (default), `openai`, or `openrouter`.
3. **Embedding key.** The key the provider needs, such as `MEMMAN_VOYAGE_API_KEY`. When the shell exports only the vendor name (`VOYAGE_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`), the prompt offers that value as its default.
4. **LLM key.** `MEMMAN_LLM_API_KEY`. Required for any endpoint outside the local machine and optional for `localhost`. On an OpenRouter endpoint the step is skipped when `MEMMAN_OPENROUTER_API_KEY` is set, and install copies that key into `MEMMAN_LLM_API_KEY`.
5. **Model.** Asked only on an endpoint other than OpenRouter, because the default model `qwen/qwen3-235b-a22b-2507` is an OpenRouter id. The model ID passes unchanged to `/chat/completions`.
6. **Backend.** `sqlite` or `postgres`. Offered only when the `memman[postgres]` extra is installed.
7. **Postgres DSN.** Asked for the `postgres` backend. The wizard connects, checks that the `pgvector` extension exists, and allows three attempts. A DSN that names a host other than `localhost` prints a hint to run Postgres behind PgBouncer in transaction-pooling mode.

Without the wizard, install refuses to finish when a required value is missing: the embedding key, the DSN for `--backend postgres`, or `MEMMAN_LLM_MODEL` on an endpoint other than OpenRouter. The error names the key.

### Uninstall

`memman uninstall` (with an optional `--target claude-code`) removes:

- the scheduled backup timer or agent,
- `~/.claude/hooks/memman/` and `~/.claude/skills/memman/`,
- the memman hooks and permission entries in `~/.claude/settings.json`,
- a block between `<!-- memman:start -->` and `<!-- memman:end -->` in `CLAUDE.md` in the current directory,
- the scheduler unit, `~/.memman/scheduler.state`, and `~/.memman/debug.state`,
- the secret keys in the env file: `MEMMAN_LLM_API_KEY`, `MEMMAN_OPENROUTER_API_KEY`, `MEMMAN_VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, and `MEMMAN_DEFAULT_POSTGRES_DSN`.

It keeps every other env-file setting, including `MEMMAN_POSTGRES_DSN_<store>`, and it keeps the stores, the queue, and the logs. When the Claude Code cleanup reports an error, uninstall stops and leaves the scheduler in place.

---

## Memory commands

```bash
memman remember "The retry cap stays at three, since a fourth try only adds load." \
  --cat decision --imp 4 --entity RetryPolicy --source agent
memman recall "retry cap" --limit 10
memman recall "auth" --cat decision --source agent
memman recall "auth" --basic
memman replace <id> "The retry cap is four for batch jobs and three elsewhere."
memman supersede <old_id> <new_id>
memman unsupersede <old_id>
memman forget <id>
```

Every command that takes a memory id also accepts an unambiguous prefix of one, such as the 8-character id that recall prints. An ambiguous prefix is refused, and the error names how many ids it matches.

`remember`, `replace`, `forget`, `supersede`, and `unsupersede` refuse to run while the scheduler is stopped (see [Scheduler](#scheduler)).

### remember and replace

`remember` adds the text to the write queue and returns at once. The background worker stores it on its next drain, and recall finds it from then on. The reply is JSON: `action` (`queued`), `queue_id`, `queue_uuid`, `store`, and `quality_warnings`. `replace` adds `replaced_id`. `memman insights by-queue <queue_uuid>` finds the memories a write produced.

`quality_warnings` lists phrasing that tends to go stale, such as an instance id or the word "currently". The warnings never block the write.

| Flag       | `remember` default | `replace` default  | Meaning                                                                         |
| ---------- | ------------------ | ------------------ | ------------------------------------------------------------------------------- |
| `--cat`    | `fact`             | the target's value | Category: `preference`, `decision`, `fact`, `insight`, or `context`.            |
| `--imp`    | `3`                | the target's value | Importance, 1 to 5. A sort key for listings and ties, stored as given.          |
| `--entity` | none               | the target's value | One entity name per flag, repeatable, at most 50. A name may contain a comma.   |
| `--source` | `user`             | the target's value | Source: `user`, `agent`, or the location of imported material. Stored as given. |

`replace <id>` queues a successor for a current memory. When the drain stores the successor, the target becomes superseded: it keeps its content and leaves recall and every listing. A forgotten or superseded target is refused, and the error for a superseded one names its successor. Each flag left off inherits the target's value. `--entity ''` clears the entity list.

### What remember and replace refuse

Both commands check the text in this order and report the first problem:

1. **Size.** More than 1,000 UTF-8 bytes.
2. **Line number.** Text that names a line of a file, because a line number goes stale on the next edit:
   - a source, config, or doc file name followed by a line: `scripts/auth.py:88`, `config.yaml:12`, `app.py-1233`, `cli.py ~1190`,
   - `line N` or `lines N`,
   - a bare `:N` or `L123` after the start, a space, `(`, `,`, or a semicolon: `at :1774`, `emsx.py L419`.

   A host port (`localhost:6379`, `db.example.com:5432`), an image tag (`python:3.11`), a clock time (`14:18`), `code:404`, a slice (`[:80]`), and `DISPLAY=:99` pass. A port without its host (`:9222`) is refused.
3. **Author.** Text whose first word is the value of `MEMMAN_AUTHOR`, ignoring case. The author field already records who wrote it. The check runs only when `MEMMAN_AUTHOR` is set, so the login-name fallback never refuses.
4. **Line break.** Text that spans several lines.
5. **Leading label.** Text that opens with at most three words, a colon, and a space: `Fix:`, `AWS gotcha:`, `User decision 2026-09-17:`. A longer phrase before the colon is allowed because it may be part of a sentence, such as "The rule is simple:". A quote or backtick ends the match, so text may start with a quoted error.

They also refuse an unknown category, an importance outside 1 to 5, an empty `--source`, more than 50 entities, and an entity name over 200 characters. `replace` checks an inherited category, importance, and source the same way.

### recall

| Flag       | Default | Description                                                |
| ---------- | ------- | ---------------------------------------------------------- |
| `--limit`  | `20`    | Maximum lines printed.                                     |
| `--cat`    | none    | Keep only this exact category.                             |
| `--source` | none    | Keep only this exact source.                               |
| `--basic`  | off     | SQL `LIKE` matching with no ranking. Lines carry no score. |

`--cat` and `--source` filter each ranking channel before its cut, so a filtered page still fills to `--limit` when enough memories match.

Recall prints one line per memory, best first, and prints nothing for an empty result:

```text
<id8> <score> <created_at> <author> <category> | <text>
```

`id8` is the first 8 characters of the id, and `score` has two decimals. `author` is `-` when unset. `text` is the summary when the memory has one, and the start of the content otherwise. `memman insights show <id>` prints the whole memory.

Compare scores only within the same result page. They have no fixed meaning across queries. Recency ranking always adds the newest memories to the candidates, so a full page does not by itself show a match. If none of the results are relevant, the query may use different wording from the stored memories. Try a query using words from the store. [Chapter 3](design/03-pipelines.md) describes the ranking.

`--basic` keeps memories in which every query word appears in the content, the entities, or the keywords, and orders them by importance, then by creation time, newest first.

**Rerank.** For a query of more than two words, a cross-encoder re-scores the top 100 candidates. The only rerank provider is Voyage (`MEMMAN_RERANK_PROVIDER=voyage`, model `MEMMAN_VOYAGE_RERANK_MODEL`, default `rerank-3-lite`), and it needs `MEMMAN_VOYAGE_API_KEY`. When the rerank call fails, recall logs a warning and keeps the blended order. `MEMMAN_RERANK_ENABLED` (default `true`) enables or disables reranking for every store, and `MEMMAN_RERANK_ENABLED_<store>` overrides it for one store:

```bash
memman config set MEMMAN_RERANK_ENABLED_work false
```

### supersede, unsupersede, and forget

- `supersede <old_id> <new_id>` marks one current memory as superseded by another. Both keep their content. It is the only way to link two memories that both exist, since `replace` always writes a new one. Both ids must be current and different. One successor can supersede several predecessors.
- `unsupersede <old_id>` returns a superseded memory to recall. Its successor must be forgotten first. The command re-embeds the content with the store's embedding model and stops if that call fails. The memory remains superseded.
- `forget <id>` soft-deletes a memory: it sets `deleted_at`, and the memory leaves recall and every listing. No command reverses a forget. A superseded memory can be forgotten.

Nothing deletes a memory on its own. The store has no cap and no retention score, so a memory stays until an operator forgets it.

---

## Insights

```bash
memman insights show <id>              # one memory as JSON, including superseded memories
memman insights show <id> --history    # the supersession chain through <id>, oldest first
memman insights by-queue <queue_uuid>  # the memories one queued write produced
memman insights review [--limit N]     # memories with quality warnings
```

- `show` accepts a forgotten memory ID only with `--history`. It then lists every memory in the chain with its `state`: `current`, `superseded`, or `forgotten`. A forgotten entry omits the content.
- `by-queue` returns `{queue_uuid, store, count, results}` for the store it searched. `count: 0` has three causes: the write is still queued, it went to another store, or its memory was forgotten. An invalid UUID is rejected.
- `review` checks current memories, newest first, against the same patterns as `quality_warnings`, and stops after `--limit` flagged memories (default 20).

---

## Re-enrichment

`memman graph rebuild` re-runs enrichment (keywords and summary) and the embedding for current memories.

```bash
memman graph rebuild               # every current memory
memman graph rebuild --stale-only  # only memories enriched under another prompt or model
memman graph rebuild --dry-run     # print the count and change nothing
```

- Both modes need a stopped scheduler (`memman scheduler stop`), except with `--dry-run`. Both run on SQLite and Postgres.
- The command works in batches of 20 and prints `{processed, remaining}`. `remaining` counts memories still waiting for enrichment after the run.
- `--stale-only` selects current memories whose `prompt_version` differs from the active one. The `prompt_version` is a hash of the enrichment prompt and `MEMMAN_LLM_MODEL`. A memory with no `prompt_version` is skipped. `memman status` reports the same count as `stale_insights`.
- `--progress-jsonl` writes one JSON progress line per memory to stderr.
- A second rebuild on the same store is refused while one runs.

---

## Embedding operations

Each store keeps the embedding model it was created with, recorded as its fingerprint. Recall, the background worker, and `graph rebuild` use the model recorded in the fingerprint. Every command that reads a store, except `doctor`, `embed status`, and `embed swap`, also builds the `MEMMAN_EMBED_PROVIDER` client, so that provider's key must be in the env file. [Chapter 4](design/04-lifecycle.md#43-embedding-support) describes the fingerprint.

```bash
memman embed status                                             # fingerprint, key check, swap progress
memman embed swap --to voyage-3-large                           # this store, provider from MEMMAN_EMBED_PROVIDER
memman embed swap --to text-embedding-3-small --provider openai
memman embed swap --resume                                      # continue a swap
memman embed swap --abort                                       # discard a swap before cutover
memman embed reembed                                            # every SQLite store, MEMMAN_EMBED_PROVIDER
memman embed reembed --dry-run                                  # count what would change
```

**`embed swap`** moves one store (the store the global flags select) to a new model.

- It needs a stopped scheduler, except with `--abort`. Recall keeps reading the old vectors while the swap writes new ones into the `embedding_pending` column in batches of `MEMMAN_EMBED_SWAP_BATCH_SIZE` (default 200).
- On Postgres, the swap first builds an HNSW index on the new column.
- The final switch, called cutover, replaces the old vectors in one transaction. Returning to the old model requires another full swap.
- `--resume` continues an interrupted swap from its recorded cursor. `--abort` drops the new column and the swap state.
- `--provider` defaults to `MEMMAN_EMBED_PROVIDER`. The swap leaves `MEMMAN_EMBED_PROVIDER` unchanged.

**`embed reembed`** moves every SQLite store under the data directory to the `MEMMAN_EMBED_PROVIDER` client.

- It needs a stopped scheduler, except with `--dry-run`. It refuses to run when the selected store is on Postgres, and it skips Postgres stores.
- For each store, every current memory whose vector is not on the target model is re-embedded, and the store gets the new fingerprint. An empty store gets only the fingerprint.
- A second run resumes an interrupted one.

To change the global embedding provider:

```bash
memman config set MEMMAN_EMBED_PROVIDER openai
memman config set MEMMAN_OPENAI_EMBED_API_KEY sk-...
memman scheduler stop
memman embed reembed
memman scheduler start
```

---

## Store management

A store is a named, isolated set of memories: one SQLite file or one Postgres schema. [Chapter 2](design/02-concepts.md) explains why stores exist.

```bash
memman store list            # JSON: {stores, active}
memman store create work
memman store use work        # write "work" to the active-store file
memman store remove old-project [--yes]
```

- A store name starts with a letter or digit and continues with letters, digits, `_`, or `-`. A Postgres store also needs a name that is a valid SQL identifier.
- `store use` accepts only an existing store.
- `store remove` asks first unless `--yes` is given. It refuses the store named in the active-store file. It deletes the store's data, its queued writes, and its `MEMMAN_BACKEND_<store>`, `MEMMAN_POSTGRES_DSN_<store>`, and `MEMMAN_RERANK_ENABLED_<store>` keys.
- `memman store` with no subcommand runs `store list`.

**Which store a command uses**, from highest priority to lowest:

1. the `--store <name>` flag,
2. the `MEMMAN_STORE` environment variable,
3. the active-store file `<data dir>/active`,
4. `default`.

**Per-directory stores.** memman reads `MEMMAN_STORE` from the process environment, so a tool that sets variables per directory switches the store on `cd`.

| Mechanism               | Setup                                                    | Scope                                                           |
| ----------------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| `direnv`                | `.envrc` in the project holds `export MEMMAN_STORE=work` | Every shell, agent, and subprocess started in the directory     |
| `--store` flag          | `--store work` on each command                           | One command                                                     |
| Project `CLAUDE.md`     | A directive telling the agent to pass `--store work`     | Claude Code sessions only                                       |
| `memman store use work` | Writes the global active-store file                      | Every caller on the host. The most recent `use` sets the store. |

Use named stores for separate projects. The scheduler processes only the queue in its configured data directory. Setting a different `MEMMAN_DATA_DIR` for each project creates queues that the scheduler does not process.

### Migrating between SQLite and Postgres

`memman migrate` copies stores between backends in either direction.

```bash
memman migrate --store work --dry-run    # SQLite -> Postgres, print the plan only
memman migrate --store work              # SQLite -> Postgres, asks first
memman migrate --store work --to sqlite  # Postgres -> SQLite
memman migrate --all --yes               # every store, no prompt
```

| Flag           | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `--store NAME` | The store to migrate. Required unless `--all` is given.       |
| `--all`        | Every store in the data directory.                            |
| `--to NAME`    | Target backend, `postgres` or `sqlite`. Default `postgres`.   |
| `--dry-run`    | Print the plan and change nothing. Only with `--to postgres`. |
| `--yes`        | Skip the confirmation prompt.                                 |

| Direction       | Source                                                                                         | Destination                       | Resulting backend settings                                                 |
| --------------- | ---------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| `--to postgres` | SQLite store, moved to `archive/<store>/<YYYYMMDD>_<NN>/`                                      | `store_<name>` schema in Postgres | `MEMMAN_BACKEND_<store>=postgres`, plus `MEMMAN_POSTGRES_DSN_<store>`      |
| `--to sqlite`   | Postgres `store_<name>`, dumped to `archive/<store>/<YYYYMMDD>_<NN>/dump.pgdump`, then dropped | New SQLite store                  | `MEMMAN_BACKEND_<store>=sqlite`, and `MEMMAN_POSTGRES_DSN_<store>` removed |

- Both directions need `pg_dump` on `PATH` and hold the drain lock, so a scheduled drain cannot run during the copy.
- The DSN for `--to postgres` is `MEMMAN_POSTGRES_DSN_<store>`, then `MEMMAN_DEFAULT_POSTGRES_DSN`. `--all` needs `MEMMAN_DEFAULT_POSTGRES_DSN`.
- The command prints a plan, with the DSN password hidden, and asks for confirmation. For `--to postgres` the plan also names the state of each target schema: `ABSENT` (created), `EMPTY` (recreated), or `POPULATED` (dropped with `CASCADE` and recreated). `--to sqlite` refuses a store whose SQLite directory already exists.
- A store already on the target backend is skipped with a message.
- A store whose name is not a valid Postgres identifier is refused, or skipped under `--all`, and the message names a fix, such as a portable name to create and migrate.
- To reverse the change, migrate in the other direction. `memman doctor` checks the result: its `stale_post_migrate_source` check warns when SQLite files remain in a store that routes to Postgres.

---

<a id="observability"></a>

## Status and logs

```bash
memman status                            # statistics for the selected store (JSON)
memman doctor [--text]                   # health checks (JSON, or colored text)
memman log list                          # operation log (JSON, last 20)
memman log list --limit 50               # more entries
memman log list --since 7d               # entries from the last 7 days
memman log list --since 7d --stats       # counts by operation
memman log list --text                   # text table
memman log worker [--errors] [--lines N]
memman log worker --stack [--lines N]
```

**`status`** prints the store name, its backend, the backends in use, counts of current, superseded, and forgotten memories, `stale_insights` (the count `graph rebuild --stale-only` would process), the oplog size, counts by category, the top entities, and the storage path.

**`doctor`** exits 1 when any check fails and 0 otherwise. It makes one live LLM call and one live embedding call.

| Group              | Checks                                                                                                                                                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store              | `integrity`, `schema_columns`, `enrichment_coverage`, `oplog_delta_coverage`, `supersession_integrity`, `partial_index_predicates`, `embedding_consistency`, `embed_fingerprint`, `no_stale_swap_meta`, `provenance_drift` |
| Queue and schedule | `queue_schema`, `queue_backlog`, `scheduler_heartbeat`, `drain_heartbeat`, `scheduler_state`                                                                                                                               |
| Configuration      | `env_completeness`, `per_store_keys`, `env_permissions`, `stale_post_migrate_source`, `claude_hooks`, `optional_extras`                                                                                                    |
| Providers          | `llm_probe`, `embed_probe`                                                                                                                                                                                                 |

A store with no memories skips `integrity`, `enrichment_coverage`, `oplog_delta_coverage`, `embedding_consistency`, and `provenance_drift`.

**`log list`** prints the operation log as JSON, 20 entries by default. `--since` takes a count and a unit: `7d`, `24h`, or `30m`. `--stats` groups the entries by operation. `--text` prints a table.

**`log worker`** prints the last 50 lines (`--lines N`) of a worker log:

| Flag       | File                                                |
| ---------- | --------------------------------------------------- |
| (none)     | `~/.memman/logs/enrich.log`, the worker's stdout    |
| `--errors` | `~/.memman/logs/enrich.err`, the worker's stderr    |
| `--stack`  | `<data dir>/logs/memman.log` and its rotated copies |

`memman.log` holds the tracebacks that a one-line error leaves out. `--stack` and `--errors` cannot be combined. `enrich.log` and `enrich.err` stay under `~/.memman/logs` regardless of `--data-dir`.

---

## Scheduler

The scheduler starts a drain of the write queue every 60 seconds by default. It is a systemd timer on Linux, a launchd agent on macOS, or a `memman scheduler serve` process. Its state is `started` or `stopped`, kept in `~/.memman/scheduler.state`.

```bash
memman scheduler status [--text]      # platform, state, interval, next run, log paths, last drain
memman scheduler start [--text]       # accept writes and resume drains
memman scheduler stop [--text]        # refuse writes and pause drains
memman scheduler trigger              # start a drain now and return at once
memman scheduler interval [--seconds N]
memman scheduler install [--interval N] [--llm-endpoint URL] [--embed-provider NAME]
memman scheduler uninstall
memman scheduler serve [--interval N] [--once]
memman scheduler debug on|off|status
```

**Recall remains available while the scheduler is stopped.** `remember`, `replace`, `forget`, `supersede`, and `unsupersede` exit with status 1 and report that writes are disabled. The error names `memman scheduler start`, which enables writes.

`scheduler trigger` refuses in the same way. A running drain finishes the current memory before stopping, and a `serve` process exits. Three commands require a stopped scheduler: `graph rebuild`, `embed swap`, and `embed reembed`.

- **`trigger`** asks systemd or launchd to start a drain and returns `dispatched` without waiting. `memman log worker` shows the outcome. In serve mode `trigger` refuses, and `memman scheduler serve --once` runs one drain.
- **`interval`** prints the interval, or sets it with `--seconds N`. systemd and launchd need at least 60 seconds. In serve mode the command only records the value. The serve loop takes its interval from `--interval`, then `MEMMAN_INTERVAL`, so a new value applies only when `memman scheduler serve` restarts with `--interval N`. In serve mode an interval of 0 drains without pause.
- **`install`** installs only the scheduler unit, with no Claude Code integration. It fills every missing setting in the env file the same way `memman install` does, and refuses a flag that conflicts with the file. `--interval` defaults to 60 and must be at least 60.
- **`uninstall`** removes the scheduler unit, clears `scheduler.state` and `debug.state`, and strips the secret keys from the env file. It leaves the Claude Code integration in place.
- **`serve`** runs drains in a loop as a long-lived process, such as a container's main process. Hosts without systemd or launchd set `MEMMAN_SCHEDULER_KIND=serve`. The interval comes from `--interval`, then `MEMMAN_INTERVAL` in the env file, then 60. `--once` runs one drain and exits. On SIGTERM or SIGINT the drain stops after the memory in hand, and the process exits 0.
- **`debug on`** writes `~/.memman/debug.state`, and later drains write a trace to `~/.memman/logs/debug.log` at mode 0600. The trace holds raw LLM requests and responses, including memory content. `debug off` stops the trace and keeps the file. `MEMMAN_DEBUG` in the environment overrides the state file.

### Queue

```bash
memman scheduler queue list [--limit N]    # status counts and recent writes (default 50)
memman scheduler queue failed [--limit N]  # failed writes (default 50)
memman scheduler queue show <row_id>       # one write in full
memman scheduler queue retry <row_id>      # return one failed write to pending
memman scheduler queue retry --all-stale   # return every stale write to pending
memman scheduler queue purge --done        # delete done writes
memman scheduler queue purge --stale       # delete stale writes
```

`memman scheduler queue` with no subcommand runs `queue list`. Each queued write has one status:

| Status    | Meaning                                                                                                                                                                                |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pending` | Waiting for a drain, or claimed by one. A claim older than 600 seconds is taken over by the next drain.                                                                                |
| `done`    | Stored. The worker deletes done writes about a minute after the drain.                                                                                                                 |
| `failed`  | Five attempts failed. The waits between attempts are 60, 120, 240, and 480 seconds. The text stays in the queue, and no automatic step deletes it. `queue retry <row_id>` requeues it. |
| `stale`   | A pending write never attempted and more than 7 days old when `scheduler start` runs or a `serve` process starts. After each drain, the worker returns every stale write to pending.   |

---

## Backup

`memman backup` writes snapshots of every store to a directory outside `~/.memman`, such as a Dropbox folder, on a cron schedule. It rotates old bundles and rebuilds a working setup after the loss of `~/.memman`. The target belongs outside `~/.memman`, because losing the host could also lose that directory. memman does not check the target path. Snapshots run while the worker keeps draining: SQLite through the `sqlite3` online backup API and Postgres through `pg_dump -Fc`.

```bash
memman backup run [TARGET]                          # one bundle now (TARGET or MEMMAN_BACKUP_TARGET)
memman backup schedule '<cron>' TARGET [--keep N]   # install a scheduled backup
memman backup unschedule                            # remove the schedule and keep its settings
memman backup list [TARGET]                         # bundles at TARGET, read from their manifests
memman backup status                                # cron, target, keep, last run, next run, latest bundle
memman backup restore BUNDLE [--yes]                # rebuild stores and settings from a bundle
```

`memman backup` with no subcommand runs `backup status`.

**Schedule.** The cron string has five fields (`min hour dom month dow`) in local time. `backup schedule` writes `MEMMAN_BACKUP_CRON`, `MEMMAN_BACKUP_TARGET`, and `MEMMAN_BACKUP_KEEP` (default 7) to the env file, creates the target directory, and installs a systemd timer (`OnCalendar=` with `Persistent=true`, so a run missed during sleep happens at wake) or a launchd agent (`StartCalendarInterval`). In serve mode the serve loop matches the cron itself. systemd treats a schedule that restricts both day of month and day of week as AND, where cron uses OR, and `backup schedule` warns about it. Each run keeps the newest `MEMMAN_BACKUP_KEEP` bundles.

**Bundle.** Each bundle is one `.tar.gz` plus a `<bundle>.manifest.json` beside it, which `backup list` reads. A store whose snapshot fails is marked `failed` in the manifest, and the rest of the bundle completes. The bundle holds `queue.db`, copied before the stores, so a write still waiting for its drain survives and drains on the restored host. The manifest records the pending count as `queue_pending`. In serve mode the loop drains the queue before the snapshot.

**Excluded settings.** The bundle's `env.nonsecret` member leaves out the four API keys, `MEMMAN_DEFAULT_POSTGRES_DSN`, every `MEMMAN_POSTGRES_DSN_<store>`, and the host's own `MEMMAN_BACKUP_*` keys. It keeps per-store backend keys and model and provider settings.

**Restore.**

1. Refuses a bundle with Postgres stores when `pg_restore` is not on `PATH`.
2. Asks first unless `--yes` is given, then holds the drain lock and refuses a bundle whose format version differs from this memman's.
3. Merges the non-secret settings into the env file, so per-store backend keys are in place.
4. Restores each store using the `backend` in its manifest entry: a file copy for SQLite, `pg_restore` for Postgres with the DSN configured on this host.
5. Restores `queue.db` and the active-store file.

The reply lists `restored`, `failed`, `pg_restore_skipped` (Postgres stores with no DSN here), `embed_mismatch` (stores whose fingerprint differs from this host's embedding settings), `queue_restored`, `active_store`, and `secret_keys_needed` (secret keys missing from this host's env file).

```bash
memman backup schedule '0 3 * * *' ~/Dropbox/code/archive/
memman backup status
memman backup restore ~/Dropbox/code/archive/memman-backup-<host>-<stamp>.tar.gz --yes
```

`memman uninstall` removes the backup timer or agent and keeps the `MEMMAN_BACKUP_*` keys, so `memman backup schedule` can reinstall it.

---

## Configuration

**The env file.** memman reads every installed setting from `$MEMMAN_DATA_DIR/env` (default `~/.memman/env`), a `KEY=VALUE` file at mode 0600. The `--data-dir` flag moves the stores, the queue, and `memman.log`, but settings are still read from this file. Blank lines and `#` comments are skipped, one pair of surrounding quotes is stripped, and `${VAR}` is not expanded. At run time memman ignores the shell for these keys, so a stale export cannot override the file. Each command reads the file when it starts, and a `serve` process reads it again before each drain.

**Install precedence.** `memman install` sets each key from the first source that has a value:

1. the value already in the env file, which install never replaces,
2. an install flag or a wizard answer,
3. the shell: the `MEMMAN_` name, then the vendor name for three keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, and `OPENAI_API_KEY` for `MEMMAN_OPENAI_EMBED_API_KEY`),
4. the included default (`INSTALL_DEFAULTS` in `src/memman/config.py`).

**Process-control variables.** `MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_AUTHOR`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`, and `MEMMAN_WORKER` are never written to the env file. The component that uses each one reads it from the process environment. `MEMMAN_AUTHOR` names who issues a write and falls back to the login name. memman records it on the queued write so the drain preserves the identity of the user who submitted it.

**Reading and changing settings.**

```bash
memman config show                       # every known variable (secrets redacted), per-store keys, scheduler state
memman config get KEY                    # one value, or exit 1 when unset
memman config set KEY VALUE              # write one value
memman config set-pg-dsn --default       # prompt for a DSN, write MEMMAN_DEFAULT_POSTGRES_DSN
memman config set-pg-dsn --store work    # prompt for a DSN, write MEMMAN_POSTGRES_DSN_work
```

- `config get` redacts API keys and the password in a DSN.
- `config set` accepts every installable key and the three per-store forms (`MEMMAN_BACKEND_<store>`, `MEMMAN_POSTGRES_DSN_<store>`, `MEMMAN_RERANK_ENABLED_<store>`). It refuses the bare names `MEMMAN_BACKEND` and `MEMMAN_POSTGRES_DSN` and names the key to use instead.

### Backend selection

A store's backend is `MEMMAN_BACKEND_<store>`, then `MEMMAN_DEFAULT_BACKEND`, then `sqlite`. The first drain that writes to a store records `MEMMAN_BACKEND_<store>` from the default, together with `MEMMAN_POSTGRES_DSN_<store>` from `MEMMAN_DEFAULT_POSTGRES_DSN` when the default is `postgres`. A later change to `MEMMAN_DEFAULT_BACKEND` therefore moves no store that has been written to. `memman migrate` moves a store and its data.

A Postgres store lives in the schema `store_<name>`. Its DSN is `MEMMAN_POSTGRES_DSN_<store>`, then `MEMMAN_DEFAULT_POSTGRES_DSN`. The write queue is always SQLite, at `<data dir>/queue.db`.

### Postgres DSN

A DSN is a libpq connection URI: `postgresql://[user[:password]@][host][:port]/[dbname][?param=value&...]`.

`memman config set-pg-dsn` prompts for host (default `localhost`), port (default `5432`), user, password (hidden), and database name (default `memman`). It URL-encodes the user and password and writes the URI under the key its flag names. Exactly one of `--default` and `--store NAME` is required. An empty password produces a DSN without one, and libpq then reads the password from `~/.pgpass`, `PGSERVICE`, or `PGPASSWORD`. The command does not test the connection. `memman doctor` and `memman migrate --dry-run` do.

| Scenario     | DSN                                                      | Notes                                                    |
| ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| Local        | `postgresql://memman@localhost/memman`                   | No password                                              |
| Inline       | `postgresql://memman:s3cret@db.internal:5432/memman`     | URL-encode `:`, `@`, and `/` in the password             |
| `~/.pgpass`  | `postgresql://memman@db.internal:5432/memman`            | No password in the URI. The safer choice on shared hosts |
| TLS required | `postgresql://memman@db.internal/memman?sslmode=require` | Any libpq parameter works, such as `application_name`    |

> **Security.** memman stores every DSN in plain text in the env file at mode 0600. Root and any process running as the same user can read it.

<a id="runtime-tunables"></a>

### Runtime settings

These variables are not installable. The component that uses each one reads it from the process environment, and `memman config show` lists it.

| Variable                          | Default        | Description                                                                                                                           |
| --------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_REINDEX_TIMEOUT`          | `180`          | Seconds allowed for the Postgres HNSW index build when a store opens. A build that times out is dropped and retried on the next open. |
| `MEMMAN_EMBED_SWAP_BATCH_SIZE`    | `200`          | Memories per batch in `memman embed swap`.                                                                                            |
| `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` | `0` (no limit) | Seconds allowed for the Postgres HNSW index build at the start of `memman embed swap`.                                                |

### Variable reference

[CONTRIBUTING.md](../CONTRIBUTING.md#variable-reference) lists every variable with its type, default, and purpose.

---

## Architecture

[Chapter 3](design/03-pipelines.md) describes the write and recall pipelines.
