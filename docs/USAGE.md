# memman usage and reference

This guide covers setup, memory commands, and maintenance. The [README](../README.md) is the short introduction, and the [design guide](DESIGN.md) explains the implementation.

## Find a task

| Task                                    | Section                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| Install or change providers             | [Installation](#install-and-uninstall), [provider setup](#provider-setup)      |
| Save, find, correct, or forget a memory | [Memory commands](#memory-commands)                                            |
| Inspect memory history or quality       | [Insights](#insights)                                                          |
| Separate projects or change databases   | [Store management](#store-management)                                          |
| Check health or investigate a failure   | [Status and logs](#status-and-logs), [troubleshooting](#troubleshooting)       |
| Control the worker or retry writes      | [Scheduler](#scheduler), [queue](#queue)                                       |
| Rebuild summaries or vectors            | [Re-enrichment](#re-enrichment), [embedding operations](#embedding-operations) |
| Back up or restore stores               | [Backup](#backup)                                                              |
| Read or change settings                 | [Configuration](#configuration)                                                |

Examples use `<id>` and `<name>` as placeholders. Square brackets in a synopsis mark an optional argument.

## Global flags

A global flag goes before the subcommand: `memman --store work recall "retry cap"`. The `--store` option of `memman migrate` and `memman config set-pg-dsn` is a separate subcommand option with its own meaning.

| Flag                | Default     | Description                                                                                                                                                        |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--store <name>`    | none        | Store to use. Takes precedence over `MEMMAN_STORE` and the active-store file (see [Store management](#store-management)).                                          |
| `--data-dir <path>` | `~/.memman` | Data directory for the stores, the queue, `memman.log`, and the env file. Falls back to `MEMMAN_DATA_DIR`, then `~/.memman` (see [Configuration](#configuration)). |
| `--verbose` / `-v`  | off         | INFO-level logging to stderr.                                                                                                                                      |
| `--debug`           | off         | DEBUG-level logging to stderr. Takes precedence over `--verbose`.                                                                                                  |
| `--version`         |             | Print the version and exit.                                                                                                                                        |

Without `--verbose` or `--debug`, the stderr level is `MEMMAN_LOG_LEVEL` (default `WARNING`).

---

## Install and uninstall

`pipx install memman` installs the package, and `memman install` configures it. `pipx install 'memman[postgres]'` adds Postgres support. memman needs Python 3.11+.

The default setup uses OpenRouter for summaries and Voyage for embeddings and reranking, so the wizard asks for both keys. [Provider setup](#provider-setup) lists the alternatives.

```bash
memman install                        # interactive wizard in a terminal
memman install --claude-code          # install into ~/.claude even when Claude Code is not detected
memman install --no-wizard --backend postgres --pg-dsn postgresql://memman@localhost/memman
memman uninstall
memman uninstall --claude-code
```

**Install flags:**

| Flag                    | Effect                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--claude-code`         | Install into `~/.claude`, whether or not Claude Code is detected.                                                                                                         |
| `--backend NAME`        | Default storage backend, `sqlite` or `postgres`. Skips the backend prompt.                                                                                                |
| `--pg-dsn URL`          | Postgres DSN. Install connects, checks for `pgvector`, and stops when either fails. Required with `--backend postgres` when no prompt runs and the env file holds no DSN. |
| `--llm-endpoint URL`    | LLM endpoint URL. Skips the endpoint prompt. Must start with `http://` or `https://`.                                                                                     |
| `--embed-provider NAME` | Embedding provider: `voyage`, `openai`, or `openrouter`. Skips the provider prompt.                                                                                       |
| `--no-wizard`           | Skip all prompts, even in a terminal. Flags, the shell, and defaults supply every value.                                                                                  |

### Installed files and services

Installation writes settings, configures a worker, and adds the Claude Code integration when detected or requested.

| Path                                                   | Purpose                                         |
| ------------------------------------------------------ | ----------------------------------------------- |
| `~/.memman/env`                                        | Settings and API keys, mode 0600.               |
| `~/.claude/skills/memman/SKILL.md`                     | Symlink to the packaged agent manual.           |
| `~/.claude/hooks/memman/*.sh`                          | Symlinks to five lifecycle hooks.               |
| `~/.claude/settings.json`                              | Hook registrations and allowed memory commands. |
| `~/.config/systemd/user/memman-enrich.{timer,service}` | Linux worker schedule.                          |
| `~/Library/LaunchAgents/com.memman.enrich.plist`       | macOS worker schedule.                          |
| `~/.memman/logs/`                                      | Worker output.                                  |

Claude Code is detected by a `claude` binary on `PATH` or an existing `~/.claude` directory. Without either, installation configures only the scheduler unless `--claude-code` is passed. Start a new Claude Code session to load the hooks.

On a host without systemd or launchd, set `MEMMAN_SCHEDULER_KIND=serve` in the process environment and run `memman scheduler serve` to process writes.

Existing env-file values take precedence. Installation refuses conflicting flags and prints the `config set` command needed to change the value. It also checks prerequisites and, for OpenRouter, model availability. A catalog outage is reported but does not stop installation.

Run `memman install` after a package upgrade to refresh hook registrations, scheduler units, and new default settings. The [integration chapter](design/05-integration.md) describes the installed hooks and permission entries.

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

`memman uninstall` (with an optional `--claude-code`) removes:

- the scheduled backup timer or agent,
- `~/.claude/hooks/memman/` and `~/.claude/skills/memman/`,
- the memman hooks and permission entries in `~/.claude/settings.json`,
- the scheduler unit, `~/.memman/scheduler.state`, and `~/.memman/debug.state`,
- the secret keys in the env file: `MEMMAN_LLM_API_KEY`, `MEMMAN_OPENROUTER_API_KEY`, `MEMMAN_VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, and `MEMMAN_DEFAULT_POSTGRES_DSN`.

It keeps every other env-file setting, including `MEMMAN_POSTGRES_DSN_<store>`, and it keeps the stores, the queue, and the logs. When the Claude Code cleanup reports an error, uninstall stops and leaves the scheduler in place.

---

## Provider setup

memman uses separate services for summaries, embeddings, and reranking. The defaults below are the models this version ships. The [README cost table](../README.md#cost) estimates their combined cost.

### LLM providers

The enrichment client uses the OpenAI-compatible `/chat/completions` protocol. Set the endpoint, API key, and model together; model IDs are specific to the endpoint.

| Endpoint                | `MEMMAN_LLM_ENDPOINT`          | Model selection                                               |
| ----------------------- | ------------------------------ | ------------------------------------------------------------- |
| OpenRouter              | `https://openrouter.ai/api/v1` | Defaults to `qwen/qwen3-235b-a22b-2507`.                      |
| OpenAI                  | `https://api.openai.com/v1`    | Set a supported model explicitly.                             |
| Ollama                  | `http://localhost:11434/v1`    | Set a locally available model; the wizard allows a blank key. |
| Other compatible server | Its chat API base URL          | Set the model and authentication required by that server.     |

```bash
memman config set MEMMAN_LLM_ENDPOINT https://api.openai.com/v1
memman config set MEMMAN_LLM_API_KEY <api-key>
memman config set MEMMAN_LLM_MODEL <model-id>
```

On OpenRouter, installation can copy `MEMMAN_OPENROUTER_API_KEY` into `MEMMAN_LLM_API_KEY` when the latter is unset. Requests use the configured provider restrictions and zero-data-retention setting. The [routing reference](design/03-pipelines.md#llm-routing) documents those defaults and the daily availability check.

### Embedding providers

| Provider     | Shipped model            | Key                           | Endpoint setting                                                      |
| ------------ | ------------------------ | ----------------------------- | --------------------------------------------------------------------- |
| `voyage`     | `voyage-3-lite`          | `MEMMAN_VOYAGE_API_KEY`       | Fixed Voyage endpoint.                                                |
| `openai`     | `text-embedding-3-small` | `MEMMAN_OPENAI_EMBED_API_KEY` | `MEMMAN_OPENAI_EMBED_ENDPOINT`, default `https://api.openai.com`.     |
| `openrouter` | `baai/bge-m3`            | `MEMMAN_OPENROUTER_API_KEY`   | `MEMMAN_OPENROUTER_ENDPOINT`, default `https://openrouter.ai/api/v1`. |
| `ollama`     | `nomic-embed-text`       | None                          | `MEMMAN_OLLAMA_HOST`, default `http://localhost:11434`.               |

The wizard offers the first three. Configure Ollama afterward with `memman config set MEMMAN_EMBED_PROVIDER ollama`. Each provider has a model setting: `MEMMAN_VOYAGE_EMBED_MODEL`, `MEMMAN_OPENAI_EMBED_MODEL`, `MEMMAN_OPENROUTER_EMBED_MODEL`, or `MEMMAN_OLLAMA_EMBED_MODEL`.

Existing stores keep their recorded model until an explicit [swap or re-embed](#embedding-operations).

### Reranker

Voyage is the only reranker. It uses its own model setting and needs a Voyage key even when embeddings use another provider.

| Setting                         | Default         | Purpose                     |
| ------------------------------- | --------------- | --------------------------- |
| `MEMMAN_RERANK_ENABLED`         | `true`          | Enable reranking globally.  |
| `MEMMAN_RERANK_ENABLED_<store>` | Global value    | Override for one store.     |
| `MEMMAN_VOYAGE_RERANK_MODEL`    | `rerank-3-lite` | Select the reranking model. |
| `MEMMAN_VOYAGE_API_KEY`         | None            | Authenticate requests.      |

```bash
memman config set MEMMAN_RERANK_ENABLED false
```

### Where keys are needed

At runtime, memman reads provider settings from `<data dir>/env` and ignores the shell. `config set` changes them. Installation can import keys from the shell ([configuration precedence](#configuration)).

| Operation                                                                        | Credentials                                             | If unavailable                                                     |
| -------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| Queue `remember`                                                                 | No model key required.                                  | Model work waits for the worker.                                   |
| Open a normal store session, including `replace`, `forget`, and `recall --basic` | Global embedding provider's key where required.         | The Voyage and OpenRouter clients stop the command.                |
| Embed recall query                                                               | Store fingerprint's provider key.                       | Recall warns and falls back to keyword and recency ranking.        |
| Embed a queued memory                                                            | Store fingerprint's provider key.                       | Missing bound credentials fail the write; inspect the queue error. |
| Generate a summary                                                               | `MEMMAN_LLM_API_KEY`, unless the endpoint permits none. | A rejected request leaves the memory without a summary.            |
| Rerank recall candidates                                                         | `MEMMAN_VOYAGE_API_KEY`.                                | Recall warns and preserves its pre-rerank order.                   |

A normal store session builds the global embedding client as well as the store-bound one, so a command can need the key of a global provider that no store uses. Opening a new store, or one whose model's vector size is not built in, also sends a probe embedding.

`remember` uses a separate related-memory lookup that calls no model. Diagnostics and operations such as `embed status`, `embed swap`, `migrate`, and `backup` bypass normal fingerprint initialization. The `openai` client starts without a key, and its first request fails instead. `doctor` checks provider configuration and connectivity.

---

## Memory commands

```bash
memman remember "The retry cap stays at three, since a fourth try only adds load." \
  --cat decision
memman recall "retry cap" --limit 10
memman recall "auth" --basic
memman replace <id> "The retry cap is four for batch jobs and three elsewhere."
memman forget <id>
```

Every command that takes a memory id also accepts an unambiguous prefix of one, such as the 8-character id that recall prints. An ambiguous prefix is refused, and the error names how many ids it matches.

`remember`, `replace`, and `forget` refuse to run while the scheduler is stopped (see [Scheduler](#scheduler)).

### remember and replace

`remember` queues a new memory. The worker stores it on a later drain; only then can recall find it. Submission returns one line of JSON:

| Field              | Meaning                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `action`           | `queued`: submission succeeded, processing is still pending.                   |
| `id`               | Persistent memory UUID, assigned now.                                          |
| `queue_id`         | Temporary numeric queue entry ID, used by queue commands.                      |
| `store`            | Destination store.                                                             |
| `quality_warnings` | Advisory warnings about potentially temporary information.                     |
| `related`          | Up to three current memories with overlapping wording. Present for `remember`. |
| `related_error`    | Replaces `related` if that lookup failed. The write remains queued.            |
| `replaced_id`      | Target memory ID. Present for `replace`.                                       |

`insights show <id>` reports when a write is still queued. Use `scheduler queue show <queue_id>` to inspect its processing state.

`related` lists stored claims the new text may correct. The lookup favors focused word overlap, reads only memories within the 1,000-byte input limit, and calls no model. It can return an empty list, and its failure never undoes the submission.

| Flag    | `remember` default | `replace` default | Values                                                 |
| ------- | ------------------ | ----------------- | ------------------------------------------------------ |
| `--cat` | `fact`             | Target's category | `preference`, `decision`, `fact`, `insight`, `context` |

`replace <id> "<text>"` checks its target in the store, queues a successor, and preserves the old memory's history. Unlike `remember`, it runs normal [store-opening checks](#where-keys-are-needed), which can require credentials or a probe embedding. Once the worker commits it, recall returns the new version and excludes the old one.

| Target state                               | Behavior                                                                                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Current memory                             | Queue a replacement.                                                                                                                                               |
| Write still queued in this store           | Queue the successor and wait for the target to finish processing.                                                                                                  |
| Forgotten or already replaced              | Refuse; an already-replaced error names the successor.                                                                                                             |
| Replacement already queued                 | Refuse and quote the queued replacement. The fix replaces that queued id with text that keeps both corrections, since a second replacement would retire the first. |
| Target unavailable when the worker commits | Save the new memory without a replacement link; the worker result reports `target_gone`.                                                                           |

Replacements are processed in queue order within a store. To correct a previous correction, replace its successor. `insights show <id> --history` displays the chain.

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

They also refuse an unknown category. `replace` checks an inherited category the same way.

### recall

| Flag      | Default | Description                                                |
| --------- | ------- | ---------------------------------------------------------- |
| `--limit` | `20`    | Maximum lines printed.                                     |
| `--basic` | off     | SQL `LIKE` matching with no ranking. Lines carry no score. |

Recall prints one line per memory, best first, and prints nothing for an empty result:

```text
<id8> <score> <created_at> <author> <category> | <text>
```

`id8` is the first 8 characters of the id, and `score` has two decimals. `author` is `-` when unset. `text` is the summary when the memory has one, and the start of the content otherwise. `memman insights show <id>` prints the whole memory.

A score ranks a row against its siblings in one response and means nothing across queries. Recency can surface a memory with no keyword or semantic match, so each returned text needs its own relevance check. [Chapter 3](design/03-pipelines.md#34-read-pipeline-recall) explains the ranking.

`--limit 0` means unlimited results on the scored path but no results with `--basic`. A scored limit over 100 may include both reranked and remaining candidates; their scores use different scales.

`--basic` requires every query word to appear as a substring of the content and returns newest memories first. It skips query embedding and reranking. Store-opening checks still run, so it can require credentials or an embedding probe; see [where keys are needed](#where-keys-are-needed).

**Rerank.** For a query of more than two words, a cross-encoder re-scores the top 100 candidates. The reranker is Voyage (model `MEMMAN_VOYAGE_RERANK_MODEL`, default `rerank-3-lite`), and it needs `MEMMAN_VOYAGE_API_KEY`. When the rerank call fails, recall logs a warning and keeps the blended order. `MEMMAN_RERANK_ENABLED` (default `true`) enables or disables reranking for every store, and `MEMMAN_RERANK_ENABLED_<store>` overrides it for one store:

```bash
memman config set MEMMAN_RERANK_ENABLED_work false
```

### forget

`forget <id>` sets `deleted_at` and excludes the memory from recall. The row stays stored, and no command reverses the action.

A replaced memory can be forgotten. A current memory whose predecessor is not forgotten cannot, and the error names `replace`, which keeps the correction chain intact. Queued writes cannot be forgotten.

Nothing expires on its own. `insights review` flags memories that may need updating or removal.

---

## Insights

```bash
memman insights show <id>              # one memory as JSON, including replaced memories
memman insights show <id> --history    # the replacement chain through <id>, oldest first
memman insights review [--limit N]     # memories with quality warnings
```

- `show` accepts a forgotten memory ID only with `--history`. It then lists every memory in the chain with its `state`: `current`, `replaced`, or `forgotten`. A forgotten entry omits the content. An ID still in the write queue reports that processing is pending.
- `review` checks current memories, newest first, against the same patterns as `quality_warnings`, and stops after `--limit` flagged memories (default 20).

---

## Re-enrichment

`memman enrich` re-runs enrichment (summary) and the embedding for current memories.

```bash
memman enrich --dry-run     # print the count and change nothing, scheduler running
memman scheduler stop
memman enrich --stale-only  # rebuild outdated or incomplete generated fields
memman scheduler start
```

- Without `--stale-only`, `enrich` processes every current memory. Both modes support SQLite and Postgres and require a stopped scheduler, except with `--dry-run`.
- The command works in batches of 20 and prints `{processed, remaining}`. `remaining` counts memories still waiting for enrichment after the run.
- `--stale-only` selects outdated prompt/model versions and incomplete enrichment. It skips an enriched memory with no version marker. `status` reports the same selection count as `stale_insights`.
- `--progress-jsonl` writes one JSON progress line per memory to stderr.
- A second rebuild on the same store is refused while one runs.

---

## Embedding operations

Each store records its embedding provider, model, and vector dimension in a **fingerprint**. Recall and the worker use that recorded model. Changing `MEMMAN_EMBED_PROVIDER` alone does not convert existing vectors. [Credential requirements](#where-keys-are-needed) and the [embedding design](design/04-lifecycle.md#43-embedding-support) give the detail.

| Command                                   | Scope                             | Purpose                                                              |
| ----------------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| `embed status`                            | Selected store                    | Show fingerprint, credentials, and swap progress.                    |
| `embed swap --to MODEL [--provider NAME]` | Selected SQLite or Postgres store | Change model while recall continues using old vectors until cutover. |
| `embed swap --resume`                     | Selected store                    | Continue an interrupted swap.                                        |
| `embed swap --abort`                      | Selected store                    | Discard pending vectors and swap state, before cutover only.         |
| `embed reembed [--dry-run]`               | All SQLite stores                 | Rewrite vectors using global provider settings.                      |

To change one store's model after configuring the target provider's credentials:

```bash
memman scheduler stop
memman --store work embed swap --to text-embedding-3-small --provider openai
memman scheduler start
```

After a failed swap, `embed status` shows its state, and `--resume` or `--abort` settles it before writes restart.

**`embed swap`** moves one store (the store the global flags select) to a new model.

- It needs a stopped scheduler, except with `--abort`. Recall keeps reading the old vectors while the swap writes new ones into the `embedding_pending` column in batches of `MEMMAN_EMBED_SWAP_BATCH_SIZE` (default 200).
- On Postgres, the swap first builds an HNSW index on the new column.
- The final switch, called cutover, replaces the old vectors in one transaction. Returning to the old model requires another full swap.
- One swap or abort runs per store at a time. A second one, from any shell, refuses while the first holds the store's swap lock.
- `--resume` continues an interrupted swap from its recorded cursor. `--abort` discards pending vectors and swap state. Once a swap reaches cutover, `--abort` refuses and `--resume` finishes it, because the cutover may already have committed.
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

Named stores separate projects. The scheduler drains only the queue in its configured data directory, so a separate `MEMMAN_DATA_DIR` per project creates queues that no scheduler drains.

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
- The command prints a plan, with the DSN password hidden, and asks for confirmation. For `--to postgres` the plan also labels each target schema `[will create]`, `[EMPTY, will recreate]`, or `[POPULATED, will DROP CASCADE and recreate]`. `--to sqlite` refuses a store whose SQLite directory already exists.
- A store already on the target backend is skipped with a message.
- A store with an embedding swap in flight is refused. The message names the fix: `memman --store <store> embed swap --resume` finishes the swap, and before cutover `--abort` discards it.
- A store whose name is not a valid Postgres identifier is refused, or skipped under `--all`, and the message names a fix, such as a portable name to create and migrate.
- To reverse the change, migrate in the other direction. `memman doctor` checks the result: its `stale_post_migrate_source` check warns when SQLite files remain in a store that routes to Postgres.

---

## Status and logs

```bash
memman status                            # statistics for the selected store (JSON)
memman doctor [--text]                   # health checks (JSON, or colored text)
memman log list                          # operation log (JSON, last 20)
memman log list --limit 50               # more entries
memman log list --since 7d               # entries from the last 7 days
memman log list --since 7d --stats       # counts by operation
memman log list --text                   # text table
memman log calls [--since 7d]            # agent-verb calls per date and verb
memman log worker [--errors] [--lines N]
memman log worker --stack [--lines N]
```

**`status`** prints the store name, its backend, the backends in use, counts of current, replaced, and forgotten memories, `stale_insights` (the count `enrich --stale-only` would process), the oplog size, counts by category, and the storage path.

**`doctor`** exits 1 when any check fails and 0 otherwise. It makes one live LLM call and two to four live embedding calls: `embed_probe` sends an availability probe and a test embed, and `embed_fingerprint` sends an availability probe for the store's recorded model, plus a size probe when that model's vector size is not built in.

| Group              | Checks                                                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store              | `integrity`, `enrichment_coverage`, `replacement_integrity`, `embedding_consistency`, `embed_fingerprint`, `no_stale_swap_meta`, `provenance_drift` |
| Queue and schedule | `queue_backlog`, `scheduler_heartbeat`, `drain_heartbeat`, `scheduler_state`                                                                        |
| Configuration      | `env_completeness`, `per_store_keys`, `env_permissions`, `stale_post_migrate_source`, `claude_hooks`, `optional_extras`                             |
| Providers          | `llm_probe`, `embed_probe`                                                                                                                          |

A store with no memories skips `integrity`, `enrichment_coverage`, `embedding_consistency`, and `provenance_drift`.

`enrichment_coverage` warns on any stranded memory, reports the count as `stranded`, and names `memman enrich --stale-only` as the fix.

**`log list`** prints the operation log as JSON, 20 entries by default. `--since` takes a count and a unit: `7d`, `24h`, or `30m`. `--stats` groups the entries by operation. `--text` prints a table.

### Agent call log

`log calls` reports call counts by UTC date and command as JSON. It covers `recall`, `remember`, `replace`, `forget`, `insights show`, `insights review`, `status`, and `doctor`. `--since` accepts the same time windows as `log list`.

Each call appends a record to `<data dir>/logs/calls.log`:

```text
<UTC start>|<verb>|<store>|<exit code>|<ms>
```

Arguments and memory text are excluded. Invalid store names become `?`. Commands rejected during argument parsing, hooks, and worker activity produce no entry. Malformed lines are counted under `meta.malformed` and excluded from command totals. The file is not automatically rotated or trimmed.

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

**Recall remains available while the scheduler is stopped.** `remember`, `replace`, and `forget` exit with status 1 and report that writes are disabled. The error names `memman scheduler start`, which enables writes.

`scheduler trigger` refuses in the same way. A running drain finishes the current memory before stopping, and a `serve` process exits. Three commands require a stopped scheduler: `enrich`, `embed swap`, and `embed reembed`.

| Command                       | Behavior                                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `trigger`                     | Ask systemd or launchd to start a drain; return `dispatched` immediately. Use `log worker` to inspect the result. In serve mode, use `serve --once` instead. |
| `interval`                    | Show the interval, or set it with `--seconds N`. systemd and launchd require at least 60 seconds.                                                            |
| `install`                     | Install only the scheduler, filling missing settings and refusing conflicting flags. `--interval` defaults to 60 and must be at least 60.                    |
| `uninstall`                   | Remove the scheduler unit and state files, and strip secret settings. Keep the Claude Code integration.                                                      |
| `serve`                       | Run drains in the foreground; `--once` processes one drain and exits.                                                                                        |
| `debug on` / `off` / `status` | Control or inspect worker tracing.                                                                                                                           |

**Serve mode.** Set `MEMMAN_SCHEDULER_KIND=serve` on hosts without systemd or launchd. The loop reads its interval from `--interval`, then the installed `MEMMAN_INTERVAL`, then 60. An interval of zero drains without pausing. In this mode `scheduler interval` only records the value, and a new interval takes effect when the loop restarts with `--interval N`. SIGTERM or SIGINT stops it after the current memory, with exit code zero.

**Debug output.** `debug on` updates `~/.memman/debug.state`. Later drains write `~/.memman/logs/debug.log` at mode 0600, including raw model requests, responses, and memory content. `debug off` stops tracing and retains the file. The process-environment variable `MEMMAN_DEBUG` overrides the state file.

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

| Status    | Meaning                                                                                                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pending` | Waiting for a drain, or claimed by one. A claim older than 600 seconds is taken over by the next drain.                                                                                           |
| `done`    | Stored. Later maintenance deletes completed entries older than 60 seconds.                                                                                                                        |
| `failed`  | Five attempts failed. The waits between attempts are 60, 120, 240, and 480 seconds. The text stays in the queue, and no automatic step deletes it. `queue retry <row_id>` requeues it.            |
| `stale`   | A pending write never attempted and more than 7 days old when `scheduler start` runs or a `serve` process starts. Drain maintenance returns stale entries to pending when its time budget allows. |

---

## Backup

`memman backup` snapshots all stores and the write queue, either on demand or on a schedule. A target belongs outside the source data directory, on storage that survives its loss. memman does not check the target location. Snapshots use SQLite's online backup API or Postgres `pg_dump -Fc` while the worker continues running.

```bash
memman backup run [TARGET]                          # one bundle now (TARGET or MEMMAN_BACKUP_TARGET)
memman backup schedule '<cron>' TARGET [--keep N]   # install a scheduled backup
memman backup unschedule                            # remove the schedule and keep its settings
memman backup list [TARGET]                         # bundles at TARGET, read from their manifests
memman backup status                                # cron, target, keep, last run, next run, latest bundle
memman backup restore BUNDLE [--yes]                # rebuild stores and settings from a bundle
```

`memman backup` with no subcommand runs `backup status`.

### Scheduling and bundles

The cron expression has five fields (`min hour dom month dow`) and uses local time. `backup schedule` saves its settings, creates the target directory, and configures the host scheduler. Each run keeps the newest `MEMMAN_BACKUP_KEEP` bundles, default 7.

| Scheduler  | Backup behavior                                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| systemd    | Persistent timer catches runs missed while asleep. Restricting both day-of-month and day-of-week means AND, where cron means OR, and memman warns. |
| launchd    | Uses `StartCalendarInterval`.                                                                                                                      |
| Serve loop | Matches the cron expression itself and drains writes before taking a snapshot.                                                                     |

A bundle consists of a `.tar.gz` archive and an adjacent manifest used by `backup list`. Store snapshot failures are listed in the manifest while other snapshots continue.

The queue is copied before the stores, preserving pending writes for the restored installation. The manifest records their count as `queue_pending`. Queue UUIDs prevent restored writes from duplicating memories already captured in a store snapshot.

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

### Settings file

Installed settings come from `<data dir>/env`, defaulting to `~/.memman/env`. Shell exports do not override those settings at runtime. Each command reads the file at startup; a serve process reloads it before each drain.

The file uses `KEY=VALUE` entries and mode 0600. Blank lines and comments are ignored, surrounding quotes are stripped, and `${VAR}` is not expanded. `--data-dir` moves the settings file together with the stores, queue, and worker log.

**Install precedence.** `memman install` sets each key from the first source that has a value:

1. the value already in the env file, which install never replaces,
2. an install flag or a wizard answer,
3. the shell: the `MEMMAN_` name, then the vendor name for three keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, and `OPENAI_API_KEY` for `MEMMAN_OPENAI_EMBED_API_KEY`),
4. the included default (`INSTALL_DEFAULTS` in `src/memman/config.py`).

### Process-control variables

These come from the process environment and are not installed in the settings file:

| Variable                | Purpose                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`       | Select the settings and data directory.                           |
| `MEMMAN_STORE`          | Select a store for this process.                                  |
| `MEMMAN_AUTHOR`         | Identify the caller submitting a memory; default to the OS login. |
| `MEMMAN_DEBUG`          | Override the debug state file.                                    |
| `MEMMAN_SCHEDULER_KIND` | Override scheduler detection, including serve mode.               |
| `MEMMAN_WORKER`         | Mark a worker process for logging.                                |

The queue preserves the author supplied at submission time.

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

## Troubleshooting

| Symptom                                      | Check                                           | Next step                                                                               |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| A new memory does not appear in recall       | `scheduler status`, then `scheduler queue list` | Wait for processing; inspect a failed entry and fix its reported error before retrying. |
| Writes report that the scheduler is stopped  | `scheduler status`                              | Run `scheduler start` when maintenance is finished.                                     |
| Recall warns about embedding or reranking    | `embed status`, then provider settings          | Restore the required key or endpoint; reranking can be disabled separately.             |
| Even `recall --basic` fails on a missing key | Global embedding provider settings              | Supply the key required during store opening.                                           |
| Doctor reports incomplete enrichment         | `doctor --text`                                 | Stop the scheduler, run `enrich --stale-only`, then restart it.                         |
| A model swap was interrupted                 | `embed status`                                  | Resume or abort the swap.                                                               |
| Claude Code has no memory reminders          | `doctor --text`                                 | Re-run `memman install` and start a new session.                                        |

Commands in this table take the `memman` prefix. `doctor` makes live provider probes, and the queue and worker logs show processing without them.
