# memman usage and reference

This guide covers setup, memory commands, and maintenance. The [README](../README.md) is the short introduction, and the [design guide](DESIGN.md) explains the implementation.

## Task index

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

A global flag goes before the subcommand: `memman --store work recall "retry cap"`. The agent verbs other than `store branch`, `store merge`, and `store drop` also take `--store` after the verb, `memman recall --store work "retry cap"`, with the same effect. memman refuses a command that gives both flags with different names. The `--store` option of `memman migrate` and `memman config set-pg-dsn` is a separate subcommand option with its own meaning.

| Flag                | Default     | Description                                                                                                                                                        |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--store <name>`    | none        | Store to use, before or after an agent verb. Takes precedence over `MEMMAN_STORE` and the active-store file (see [Store management](#store-management)).           |
| `--data-dir <path>` | `~/.memman` | Data directory for the stores, the queue, `memman.log`, and the env file. Falls back to `MEMMAN_DATA_DIR`, then `~/.memman` (see [Configuration](#configuration)). |
| `--verbose` / `-v`  | off         | INFO-level logging to stderr.                                                                                                                                      |
| `--debug`           | off         | DEBUG-level logging to stderr. Takes precedence over `--verbose`.                                                                                                  |
| `--version`         |             | Print the version and exit.                                                                                                                                        |

Without `--verbose` or `--debug`, the stderr level is `MEMMAN_LOG_LEVEL` (default `WARNING`).

---

## Install and uninstall

`pipx install memman` installs the package, and `memman install` configures it. `pipx install 'memman[postgres]'` adds Postgres support. memman needs Python 3.11+.

The default setup uses one OpenRouter endpoint and key for summaries, embeddings, and reranking, so the wizard asks for one key. [Provider setup](#provider-setup) lists the alternatives.

```bash
memman install                        # interactive wizard in a terminal
memman install --claude-code          # install into ~/.claude even when Claude Code is not detected
memman install --codex                # install the Codex memory skill explicitly
memman install --no-wizard --backend postgres --pg-dsn postgresql://memman@localhost/memman
memman uninstall
memman uninstall --claude-code
```

**Install flags:**

| Flag             | Effect                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--claude-code`  | Install into `~/.claude`, whether or not Claude Code is detected.                                                                                                         |
| `--codex`        | Install the Codex skill and command rules, whether or not Codex is detected.                                                                                              |
| `--backend NAME` | Default storage backend, `sqlite` or `postgres`. Skips the backend prompt.                                                                                                |
| `--pg-dsn URL`   | Postgres DSN. Install connects, checks for `pgvector`, and stops when either fails. Required with `--backend postgres` when no prompt runs and the env file holds no DSN. |
| `--endpoint URL` | Endpoint URL for the LLM, embeddings, and rerank. Skips the endpoint prompt. Must start with `http://` or `https://`.                                                     |
| `--no-wizard`    | Skip all prompts, even in a terminal. Flags, the shell, and defaults supply every value.                                                                                  |

### Installed files and services

Installation writes settings, configures a worker, and adds detected agent integrations. Passing `--claude-code` or `--codex` selects only the named integration. Passing both installs both.

| Path                                                   | Purpose                                         |
| ------------------------------------------------------ | ----------------------------------------------- |
| `~/.memman/env`                                        | Settings and API keys, mode 0600.               |
| `~/.claude/skills/memman/SKILL.md`                     | Symlink to the packaged agent manual.           |
| `~/.claude/hooks/memman/*.sh`                          | Symlinks to five lifecycle hooks.               |
| `~/.claude/settings.json`                              | Hook registrations and allowed memory commands. |
| `~/.agents/skills/memman`                              | Symlink to the packaged Codex memory skill.     |
| `~/.codex/rules/memman.rules`                          | Codex rules that allow the memory commands.     |
| `~/.config/systemd/user/memman-enrich.{timer,service}` | Linux worker schedule.                          |
| `~/Library/LaunchAgents/com.memman.enrich.plist`       | macOS worker schedule.                          |
| `~/.memman/logs/`                                      | Worker output.                                  |

Claude Code is detected by a `claude` binary on `PATH` or an existing `~/.claude` directory. The next new Claude Code session loads the hooks. If no supported agent is detected or explicitly selected, installation configures only the scheduler.

On a host without systemd or launchd, the operator sets `MEMMAN_SCHEDULER_KIND=serve` in the process environment and runs `memman scheduler serve` to process writes.

Existing env-file values take precedence. Installation refuses conflicting flags and prints the `config set` command needed to change the value. It also checks prerequisites and, for OpenRouter, model availability. A catalog outage is reported but does not stop installation.

After a package upgrade, `memman install` refreshes hook registrations, scheduler units, and new default settings. The [integration chapter](design/05-integration.md) describes the installed hooks and permission entries.

### Codex

Codex is detected by a `codex` binary on `PATH`, an existing `CODEX_HOME` directory (default `~/.codex`), or the installed memory skill. The skill goes in `~/.agents/skills/memman`, the Codex user skill directory, whatever `CODEX_HOME` holds.

```bash
memman install --codex
```

In Codex, `$memman` invokes the skill, as in `Use $memman to recall our deployment decisions`. A new session loads a skill the running one does not show. Codex gets no lifecycle hooks: the skill alone tells the agent when to recall and save, and the agent runs the same CLI.

The Codex sandbox blocks writes to the memman data directory and the provider network calls, so a sandboxed `memman` call fails or stops for approval. Installation therefore writes `$CODEX_HOME/rules/memman.rules`, one `prefix_rule(..., decision="allow")` line for each of the eleven verbs Claude Code also allows: `doctor`, `forget`, `insights review`, `insights show`, `recall`, `remember`, `replace`, `status`, `store branch`, `store drop`, and `store merge`. Codex runs an allowed verb outside the sandbox with no prompt. An interactive installation lists the rules and asks first. With `--no-wizard` or no terminal, it writes them without a prompt. A `--store` after the verb, as in `memman recall --store NAME`, matches the verb's rule. A call that puts a global option first, such as `memman --store NAME recall`, matches no rule and still prompts. Installation leaves `config.toml`, `AGENTS.md`, and every other rules file unchanged.

Codex needs `memman` on its shell's `PATH`. For a custom data directory, the Codex environment sets `MEMMAN_DATA_DIR`, or the call passes `memman --data-dir PATH ...`.

Store selection is the same for both agents: `--store NAME`, before or after the verb, takes precedence over `MEMMAN_STORE`, then the saved active store. Codex's shell must inherit `MEMMAN_STORE` to use it. A store name starts with a letter or digit and holds only letters, digits, dashes, and underscores, so it can never name a filesystem path. The selected store must exist: every memory verb refuses a store that does not, and creates nothing.

Both agents share the same stores and background worker. Reinstallation refreshes the skill link and the rules file. A different skill already named `memman` stays in place, and installation reports the conflict. `memman doctor` reports a broken Codex skill link and how to repair it.

### Install wizard

The wizard runs only in a terminal and only without `--no-wizard`. Each step prompts only when no flag supplies the value and the env file lacks it. The key and model steps also skip the prompt when the shell exports the `MEMMAN_` variable. Key prompts hide the input.

1. **Endpoint.** Any OpenAI-compatible URL that answers `/chat/completions`, `/embeddings`, and `/rerank`. The default is `https://openrouter.ai/api/v1`.
2. **API key.** `MEMMAN_API_KEY`. Required for any endpoint outside the local machine and optional for `localhost`. On an OpenRouter endpoint with no key set, the prompt offers the shell's `OPENROUTER_API_KEY` as its default.
3. **Models.** Asked only on an endpoint other than OpenRouter, because the shipped defaults for `MEMMAN_LLM_MODEL`, `MEMMAN_EMBED_MODEL`, and `MEMMAN_RERANK_MODEL` are OpenRouter ids. Each model ID passes unchanged to the endpoint.
5. **Model.** Asked only on an endpoint other than OpenRouter, because the default model `qwen/qwen3-235b-a22b-2507` is an OpenRouter id. The model ID passes unchanged to `/chat/completions`.
4. **Backend.** `sqlite` or `postgres`. Offered only when the `memman[postgres]` extra is installed.
5. **Postgres DSN.** Asked for the `postgres` backend. The wizard connects, checks that the `pgvector` extension exists, and allows three attempts. A DSN that names a host other than `localhost` prints a hint to run Postgres behind PgBouncer in transaction-pooling mode.

Without the wizard, installation stops when a required value is missing: `MEMMAN_API_KEY` on a non-loopback endpoint, the DSN for `--backend postgres`, or any of `MEMMAN_LLM_MODEL`, `MEMMAN_EMBED_MODEL`, and `MEMMAN_RERANK_MODEL` on an endpoint other than OpenRouter. The error names the key.

### Uninstall

`memman uninstall` removes every detected integration, or only those that `--claude-code` and `--codex` name. For Claude Code it removes:

- `~/.claude/hooks/memman/` and `~/.claude/skills/memman/`,
- the memman hooks and permission entries in `~/.claude/settings.json`.

For Codex it removes the skill link `~/.agents/skills/memman` and `$CODEX_HOME/rules/memman.rules`.

When no memman integration remains, uninstall also removes the shared services:

- the scheduled backup timer or agent,
- the scheduler unit, `~/.memman/scheduler.state`, and `~/.memman/debug.state`,
- the secret keys in the env file: `MEMMAN_API_KEY` and `MEMMAN_DEFAULT_POSTGRES_DSN`.

A selective uninstall that leaves another integration installed keeps the shared services.

It keeps every other env-file setting, including `MEMMAN_POSTGRES_DSN_<store>`, and it keeps the stores, the queue, and the logs. When an integration cleanup reports an error, uninstall stops and leaves the scheduler and backup schedule in place. Uninstall checks for a Codex skill conflict before it changes any integration or setting.

---

## Provider setup

memman sends summaries, embeddings, and reranking to one OpenAI-compatible endpoint with one key. The defaults below are the models included in this version. The [README cost table](../README.md#cost) estimates their combined cost.

| Setting               | Default                        | Purpose                                                              |
| --------------------- | ------------------------------ | -------------------------------------------------------------------- |
| `MEMMAN_ENDPOINT`     | `https://openrouter.ai/api/v1` | Base URL. Must answer `/chat/completions`, `/embeddings`, `/rerank`. |
| `MEMMAN_API_KEY`      | None                           | Secret. Required off a loopback endpoint.                            |
| `MEMMAN_LLM_MODEL`    | `qwen/qwen3-235b-a22b-2507`    | Summary model.                                                       |
| `MEMMAN_EMBED_MODEL`  | `voyageai/voyage-4-lite`       | Embedding model. Returns 1024-dimension vectors.                     |
| `MEMMAN_RERANK_MODEL` | `voyageai/rerank-3-lite`       | Rerank model.                                                        |

Model IDs are specific to the endpoint. The shipped defaults are OpenRouter ids, so an install on another endpoint requires all three model settings.

```bash
memman config set MEMMAN_ENDPOINT https://api.openai.com/v1
memman config set MEMMAN_API_KEY <api-key>
memman config set MEMMAN_LLM_MODEL <model-id>
memman config set MEMMAN_EMBED_MODEL <model-id>
memman config set MEMMAN_RERANK_MODEL <model-id>
```

At install on an OpenRouter endpoint, a blank `MEMMAN_API_KEY` is seeded from the shell's `OPENROUTER_API_KEY`.

### Voyage models through OpenRouter

OpenRouter runs Voyage models on the operator's own key (OpenRouter BYOK). OpenRouter's Voyage provider calls MongoDB's Atlas Embedding and Reranking API, so the BYOK key must be a MongoDB Atlas model API key (Atlas project, AI Model APIs). A voyageai.com dashboard key does not work. Atlas has its own training opt-out, the organization setting "Help Improve Voyage AI Models", which is on by default.

Existing stores keep their recorded embedding model until an explicit [swap or re-embed](#embedding-operations).

### Reranker

The reranker posts `{model, query, documents, top_n}` to `<MEMMAN_ENDPOINT>/rerank`.

| Setting                         | Default                  | Purpose                     |
| ------------------------------- | ------------------------ | --------------------------- |
| `MEMMAN_RERANK_ENABLED`         | `true`                   | Enable reranking globally.  |
| `MEMMAN_RERANK_ENABLED_<store>` | Global value             | Override for one store.     |
| `MEMMAN_RERANK_MODEL`           | `voyageai/rerank-3-lite` | Select the reranking model. |

```bash
memman config set MEMMAN_RERANK_ENABLED false
```

### Credential requirements

At runtime, memman reads endpoint settings from `<data dir>/env` and ignores the shell. `config set` changes them. Installation can import keys from the shell ([configuration precedence](#configuration)).

| Operation                                                                        | Credentials                                         | If unavailable                                                         |
| -------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| Queue `remember`                                                                 | No model key required.                              | Model work waits for the worker.                                       |
| Open a normal store session, including `replace`, `forget`, and `recall --basic` | `MEMMAN_API_KEY`, unless the endpoint permits none. | The embedding client stops the command.                                |
| Embed recall query                                                               | `MEMMAN_API_KEY`, unless the endpoint permits none. | Recall warns and falls back to keyword and recency ranking.            |
| Embed a queued memory                                                            | `MEMMAN_API_KEY`, unless the endpoint permits none. | Missing credentials fail the write; the queue entry records the error. |
| Generate a summary                                                               | `MEMMAN_API_KEY`, unless the endpoint permits none. | A rejected request leaves the memory without a summary.                |
| Rerank recall candidates                                                         | `MEMMAN_API_KEY`, unless the endpoint permits none. | Recall warns and preserves its pre-rerank order.                       |

A normal store session builds the global embedding client as well as the store-bound one, so a command can need the key even when no store uses the global model. Creating a store, or opening one whose model's vector size is not built in, also sends a probe embedding.

`remember` uses a separate related-memory lookup that calls no model. Diagnostics and operations such as `embed status`, `embed swap`, `migrate`, and `backup` bypass normal fingerprint initialization. `doctor` checks endpoint configuration and connectivity.

---

## Memory commands

```bash
memman remember "The retry cap stays at three, since a fourth try only adds load."
memman recall "retry cap" --limit 10
memman recall "auth" --basic
memman replace <id> "The retry cap is four for batch jobs and three elsewhere."
memman forget <id>
```

Every command that takes a memory id also accepts an unambiguous prefix of one, such as the 8-character id that recall prints. An ambiguous prefix is refused, and the error names how many ids it matches.

`remember`, `replace`, and `forget` refuse to run while the scheduler is stopped (see [Scheduler](#scheduler)).

### remember and replace

`remember` queues a new memory. The worker stores it on its next drain, a pass over the write queue that the [scheduler](#scheduler) starts on a timer. Recall finds the memory only after that. Submission returns one line of JSON:

| Field              | Meaning                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `action`           | `queued`: submission succeeded, processing is still pending.                   |
| `id`               | Persistent memory UUID, assigned at submission.                                |
| `queue_id`         | Temporary numeric queue entry ID, used by queue commands.                      |
| `store`            | Destination store.                                                             |
| `quality_warnings` | Advisory warnings about potentially temporary information.                     |
| `related`          | Up to three current memories with overlapping wording. Present for `remember`. |
| `related_error`    | Replaces `related` if that lookup failed. The write remains queued.            |
| `replaced_id`      | Target memory ID. Present for `replace`.                                       |

`insights show <id>` reports when a write is still queued. `scheduler queue show <queue_id>` reports its processing state.

`related` lists stored claims the new text may correct. The lookup favors focused word overlap, reads only memories within the 1,000-byte input limit, and calls no model. It can return an empty list, and its failure never undoes the submission.

`replace <id> "<text>"` checks its target in the store, queues a successor, and preserves the old memory's history. Unlike `remember`, it runs normal [store-opening checks](#credential-requirements), which can require credentials or a probe embedding. Once the worker commits it, recall returns the new version and excludes the old one.

| Target state                               | Behavior                                                                                                                                                                     |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current memory                             | Queue a replacement.                                                                                                                                                         |
| Write still queued in this store           | Queue the successor and wait for the target to finish processing.                                                                                                            |
| Forgotten or already replaced              | Refuse; an already-replaced error names the successor.                                                                                                                       |
| Replacement already queued                 | Refuse and quote the queued replacement. The remedy is to replace that queued id with text that keeps both corrections, because a second replacement would retire the first. |
| Target unavailable when the worker commits | Save the new memory without a replacement link; the worker result reports `target_gone`.                                                                                     |

Replacements are processed in queue order within a store. A correction of a previous correction replaces its successor. `insights show <id> --history` displays the chain.

### Rejected input

Both commands check the text in this order and report the first problem:

1. **Size.** More than 1,000 UTF-8 bytes.
2. **Line number.** Text that names a line of a file, because a line number is outdated after the next edit:
   - a source, config, or doc file name followed by a line: `scripts/auth.py:88`, `config.yaml:12`, `app.py-1233`, `cli.py ~1190`,
   - `line N` or `lines N`,
   - a bare `:N` or `L123` after the start, a space, `(`, `,`, or a semicolon: `at :1774`, `emsx.py L419`.

   A host port (`localhost:6379`, `db.example.com:5432`), an image tag (`python:3.11`), a clock time (`14:18`), `code:404`, a slice (`[:80]`), and `DISPLAY=:99` are accepted. A port without its host (`:9222`) is refused.
3. **Author.** Text whose first word is the value of `MEMMAN_AUTHOR`, ignoring case. The author field already records who wrote it. The check runs only when `MEMMAN_AUTHOR` is set, so the login-name fallback never refuses.
4. **Line break.** Text that spans several lines.
5. **Leading label.** Text that opens with at most three words, a colon, and a space: `Fix:`, `AWS gotcha:`, `User decision 2026-09-17:`. A longer phrase before the colon is allowed because it may be part of a sentence, such as "The rule is simple:". A quote or backtick ends the match, so text may start with a quoted error.

### recall

| Flag      | Default                          | Description                                                |
| --------- | -------------------------------- | ---------------------------------------------------------- |
| `--limit` | `MEMMAN_RECALL_LIMIT`, else `20` | Maximum lines printed.                                     |
| `--basic` | off                              | SQL `LIKE` matching with no ranking. Lines carry no score. |

Recall prints one line per memory, best first, and prints nothing for an empty result:

```text
<id8> <score> <created_at> <author> | <text>
```

`id8` is the first 8 characters of the id, `score` has two decimals, and `created_at` is the UTC date `YYYY-MM-DD`. `author` is `-` when unset. `text` is the summary when the memory has one, and the start of the content otherwise. `memman insights show <id>` prints the whole memory.

A score ranks a row against its siblings in one response and means nothing across queries. Recency can return a memory with no keyword or semantic match, so each returned text needs its own relevance check. [Chapter 3](design/03-pipelines.md#34-read-pipeline-recall) explains the ranking.

`--limit 0` means unlimited results on the scored path but no results with `--basic`. A scored limit over 100 may include both reranked and remaining candidates; their scores use different scales.

Without `--limit`, recall prints up to `MEMMAN_RECALL_LIMIT` lines, or 20 when the key is unset. A smaller value shortens every page an agent reads into its context. The key takes the same values as `--limit`, and an explicit `--limit` overrides it. A non-integer value stops recall with an error that names the key:

```bash
memman config set MEMMAN_RECALL_LIMIT 8
```

`--basic` requires every query word to appear as a substring of the content and returns newest memories first. It skips query embedding and reranking. Store-opening checks still run, so it can require credentials or an embedding probe; see [credential requirements](#credential-requirements).

**Rerank.** For a query of more than two words, a cross-encoder re-scores the top 100 candidates. The reranker uses `MEMMAN_RERANK_MODEL` (default `voyageai/rerank-3-lite`) on the shared endpoint with `MEMMAN_API_KEY`. When the rerank call fails, recall logs a warning and keeps the blended order. `MEMMAN_RERANK_ENABLED` (default `true`) enables or disables reranking for every store, and `MEMMAN_RERANK_ENABLED_<store>` overrides it for one store:

```bash
memman config set MEMMAN_RERANK_ENABLED_work false
```

### forget

`forget <id>` sets `deleted_at` and excludes the memory from recall. The row stays stored, and no command reverses the action.

A replaced memory can be forgotten. A current memory whose predecessor is not forgotten cannot, and the error names `replace`, which keeps the correction chain intact. Queued writes cannot be forgotten.

No memory expires automatically. `insights review` flags memories that may need updating or removal.

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

Each store records its embedding model and vector dimension in a **fingerprint**. Recall and the worker use that recorded model on the shared endpoint. Changing `MEMMAN_EMBED_MODEL` alone does not convert existing vectors. [Credential requirements](#credential-requirements) and the [embedding design](design/04-lifecycle.md#43-embedding-support) give the detail.

| Command                     | Scope                                            | Purpose                                                              |
| --------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| `embed status`              | Selected store                                   | Show fingerprint, credentials, and swap progress.                    |
| `embed swap --to MODEL`     | Selected SQLite or Postgres store                | Change model while recall continues using old vectors until cutover. |
| `embed swap --resume`       | Selected store                                   | Continue an interrupted swap.                                        |
| `embed swap --abort`        | Selected store                                   | Discard pending vectors and swap state, before cutover only.         |
| `embed reembed [--dry-run]` | SQLite stores, Postgres-parent branches excepted | Rewrite vectors using `MEMMAN_EMBED_MODEL`.                          |

To change one store's model:

```bash
memman scheduler stop
memman --store work embed swap --to <model-id>
memman scheduler start
```

After a failed swap, `embed status` shows its state, and `--resume` or `--abort` resolves it before writes restart.

**`embed swap`** converts one store (the store the global flags select) to a new model.

- It needs a stopped scheduler, except with `--abort`. Recall keeps reading the old vectors while the swap writes new ones into the `embedding_pending` column in batches of `MEMMAN_EMBED_SWAP_BATCH_SIZE` (default 200).
- On Postgres, the swap first builds an HNSW index on the new column.
- The final switch, called cutover, replaces the old vectors in one transaction. Returning to the old model requires another full swap.
- One swap or abort runs per store at a time. A second one, from any shell, refuses while the first holds the store's swap lock.
- `--resume` continues an interrupted swap from its recorded cursor. `--abort` discards pending vectors and swap state. Once a swap reaches cutover, `--abort` refuses and `--resume` finishes it, because the cutover may already have committed.
- The swap leaves `MEMMAN_EMBED_MODEL` unchanged.

**`embed reembed`** converts every SQLite store under the data directory to the `MEMMAN_EMBED_MODEL` client.

- It needs a stopped scheduler, except with `--dry-run`. It refuses to run when the selected store is on Postgres, and it skips Postgres stores.
- A branch keeps its parent's model, so `embed reembed` skips a branch whose parent is not a local SQLite store. Such a branch needs `embed swap` after its parent swaps.
- For each store, every current memory whose vector is not on the target model is re-embedded, and the store gets the new fingerprint. An empty store gets only the fingerprint.
- A second run resumes an interrupted one.

To change the global embedding model:

```bash
memman config set MEMMAN_EMBED_MODEL <model-id>
memman scheduler stop
memman embed reembed
memman scheduler start
```

---

## Store management

A store is a named, isolated set of memories: one SQLite file or one Postgres schema. [Chapter 2](design/02-concepts.md) explains why stores exist.

```bash
memman store list            # JSON: {stores, active, branches}
memman store create work
memman store use work        # write "work" to the active-store file
memman store remove old-project [--yes]
memman store branch work rearch        # a branch of work, named work__rearch_<4 hex>
memman store merge work__rearch_7f3a   # replay the branch into work, then delete it
memman store drop work__rearch_7f3a    # list the branch's own rows, then delete it
```

- A store name starts with a letter or digit and continues with letters, digits, `_`, or `-`. A Postgres store also needs a name that is a valid SQL identifier. `store create` refuses a name holding `__`, which marks a branch.
- A store exists once `store create` has made it. Every command that opens a store, `remember` and `replace` included, refuses a store that does not exist with `store "<name>" does not exist (create it with memman store create <name>)` and writes nothing: no directory, no Postgres schema, no env key. `memman install` creates the store the active-store file names.
- `store use` accepts only an existing store, and refuses a branch.
- `store remove` asks first unless the call passes `--yes`. It refuses the store named in the active-store file. It deletes the store's data, its queued writes, and its `MEMMAN_BACKEND_<store>`, `MEMMAN_POSTGRES_DSN_<store>`, and `MEMMAN_RERANK_ENABLED_<store>` keys. It refuses a branch and names `store merge` and `store drop`. It refuses a store that holds a `branch_token:<branch>` meta key, the mark of an open branch on this host or another. It lists each such key and marks a key that has no local branch. When it cannot read a SQLite store's file, it skips these checks and removes the store.
- `memman store` with no subcommand runs `store list`.

**Store selection**, from highest priority to lowest:

1. the `--store <name>` flag, before or after an agent verb,
2. the `MEMMAN_STORE` environment variable,
3. the active-store file `<data dir>/active`,
4. `default`.

### Store branches

A branch is an empty SQLite store layered over its live parent store. It ends with `store merge` or `store drop`. It holds one research thread's writes apart from the parent, while recall on the branch also reads the parent's current rows. [Chapter 3](design/03-pipelines.md#36-store-branches) describes the three flows.

| Command                         | Effect                                                                                                                                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `store branch <parent> <label>` | Creates an empty `<parent>__<label>_<4 hex>` on SQLite, whatever the parent's backend. Prints `store`, `parent`, `path`, `created_at`, and `instruction`.                                                                                               |
| `store merge <branch>`          | Copies the branch's own rows into the parent, repeats the branch's `replace` and `forget` on parent rows, deletes the branch. Prints `copied`, `retired`, and `conflicts`.                                                                              |
| `store drop <branch>`           | Deletes the branch. Prints `dropped`, the `{id, content, replaces}` of each current branch row, so the agent can re-save a claim unrelated to the thread with `remember --store <parent>`. `replaces` names the parent row the row corrects, else null. |

- The parent of a branch is the `branch_parent` entry in its `meta` table, which `status` and `store list` print. `merge` and `drop` refuse any store without it, so no agent verb can delete an ordinary store. `store branch` refuses a branch as parent. Any number of branches of one parent can stay open. Siblings see each other's rows only after one merges.
- The `instruction` line reads "Use memman store <branch> for this thread: pass --store <branch> to every memman recall, remember, replace, forget and insights show call." A session that reads the line passes `--store <branch>` to every memory verb. A session without it uses the parent.
- A `--store <branch>` recall ranks the branch's rows and the parent's current rows in one pass. A row the branch holds hides the parent row with the same id. Recall refuses when the parent's embedding fingerprint differs from the branch's, and the message names the branch's embed swap. Recall also refuses when the parent's `meta` lacks `branch_token:<branch>`, as after a parent removed and recreated under the same name, pointed at another database, or restored from an older backup. The message names the fix: repair a parent that points at the wrong database, else `store drop <branch>`, which lists the branch rows to remember again in the parent. `status` counts the rows recall sees.
- `remember`, `replace`, and `forget` with `--store <branch>` write only the branch. A `replace` or `forget` of a current parent row copies the row into the branch and retires the copy there. A `replace` or `forget` of a parent row the parent already retired fails and names the current head.
- `merge` refuses a store that is not a branch, the active branch, a missing parent, queued writes for the branch, a pending or failed parent `replace` whose target's chain holds a row the branch retired, an embed swap or re-embed in progress in either store, a branch whose embedding fingerprint differs from the parent's, and a parent that does not hold the branch's token. After the checks it refuses every further write to the branch. Each `conflicts` entry is a copied row the branch and the parent retired differently. The parent keeps its own state, and the operator or agent settles the entry with `replace` or `forget`.
- A `merge` that stops part way says to re-run `merge`, which finishes it. A re-run after the parent commit writes nothing more to the parent and reports the conflicts as the parent holds them. `drop` refuses a branch whose merge stopped after the parent commit, the active branch, and a branch with queued writes. A missing parent does not stop `drop`. An unreachable parent stops it only for a branch whose merge started.
- A branch cannot be the active store. `migrate` skips it, and `backup` bundles it like any store. `doctor` lists each branch with its parent and creation time. On a branch whose parent is missing or unreachable, `doctor` reports a failed `integrity` check and runs the other checks, `branches` included.

**Per-directory stores.** memman reads `MEMMAN_STORE` from the process environment, so a tool that sets variables per directory switches the store on a directory change.

| Mechanism                          | Setup                                                    | Scope                                                           |
| ---------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| `direnv`                           | `.envrc` in the project holds `export MEMMAN_STORE=work` | Every shell, agent, and subprocess started in the directory     |
| `--store` flag                     | `--store work` on each command                           | One command                                                     |
| Project `CLAUDE.md` or `AGENTS.md` | A directive telling the agent to pass `--store work`     | Sessions that read that file                                    |
| `memman store use work`            | Writes the global active-store file                      | Every caller on the host. The most recent `use` sets the store. |

Named stores separate projects. The scheduler processes only the queue in its configured data directory, so a separate `MEMMAN_DATA_DIR` per project creates queues that no scheduler processes.

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
- A migration in the other direction reverses the change. `memman doctor` checks the result: its `stale_post_migrate_source` check warns when SQLite files remain in a store that routes to Postgres.

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

**`status`** prints the store name, its backend, the backends in use, counts of current, replaced, and forgotten memories, `stale_insights` (the count `enrich --stale-only` would process), the oplog size, and the storage path.

**`doctor`** exits 1 when any check fails and 0 otherwise. It makes one live LLM call and two to four live embedding calls: `embed_probe` sends an availability probe and a test embed, and `embed_fingerprint` sends an availability probe for the store's recorded model, plus a size probe when that model's vector size is not built in.

| Group              | Checks                                                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store              | `integrity`, `enrichment_coverage`, `replacement_integrity`, `embedding_consistency`, `embed_fingerprint`, `no_stale_swap_meta`, `provenance_drift` |
| Queue and schedule | `queue_backlog`, `scheduler_heartbeat`, `drain_heartbeat`, `scheduler_state`                                                                        |
| Configuration      | `env_completeness`, `per_store_keys`, `env_permissions`, `stale_post_migrate_source`, `claude_hooks`, `codex_skill`, `optional_extras`              |
| Providers          | `llm_probe`, `embed_probe`                                                                                                                          |

A store with no memories skips `integrity`, `enrichment_coverage`, `embedding_consistency`, and `provenance_drift`.

`enrichment_coverage` warns on any memory whose enrichment never completed, reports the count as `stranded`, and names `memman enrich --stale-only` as the fix.

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
memman scheduler install [--interval N] [--endpoint URL]
memman scheduler uninstall
memman scheduler serve [--interval N] [--once]
memman scheduler debug on|off|status
```

**Recall remains available while the scheduler is stopped.** `remember`, `replace`, and `forget` exit with status 1 and report that writes are disabled. The error names `memman scheduler start`, which enables writes.

`scheduler trigger` refuses in the same way. A running drain finishes the current memory before stopping, and a `serve` process exits. Three commands require a stopped scheduler: `enrich`, `embed swap`, and `embed reembed`.

| Command                       | Behavior                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `trigger`                     | Ask systemd or launchd to start a drain; return `dispatched` immediately. `log worker` shows the result. Serve mode uses `serve --once` instead. |
| `interval`                    | Show the interval, or set it with `--seconds N`. systemd and launchd require at least 60 seconds.                                                |
| `install`                     | Install only the scheduler, filling missing settings and refusing conflicting flags. `--interval` defaults to 60 and must be at least 60.        |
| `uninstall`                   | Remove the scheduler unit and state files, and strip secret settings. Keep the Claude Code integration.                                          |
| `serve`                       | Run drains in the foreground; `--once` processes one drain and exits.                                                                            |
| `debug on` / `off` / `status` | Control or inspect worker tracing.                                                                                                               |

**Serve mode.** Hosts without systemd or launchd use `MEMMAN_SCHEDULER_KIND=serve`. The loop reads its interval from `--interval`, then the installed `MEMMAN_INTERVAL`, then 60. An interval of zero runs drains without a pause. In this mode `scheduler interval` only records the value, and a new interval takes effect when the loop restarts with `--interval N`. SIGTERM or SIGINT stops it after the current memory, with exit code zero.

**Debug output.** `debug on` updates `~/.memman/debug.state`. Later drains write `~/.memman/logs/debug.log` at mode 0600, including raw model requests, responses, and memory content. `debug off` stops tracing and retains the file. The process-environment variable `MEMMAN_DEBUG` overrides the state file.

### Queue

```bash
memman scheduler queue list [--limit N]    # status counts and recent writes (default 50)
memman scheduler queue failed [--limit N]  # failed writes (default 50)
memman scheduler queue show <row_id>       # one write in full
memman scheduler queue retry <row_id>      # return one failed write to pending
memman scheduler queue purge --done        # delete done writes
```

`memman scheduler queue` with no subcommand runs `queue list`. Each queued write has one status:

| Status    | Meaning                                                                                                                                                                                |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pending` | Waiting for a drain, or claimed by one. A claim older than 600 seconds is taken over by the next drain. A write keeps this status however long the scheduler stays stopped.            |
| `done`    | Stored. Later maintenance deletes completed entries older than 60 seconds.                                                                                                             |
| `failed`  | Five attempts failed. The waits between attempts are 60, 120, 240, and 480 seconds. The text stays in the queue, and no automatic step deletes it. `queue retry <row_id>` requeues it. |

---

## Backup

`memman backup` snapshots all stores and the write queue, either on demand or on a schedule. The target belongs outside the source data directory, on storage that survives the loss of that directory. memman does not check the target location. Snapshots use SQLite's online backup API or Postgres `pg_dump -Fc` while the worker continues running.

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

| Scheduler  | Backup behavior                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| systemd    | A persistent timer runs a backup missed while the host was asleep. Restricting both day-of-month and day-of-week means AND, where cron means OR, and memman warns. |
| launchd    | Uses `StartCalendarInterval`.                                                                                                                                      |
| Serve loop | Matches the cron expression itself and drains writes before taking a snapshot.                                                                                     |

A bundle consists of a `.tar.gz` archive and an adjacent manifest used by `backup list`. Store snapshot failures are listed in the manifest while other snapshots continue.

The queue is copied before the stores, preserving pending writes for the restored installation. The manifest records their count as `queue_pending`. Queue UUIDs prevent restored writes from duplicating memories already captured in a store snapshot.

**Excluded settings.** The bundle's `env.nonsecret` member leaves out `MEMMAN_API_KEY`, `MEMMAN_DEFAULT_POSTGRES_DSN`, every `MEMMAN_POSTGRES_DSN_<store>`, and the host's own `MEMMAN_BACKUP_*` keys. It keeps per-store backend keys and model and endpoint settings.

**Restore.**

1. Refuses a bundle with Postgres stores when `pg_restore` is not on `PATH`.
2. Asks first unless `--yes` is given, then holds the drain lock and refuses a bundle whose format version differs from this memman's.
3. Merges the non-secret settings into the env file, so per-store backend keys are in place.
4. Restores each store using the `backend` in its manifest entry: a file copy for SQLite, `pg_restore` for Postgres with the DSN configured on this host.
5. Restores `queue.db` and the active-store file.

The output lists `restored`, `failed`, `pg_restore_skipped` (Postgres stores with no DSN on this host), `embed_mismatch` (stores whose fingerprint differs from this host's embedding settings), `queue_restored`, `active_store`, and `secret_keys_needed` (secret keys missing from this host's env file).

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
3. the shell: the `MEMMAN_` name, then, for `MEMMAN_API_KEY` on an OpenRouter endpoint, `OPENROUTER_API_KEY`,
4. the included default (`INSTALL_DEFAULTS` in `src/memman/config.py`).

### Process-control variables

These come from the process environment and are not installed in the settings file:

| Variable                | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`       | Select the settings and data directory.                                 |
| `MEMMAN_STORE`          | Select a store for this process.                                        |
| `MEMMAN_AUTHOR`         | Identify the caller submitting a memory; defaults to the OS login name. |
| `MEMMAN_DEBUG`          | Override the debug state file.                                          |
| `MEMMAN_SCHEDULER_KIND` | Override scheduler detection, including serve mode.                     |
| `MEMMAN_WORKER`         | Mark a worker process for logging.                                      |

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

A store's backend is `MEMMAN_BACKEND_<store>`, then `MEMMAN_DEFAULT_BACKEND`, then `sqlite`. The first drain that writes to a store records `MEMMAN_BACKEND_<store>` from the default, together with `MEMMAN_POSTGRES_DSN_<store>` from `MEMMAN_DEFAULT_POSTGRES_DSN` when the default is `postgres`. A later change to `MEMMAN_DEFAULT_BACKEND` therefore moves no store a drain has written to. `memman migrate` moves a store and its data. `store branch` writes `MEMMAN_BACKEND_<branch>=sqlite` itself, whatever the default, because a branch is always SQLite.

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

| Symptom                                           | Check                                           | Next step                                                                                                                                   |
| ------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| A new memory does not appear in recall            | `scheduler status`, then `scheduler queue list` | Wait for processing; inspect a failed entry and fix its reported error before retrying.                                                     |
| Writes report that the scheduler is stopped       | `scheduler status`                              | Run `scheduler start` when maintenance is finished.                                                                                         |
| Recall warns about embedding or reranking         | `embed status`, then endpoint settings          | Restore the required key or endpoint; reranking can be disabled separately.                                                                 |
| Even `recall --basic` fails on a missing key      | Global embedding model and endpoint             | Supply the key required during store opening.                                                                                               |
| Doctor reports incomplete enrichment              | `doctor --text`                                 | Stop the scheduler, run `enrich --stale-only`, then restart it.                                                                             |
| A model swap was interrupted                      | `embed status`                                  | Resume or abort the swap.                                                                                                                   |
| Claude Code has no memory reminders               | `doctor --text`                                 | Re-run `memman install` and start a new session.                                                                                            |
| A memory verb reports that a store does not exist | `store list`                                    | Create it with `store create NAME`, or correct `--store`, `MEMMAN_STORE`, or a stale branch instruction line so it names an existing store. |

Commands in this table take the `memman` prefix. `doctor` makes live provider probes, and the queue and worker logs show processing without them.
