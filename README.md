# memman

**LLM-supervised persistent memory for coding agents.**

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### Memory categories

| Category     | Captures                                | Example                                                           |
| ------------ | --------------------------------------- | ----------------------------------------------------------------- |
| `preference` | User preferences and style              | "Prefers snake_case, dislikes ORMs"                               |
| `decision`   | Architectural choices and their reasons | "Chose SQLite for its embedded database and lack of dependencies" |
| `fact`       | Lasting facts about systems or domains  | "API rate limit is 100 req/s"                                     |
| `insight`    | Conclusions drawn from several sources  | "Combining search rankings gives better results here"             |
| `context`    | Project background and user environment | "Monorepo, deploys to AWS ECS"                                    |

See [Design & Architecture](docs/DESIGN.md) for details.

## How it works

The coding agent runs Memman. Claude Code [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) remind the agent to recall at session start, on each prompt, and before a delegation, and to store its conclusions when it leaves plan mode.

Five hook scripts respond to Claude Code events:

| Hook script      | Event                        | Role                                                                                    |
| ---------------- | ---------------------------- | --------------------------------------------------------------------------------------- |
| `prime.sh`       | `SessionStart`               | prints the status line, any model notice, a recall hint after compaction, and the guide |
| `user_prompt.sh` | `UserPromptSubmit`           | reminds the agent to recall before answering                                            |
| `task_recall.sh` | `PreToolUse` (Agent or Task) | reminds the agent to recall before the next delegation                                  |
| `compact.sh`     | `PreCompact`                 | writes a flag for the post-compaction recall hint                                       |
| `exit_plan.sh`   | `PreToolUse` (ExitPlanMode)  | reminds the agent to store plan conclusions                                             |

### Inside Claude Code vs outside

During a turn, the agent queues writes and recalls stored memories. A background worker enriches queued writes and creates their embeddings.

```
 Inside the turn                          Background worker
+----------------------------------+     +-----------------------------------+
| memman remember                  |     | a drain runs every 60 s (default) |
|   append the write to queue.db --+---->|   claim each queued write         |
|                                  |     |   enrich it with the LLM          |
| memman recall                    |     |   embed it                        |
|   read the store <---------------+-----+-- store the memory                |
|   embed the query, rerank        |     |                                   |
+----------------------------------+     +-----------------------------------+
```

| Step                    | Where   | Network calls                                 | Notes                                                 |
| ----------------------- | ------- | --------------------------------------------- | ----------------------------------------------------- |
| `memman recall --basic` | inside  | none on a store with an embedding fingerprint | keyword match on the local store, no score            |
| `memman recall`         | inside  | query embedding and reranking                 | rerank skips a query of two words or fewer            |
| `memman remember`       | inside  | none                                          | appends to `queue.db`                                 |
| drain trigger           | outside | none                                          | systemd or launchd timer, or `memman scheduler serve` |
| enrichment              | outside | one LLM call                                  | adds a summary                                        |
| embedding               | outside | one embedding call                            | vector for semantic search                            |
| DB write                | outside | none                                          | makes the memory recallable                           |

This split determines when model calls run and when memories become available:

- **Model calls.** The agent's turn never enriches or embeds a write. `remember` appends to the queue and makes no network call. `recall` reads the local store and, on its default path, embeds the query and reranks the top results. `--basic` makes neither call. Opening a store needs the Voyage or OpenRouter key when either provider is in use, including with `--basic` ([Where keys are needed](#where-keys-are-needed)).
- **Recallable after a drain.** A queued write is not recallable until a drain stores it. After that, recall returns it in the same session or any later one.

## Features

- **Built for coding agents** - stores decisions, preferences, and facts from one Claude Code session for recall in later sessions.
- **Recall reminders** - five lifecycle hooks remind the agent to recall and to store.
- **LLM-supervised** - the host LLM decides what to remember and forget. A worker model handles enrichment. No LLM judges a write.
- **Combined search rankings** - Reciprocal Rank Fusion (RRF) combines keyword, vector, and recency rankings. A reranker reorders the top results of a query longer than two words.
- **Explicit replacements** - a write adds a row or replaces the row named by `replace <id>`. Only `replace` and `supersede` retire a row. A retired row keeps its content and records its successor in `superseded_by`. Recall skips it. `memman insights show <id> --history` shows the chain of replacements.
- **Nothing expires** - a store has no size cap, and nothing expires or is pruned on its own. `memman forget <id>` is the only command that removes a single memory. `memman insights review` flags temporary information to help with that decision.
- **Embedding providers** - the registered providers are `voyage`, `openai` (any OpenAI-compatible endpoint), `openrouter`, and `ollama`. Each store's `meta.embed_fingerprint` binds it to one model, so one process serves stores on different models. `memman embed swap` and `memman embed reembed` move stores to a new model ([Embedding operations](docs/USAGE.md#embedding-operations)).
- **Storage options** - SQLite by default. The `memman[postgres]` extra adds Postgres with pgvector, and `memman migrate` moves a store between the two in one command ([Usage](docs/USAGE.md#migrating-between-sqlite-and-postgres)).
- **External scheduled backups** - `memman backup schedule '<cron>' <dir>` writes every store to an outside directory on a cron schedule and keeps the last N bundles. Bundles leave out secrets. `memman backup restore` rebuilds a working store after the loss of `~/.memman/` ([Backup](docs/USAGE.md#backup)).

## Install

> [!IMPORTANT]
> **memman's API calls are billed separately from the agent's.** Claude Code keeps its own Claude login, which memman never reads or bills against. memman's keys pay for the worker's enrichment and embedding calls and for the calls recall makes to rank results. A Claude Pro or Max subscription does not cover them, because a chat subscription and a developer API bill separately. [Where keys are needed](#where-keys-are-needed) lists the key each step uses.

```bash
pipx install memman
# or, with the optional Postgres backend:
# pipx install 'memman[postgres]'
memman install
```

In a terminal, `memman install` runs a wizard. It asks for the LLM endpoint, the embedding provider, the keys those two need, and the storage backend. It asks for the reranker's Voyage key only when Voyage embeddings were chosen ([Reranker](#reranker)). A loopback LLM endpoint (Ollama, local vLLM or LiteLLM) may leave the API key blank. A headless install passes `--no-wizard` and takes the keys from the shell or from an existing `~/.memman/env`. [Variable reference](CONTRIBUTING.md#variable-reference) lists every key.

Installation creates or updates these paths:

| Path                                                   | What                                                         | Form                            |
| ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------- |
| `~/.claude/skills/memman/SKILL.md`                     | the skill: the full manual the agent loads on demand         | symlink into installed package  |
| `~/.claude/hooks/memman/*.sh`                          | five hook scripts                                            | symlinks into installed package |
| `~/.claude/settings.json`                              | hook registrations and `Bash(memman <verb>:*)` allow entries | JSON merge                      |
| `~/.config/systemd/user/memman-enrich.{timer,service}` | scheduler unit (Linux)                                       | unit files                      |
| `~/Library/LaunchAgents/com.memman.enrich.plist`       | scheduler agent (macOS)                                      | plist                           |
| `~/.memman/env` (mode 0600)                            | every setting, including API keys                            | created or updated in place     |
| `~/.memman/logs/`                                      | worker output                                                | directory                       |

The install needs systemd on Linux, launchd on macOS, or `MEMMAN_SCHEDULER_KIND=serve` on a host that runs `memman scheduler serve` itself. It installs into `~/.claude` when it detects Claude Code, and installs only the scheduler when it does not. `--target` skips the detection:

```bash
memman install --target claude-code
```

A new Claude Code session picks up the hooks. [Development](#development) covers editable installs and the test suite.

### Provider setup

memman calls three outside services: an LLM for enrichment, an embedding provider for vector search, and a reranker that orders recall results.

#### Where keys are needed

| What runs                                                    | Where           | Key it needs                                                                                | Without that key                                                                  |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `memman remember`                                            | inside the turn | none                                                                                        | works: it is the only memory command that opens no store                          |
| every command that opens a store, including `recall --basic` | inside the turn | the key of `MEMMAN_EMBED_PROVIDER` and of the store's own provider (`voyage`, `openrouter`) | the command stops. Voyage reports `MEMMAN_VOYAGE_API_KEY is not set in <dir>/env` |
| `recall`: rerank the top results                             | inside the turn | `MEMMAN_VOYAGE_API_KEY`                                                                     | recall keeps the order it had before reranking and logs a warning                 |
| enrichment                                                   | worker          | `MEMMAN_LLM_API_KEY` (blank for a local endpoint)                                           | the memory is stored without a summary                                            |
| embedding                                                    | worker          | the active embedding provider's key                                                         | the queued write fails, retries, and after 5 attempts stays queued as `failed`    |

- **Opening a store builds two embedding clients.** One is for `MEMMAN_EMBED_PROVIDER`. The other is for the provider the store's fingerprint names. The `voyage` and `openrouter` clients refuse to start without their key. The `openai` client starts without a key, and its first embedding call fails.
- **Every key lives in `~/.memman/env`.** memman reads its settings from that file and ignores the shell. `memman config set KEY VALUE` writes a key ([Configuration](docs/USAGE.md#configuration)).
- **Reranking uses a Voyage key whatever the embedding provider is.** Voyage is the only reranker, and reranking is on by default. `memman config set MEMMAN_RERANK_ENABLED false` turns it off. `MEMMAN_RERANK_ENABLED_<store>` sets it for one store. This is the one key whose absence degrades recall instead of stopping it.

#### LLM providers

The LLM client uses the OpenAI-compatible `/chat/completions` protocol. Any endpoint that supports it works without code changes.

| Provider                | Endpoint                       | Key (`MEMMAN_LLM_API_KEY`) |
| ----------------------- | ------------------------------ | -------------------------- |
| OpenRouter              | `https://openrouter.ai/api/v1` | `sk-or-...`                |
| OpenAI                  | `https://api.openai.com/v1`    | `sk-...`                   |
| Anthropic (OpenAI shim) | `https://api.anthropic.com/v1` | `sk-ant-...`               |
| Ollama (local)          | `http://localhost:11434/v1`    | blank                      |
| vLLM / LiteLLM          | self-hosted URL                | as required                |

Switching endpoints takes three settings, because the default model ID is an OpenRouter ID:

```bash
memman config set MEMMAN_LLM_ENDPOINT https://api.openai.com/v1
memman config set MEMMAN_LLM_API_KEY sk-...
memman config set MEMMAN_LLM_MODEL <model id>
```

`MEMMAN_LLM_MODEL` names the model, and memman never changes it on its own. On OpenRouter, the install sets `qwen/qwen3-235b-a22b-2507`, and each request routes only to the vendors in `MEMMAN_LLM_PROVIDER_ONLY` under zero data retention ([LLM routing](docs/design/03-pipelines.md#llm-routing)). memman checks at install, and once a day from the scheduler, that such a vendor serves the model and that OpenRouter lists no retirement date for it. A failed check prints a notice at session start that names the fix. On any other endpoint the wizard asks for a model id, and an install without the wizard refuses to finish without one.

#### Embedding providers

Each store records the provider, model, and vector dimension of its embeddings in `meta.embed_fingerprint`. One process can therefore serve stores bound to different providers.

| Provider     | Default model            | Settings                                                                                             |
| ------------ | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `voyage`     | `voyage-3-lite` (512)    | `MEMMAN_VOYAGE_API_KEY`                                                                              |
| `openai`     | `text-embedding-3-small` | `MEMMAN_OPENAI_EMBED_API_KEY`, and `MEMMAN_OPENAI_EMBED_ENDPOINT` (default `https://api.openai.com`) |
| `openrouter` | `baai/bge-m3`            | `MEMMAN_OPENROUTER_API_KEY` and `MEMMAN_OPENROUTER_ENDPOINT`                                         |
| `ollama`     | `nomic-embed-text`       | no key, `MEMMAN_OLLAMA_HOST` (default `http://localhost:11434`)                                      |

The wizard and `--embed-provider` offer `voyage`, `openai`, and `openrouter`. `ollama` is set only with `memman config set MEMMAN_EMBED_PROVIDER ollama`.

A store stays bound to the model its fingerprint records. A change of `MEMMAN_EMBED_PROVIDER` reaches a store only after `memman embed reembed` rewrites every SQLite store, or `memman embed swap` moves one store. Both need a stopped scheduler ([Embedding operations](docs/USAGE.md#embedding-operations)):

```bash
memman config set MEMMAN_EMBED_PROVIDER openai
memman config set MEMMAN_OPENAI_EMBED_API_KEY sk-...
memman scheduler stop
memman embed reembed
memman scheduler start
```

#### Reranker

memman includes one reranker, enabled by default. It scores the top recall results against the query so the best match comes first.

| Setting                      | Default         | What it does                                         |
| ---------------------------- | --------------- | ---------------------------------------------------- |
| `MEMMAN_RERANK_ENABLED`      | `true`          | `false` skips reranking, so no Voyage key is needed  |
| `MEMMAN_RERANK_PROVIDER`     | `voyage`        | the only registered provider                         |
| `MEMMAN_VOYAGE_API_KEY`      | -               | authenticates the reranker, whatever the embedder is |
| `MEMMAN_VOYAGE_RERANK_MODEL` | `rerank-3-lite` | model id                                             |

## Operation

### Memory shared across sessions

Every session uses the `default` store until another is chosen, so a decision remembered in one session is recalled in every later one.

### Isolation per project or agent

Named stores keep memories apart:

```bash
memman store create work                  # create a store
memman store use work                     # make it the active store
memman --store work recall "query"        # one command
MEMMAN_STORE=work memman recall "query"   # one process
```

`--store` takes precedence over `MEMMAN_STORE`, which takes precedence over the active store.

### Automatic store selection per directory

A tool that loads environment variables for each directory, such as [direnv](https://direnv.net), sets `MEMMAN_STORE` per project:

```bash
cd ~/projects/work
echo 'export MEMMAN_STORE=work' > .envrc
direnv allow
```

Every shell, agent, and subprocess started in that directory uses the `work` store. [USAGE.md](docs/USAGE.md#store-management) compares the alternatives.

### Customizing behavior

The included `guide.md` (instructions for the agent) and `SKILL.md` (full manual) live inside the installed package. A change to either belongs in the package source. An editable install (`pipx install -e .`) uses the edited files immediately.

### What `memman remember` does

`memman remember` appends a row to `queue.db` and returns. It refuses text over 1,000 bytes, text that spans lines, and other text that fails the single-memory format checks ([What remember and replace refuse](docs/USAGE.md#what-remember-and-replace-refuse)). The scheduler drains every 60 s by default (`memman scheduler interval` changes it), and a write becomes recallable once a drain stores it ([Inside Claude Code vs outside](#inside-claude-code-vs-outside)).

### Pausing the scheduler

`memman scheduler stop` sets the state to stopped and disables the systemd timer or launchd agent. While stopped, memman is recall-only: `remember`, `replace`, `supersede`, `unsupersede`, and `forget` report that the scheduler is stopped and writes are disabled. `scheduler trigger` reports the same error. `graph rebuild`, `embed reembed`, and `embed swap` run only while the scheduler is stopped. `memman scheduler start` resumes it ([Scheduler](docs/USAGE.md#scheduler)).

## Updating

```bash
pipx upgrade memman
```

Upgrading updates the hook scripts and `SKILL.md` through their symlinks into the installed package. It also updates `guide.md`, which `memman prime` reads from the package. Run `memman install` after each upgrade to update the following files and settings:

- **`~/.claude/settings.json`.** It holds the hook registrations and the allow entries. A release that adds or removes a hook or changes its matcher keeps the old registration until `memman install` rewrites the file. `memman doctor` reports a registration that differs from what install writes.
- **The scheduler unit.** A release that changes the unit takes effect only when `memman install` rewrites it.
- **New settings.** A release that adds a setting writes its default to `~/.memman/env` only at install. `memman doctor` reports a missing key.

## Uninstall

```bash
memman uninstall            # remove hooks, skill, settings entries, scheduler unit
pipx uninstall memman       # remove the memman binary
```

Either command runs alone. `memman uninstall` also removes a scheduled backup and deletes the API keys and the default Postgres DSN from `~/.memman/env`. It keeps every store, the logs, and the other settings. [Usage](docs/USAGE.md#install-and-uninstall) lists what it removes.

## Development

```bash
make dev            # editable Poetry install with dev dependencies
make test           # unit tests (pytest)
make e2e            # end-to-end tests
pipx install -e .   # editable pipx install, for the Claude Code integration
memman install      # deploy the integration
memman uninstall    # remove the integration
```

**Dependencies**: Python 3.11+, Click, httpx, tqdm, numpy. The `postgres` extra adds psycopg, psycopg-pool, and pgvector. [Where keys are needed](#where-keys-are-needed) lists the keys. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, tests, and conventions.

## Documentation

- [Design & Architecture](docs/DESIGN.md): the design chapters, from background to Claude Code integration
- [Usage & Reference](docs/USAGE.md): every command, flag, and setting
- [Contributing](CONTRIBUTING.md): development setup, schema changes, and tests
- [Diagrams](docs/diagrams/): the LLM-supervised split, system architecture, memory data model, remember pipeline, recall pipeline, and Claude Code integration

## License

[MIT](LICENSE)
