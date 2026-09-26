# memman

**LLM-supervised persistent memory for coding agents.**

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### Memory categories

| Category     | Captures                                | Example                                        |
| ------------ | --------------------------------------- | ---------------------------------------------- |
| `preference` | User-stated likes, dislikes, style      | "Prefers snake_case, dislikes ORMs"            |
| `decision`   | Architectural choices with rationale    | "Chose SQLite - zero deps, embeddable"         |
| `fact`       | Durable truths about systems/domains    | "API rate limit is 100 req/s"                  |
| `insight`    | Conclusions from multi-source reasoning | "RRF fusion beats a single ranked signal here" |
| `context`    | Project background, user environment    | "Monorepo, deploys to AWS ECS"                 |

See [Design & Architecture](docs/DESIGN.md) for details.

## How it works

Once installed, the coding agent runs memman, not the user. Claude Code [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) fire on session start and prompt submit; each reminds the agent to recall before responding and remember after.

Five hook scripts drive the Claude Code lifecycle:

| Hook script      | Event                        | Role                                                          |
| ---------------- | ---------------------------- | ------------------------------------------------------------- |
| `prime.sh`       | `SessionStart`               | loads the behavioral guide; surfaces post-compact recall hint |
| `user_prompt.sh` | `UserPromptSubmit`           | reminds the agent to recall before answering                  |
| `task_recall.sh` | `PreToolUse` (Agent or Task) | reminds the agent to recall before sub-agent delegation       |
| `compact.sh`     | `PreCompact`                 | drops a flag so the next `SessionStart` re-recalls context    |
| `exit_plan.sh`   | `PreToolUse` (ExitPlanMode)  | prompts memory storage before plan-to-execute transitions     |

### Inside Claude Code vs outside

memman splits along a hot-path boundary. The agent's turn does only fast local work; everything slow runs in a background worker.

```
┌─ Inside Claude Code (synchronous) ──┐    ┌─ Background worker ─────────────┐
│                                     │    │                                 │
│  memman recall   (local read, then  │    │  drain fires every 60 s under   │
│                   embed + rerank)   │    │  flock on ~/.memman/drain.lock  │
│  memman remember (queue append)     │ →  │                                 │
│                                     │    │                                 │
│  No enrichment                      │    │  enrich → embed → DB            │
│                                     │    │                                 │
└─────────────────────────────────────┘    └─────────────────────────────────┘
              │                                          ▲
              └──── queue.db (handoff; not recallable) ──┘
```

| Step                    | Where   | Latency       | Notes                                                                    |
| ----------------------- | ------- | ------------- | ------------------------------------------------------------------------ |
| `memman recall --basic` | inside  | ~50-200 ms    | local read only - no network on a store already stamped                  |
| `memman recall`         | inside  | network-bound | local read, plus one call to encode the query and one to reorder results |
| agent reasoning         | inside  | -             | uses recall results as context                                           |
| `memman remember`       | inside  | ~50 ms        | enqueue only - no LLM, no embed, no network                              |
| drain trigger           | outside | every 60 s+   | systemd/launchd timer or serve loop                                      |
| enrichment              | outside | network-bound | external LLM provider call                                               |
| embedding               | outside | network-bound | external embedding provider call                                         |
| DB write                | outside | ms            | makes insight visible to *future* turns                                  |

Two invariants follow from this split:

- **Hot-path discipline.** The agent's turn never checks a write for duplicates or writes to the graph. `remember` appends to a queue file and reaches no network. `recall` reads the local database and, on its default path, calls the embedding provider to encode the query and the reranker to reorder the top results; `--basic` makes neither call. Opening the store needs the embedding provider's key on every path, `--basic` included - see [Where keys are needed](#where-keys-are-needed).
- **One-way visibility.** A memory written this turn is **not** recallable later in the same turn - it lands for future sessions only.

## Features

- **Built for coding agents** - memory for Claude Code: the decisions, preferences, and facts a coding session settles, recalled in the next one.
- **Hook-driven** - five lifecycle hooks handle memory operations automatically.
- **LLM-supervised** - the host LLM decides what to remember and forget; a worker model handles enrichment. No LLM judges a write.
- **Multi-signal recall** - RRF-fused keyword, vector, and recency anchors, reranked by a cross-encoder on longer queries. Results always come back in relevance order.
- **Write once, retire deliberately** - a write adds a row, or replaces the row `replace <id>` names; only `replace` and `supersede` retire a row. A replaced or superseded memory is never deleted: it keeps its content behind `superseded_by`, leaves recall by default, and `memman insights show <id> --history` walks the chain.
- **Operator-only deletion** - a store is uncapped and nothing expires or is pruned on its own. `memman forget <id>` is the only thing that removes a memory; `memman insights review` surfaces transient content for that decision.
- **Pluggable embeddings, per-store sovereignty** - registered providers include `voyage`, `openai` (any OpenAI-compatible endpoint: OpenAI, vLLM, LiteLLM, ...), `openrouter`, and `ollama`. Each store's `meta.embed_fingerprint` is the runtime authority over its embedder, so one process can serve multiple stores with different embedders. Switch online via `memman embed swap` or offline via `memman embed reembed`.
- **Pluggable storage backend** - SQLite by default; Postgres + pgvector via the `memman[postgres]` extra. `memman migrate` copies a store between backends in a single command (idempotent, drain-lock-guarded, dry-run support).
- **External scheduled backups** - `memman backup schedule '<cron>' <dir>` snapshots every store to an external, durable directory (e.g. a Dropbox path) on a cron schedule, online and non-disruptively, with keep-last-N retention. Secrets are excluded from bundles; `memman backup restore` rebuilds a working store after total loss of `~/.memman/`. See [USAGE.md § Backup](docs/USAGE.md#backup).

## Install

> [!IMPORTANT]
> **The API keys belong to memman, not to the agent.** The agent authenticates as it always has - Claude Code runs on its own Claude login, which memman neither reads nor bills against. The keys below pay for the calls memman makes on its own behalf: the background worker that enriches and embeds each memory, and the two calls recall makes to rank results. A Claude Pro / Max or ChatGPT Plus subscription does **not** cover them, since a chat subscription and the developer APIs are billed separately. Any registered provider works (OpenRouter, OpenAI-compatible endpoints, Voyage, Ollama, ...), and an install running Ollama on both sides needs no key at all. [Where keys are needed](#where-keys-are-needed) breaks this down per command.

```bash
pipx install memman
# or, with the optional Postgres backend:
# pipx install 'memman[postgres]'
memman install
```

In a TTY, the install wizard prompts for an LLM endpoint URL and an embedding provider, then collects the keys those two need (masked input). It does not ask for the reranker's key unless Voyage embeddings were chosen; set `MEMMAN_VOYAGE_API_KEY` afterwards, or turn reranking off - see [Reranker](#reranker). Pre-seeded defaults are accepted with Enter, but any registered provider works equally well - see [Provider setup](#provider-setup) below for the full list. Loopback LLM endpoints (Ollama, local vLLM/LiteLLM) may leave the API key blank. Headless / CI installs need the keys exported (or pre-written into `~/.memman/env`) and should pass `--no-wizard`. After install, the env file at `~/.memman/env` (mode 0600) is the canonical source of truth; runtime never reads the shell for installable settings. Change a setting with `memman config set KEY VALUE`. See [CONTRIBUTING.md § Variable reference](CONTRIBUTING.md#variable-reference) for the full key list and [USAGE.md § Configuration](docs/USAGE.md#configuration) for the precedence model.

### Provider setup

memman talks to three external services: an **LLM** (enrichment), an **embedding provider** (vector search), and a **reranker** (final ordering of recall results). All three are pluggable; the embed side is also per-store via `meta.embed_fingerprint`.

#### Where keys are needed

The agent's own login is never involved. These are the calls memman makes on its own behalf:

| What runs                                                | Where           | Key it needs                                          | Without that key                                                   |
| -------------------------------------------------------- | --------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| `memman remember`                                        | inside the turn | none                                                  | works - the only verb that opens no store                          |
| every verb that opens a store, `recall --basic` included | inside the turn | the active embedding provider's key (none for Ollama) | the command stops: `MEMMAN_VOYAGE_API_KEY is not set in <dir>/env` |
| `recall` - reorder the top results                       | inside the turn | `MEMMAN_VOYAGE_API_KEY`                               | recall keeps its earlier order, and logs why                       |
| enrichment                                               | worker          | `MEMMAN_LLM_API_KEY` (blank for a local LLM)          | the row still stores, unenriched                                   |
| embedding                                                | worker          | the active embedding provider's key                   | no memory is ever stored                                           |
| `embed reembed`, `embed swap`, `migrate`                 | on demand       | the active embedding provider's key                   | the command stops with an error                                    |

Three things worth knowing before picking a provider:

- **One key gates almost everything: the one named by `MEMMAN_EMBED_PROVIDER`.** Opening a store constructs that provider's client, and the client demands its key before any query runs, so `recall`, `forget`, `replace`, `insights show`, `graph`, and `status` all exit with `MEMMAN_VOYAGE_API_KEY is not set in <dir>/env` when it is absent. `recall --basic` exits the same way - skipping the vector path does not skip opening the store. `memman remember` is the one exception, since it appends to the queue without opening a store. An Ollama embedder needs no key and satisfies the check for free.
- **The key must sit in `~/.memman/env`, not in the shell.** Runtime reads that file alone, so an exported variable does nothing. `memman config set KEY VALUE` writes it.
- **Reranking asks for a Voyage key whatever the embedding provider is.** It is on by default, and Voyage is the only reranker shipped, so an install on `openai` or `ollama` embeddings still wants `MEMMAN_VOYAGE_API_KEY`. Set it, or turn reranking off with `memman config set MEMMAN_RERANK_ENABLED false` (per store: `MEMMAN_RERANK_ENABLED_<store>`). This is the one key whose absence degrades rather than stops: every recall of more than two words silently keeps the order it had before reranking.

#### LLM providers

The LLM client speaks OpenAI-compatible `/chat/completions` against whichever endpoint is configured. Any vendor exposing an OpenAI-compat shim is reachable without code changes.

| Provider                | Endpoint                       | Key (`MEMMAN_LLM_API_KEY`) |
| ----------------------- | ------------------------------ | -------------------------- |
| OpenRouter              | `https://openrouter.ai/api/v1` | `sk-or-...`                |
| OpenAI                  | `https://api.openai.com/v1`    | `sk-...`                   |
| Anthropic (OpenAI shim) | `https://api.anthropic.com/v1` | `sk-ant-...`               |
| Ollama (local)          | `http://localhost:11434/v1`    | blank                      |
| vLLM / LiteLLM          | self-hosted URL                | as required                |

Switching is a one-env-var edit:

```bash
memman config set MEMMAN_LLM_ENDPOINT https://api.openai.com/v1
memman config set MEMMAN_LLM_API_KEY sk-...
```

`MEMMAN_LLM_MODEL` names the model, and memman never switches it on its own. On an OpenRouter endpoint, `memman config models` and the install wizard list up to three candidates in the current model's family - served under zero data retention by a vendor in `MEMMAN_LLM_PROVIDER_ONLY`, inside `MEMMAN_LLM_MAX_INPUT_PRICE` and `MEMMAN_LLM_MAX_OUTPUT_PRICE` - and write the one the operator picks. On any other endpoint the wizard prompts for the slug, and a headless install without one refuses.

#### Embedding providers

Four embed providers are registered. Each store records its active `(provider, model, dim)` triple in `meta.embed_fingerprint` so one process can serve multiple stores fingerprinted to different providers.

| Provider     | Default model            | Key                                                               |
| ------------ | ------------------------ | ----------------------------------------------------------------- |
| `voyage`     | `voyage-3-lite` (512d)   | `MEMMAN_VOYAGE_API_KEY`                                           |
| `openai`     | `text-embedding-3-small` | `MEMMAN_OPENAI_EMBED_API_KEY` + `MEMMAN_OPENAI_EMBED_ENDPOINT`    |
| `openrouter` | `baai/bge-m3` (1024d)    | reuses `MEMMAN_OPENROUTER_API_KEY` + `MEMMAN_OPENROUTER_ENDPOINT` |
| `ollama`     | `nomic-embed-text`       | local; `MEMMAN_OLLAMA_HOST` (default `http://localhost:11434`)    |

Switch on a new install:

```bash
memman config set MEMMAN_EMBED_PROVIDER openai
memman config set MEMMAN_OPENAI_EMBED_API_KEY sk-...
```

Switch a populated store: online via `memman embed swap --to <model> --provider <name>` (resumable, atomic cutover) or offline via `memman embed reembed` (requires `memman scheduler stop`). See [USAGE.md § Embedding operations](docs/USAGE.md#embedding-operations).

#### Reranker

One reranker ships, and it is on by default. It scores the top recall results against the query so the best answer sits first.

| Setting                      | Default         | What it does                                         |
| ---------------------------- | --------------- | ---------------------------------------------------- |
| `MEMMAN_RERANK_ENABLED`      | `true`          | set `false` to skip reranking and its key entirely   |
| `MEMMAN_RERANK_PROVIDER`     | `voyage`        | the only provider registered today                   |
| `MEMMAN_VOYAGE_API_KEY`      | -               | authenticates the reranker, whatever the embedder is |
| `MEMMAN_VOYAGE_RERANK_MODEL` | `rerank-3-lite` | model slug                                           |

Reranking skips itself on queries of two words or fewer, since there is little to reorder.

`pipx install` puts the `memman` binary on the PATH. `memman install` wires integration into Claude Code. The paths it writes:

| Path                                                   | What                                                               | Form                            |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------- |
| `~/.claude/skills/memman/SKILL.md`                     | command reference loaded by the agent                              | symlink into installed package  |
| `~/.claude/hooks/memman/*.sh`                          | five lifecycle hook scripts                                        | symlinks into installed package |
| `~/.claude/settings.json`                              | hook registrations + curated `Bash(memman <verb>:*)` allow entries | JSON merge                      |
| `~/.config/systemd/user/memman-enrich.{timer,service}` | scheduler unit (Linux)                                             | unit files                      |
| `~/Library/LaunchAgents/com.memman.enrich.plist`       | scheduler agent (macOS)                                            | plist                           |
| `~/.memman/env` (mode 0600)                            | canonical config file (API keys + installable knobs)               | created or updated in place     |
| `~/.memman/logs/`                                      | scheduler enrichment worker stdout/stderr                          | directory                       |

Target a specific environment:

```bash
memman install --target claude-code
```

Start a new Claude Code session to activate.

For editable installs and the test suite, see [Development](#development).

## Operation

### Memory shared across sessions

By default, all sessions use the same `default` store - a decision remembered in one session is available in every future session.

### Isolation per project or agent

Use named stores:

```bash
memman store create work        # create a new store
memman store use work           # set as default
MEMMAN_STORE=work memman recall "query"  # or use env var per-process
```

Different agents/processes can use different stores via the `MEMMAN_STORE` environment variable.

### Automatic store selection per directory

Set `MEMMAN_STORE` with a directory-scoped env loader like [direnv](https://direnv.net):

```bash
cd ~/projects/work
echo 'export MEMMAN_STORE=work' > .envrc
direnv allow
```

Every shell, agent, and subprocess started in that directory now resolves to the `work` store. For the full comparison of alternatives (`--store` flag, project `CLAUDE.md` rule, global `memman store use`) and a note on `MEMMAN_DATA_DIR`, see [USAGE.md § Stores](docs/USAGE.md#store-management).

### Customizing behavior

The shipped `guide.md` (behavioral policy) and `SKILL.md` (command reference) live inside the installed package and update on `pipx upgrade memman`. To change behavior, edit the package source (editable installs pick up changes live) or propose a change upstream.

### What `memman remember` does

`memman remember` appends a row to `queue.db` and returns in ~50 ms. The scheduler drains every 60 s; writes become recallable after the next drain. See [Inside Claude Code vs outside](#inside-claude-code-vs-outside).

### Pausing the scheduler

`memman scheduler stop` sets the persistent state to STOPPED and disables the timer on systemd/launchd hosts. While stopped, memman is recall-only: `remember`, `replace`, `supersede`, `unsupersede`, `forget`, and `graph rebuild` exit with `Scheduler is stopped; cannot <verb>`. Resume with `memman scheduler start`. See [USAGE.md § Scheduler](docs/USAGE.md#scheduler) for the full verb list.

## Updating

```bash
pipx upgrade memman
```

Hook scripts and `SKILL.md` are symlinks into the installed package, so they refresh automatically. `guide.md` is read live from the package via `importlib.resources`. A change confined to those assets propagates without re-running `memman install`.

Two things an upgrade does not carry, so re-run `memman install` after every upgrade:

- **Hook registrations.** `~/.claude/settings.json` names the events and tool matchers, and only `memman install` rewrites it. A release that adds, drops, or re-matches a hook leaves the old registration live until then - a dropped hook keeps a settings entry pointing at a symlink whose target the new package no longer ships.
- **The scheduler unit.** Its `ExecStart` line points at the old package path until `memman install` runs again. `make e2e` and `memman doctor` catch unit-file drift.

## Uninstall

```bash
memman uninstall            # remove hooks, skill, settings entries, scheduler unit
pipx uninstall memman       # remove the memman binary
```

Either can run alone. `memman uninstall` never deletes anything under `~/.memman/` - the memory store, the API keys, and the scheduler logs all survive.

## Development

```bash
make dev            # editable Poetry install with dev deps (for running tests)
make test           # unit tests (pytest)
make e2e            # end-to-end test suite
pipx install -e .   # editable pipx install (for wiring Claude Code integration)
memman install      # deploy integration
memman uninstall    # remove integration
```

**Dependencies**: Python 3.11+, Click, httpx, tqdm, numpy. **Keys**: the worker needs whatever the configured LLM endpoint asks for (`MEMMAN_LLM_API_KEY`, blank for a local endpoint) plus the active embedding provider's key. Reranking uses `MEMMAN_VOYAGE_API_KEY`, and skips itself without one. Every side is pluggable with one edit - see [Where keys are needed](#where-keys-are-needed) for what breaks without each key, and [USAGE.md § Configuration](docs/USAGE.md#configuration) for the precedence model.

## Documentation

- [Design & Architecture](docs/DESIGN.md) - philosophy, algorithms, integration design
- [Usage & Reference](docs/USAGE.md) - CLI commands, configuration, embedding support
- [Architecture Diagrams](docs/diagrams/) - system architecture, pipelines, lifecycle management

## License

[MIT](LICENSE)
