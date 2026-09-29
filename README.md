# memman

**Persistent memory for Claude Code.**

memman saves decisions, preferences, and project knowledge so the agent can find them in later sessions. The agent chooses what to remember and when to recall it. memman handles storage, search, and background processing.

Storing and recalling 1,000 memories costs **under $1** on the default models ([Cost](#cost)).

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Install

memman needs Python 3.11+ on Linux or macOS. The default setup uses SQLite for storage, OpenRouter for summaries, and Voyage for embeddings and reranking, so the install asks for an OpenRouter key and a Voyage key.

```bash
pipx install memman
memman install
```

In a terminal, `memman install` runs a wizard that configures the providers, saves every setting in `~/.memman/env`, and installs a background worker. When it detects Claude Code, it also installs the hooks and the agent's instructions. A new Claude Code session loads them.

The worker runs as a systemd timer on Linux or a launchd agent on macOS. On a host with neither, the operator sets `MEMMAN_SCHEDULER_KIND=serve` and runs `memman scheduler serve`. [Installation](docs/USAGE.md#install-and-uninstall) covers headless installs and the optional Postgres backend, and [Provider setup](docs/USAGE.md#provider-setup) covers other providers.

## Usage

The agent runs these commands through its Bash tool. They behave the same in an interactive shell:

```bash
memman remember "The billing service retries failed requests at most three times." --cat decision
# The background worker stores the write on its next run, every 60 seconds by default.
memman recall "billing service retry limit"
```

`remember` prints the memory id immediately. `recall` returns the memory after the worker has stored it. The inspection and correction commands take the id that `remember` or `recall` prints:

```bash
memman insights show <id>
memman replace <id> "The billing service retries batch requests four times and other requests three times."
memman insights show <id> --history
```

Each memory holds one self-contained claim, on one line, within 1,000 UTF-8 bytes.

| Category         | Captures                           | Example                                                   |
| ---------------- | ---------------------------------- | --------------------------------------------------------- |
| `preference`     | User preferences and style         | "Prefers explicit SQL over an ORM."                       |
| `decision`       | Choices and their reasons          | "The cache uses SQLite to avoid running another service." |
| `fact` (default) | Knowledge about systems or domains | "The billing API allows 100 requests per second."         |
| `insight`        | Conclusions drawn from evidence    | "The flaky test fails only when the cache is cold."       |
| `context`        | Project or user background         | "The billing service deploys to AWS ECS."                 |

[Memory commands](docs/USAGE.md#memory-commands) lists the input rules and output formats.

## How it works

1. **The agent decides.** Five lifecycle hooks remind the agent to recall context and to save its conclusions. The agent runs every memory command itself. No hook writes a memory.
2. **Writes enter a queue.** `remember` and `replace` queue the text and return without waiting for a model response. A background worker adds a short summary, creates an embedding, and stores the memory. The text is stored as written.
3. **Recall searches stored memories.** It combines keyword matches, vector similarity, and recency, then reranks the best candidates. `recall --basic` matches words in the text instead.

A memory stays until the agent replaces or forgets it, and nothing expires on its own. A replaced memory leaves recall, and `insights show --history` still lists it. A forgotten memory leaves recall too.

The [design guide](docs/DESIGN.md) explains the architecture and the reasons behind it.

## Separate stores

Named stores hold separate sets of memories:

```bash
memman store create work
memman --store work recall "deployment decisions"  # for one command
memman store use work                             # the default for every process
```

`MEMMAN_STORE=work` in a process's environment selects the store for that process, so two agent sessions on one host can use different stores. `--store` takes precedence over `MEMMAN_STORE`, and `MEMMAN_STORE` takes precedence over the default set by `memman store use`. [Store management](docs/USAGE.md#store-management) covers per-directory selection and moving a store between SQLite and Postgres.

## Cost

On the default models, 1,000 stored memories and 1,000 recalls cost under $1:

| Step                                            | Default model               | Cost          |
| ----------------------------------------------- | --------------------------- | ------------- |
| Summaries for 1,000 memories                    | `qwen/qwen3-235b-a22b-2507` | $0.40         |
| Embeddings for 1,000 memories and 1,000 queries | `voyage-3-lite`             | under $0.01   |
| Reranking for 1,000 recalls                     | `rerank-3-lite`             | $0.30 - $0.50 |
| **Total**                                       |                             | **under $1**  |

The figures rest on these assumptions:

- **Summaries.** 1,200 input and 100 output tokens per memory, at $0.25 in and $1.00 out per million tokens ([OpenRouter pricing](https://openrouter.ai/qwen/qwen3-235b-a22b-2507)).
- **Embeddings.** Memories of 150 - 250 tokens and short queries, at $0.02 per million tokens ([Voyage pricing](https://docs.voyageai.com/docs/pricing)).
- **Reranking.** Up to 100 candidates per recall, 150 - 250 tokens per query and memory pair, at $0.02 per million tokens. A query of two words or fewer skips reranking.

Longer memories, retries, and other providers change the total. memman calls the providers with its own API keys, and a Claude Pro or Max subscription does not cover those charges. [Provider setup](docs/USAGE.md#provider-setup) lists the key each step uses and explains how to turn reranking off.

## Operations and upgrades

```bash
memman status                  # memory counts for the selected store
memman scheduler status        # worker state and last run
memman doctor --text           # health checks, including live provider probes
```

[The usage guide](docs/USAGE.md) covers backups, failed writes, model changes, and configuration. While the scheduler is stopped, memman refuses `remember`, `replace`, and `forget`. `recall` continues to work.

After a package upgrade, `memman install` refreshes the installed hooks, scheduler unit, and settings:

```bash
pipx upgrade memman
memman install
```

Two commands remove the integration and the package:

```bash
memman uninstall
pipx uninstall memman
```

`memman uninstall` keeps the stored memories, the queue, and the logs. [Uninstall](docs/USAGE.md#uninstall) lists the settings it removes.

## Documentation

- [Usage and reference](docs/USAGE.md): setup, commands, configuration, and troubleshooting.
- [Design and architecture](docs/DESIGN.md): responsibilities, storage, search, and integration.
- [Contributing](CONTRIBUTING.md): development setup, tests, and schema changes.

## License

[MIT](LICENSE)
