# memman

**Persistent memory for Claude Code and Codex.**

memman saves decisions, preferences, and project knowledge so the agent can find them in later sessions. The agent chooses what to remember and when to recall it. memman handles storage, search, and background processing.

Storing and recalling 1,000 memories costs **under $1** on the default models ([Cost](#cost)).

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Install

memman needs Python 3.11+ on Linux, macOS, or Windows under WSL2. The default setup uses SQLite for storage, and one OpenRouter endpoint and key for summaries, embeddings, and reranking, so the install asks for one key.

```bash
pipx install memman
memman install
```

In a terminal, `memman install` runs a wizard that configures the providers, saves every setting in `~/.memman/env`, and installs a background worker. When it detects Claude Code, it also installs the hooks and the agent's instructions. A new Claude Code session loads them.

For Codex, installation adds a memory skill at `~/.agents/skills/memman` and a rules file that lets Codex run the memory commands without a prompt. `memman install --codex` selects Codex explicitly. In Codex, `$memman` invokes the skill. Codex gets no lifecycle hooks, so the skill alone tells the agent when to recall and save. [Codex setup](docs/USAGE.md#codex) covers detection and permissions.

The worker runs as a systemd timer on Linux or a launchd agent on macOS. On a host with neither, the operator sets `MEMMAN_SCHEDULER_KIND=serve` and runs `memman scheduler serve`. [Installation](docs/USAGE.md#install-and-uninstall) covers headless installs and the optional Postgres backend, and [Provider setup](docs/USAGE.md#provider-setup) covers other providers.

On Windows, pipx installs memman inside a WSL2 distribution, and Claude Code or Codex runs in that same distribution. The worker needs systemd, which the distribution turns on with `systemd=true` under `[boot]` in `/etc/wsl.conf`. The data directory stays on the Linux filesystem, because SQLite file locking is unreliable on a `/mnt/c` path. WSL shuts an idle distribution down after the `instanceIdleTimeout` in `%UserProfile%\.wslconfig`, which stops the worker. Queued writes then wait until the next session starts it, and `instanceIdleTimeout=-1` under `[general]` keeps the distribution running.

## Usage

The agent runs these commands through its shell tool. They behave the same in an interactive shell:

```bash
memman remember "The billing service retries failed requests at most three times."
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

[Memory commands](docs/USAGE.md#memory-commands) lists the input rules and output formats.

## How it works

1. **The agent decides.** Claude Code's five lifecycle hooks remind the agent to recall context and save its conclusions. Codex uses the memory skill's guidance. The agent runs every memory command itself. No hook writes a memory.
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

On the default models, summaries for 1,000 stored memories cost $0.40. MongoDB Atlas bills the embedding and rerank calls that OpenRouter forwards, so this table states no price for them.

| Step                                            | Default model               | Cost            |
| ----------------------------------------------- | --------------------------- | --------------- |
| Summaries for 1,000 memories                    | `qwen/qwen3-235b-a22b-2507` | $0.40           |
| Embeddings for 1,000 memories and 1,000 queries | `voyageai/voyage-4-lite`    | billed by Atlas |
| Reranking for 1,000 recalls                     | `voyageai/rerank-3-lite`    | billed by Atlas |

The summary figure rests on these assumptions:

- **Summaries.** 1,200 input and 100 output tokens per memory, at $0.25 in and $1.00 out per million tokens ([OpenRouter pricing](https://openrouter.ai/qwen/qwen3-235b-a22b-2507)).
- **Embeddings.** Memories of 150 - 250 tokens and short queries.
- **Reranking.** Up to 100 candidates per recall, 150 - 250 tokens per query and memory pair. A query of two words or fewer skips reranking.

Longer memories, retries, and other models change the total. memman calls the endpoint with its own API key, and a Claude Pro or Max subscription does not cover those charges. [Provider setup](docs/USAGE.md#provider-setup) lists the key each step uses and explains how to turn reranking off.

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
