# memman

**Persistent memory for Claude Code and Codex.**

memman saves decisions, preferences, and project knowledge so the agent can find them in later sessions. The agent chooses what to remember and when to recall it. memman handles storage, search, and background processing.

Storing and recalling 1,000 memories costs **under $1** on the default models ([Cost](#cost)).

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **The agent decides.** Claude Code's three lifecycle hooks deliver the session guide and remind the agent to recall. The Codex skill does the same. The agent runs every memory command itself, and no hook writes a memory.
- **Queued writes.** `remember` and `replace` return at once. A background worker adds a summary and an embedding and stores the text as written.
- **Hybrid recall.** Keyword matches, vector similarity, and recency, then a reranker over the best candidates.
- **Separate stores and branches.** Named stores keep projects apart. A branch layers over a store and ends with `store merge` or `store drop`.
- **SQLite or Postgres.** SQLite by default, Postgres with `memman[postgres]`.
- **Claude Code and Codex.** One CLI, and the same stores and worker for both.

## Quickstart

memman needs Python 3.11+ on Linux, macOS, or Windows under WSL2.

```bash
pipx install memman
memman install    # one OpenRouter key, a background worker, the hooks or skill
```

```bash
memman remember "The billing service retries failed requests at most three times."
# The worker stores the write on its next run, every 60 seconds by default.
memman recall "billing service retry limit"
memman replace <id> "The billing service retries batch requests four times and other requests three times."
```

[Installation](docs/USAGE.md#install-and-uninstall) covers Codex, headless installs, WSL2, upgrades, and Postgres.

## Cost

On the default models, summaries for 1,000 stored memories cost $0.17. MongoDB Atlas bills the embedding and rerank calls that OpenRouter forwards, so this table states no price for them.

| Step                                            | Default model                    | Cost            |
| ----------------------------------------------- | -------------------------------- | --------------- |
| Summaries for 1,000 memories                    | `~anthropic/claude-haiku-latest` | $0.17           |
| Embeddings for 1,000 memories and 1,000 queries | `voyageai/voyage-4-lite`         | billed by Atlas |
| Reranking for 1,000 recalls                     | `voyageai/rerank-3-lite`         | billed by Atlas |

[Cost](docs/USAGE.md#cost) lists the assumptions behind the figure.

## Documentation

- [Usage and reference](docs/USAGE.md): setup, commands, configuration, and troubleshooting.
- [Design and architecture](docs/DESIGN.md): responsibilities, storage, search, and integration.
- [Contributing](CONTRIBUTING.md): development setup, tests, and schema changes.

## License

[MIT](LICENSE)
