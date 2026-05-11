# memman

**LLM-supervised persistent memory for AI agents.**

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### Memory categories

| Category     | Captures                                | Example                                |
| ------------ | --------------------------------------- | -------------------------------------- |
| `preference` | User-stated likes, dislikes, style      | "Prefers snake_case, dislikes ORMs"    |
| `decision`   | Architectural choices with rationale    | "Chose SQLite — zero deps, embeddable" |
| `fact`       | Durable truths about systems/domains    | "API rate limit is 100 req/s"          |
| `insight`    | Conclusions from multi-source reasoning | "Beam search outperforms BFS here"     |
| `context`    | Project background, user environment    | "Monorepo, deploys to AWS ECS"         |

See [Design & Architecture](docs/DESIGN.md) for details.

## Install

```bash
pipx install git+https://github.com/bissli/memman.git
# or, with the optional Postgres backend:
# pipx install 'git+https://github.com/bissli/memman.git#egg=memman[postgres]'
memman install
```

In a TTY, the install wizard prompts (with masked input) for `OPENROUTER_API_KEY` and `VOYAGE_API_KEY` if they are not already in the env file or the shell. Headless / CI installs need both keys exported (or pre-written into `~/.memman/env`) and should pass `--no-wizard`. After install, the env file at `~/.memman/env` (mode 0600) is the canonical source of truth; runtime never reads the shell for installable settings. Change a setting with `memman config set KEY VALUE`. See [CONTRIBUTING.md § Variable reference](CONTRIBUTING.md#variable-reference) for the full key list and [USAGE.md § Configuration](docs/USAGE.md#configuration) for the precedence model.

`pipx install` puts the `memman` binary on your PATH. `memman install` wires integration into Claude Code, [OpenClaw](https://github.com/openclaw/openclaw), and/or [NanoClaw](https://github.com/qwibitai/nanoclaw):

- skill file symlinked into `~/.claude/skills/memman/SKILL.md` (or equivalent)
- lifecycle hook scripts symlinked into `~/.claude/hooks/memman/`
- `~/.claude/settings.json` hook registrations and `Bash(memman:*)` permission
- scheduler unit (systemd timer on Linux, launchd agent on macOS)
- `~/.memman/logs/` directory (scheduler enrichment worker stdout/stderr)

Target a specific environment:

```bash
memman install --target openclaw
memman install --target claude-code
```

For NanoClaw (agents inside Linux containers), install memman on the host as above, then run the `/add-memman` skill in your NanoClaw project — it modifies the Dockerfile, adds a container skill, and wires volume mounts. Each WhatsApp group gets its own isolated store, with optional global shared memory (read-only).

Start a new Claude Code session (or restart the OpenClaw gateway) to activate.

For editable installs and the test suite, see [Development](#development).

## Updating

```bash
pipx upgrade memman
```

Hook scripts and `SKILL.md` are symlinks into the installed package, so they refresh automatically. `guide.md` is read live from the package via `importlib.resources`. Asset-only changes propagate without re-running `memman install`.

## Uninstall

```bash
memman uninstall            # remove hooks, skill, settings entries, scheduler unit
pipx uninstall memman       # remove the memman binary
```

Either can run alone. `memman uninstall` never deletes anything under `~/.memman/` — your memory store, API keys, and scheduler logs all survive.

## How it works

Once installed, the agent runs memman, not the user. Claude Code [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) (or, for OpenClaw, a `before_prompt_build` plugin) fire on session start, prompt submit, and stop; each reminds the agent to recall before responding and remember after.

Six hook scripts drive the Claude Code lifecycle:

| Hook script      | Event                       | Role                                                          |
| ---------------- | --------------------------- | ------------------------------------------------------------- |
| `prime.sh`       | `SessionStart`              | loads the behavioral guide; surfaces post-compact recall hint |
| `user_prompt.sh` | `UserPromptSubmit`          | reminds the agent to recall before answering                  |
| `stop.sh`        | `Stop`                      | reminds the agent to evaluate "remember?" after responding    |
| `task_recall.sh` | `PreToolUse` (Task)         | reminds the agent to recall before sub-agent delegation       |
| `compact.sh`     | `PreCompact`                | drops a flag so the next `SessionStart` re-recalls context    |
| `exit_plan.sh`   | `PreToolUse` (ExitPlanMode) | prompts memory storage before plan-to-execute transitions     |

### Inside Claude Code vs outside

memman splits along a hot-path boundary. The agent's turn does only fast local work; everything slow runs in a background worker.

```
┌─ Inside Claude Code (synchronous) ──┐    ┌─ Background worker ─────────────┐
│                                     │    │                                 │
│  memman recall   (SQLite read)      │    │  drain fires every 60 s under   │
│  memman remember (queue append)     │ →  │  flock on ~/.memman/drain.lock  │
│                                     │    │                                 │
│  No network, no LLM, no embeddings  │    │  LLM extraction → reconcile →   │
│                                     │    │  enrich → embed → edges → DB    │
└─────────────────────────────────────┘    └─────────────────────────────────┘
              │                                          ▲
              └──── queue.db (handoff; not recallable) ──┘
```

| Step                | Where   | Latency       | Notes                                     |
| ------------------- | ------- | ------------- | ----------------------------------------- |
| `memman recall`     | inside  | ~50–200 ms    | local SQLite read; no network             |
| agent reasoning     | inside  | —             | uses recall results as context            |
| `memman remember`   | inside  | ~50 ms        | enqueue only — no LLM, no embed, no edges |
| drain trigger       | outside | every 60 s+   | systemd/launchd timer or serve loop       |
| LLM extraction      | outside | network-bound | external LLM provider call                |
| embedding           | outside | network-bound | external embedding provider call          |
| edge inference + DB | outside | ms            | makes insight visible to *future* turns   |

Two invariants follow from this split:

- **Hot-path discipline.** Nothing the agent runs synchronously hits the network or an LLM. `recall` is a SQLite read; `remember` is a blob append.
- **One-way visibility.** A memory written this turn is **not** recallable later in the same turn — it lands for future sessions only.

### OpenClaw and NanoClaw — same split, different topology

The hot-path/background split is universal across integrations. What changes is **what triggers the recall/remember reminders** and **where the worker runs**:

| Integration | Trigger (inside)                                                          | Worker (outside)                                              | Data location                                                                                 |
| ----------- | ------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Claude Code | six lifecycle hook scripts (`prime.sh`, `user_prompt.sh`, `stop.sh`, ...) | systemd timer (Linux) or launchd agent (macOS) on host        | `~/.memman/data/default/` on host                                                             |
| OpenClaw    | `before_prompt_build` plugin injects recall/remember hints                | same host scheduler as Claude Code (shared)                   | `~/.memman/data/default/` on host                                                             |
| NanoClaw    | three hook scripts inside the container                                   | `memman scheduler serve` as PID 1 inside the *same* container | host `~/.memman/data/{group}/` volume-mounted to container `/home/node/.memman/data/default/` |

OpenClaw sits on the same host as Claude Code: install memman once on the host and the worker is shared. The agent invokes `memman` via the `exec` tool rather than Bash-hook nudges.

NanoClaw moves the hot-path boundary into the container. Agent and worker share one container; the SQLite data dir is volume-mounted from `~/.memman/data/{group}/` (rw) on the host so memory survives container restarts, and an optional `~/.memman/data/global/` is mounted read-only into every container for shared knowledge. Each WhatsApp group gets its own container and its own private store. `queue.db` is intentionally outside the volume mount — pending writes are seconds old and re-driven on the next drain tick, so a restart loses at most one cycle of unprocessed items.

## Features

- **Hook-driven** — six lifecycle hooks handle memory operations automatically.
- **LLM-supervised** — the host LLM decides what to remember and forget; a worker model handles fact extraction, reconciliation, enrichment, causal inference, and query expansion.
- **Four-graph architecture** — temporal, entity, causal, and semantic edges.
- **Intent-aware recall** — graph beam search with RRF fusion; query intent (WHY/WHEN/ENTITY/GENERAL) controls edge weights and result ordering.
- **LLM reconciliation** — each fact classified as ADD/UPDATE/DELETE/NONE against existing memories.
- **Retention lifecycle** — importance decay, access-count boosting, immunity rules, garbage collection.
- **Pluggable embeddings, per-store sovereignty** — Voyage, any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, LiteLLM, ...), or Ollama. Each store's `meta.embed_fingerprint` is the runtime authority over its embedder, so one process can serve multiple stores with different embedders. Switch online via `memman embed swap` or offline via `memman embed reembed`.
- **Pluggable storage backend** — SQLite by default; Postgres + pgvector via the `memman[postgres]` extra. `memman migrate` copies a store between backends in a single command (idempotent, drain-lock-guarded, dry-run support).

## FAQ

**Do different sessions share memory?**
By default, all sessions use the same `default` store — a decision remembered in one session is available in every future session.

**Can I isolate memory per project or agent?**
Use named stores:

```bash
memman store create work        # create a new store
memman store use work           # set as default
MEMMAN_STORE=work memman recall "query"  # or use env var per-process
```

Different agents/processes can use different stores via the `MEMMAN_STORE` environment variable.

**Install scope?**
`memman install` always installs globally at `~/.claude/` (or `~/.openclaw/`). There is no local/project mode.

**How do I customize the behavior?**
The shipped `guide.md` (behavioral policy) and `SKILL.md` (command reference) live inside the installed package and update on `pipx upgrade memman`. memman does not deploy any user-override file under `~/.memman/`. To change behavior, edit the package source (editable installs pick up changes live) or propose a change upstream.

**How does `memman remember` work?**
It appends a row to `queue.db` and returns in ~50 ms. The scheduler drains every 60 s; writes become recallable after the next drain. See [Inside Claude Code vs outside](#inside-claude-code-vs-outside).

**How do I pause the scheduler?**
`memman scheduler stop` sets the persistent state to STOPPED and disables the timer on systemd/launchd hosts. While stopped, memman is recall-only: `remember`, `replace`, `forget`, `graph link`, `graph rebuild`, and `insights protect` exit with `Scheduler is stopped; cannot <verb>`. Resume with `memman scheduler start`. See [USAGE.md § Scheduler](docs/USAGE.md#scheduler) for the full verb list.

**Upgrading?**
After `pipx upgrade memman`, re-run `memman install` to refresh the scheduler unit's `ExecStart` line. `make e2e` and `memman doctor` catch unit-file drift.

## Development

```bash
make dev            # editable Poetry install with dev deps (for running tests)
make test           # unit tests (pytest)
make e2e            # end-to-end test suite
pipx install -e .   # editable pipx install (for wiring Claude Code integration)
memman install      # deploy integration
memman uninstall    # remove integration
```

**Dependencies**: Python 3.11+, Click, httpx, cachetools, tqdm, numpy. **Required at runtime**: an LLM provider API key and an embedding provider API key. Defaults today are `OPENROUTER_API_KEY` for inference and `VOYAGE_API_KEY` for embeddings; both providers are pluggable. See [USAGE.md § Configuration](docs/USAGE.md#configuration) for the current set of supported providers.

## Documentation

- [Design & Architecture](docs/DESIGN.md) — philosophy, algorithms, integration design
- [Usage & Reference](docs/USAGE.md) — CLI commands, configuration, embedding support
- [Architecture Diagrams](docs/diagrams/) — system architecture, pipelines, lifecycle management

## References

- **MAGMA** — Jiang et al. [A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236). 2025. Four-graph model (temporal, entity, causal, semantic) with intent-adaptive retrieval and beam search traversal.
- **RRF** — Cormack, Clarke & Buttcher. [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114). SIGIR 2009. Multi-signal anchor fusion with k=60.

## License

[MIT](LICENSE)
