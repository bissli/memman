# MemMan

**LLM-supervised persistent memory for AI agents.**

[![CI](https://github.com/bissli/memman/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/memman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### Memory Categories

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
export OPENROUTER_API_KEY=...   # required for the background enrichment worker
export VOYAGE_API_KEY=...       # required for embeddings
memman install
```

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

For NanoClaw (agents running inside Linux containers), install memman on the host as above, then run the `/add-memman` skill in your NanoClaw project — it modifies the Dockerfile, adds a container skill, and wires volume mounts. Each WhatsApp group gets its own isolated store, with optional global shared memory (read-only).

Start a new Claude Code session (or restart the OpenClaw gateway) to activate.

### Development

```bash
git clone https://github.com/bissli/memman.git && cd memman
pipx install -e . --force      # editable install — Python and asset edits go live
memman install
```

Or for running the tests without wiring an integration:

```bash
make dev && make test
```

## Updating

```bash
pipx upgrade memman
```

That's it. Hook scripts and `SKILL.md` are symlinks into the installed package, so they refresh automatically. `guide.md` is read live from the package via `importlib.resources` — `memman guide` prints it on demand, `prime.sh` invokes it at each SessionStart. Asset-only changes propagate without re-running `memman install`.

## Uninstall

```bash
memman uninstall            # remove hooks, skill, settings entries, scheduler unit
pipx uninstall memman       # remove the memman binary
```

Either can run alone. `memman uninstall` never deletes anything under `~/.memman/` — your memory store, API keys, and scheduler logs all survive.

## How It Works

Once set up, memory operates transparently via Claude Code's [hook system](https://docs.anthropic.com/en/docs/claude-code/hooks):

```
Session starts
    │
    ▼
  Prime (SessionStart) ─── prime.sh ──→ memman prime (status + guide + compact hint)
    │
    ▼
  User sends message
    │
    ▼
  Remind (UserPromptSubmit) ─── user_prompt.sh ──→ remind agent to recall & remember
    │
    ▼
  LLM generates response (guided by skill + guide.md rules)
    │
    ▼
  Nudge (Stop) ─── stop.sh ──→ remind agent to remember
    │
    ▼
  (before delegating to sub-agents)
  Recall (PreToolUse) ─── task_recall.sh ──→ remind agent to recall before delegation
    │
    ▼
  (when context compacts)
  Compact (PreCompact) ─── compact.sh ──→ flag file for post-compact recall
    │
    ▼
  (before exiting plan mode)
  ExitPlan (PreToolUse) ─── exit_plan.sh ──→ prompt memory storage before transition
```

Six hooks drive the lifecycle. **Prime** loads the behavioral guide at session start. **Remind** and **Nudge** prompt the agent to recall and remember before/after each response. **Compact** bridges context across compaction via a flag file that Prime detects on the next SessionStart. **Recall** fires before sub-agent delegation. **ExitPlan** prompts memory storage before plan-to-execute transitions.

You don't run memman commands yourself. The agent does — driven by hooks and guided by the skill and behavioral guide.

## Features

- **Hook-driven** — six lifecycle hooks handle all memory operations automatically
- **LLM-supervised** — the host LLM decides what to remember and forget; Haiku handles fact extraction, reconciliation, enrichment, causal inference, and query expansion
- **Four-graph architecture** — temporal, entity, causal, and semantic edges
- **Intent-aware recall** — graph beam search with RRF fusion; query intent (WHY/WHEN/ENTITY/GENERAL) controls edge weights and result ordering
- **LLM reconciliation** — each fact classified as ADD/UPDATE/DELETE/NONE against existing memories
- **Retention lifecycle** — importance decay, access-count boosting, immunity rules, garbage collection
- **Voyage embeddings** — 512-dim vectors via Voyage AI for semantic search and edge creation

## FAQ

**Do different sessions share memory?**
Yes. By default, all sessions use the same `default` store — a decision remembered in one session is available in every future session.

**Can I isolate memory per project or agent?**
Yes. Use named stores to separate memory:

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
It is a fast queue-append (~50 ms). A scheduler-driven worker drains the queue and runs the full pipeline — fact extraction, reconciliation, enrichment, causal inference, embedding — out of band. Three scheduler kinds: a `systemctl --user` timer on Linux hosts, a launchd agent on macOS hosts, and `memman scheduler serve` (a long-running drain loop) on hosts that opt in via `MEMMAN_SCHEDULER_KIND=serve` — typically containers running it as PID 1. Newly stored memories become recallable on the next drain tick (default 60 s).

**How do I pause the scheduler?**
`memman scheduler stop` flips the persistent state to STOPPED. The serve loop polls the state every iteration AND mid-drain, so pause is observed within seconds even during long drains; on systemd/launchd hosts the timer is also disabled. While stopped, memman is **recall-only**: `remember`, `replace`, `forget`, `graph link`, `graph rebuild`, and `insights protect` reject with `Scheduler is stopped; cannot <verb>. Run 'memman scheduler start' to enable.` `memman scheduler start` resumes (and on serve hosts the operator must re-run `memman scheduler serve`). `memman scheduler interval --seconds N` changes the cadence (min 60 s for systemd/launchd; serve mode accepts any non-negative value, with `0` meaning continuous — drains run back-to-back, throttled by a 100 ms idle backoff when the queue is empty). Drains never overlap: the next trigger waits for the previous to finish, enforced by a process-level `fcntl.flock` on `~/.memman/drain.lock`. `memman log worker` prints the tail of the worker's output; add `--errors` for stderr. Logs live at `~/.memman/data/logs/memman.log`.

**Upgrading?**
After `pipx upgrade memman`, re-run `memman install` to refresh the scheduler unit's `ExecStart` line. `make e2e` and `memman doctor` will catch unit-file drift.

## Configuration

See [Usage & Reference](docs/USAGE.md#configuration) for all environment variables, API keys, and optional overrides.

## Development

```bash
make dev            # editable Poetry install with dev deps (for running tests)
make test           # unit tests (pytest)
make e2e            # end-to-end test suite
pipx install -e .   # editable pipx install (for wiring Claude Code integration)
memman install      # deploy integration
memman uninstall    # remove integration
```

**Dependencies**: Python 3.11+, Click, httpx, cachetools, tqdm. **Required**: `OPENROUTER_API_KEY` (LLM inference via OpenRouter's ZDR-enforced routing) and `VOYAGE_API_KEY` (embeddings).

## Documentation

- [Design & Architecture](docs/DESIGN.md) — philosophy, algorithms, integration design
- [Usage & Reference](docs/USAGE.md) — CLI commands, embedding support, architecture overview
- [Architecture Diagrams](docs/diagrams/) — system architecture, pipelines, lifecycle management

## References

- **MAGMA** — Jiang et al. [A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236). 2025. Four-graph model (temporal, entity, causal, semantic) with intent-adaptive retrieval and beam search traversal.
- **RRF** — Cormack, Clarke & Buttcher. [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114). SIGIR 2009. Multi-signal anchor fusion with k=60.

## License

[MIT](LICENSE)
