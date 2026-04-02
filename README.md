# Mnemon

**LLM-supervised persistent memory for AI agents.**

[![CI](https://github.com/bissli/mnemon/actions/workflows/ci.yml/badge.svg)](https://github.com/bissli/mnemon/actions/workflows/ci.yml)
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

## Quick Start

### Install

**From source (Poetry)**:

```bash
git clone https://github.com/bissli/mnemon.git && cd mnemon
make install
```

**Development**:

```bash
git clone https://github.com/bissli/mnemon.git && cd mnemon
make dev
```

### Claude Code

```bash
mnemon setup
```

`mnemon setup` auto-detects Claude Code, then interactively deploys skill, hooks, and behavioral guide. Start a new session — memory is active.

### [OpenClaw](https://github.com/openclaw/openclaw)

```bash
mnemon setup --target openclaw --yes
```

Deploys skill, hook, plugin, and behavioral guide to `~/.openclaw/`. Restart the gateway to activate.

### [NanoClaw](https://github.com/qwibitai/nanoclaw)

NanoClaw runs agents inside Linux containers. Use the `/add-mnemon` skill to integrate:

1. Install mnemon on the host (see above)
2. In your NanoClaw project, run `/add-mnemon` — Claude Code will modify the Dockerfile, add a container skill, and set up volume mounts
3. Each WhatsApp group gets its own isolated memory store, with optional global shared memory (read-only)

The skill is available at `.claude/skills/add-mnemon/` in the NanoClaw repo.

### Uninstall

```bash
mnemon setup --eject
```

## How It Works

Once set up, memory operates transparently via Claude Code's [hook system](https://docs.anthropic.com/en/docs/claude-code/hooks):

```
Session starts
    │
    ▼
  Prime (SessionStart) ─── prime.sh ──→ load guide.md (memory execution manual)
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

You don't run mnemon commands yourself. The agent does — driven by hooks and guided by the skill and behavioral guide.

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
mnemon store create work        # create a new store
mnemon store set work           # set as default
MNEMON_STORE=work mnemon recall "query"  # or use env var per-process
```

Different agents/processes can use different stores via the `MNEMON_STORE` environment variable.

**Local or global mode?**
`mnemon setup` defaults to **global** (user-wide `~/.claude/`), activating mnemon across all projects. **Local** (project-scoped `.claude/`) can be selected interactively.

**How do I customize the behavior?**
Edit `~/.mnemon/prompt/guide.md`. This file controls when the agent recalls memories and what it considers worth remembering. The skill file (`SKILL.md`) is auto-deployed and should not need manual editing.

**What is sub-agent delegation?**
The host model decides *what* to remember, then delegates the actual `mnemon remember` execution to a lightweight sub-agent. This saves tokens and keeps memory operations out of the main context.

## Configuration

See [Usage & Reference](docs/USAGE.md#configuration) for all environment variables, API keys, and optional overrides.

## Development

```bash
make dev            # editable install with dev deps
make test           # unit tests (pytest)
make e2e            # end-to-end test suite
make install        # production install (~/.local/share/mnemon/venv)
mnemon setup     # interactive setup
mnemon setup --eject  # remove all integrations
```

**Dependencies**: Python 3.11+, Click, httpx, tqdm. **Required**: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`.

## Documentation

- [Design & Architecture](docs/DESIGN.md) — philosophy, algorithms, integration design
- [Usage & Reference](docs/USAGE.md) — CLI commands, embedding support, architecture overview
- [Architecture Diagrams](docs/diagrams/) — system architecture, pipelines, lifecycle management

## References

- **MAGMA** — Jiang et al. [A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236). 2025. Four-graph model (temporal, entity, causal, semantic) with intent-adaptive retrieval and beam search traversal.
- **RRF** — Cormack, Clarke & Buttcher. [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114). SIGIR 2009. Multi-signal anchor fusion with k=60.

## License

[MIT](LICENSE)
