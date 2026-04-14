---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle.
---

# memman

`memman` is a CLI on PATH — run commands directly via Bash (e.g., `memman recall ...`).

## Workflow

1. **Remember**: `memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent`
   - Diff is built-in: duplicates skipped, conflicts auto-replaced.
   - Output includes `action` (added/updated/skipped/replaced), `enrichment` (keywords, summary, entities), and `edges_created` (temporal, entity, causal).
   - All edge creation, LLM enrichment, and causal inference run inline before `remember` returns.
   - **Replace**: `memman replace <id> "<new content>"` — deterministic replacement by ID. Inherits metadata (cat/imp/tags/entities/source) from original unless overridden. Carries `access_count` forward.
2. **Link** (manual linking when you identify relationships):
   - Syntax: `memman link <id> <target> --type <causal|semantic> --weight <0-1> [--meta '<json>']`
   - For causal links, pass sub_type via `--meta`: `memman link <id> <target> --type causal --meta '{"sub_type": "causes"}'` (values: `causes`, `enables`, `prevents`)
3. **Recall**: `memman recall "<query>" --limit 10`

## Commands

```bash
memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent
memman link <id1> <id2> --type <type> --weight <0-1> [--meta '<json>']
memman recall "<query>" --limit 10
memman search "<query>" --limit 10
memman replace <id> "<new content>" [--cat] [--imp] [--tags] [--entities] [--source]
memman forget <id>
memman related <id> --edge causal
memman gc --threshold 0.4
memman gc --keep <id>
memman gc --review
memman graph rebuild
memman graph reindex
memman status
memman doctor
memman log [--since 7d] [--group-by operation] [--stats]
memman store list
memman store create <name>
memman store set <name>
memman store remove <name>
```

## Guardrails

- Never run `remember` or `link` in the main conversation — always delegate to a sub-agent.
- Do not store secrets, passwords, or tokens.
- Categories (`--cat`):
  - `preference` — user-stated likes, dislikes, style choices ("I prefer X over Y")
  - `decision` — architectural/design choices with rationale ("chose X because Y")
  - `fact` — discovered truths about systems, tools, domains
  - `insight` — non-trivial conclusions from multi-source reasoning
  - `context` — background about user, project, environment
- Edge types: `temporal` · `semantic` · `causal` · `entity`
- Max 8,000 chars per insight.

## Execution

- **Batching**: at decision boundaries, accumulate multiple memories in a single
  sub-agent invocation. Provide a bulleted list of what to store (content, category,
  importance, entities, create/update intent). Do not write CLI commands — the
  sub-agent reads this skill doc and executes the correct commands.
- **Quality warnings**: after `remember` runs, check `quality_warnings` in the output.
  If warnings are present, evaluate whether to revise (trim transient content and
  re-run) or accept if the warning is a false positive.
