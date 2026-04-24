---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle.
---

# memman

`memman` is a CLI on PATH — run commands directly via Bash (e.g., `memman recall ...`).

## Workflow

1. **Remember**: `memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent`
   - Diff is built-in: duplicates skipped, conflicts auto-replaced.
   - Returns `{"action": "queued", "queue_id": N, "store": ...}` immediately.
   - Full fact extraction, enrichment, embeddings, and causal inference
     run out-of-band (scheduler drains the queue periodically).
   - **Replace**: `memman replace <id> "<new content>"` — deterministic
     replacement by ID. Inherits metadata (cat/imp/tags/entities/source)
     from original unless overridden. Carries `access_count` forward.
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
memman log [--limit N] [--since 7d] [--group-by operation] [--stats]
memman store list
memman store create <name>
memman store set <name>
memman store remove <name>
memman queue list | list-failed | cat <id> | retry <id> | purge --done
memman scheduler status | enable | disable | interval [--seconds N] | logs [--errors] | trigger
```

The `memman queue` group inspects and manages the deferred-write queue
(`~/.memman/queue.db`). The `memman scheduler` group controls the
already-installed background enrichment unit: `enable`/`disable`
pause/resume without touching unit files; `interval --seconds N`
updates the cadence; `logs` tails `~/.memman/logs/enrich.{log,err}`;
`status` shows install state, next run, current interval, and log paths.

## Guardrails

- Do not store secrets, passwords, or tokens.
- Edge types: `temporal` · `semantic` · `causal` · `entity`.
- Max 8,000 chars per insight.
- `--cat` values: `preference`, `decision`, `fact`, `insight`,
  `context` (see the behavioral guide for when to pick which).

## Execution

- **Batching**: at decision boundaries, emit one `memman remember`
  call per distinct memory. The worker's `extract_facts` pass will
  split multi-fact blobs into atomic insights — you can therefore
  group related claims into a single self-contained paragraph if
  that reads better.
- Use `memman queue cat <id>` to inspect raw text and `memman queue
  list` to see processing status.
