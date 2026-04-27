---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle.
---

# memman

`memman` is a CLI on PATH — invoke commands directly via Bash. Memory is
organized into typed insights and a graph of edges between them. Writes
are queued and enriched in the background; reads are intent-aware.

## Storing what you learn

Store one self-contained fact per call. Pick the most accurate `--cat`.

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --tags "t1,t2" --entities "e1,e2" --source agent
```

Categories: `preference` · `decision` · `fact` · `insight` · `context` · `general`.
Importance is 2 (passing mention) to 5 (architectural / strong preference). The extraction worker silently floors importance at 2 — `--imp 1` becomes `--imp 2`.

To correct a stored insight by ID without losing its `access_count` and
edges:

```bash
memman replace <id> "<new content>"
```

`replace` inherits the original's category, importance, tags, entities,
and source unless you override per-flag.

## Recalling what you know

Default recall does LLM query expansion + vector + graph traversal:

```bash
memman recall "<query>" --limit 10
```

Add `--intent WHY|WHEN|ENTITY` to bias the ranking when intent is
unambiguous (cause/effect, timeline, entity-centric). Add `--cat` or
`--source` to filter.

For a fast token-only lookup that skips LLM expansion (cheaper, no
network cost, no ranking by importance):

```bash
memman recall "<keyword>" --basic
```

Read a single insight by ID:

```bash
memman insights show <id>
```

## Forgetting and protecting

```bash
memman forget <id>                    # soft-delete
memman insights protect <id>          # boost retention (opposite of forget)
memman insights candidates            # list low-retention candidates (read-only)
memman insights review                # scan for content quality issues
```

`insights candidates` and `insights review` only surface candidates —
neither deletes anything. Use `forget <id>` to actually remove.

## Working with relationships

The graph holds typed edges between insights. Auto-edges (semantic,
temporal, entity) are computed during enrichment; manual links express
relationships you've identified:

```bash
memman graph link <src> <tgt> --type semantic --weight 0.85
memman graph link <src> <tgt> --type causal --weight 0.8 \
    --meta '{"sub_type": "causes"}'
```

Causal `sub_type` values: `causes` · `enables` · `prevents`.

Traverse from any insight:

```bash
memman graph related <id> --depth 2
memman graph related <id> --edge causal
```

## Inspecting the system

```bash
memman status                         # insight count, store, scheduler state
memman doctor                         # health check (sqlite, queue, keys, scheduler)
```

## Operator commands the agent rarely runs

| Command                                              | Purpose                             |
| ---------------------------------------------------- | ----------------------------------- |
| `memman log list [--since 7d --stats --text]`        | Operation audit log                 |
| `memman scheduler status`                            | Worker state, queue depth, next run |
| `memman scheduler queue list`                        | Inspect deferred-write queue        |
| `memman store list` / `use <name>` / `create <name>` | Multi-store management              |
| `memman config show`                                 | Effective settings (env + on-disk)  |

## Guardrails

- Never store secrets, passwords, or tokens.
- Max 8,000 characters per insight; chunk longer content.
- One self-contained fact per `remember` call. The enrichment worker
  splits multi-fact blobs into atomic insights, so a tight paragraph is
  fine — but write each call with one clear claim in mind.
- `--source agent` when storing on behalf of the user; `--source user`
  is the default for direct user statements.
