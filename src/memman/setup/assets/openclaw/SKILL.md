---
name: memman
description: "Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle."
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      bins: ["memman"]
---

# memman

`memman` is a CLI on PATH — invoke commands directly via the `exec`
tool. Memory is organized into typed insights and a graph of edges
between them. In this OpenClaw integration, all enrichment runs inline
before `remember` returns (no background worker).

## Storing what you learn

Store one self-contained fact per call. Pick the most accurate `--cat`.

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --tags "t1,t2" --entities "e1,e2" --source agent
```

Categories: `preference` · `decision` · `fact` · `insight` · `context`.
Importance is 1 (passing mention) to 5 (architectural / strong preference).

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
unambiguous. Add `--cat` or `--source` to filter.

Fast token-only lookup that skips LLM expansion:

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
memman insights protect <id>          # boost retention
memman insights candidates            # list low-retention candidates
memman insights review                # scan for content quality issues
```

`insights candidates` and `insights review` only surface candidates —
neither deletes anything. Use `forget <id>` to actually remove.

## Working with relationships

```bash
memman graph link <src> <tgt> --type semantic --weight 0.85
memman graph link <src> <tgt> --type causal --weight 0.8 \
    --meta '{"sub_type": "causes"}'
memman graph related <id> --depth 2
```

Causal `sub_type` values: `causes` · `enables` · `prevents`.

## Inspecting the system

```bash
memman status                         # insight count, store
memman doctor                         # health check
memman log list [--since 7d --stats]  # operation audit log
```

## Guardrails

- Use the `exec` tool to run memman commands.
- Never store secrets, passwords, or tokens.
- Max 8,000 characters per insight; chunk longer content.
- One self-contained fact per `remember` call.
