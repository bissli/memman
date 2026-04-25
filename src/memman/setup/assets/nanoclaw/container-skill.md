---
name: memman
description: Persistent graph-based memory. Recall context before responding, remember insights after. Each group has private memory; global memory is read-only.
---

# memman — Persistent Memory

`memman` is a CLI on PATH inside the container. Memory is organized into
typed insights and a graph of edges between them. In this NanoClaw
integration, all enrichment runs inline before `remember` returns.

## Memory stores

- **Private** (default): per-group, read-write. All writes go here.
- **Global**: shared across all groups, read-only. Append `--store global --readonly` to read it.

Never write to the global store — the mount is read-only.

## Recall — before responding

**Default: recall on every new user message**, unless ALL of these apply:
- Direct follow-up within a topic already fully in context
- No reference to past sessions, decisions, or preferences
- No knowledge dependency beyond the current conversation

```bash
memman recall "<query>" --limit 5
memman recall "<query>" --store global --readonly --limit 5
```

Craft a focused, keyword-rich query — do not pass the raw user prompt.

## Remember — after responding

Run this decision tree after every substantive response:

**Step 1 — Does this exchange contain any of these?**
  a) User directive — preference, decision, correction, explicit "remember this"
  b) Reasoning conclusion — non-trivial judgment from multi-source synthesis
  c) Durable observed state — system fact, environment detail, architectural finding
  → No to all → STOP.

**Step 2 — Does a highly overlapping memory already exist?**
  → Yes, incremental new info → UPDATE (merge into existing)
  → Yes, but contradicts/supersedes → REPLACE
  → No significant overlap → CREATE

**Step 3 — Is it worth storing?**
  Rebuilding from scratch costs more than storing + recalling?
  - Single-query public facts → No
  - Multi-source synthesis with non-obvious conclusions → Yes
  - User-specific context no search engine can recover → Yes
  → No → STOP.

**What to store**: conclusions and user-specific context, not raw facts.

## Storing what you learn

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --tags "t1,t2" --entities "e1,e2" --source agent
```

Categories: `preference` · `decision` · `fact` · `insight` · `context`.

Correct an existing insight by ID:

```bash
memman replace <id> "<new content>"
```

`replace` inherits metadata from the original unless overridden.

## Recalling and inspecting

```bash
memman recall "<query>" --limit 10                    # smart recall
memman recall "<keyword>" --basic                      # fast token-only
memman insights show <id>                              # read by ID
```

Add `--intent WHY|WHEN|ENTITY` to bias ranking when intent is unambiguous.

## Forgetting and protecting

```bash
memman forget <id>                    # soft-delete
memman insights protect <id>          # boost retention
memman insights candidates            # list low-retention candidates
memman insights review                # scan content quality issues
```

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
memman log list [--since 7d]          # operation audit log
```

## Guardrails

- Never store secrets, passwords, or tokens.
- Never write to the global store — it is mounted read-only.
- Max 8,000 characters per insight.
- One self-contained fact per `remember` call.
