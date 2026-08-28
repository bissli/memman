---
name: memman
description: Persistent graph-based memory. Recall context before responding, remember insights after. Each group has private memory; global memory is read-only.
---

# memman — Persistent Memory

`memman` is a CLI on PATH inside the container. Memory is organized into
typed insights and a graph of edges between them. PID 1 of the container
is `memman scheduler serve`, which drains the write queue every 60
seconds. From the agent's perspective `remember` returns immediately
with `{action: queued, queue_id}`; the new insight becomes recallable
within the next drain interval.

If `memman scheduler stop` is run inside the container, memman becomes
recall-only and the serve loop exits at its next iteration. Because the
serve loop is PID 1, the container also exits. To resume, restart the
container — do not invoke `scheduler stop` inside a container where
serve is PID 1 unless you intend to terminate the container.

## Memory stores

- **Private** (default): per-group, read-write. All writes go here.
- **Global**: shared across all groups, read-only. Append `--store global` to read it.

Never write to the global store — the mount is read-only.

## Recall — before responding

**Default: recall on every new user message**, unless ALL of these apply:
- Direct follow-up within a topic already fully in context
- No reference to past sessions, decisions, or preferences
- No knowledge dependency beyond the current conversation

```bash
memman recall "<query>" --limit 5
memman --store global recall "<query>" --limit 5
```

The cross-encoder reranker runs by default on multi-token queries
(auto-skipped on 1-2 token queries).

Note: `--store` is a root-group flag and must come **before** the subcommand name (e.g. `recall`).

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
memman remember "<fact>" --cat <category> --imp <1-5> --entities "e1,e2" --source agent --session $SESSION_ID
```

Always pass `--session` with your session id — it links the
session's writes into one temporal chain; a write without it joins
no chain.

Categories: `preference` · `decision` · `fact` · `insight` · `context` · `general`.

Importance is 2 (passing mention) to 5 (architectural / strong preference). The extraction worker silently floors importance at 2 — `--imp 1` becomes `--imp 2`.

A write is not guaranteed to land. The worker drops content its
extractor judges trivial, folds a fact that merely restates a stored
insight into that insight, and deletes a stored insight the new text
contradicts. All three complete as `done`, so the queue reports
success either way. When nothing at all was stored, the write is filed
in the skipped ledger: read it back with `memman scheduler queue
skipped`, which keeps the full content and the reason. A write that
stored even one fact is not filed, so a single folded fact in a
multi-fact write leaves no ledger row. To store text verbatim and
bypass all three, pass `--no-reconcile`.

Correct an existing insight by ID:

```bash
memman replace <id> "<new content>"
```

`replace` inherits metadata from the original unless overridden.

## Recalling and inspecting

```bash
memman recall "<query>" --limit 10                     # smart recall + cross-encoder rerank
memman recall "<keyword>" --basic                      # fast token-only
memman recall "<query>" --limit 10 --brief             # id/category/importance/summary only
memman insights show <id>                              # read by ID
```

`--brief` works on both paths. A row left without a summary falls back
to its content instead, so no row comes back blank. `truncated: true`
marks the rows whose content was cut at 200 characters.

Add `--intent WHY|WHEN|ENTITY` to bias ranking when intent is unambiguous.

Recall always returns rows, so check `meta.sparse` before trusting
them. It is set when the set is empty, when it holds fewer than
`limit // 2` rows, or when no candidate matched a query token. A
`sparse` response means nothing relevant is stored, unless the query
shared no literal token with a row the vector search did find; re-ask
in the store's own words before concluding it is empty.
`--min-score` drops rows whose keyword plus similarity sum is under
the floor (0.0 to 2.0, `0.0` = off, rejected with `--basic`). No value
is worth copying; the usable band depends on the embedder and store.

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
memman doctor                         # health check (sqlite, queue, keys, scheduler, env_completeness)
memman log list [--since 7d]          # operation audit log
```

## Guardrails

- Never store secrets, passwords, or tokens.
- Never write to the global store — it is mounted read-only.
- Max 8,000 characters per insight.
- One self-contained fact per `remember` call.
