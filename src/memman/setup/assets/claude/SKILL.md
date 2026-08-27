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
Writes link into one temporal chain by session, which is what WHEN
recall walks. Omit `--session`: it reads `$CLAUDE_CODE_SESSION_ID` by
itself. Pass it only to pin a different id.

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --entities "e1,e2" --source agent
```

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

To correct a stored insight by ID without losing its `access_count` and
edges (`corroboration_count` — the count of exact restatements shown in
recall/get JSON, though not under `--brief` — resets, since the
successor is a new row identity):

```bash
memman replace <id> "<new content>"
```

`replace` inherits the original's category, importance, entities,
and source unless you override per-flag. `--session` does not inherit:
the successor is written into today's chain. It also keeps the
replaced row's edges, so it stays linked to the original's chain as
well, bridging the two.

## Recalling what you know

Recall: vector + graph traversal + cross-encoder reranker. Reranker
runs by default on multi-token queries and auto-skips on 1-2 token
queries.

```bash
memman recall "<query>" --limit 10
```

Add `--intent WHY|WHEN|ENTITY` to bias the ranking when intent is
unambiguous (cause/effect, timeline, entity-centric). Add `--cat` or
`--source` to filter.

For a fast token-only lookup that skips graph and reranking (cheap,
no network cost; rows come back ranked by importance, then recency):

```bash
memman recall "<keyword>" --basic
```

Add `--brief` to cut each insight to `id`, `category`, `importance`,
and `summary`. Use it when scanning for which insight to open rather
than reading the insights themselves. It works on both paths; on the
ranked path the `score`, `intent`, and `signals` keys around each
insight are kept. A row left without a summary falls back to its
content instead, so no row comes back blank. `truncated: true` marks
the rows whose content was cut at 200 characters -- those, and only
those, have more to read via `memman insights show <id>`.

```bash
memman recall "<query>" --limit 10 --brief
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
memman doctor                         # health check (sqlite, queue, keys, scheduler, env_completeness)
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
- No session, no temporal chain. You do not have to pass one:
  `--session` reads `$MEMMAN_SESSION_ID`, then
  `$CLAUDE_CODE_SESSION_ID`. Claude Code exports that second one into
  every Bash call, a subagent's included, with the parent's id. An
  explicit `--session <id>` beats both.
