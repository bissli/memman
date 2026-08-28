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
between them.

OpenClaw is host-resident, so the host's systemd or launchd-driven
worker drains queued writes. `memman remember` returns as soon as the
write is queued; recall reads the latest committed state. If
`memman scheduler stop` is ever run on the host, memman becomes
recall-only — every write returns a clear error pointing at
`memman scheduler start`.

## Storing what you learn

Store one self-contained fact per call. Pick the most accurate `--cat`.
**Always pass `--session` with your session id** — it links the
session's writes into one temporal chain; a write without it joins
no chain.

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --entities "e1,e2" --source agent --session $SESSION_ID
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
recall/get JSON — resets, since the successor is a new row identity):

```bash
memman replace <id> "<new content>" --session $SESSION_ID
```

`replace` inherits the original's category, importance, entities,
and source unless you override per-flag; `--session` follows the
same rule as `remember`.

## Recalling what you know

Recall: vector + graph traversal + cross-encoder reranker. Reranker
runs by default on multi-token queries and auto-skips on 1-2 token
queries.

```bash
memman recall "<query>" --limit 10
```

Add `--intent WHY|WHEN|ENTITY` to bias the ranking when intent is
unambiguous. Add `--cat` or `--source` to filter.

Recall always returns rows, so check `meta.sparse` before trusting
them. It is set when the set is empty, when it holds fewer than
`limit // 2` rows, or when no candidate matched a query token -- a
full page of the nearest unrelated memories. On a
`sparse` response, say nothing relevant is stored; do not reason from
the rows as if they answered the query. It reads literal tokens, so a
paraphrase sharing no word with a row the vector search did find trips
it -- re-ask in the store's own words before concluding it is empty.

`--min-score` drops rows whose keyword plus similarity sum is under
the floor (0.0 to 2.0, `0.0` = off, rejected with `--basic`). Leave it
off by default: the deep tail of a recall is often where the useful
row sits. There is no value worth copying -- the usable band depends
on the embedder and the store, so find it by running the query with
and without a floor.

Fast token-only lookup that skips graph and reranking; rows come back
ranked by importance, then recency:

```bash
memman recall "<keyword>" --basic
```

Add `--brief` to cut each insight to `id`, `category`, `importance`,
and `summary`, on both paths; the ranked path keeps the `score`,
`intent`, and `signals` keys around each insight. A row left without a
summary falls back to its content instead. `truncated: true` marks the
rows whose content was cut at 200 characters -- those, and only those,
have more to read.

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
