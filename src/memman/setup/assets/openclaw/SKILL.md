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

`memman` is a CLI on PATH. Invoke commands directly via the `exec`
tool. Memory is typed insights and a graph of edges between them.

OpenClaw is host-resident, so the host's systemd or launchd-driven
worker drains queued writes. `memman remember` returns as soon as the
write is queued; recall reads the latest committed state. If
`memman scheduler stop` is ever run on the host, memman becomes
recall-only: every write returns an error pointing at
`memman scheduler start`.

## Storing what you learn

Store one self-contained fact per call. Pick the most accurate `--cat`.
**Always pass `--session` with the session id**: it links the
session's writes into one temporal chain. A write without it joins
no chain.

```bash
memman remember "<fact>" --cat <category> --imp <1-5> --entity e1 --entity e2 --source agent --session $SESSION_ID
```

Categories: `preference` · `decision` · `fact` · `insight` · `context`.
`--imp` is a sort key for listings and tie-breaks (1-5, default 3),
stored as passed. Pass 5 for a fact the whole system rests on.

A write is not guaranteed to land. The worker drops content its
extractor judges trivial, folds a fact that merely restates a stored
insight into that insight, and supersedes a stored insight the new text
contradicts (the old row keeps its content behind `superseded_by` and
leaves recall). A fact that contradicts several stored insights
supersedes each of them. All three complete as `done`, so the queue
reports success either way. When nothing at all was stored, the write is
filed in the skipped ledger: read it back with `memman scheduler queue
skipped`, which keeps the full content and the reason. A write that
stored even one fact is not filed, so a single folded fact in a
multi-fact write leaves no ledger row. To store text verbatim and bypass
all three, pass `--no-reconcile`.

To correct a stored insight by ID and keep its `access_count` and
edges:

```bash
memman replace <id> "<new content>" --session $SESSION_ID
```

`corroboration_count` (restatements seen, byte-identical or reworded;
shown in recall/get JSON) resets, since the successor is a new row
identity. `replace` inherits the original's category, importance,
entities, and source unless a flag overrides one. `--session` follows
the same rule as `remember`.

The original is superseded, not deleted: it keeps its content behind
`superseded_by`, leaves every recall and listing, and `memman insights
show <id> --history` reads the chain back. When the correction was
stored as its own insight before the link was noticed, link the two
existing rows instead of writing a third:

```bash
memman supersede <old_id> <new_id>
```

## Recalling what you know

Recall: vector + graph traversal + cross-encoder reranker. Reranker
runs by default on multi-token queries and auto-skips on 1-2 token
queries.

```bash
memman recall "<query>" --brief --limit 20 --session <id>
```

Add `--intent WHY|WHEN|ENTITY` to bias the ranking when intent is
unambiguous. Add `--cat` or `--source` to filter (`--source` is an exact match).

Recall returns rows even when nothing matches: a recency channel
seeds the newest insights as anchors regardless. An empty `results`
therefore means the store itself is empty, not that the query failed.
A full page is not evidence that anything on it is relevant. A page
that looks thin usually is not, because the store nearly always holds
something bearing on a query drawn from the same work. Judge each row
on its merits against the query. Each row carries its own `score` and
per-channel `signals`; compare them WITHIN the page and never against
a fixed number, because the scale belongs to whichever reranker is
configured. Report that nothing relevant is stored only when no row
bears on the query.

`--basic` returns before ranking, so it carries no `score` and no
`signals` to judge with at all. If a paraphrase returns nothing that
bears on the query, re-ask in the store's own words before concluding
it is empty.

Rows come back in relevance order at every `--limit`, so the first `n`
of a page of `m` are exactly a page of `n`.

`--min-score` drops rows whose keyword plus similarity sum is under
the floor (0.0 to 2.0, `0.0` = off). `--basic` rejects it: a filter
that quietly did nothing would certify rows it never checked, unlike
`--intent` and `--expand`, which `--basic` names in `meta.ignored`
instead. Leave it off by default: the deep tail of a recall is often
where the useful row sits. No value is worth copying. The usable band
depends on the embedder and the store, so find it by running the
query with and without a floor.

Fast token-only lookup that skips graph and reranking; rows come back
ranked by importance, then recency:

```bash
memman recall "<keyword>" --basic
```

Add `--brief` to cut each insight to `id`, `category`, `importance`,
`created_at`, and `summary`, on both paths; the ranked path keeps the
`score`, `intent`, and `signals` keys around each insight. A row with
no summary falls back to its content, so no row comes back blank.
`truncated: true` marks a fallback row whose content ran past the
cut: the text is a raw 200-character content prefix. The marker's
ABSENCE does not prove the row is whole: a summarized row carries no
marker however much its summary left out. `memman insights show <id>`
reads the rest of any row worth more than a scan.

A brief row carries `created_at`, so a WHEN query reconstructs a
timeline by sorting on that field rather than by reading row order,
which is relevance-ordered on every path.

Read a single insight by ID:

```bash
memman insights show <id>
```

`remember` and `replace` return a `queue_uuid`, stamped on every insight
that write produces. It answers "where did my write land" once the
scheduler has drained:

```bash
memman insights by-queue <queue_uuid>
```

`count: 0` has three causes: the write is still queued, it stored
nothing (see `memman scheduler queue skipped`), or it went to a
different store -- the queue is global while this reads one store.

## Forgetting

```bash
memman forget <id>                    # soft-delete
memman insights review                # scan for content quality issues
```

`insights review` only surfaces rows. It deletes nothing. Use
`forget <id>` to remove. Nothing else deletes: the store is
uncapped and a stored insight persists until someone forgets it.
Supersession (`replace`, `supersede`, a reconcile merge) hides without
deleting; `memman unsupersede <id>` brings a superseded row back once
its successor has been forgotten.

## Working with relationships

The graph holds three edge types between insights: `temporal` (same
session chain, or close in time), `semantic` (similar content), and
`entity` (a shared entity). The worker computes all three during
enrichment. A manual link adds an edge the worker would not find on
its own; `graph related` walks the edges out from an insight:

```bash
memman graph link <src> <tgt> --type semantic --weight 0.85
memman graph link <src> <tgt> --type entity --weight 0.8
memman graph related <id> --depth 2
```

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
- `--source agent` for the agent's own conclusion, a locator (URL,
  script, dataset pull) for imported material; `user`, the default, is
  for the user's words. Recall's `--source` filter is an exact match on
  that string.
