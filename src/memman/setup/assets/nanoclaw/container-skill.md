---
name: memman
description: Persistent graph-based memory. Recall context before responding, remember insights after. Each group has private memory; global memory is read-only.
---

# memman — Persistent Memory

## Memory Stores

- **Private** (default): Per-group, read-write. All writes go here.
- **Global**: Shared across all groups, read-only. Use `--store global --readonly` to query.

## Recall — before responding

**Default: recall on every new user message**, unless ALL of these apply:
- Direct follow-up within a topic already fully in context
- No reference to past sessions, decisions, or preferences
- No knowledge dependency beyond the current conversation

To recall:
```bash
memman recall "<query>" --limit 5
# Also check shared memory:
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

## Workflow

1. **Remember**: `memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent`
   - Diff is built-in: duplicates skipped, conflicts auto-replaced.
   - Output includes `action` (added/updated/skipped/replaced), `enrichment` (keywords, summary, entities), and `edges_created` (temporal, entity, causal).
   - All edge creation, LLM enrichment, and causal inference run inline before `remember` returns.
   - **Replace**: `memman replace <id> "<new content>"` — deterministic replacement by ID. Inherits metadata from original unless overridden. Carries `access_count` forward.
2. **Link** (manual linking when you identify relationships):
   - Syntax: `memman link <id> <target> --type <causal|semantic> --weight <0-1> [--meta '<json>']`
   - For causal links, pass sub_type via `--meta`: `memman link <id> <target> --type causal --meta '{"sub_type": "causes"}'` (values: `causes`, `enables`, `prevents`)
3. **Recall**: `memman recall "<query>" --limit 10`

## Commands

```bash
memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent
memman link <id1> <id2> --type <type> --weight <0-1> [--meta '<json>']
memman recall "<query>" --limit 10
memman recall "<query>" --store global --readonly --limit 10
memman search "<query>" --limit 10
memman replace <id> "<new content>" [--cat] [--imp] [--tags] [--entities] [--source]
memman forget <id>
memman related <id> --edge causal
memman gc --threshold 0.4
memman gc --keep <id>
memman graph rebuild
memman graph reindex
memman status
memman doctor
memman log [--limit N] [--since 7d]
```

## Guardrails

- Do not store secrets, passwords, or tokens.
- Never write to the global store — it is mounted read-only.
- Categories: `preference` · `decision` · `insight` · `fact` · `context`
- Edge types: `temporal` · `semantic` · `causal` · `entity`
- Max 8,000 chars per insight.
