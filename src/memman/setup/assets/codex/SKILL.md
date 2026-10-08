---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, manage lifecycle.
---

# memman

`memman` is a CLI on PATH, run through the shell. A memory is one
stored insight. A write goes to a queue. A
background worker stores and enriches it on its next drain, one
worker run that processes the queued writes.

## Codex

Codex has no memman hooks: nothing injects a guide at session start
or reminds the agent at a new task, a plan, or after compaction. The
recall and write rules below bind whenever this skill is loaded.

The Codex sandbox blocks memman's writes to its data dir and its
provider network calls. `memman install --codex` writes allow rules to
`$CODEX_HOME/rules/memman.rules` for the agent verbs. An allowed verb
runs outside the sandbox with no approval prompt. A `--store` after
the verb (`memman recall --store NAME ...`) matches the verb's rule.
Any other command, or a call with a global option before the verb
(`memman --store NAME recall ...`), matches no rule: the agent
requests outside-sandbox execution through the shell tool
(`sandbox_permissions="require_escalated"`), subject to the session's
approval policy.

## Remembering

Store one thought per call, written as one paragraph that opens on
its subject, with no label prefix, list, or line break. A thought is
one thing that can become outdated on its own. If half could become
false while the rest stays true, that is two memories, and a
paragraph holding two independent thoughts is two memories. Do not
split what becomes false together: a decision and its reason, a rule
and the value it constrains, a constraint and its rationale stay in
the one paragraph, however many sentences it takes. Several calls per
turn is normal. Never merge unrelated thoughts to look tidy, or pad
one to look substantial. When unsure, write the smaller memory. A
too-small memory stays retrievable and replaces cleanly. A too-large
one forces a rewrite and drops clauses.

```bash
memman remember "<thought>"
```

### Timing

Store a user directive - a stated preference, a decision, a
correction, or "remember this" - at once, even mid-conversation.
The stability test for everything else: would this be worth storing
as-is if the exchange stopped here? If yes, store it. If the next
exchange might change it, defer.

After each response, the agent runs this check, biased toward
capturing: when in doubt, store.

Always store:

- a user directive: explicit preference, decision, correction, or
  "remember this"
- a reasoning conclusion: a non-trivial judgment from multi-source
  synthesis
- a durable system or architectural fact discovered this session
- user-specific context no search engine can recover

Store unless trivial:

- a casual preference revealed in passing ("I usually...", "I
  prefer...", "I don't like...")
- a topic explored, with its conclusion or current understanding
- a useful framing or analogy the user offered
- background context about the user's projects, tools, or setup

None of the above: stop.

Never stored, at any tier. The recoverability test: can this fact be
recovered from the project's code, config, IaC state, or cloud
account? If yes, do not store it.

- a bug or issue discovery (its resolution is stored)
- a state snapshot: line numbers, line counts, file sizes, resource
  counts, instance IDs
- a deployment or verification receipt ("all verified", "deployed
  via", "state clean")
- a temporal observation ("currently", "not yet", "TODO", "should be
  changed to")
- an intermediate finding that will shift once the task completes

Mixed content: strip line numbers, counts, sizes, and other state
snapshots; keep the file path or symbol that locates the claim, and
keep the reasoning and conclusions.

### Corrections

A correction goes through `memman replace <id> "<new text>"` on the
row it corrects. `remember` only adds a row and retires nothing, so a
correction written with `remember` leaves the outdated row current in
recall. When that has happened, replace the outdated row with the
current claim.

```bash
memman replace <id> "<new content>"
```

The id is one of:

- the row's id8 on a recall page this session
- the `id` an earlier `remember` or `replace` printed, which
  addresses the agent's own write even while it is still queued

With neither in hand, recall the corrected topic first. When no row
holds the outdated claim, store the correction with `remember`.

The replacement restates every claim of the old row that is still
true, because the old row leaves recall whole. A stored memory is one
claim, which keeps this cheap. When an outdated row holds more
still-true claims than fit one 1,000-byte replacement, the
replacement carries the corrected claim and the other claims go in as
their own `remember` writes in the same turn.

Each of these is a correction:

- a settled open question corrects the row that left it open
- a memory recording a change (a migration ran, a value moved, a step
  finished) corrects the row that stated the old state
- a later write in the same session that changes an earlier claim is
  a `replace` of that write; one that only adds a claim carries that
  claim alone
- the work at hand (the code, a command's output, the user) showing
  a recalled row false corrects that row, with or without a write on
  that topic

`forget` removes a row that should never have existed, or an outdated
row whose every still-true claim a queued write already holds. Every
other correction goes through `replace`. `forget` refuses a queued
id, so a write still queued is forgotten after the drain.

### The `related` list

`remember` replies with `related`: up to three current rows of at
most 1,000 bytes, each as `<id8> <content>`, ranked by the words they
share with the new text. On equal overlap a short row outranks a long
one.

The agent acts on a related row only when one of its sentences is now
false. A row the new text repeats, narrows, or extends stays. The new
row is already queued, so an outdated related row takes one of two
paths:

- `memman forget <id>` when the new row holds every claim the related
  row still has right
- `replace` with only the related row's still-true claims otherwise

Either way the new claim lives in one row. A related row that
replaced a row not yet forgotten refuses `forget` (see Forgetting),
and the refusal names `replace`. Following the refusal leaves the new
claim in two current rows, which duplicates it and loses nothing.
When every related row is outdated, the agent recalls the topic for
the rest.

### The text

The text stores conclusions AND enough context to understand them. It
is self-contained: the agent replaces every "that", "this", and "it"
with its subject before the call. It never opens with a label such
as `Fix:` or `Decision:`, nor with who wrote it or when: `author`
and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

An event date, or the name of another person the fact is about, stays
in the text. The agent runs `memman remember` directly in the current
turn, never through a sub-agent.

A behavioral rule - universal language such as "never", "always", or
"mandatory", with no project-specific entity - goes to the project
AGENTS.md under a `## Directives` section instead of `memman
remember`. The agent creates the section if absent. A directive needs
guaranteed recall, which AGENTS.md gets by loading at session start
and a ranked recall page does not. The user prunes AGENTS.md
periodically, so no confirmation is needed.

### The write pipeline

`memman remember` queues the write, then reads the store to list
`related`. When that read fails the reply carries `related_error`
in its place and the write stays queued. The full pipeline -
summary enrichment, then embedding - runs out of band in a worker
the scheduler starts on a timer (systemd on Linux, launchd on macOS,
`memman scheduler serve` in containers). A write is visible to
`memman recall` once a drain stores it, in this session or a later
one.

`memman enrich` re-enriches every stored insight - summary and
vector - after a model or prompt change or to repair partial
enrichment.

The worker stores the text as written, as one memory; no model
rewords, splits, or judges it. Every write is stored as its own row:
a second write of the same text is a second row.

`replace` refuses a forgotten target. It refuses a target already
replaced, and the message names the current row at the end of its
chain, which is the row to replace instead, or says the chain ends in
a forgotten row. It refuses a target, stored or queued, that a queued
`replace` in the same store already names, and the message quotes
that replace's id and full text, so an agent in another session or
past a compaction sees the first correction. The fix is to `replace`
the queued replace's id with text that keeps its correction and adds
the second: on the drain, a second replace of the same target retires
the first, and any claim only the first text held is lost.

On the drain the worker stores the replacement under the `id` that
`replace` printed and retires the old row, which keeps its content behind
`replaced_by` and drops out of every recall and listing. A `replace`
waits behind its queued target and behind every earlier `replace` in
the store, so the store holds replacements in queue order whatever
retries they take. `memman insights show <id> --history` lists the chain of
replacements through a row, old text included. A wrong correction is
itself an outdated row: replace it with the right text.

## Recall

Recall runs on every new user message and before each new task or
phase, unless all three hold: the message is a direct follow-up within
a topic already in context, it refers to no past session, decision, or
preference, and it depends on nothing outside the current
conversation. Recall always runs before:

- launching an explore, plan, or code agent
- a web search, since stored context sharpens the query
- an architectural or design decision
- writing code that touches a pattern discussed in a past session

The query is focused and keyword-rich, never the raw user prompt.

Recall fuses keyword, vector, and recency anchors, blends keyword,
similarity, and the fused-anchor score, and reranks with a
cross-encoder. The reranker runs by default on queries of three or
more words, stopwords counted.

```bash
memman recall "<query>"
```

The page is one plain-text line per row, best first, and nothing
else:

```
<id8> <score> <created_at> <author> | <text>
```

- `id8`: the first eight characters of the id. Every id-taking
  command (`memman insights show <id8>`, `replace`, `forget`)
  resolves an unambiguous prefix.
- `score`: two decimals. Compare it only against the other scores on
  the same page, never against a fixed number and never across
  pages: the scale belongs to whichever reranker is configured.
- `created_at`: the UTC date, `YYYY-MM-DD`, with no time of day. A
  timeline question sorts on this field, since row order is relevance
  order. Rows from one day tie. `memman insights
  show <id8>` carries the full timestamp when same-day order matters.
- `author`: `MEMMAN_AUTHOR` from the writer's environment, else the
  OS username; `-` when the row has none.
- `text`: the stored summary, else the content. Either folds every
  whitespace run to one space, then keeps its first 200 characters
  with `...` when longer. A summary that fits carries no marker
  however much of the content it left out.

The page is for choosing which row to open: `memman insights show
<id8>` reads the rest of any row worth more than a scan. `--limit`
defaults to `MEMMAN_RECALL_LIMIT`, or 20 when unset. A wide page costs
little and carries more relevant material than a narrow one, so scan
the wide page and open the rows worth reading. Rows come back in
relevance order at every `--limit`, so the first `n` of a page of `m`
are exactly a page of `n`.

Recall prints rows even when nothing matches: a recency channel adds
the newest rows as anchors whatever the query. A scored page with no
line therefore means the store holds no memory, not that the query
failed. A full page is not evidence that anything on it is relevant.
Read a thin-looking page in full: rows from the same work often bear
on the query. Report that nothing relevant is stored only when no
row bears on the query. If a
paraphrase returns nothing that bears on the query, re-ask in the
store's own words before concluding it is empty.

Rows assert, and AGENTS.md directs. A row recording a decision is history
with its rationale, and a rule to follow goes in AGENTS.md. A row that
names a file path or a symbol is a claim about the code at the row's
`created_at`. Before acting on it, check the path's history since
that date with `git log --since=<created_at> -- <path>` from the
project directory. An empty result means the path did not change OR
the path is not in this repo, since a store can hold rows from
several repos. `git log -1 -- <path>` confirms the path exists here
before an empty result is taken to mean the row is current.

For a keyword-only lookup with no query embed and no rerank, newest
first:

```bash
memman recall "<keyword>" --basic
```

`--basic` computes no score, so it prints the same line without the
score field. It has no recency channel: an empty `--basic` page means
no row matched the keyword.

```bash
memman insights show <id>
```

`remember` and `replace` print the `id` the drain stores the row
under, and a `queue_id` that addresses the queue row only. `memman
insights show <id>` answers where a write is stored once the
scheduler has drained. For a write still queued it answers that the
write is still queued and is stored on the next drain. A write that
fails every drain attempt moves to queue status `failed`. For a
failed write, `insights show <id>` and `replace <id>` answer `insight
<id> not found`. A `replace` queued behind a target that then fails
is stored as a plain add with no link, and its result names the
target under `target_gone`. `memman doctor` warns on failed rows.
`memman scheduler queue retry <queue_id>` requeues one, where
`<queue_id>` is the `queue_id` that `remember` or `replace` printed.
Once the retry is stored, the row sits under the printed `id`. When
the retried target of an unlinked `replace` is stored, the topic has
two current rows, so replace one of them.

## Forgetting

```bash
memman forget <id>                    # soft-delete
memman insights review                # scan for content quality issues
```

`insights review` only lists rows. It deletes nothing. Use
`forget <id>` to remove. Nothing else deletes: the store is
uncapped and a stored insight persists until someone forgets it.
`forget` refuses a current row that replaced a row not
yet forgotten, because that row stays retired either way. Once every
row it replaced is forgotten, `forget` takes it.

## Status and health

```bash
memman status                         # store, backend, insight counts, oplog size
memman doctor                         # health checks, each pass/warn/fail
```

Every memory verb refuses a store that does not exist and writes
nothing. Only `memman install`, `memman store create <name>` or
`store branch` makes a store.

## Branches

A branch is an empty local store layered over a parent. It holds one
thread's writes and reads the parent live. The parent gains only a
token and keeps serving every other session. A branch starts and ends
only when the user asks: `store merge` keeps the thread's memories,
`store drop` discards them.

```bash
memman store branch <parent> <label>  # <parent>: the `store` field of `memman status`
memman store merge <branch>
memman store drop <branch>
```

The name is `<parent>__<label>_<4 hex>`, and the label takes no `__`.
`store branch` prints an `instruction` line. Paste it into the notes
the thread's next session reads, and remove it after the merge or
drop. With no such notes, instruction or framework, ask the user how
to carry it to future sessions, or whether future sessions should know
of the branch at all. Every memory verb takes `--store <branch>` while
the line applies, and every subagent brief carries the line. A call
without `--store <branch>` reads and writes the parent. `store use`
refuses a branch, since the active file routes every session on the
host. A branch of a branch is refused.

Recall on the branch ranks its rows with the parent's current rows,
later parent rows included, so the thread needs no parent recall and
no date filter. A `replace` or `forget` of a parent row acts on a copy
in the branch until merge. On a row the parent already replaced it is
refused, and the error names the current row.

`store merge` copies the branch's rows into the parent with dates,
summaries and embeddings, and repeats each `replace` and `forget` made
on a parent row, in one parent transaction. `conflicts` holds parent
rows the two stores retired differently, and the parent keeps its
state. Each entry carries `branch_content`, `parent_content` and
`parent_head`, the parent's current row (null when the parent's chain
ends in a forgotten row). Settle each in the parent with `replace` or
`forget`. Merge refuses while the branch has queued writes and while
either store is mid embed swap or re-embed. Merge and branch recall
refuse when the parent does not hold the branch's token, as after a
parent recreated under the same name, pointed at another database,
or restored from an older backup. After an embed model change on the
parent, branch recall and merge refuse until the branch runs the
embed swap the refusal names. Every refusal names its fix. A merge
that stops part way says to re-run, which finishes it.

`store drop` prints `dropped`, one `{id, content, replaces}` per
current branch row. `replaces` names the parent row it corrects.
Re-save each claim unrelated to the thread with `memman remember
--store <parent>` (`<parent>` from the output), or `memman replace
--store <parent> <replaces>` where set, then a closing row the same
way saying why the thread was dropped. Drop refuses while the branch
has queued writes, and after a merge that stopped part way, which only
a re-run of merge ends.

A branch is local to its host, whatever the parent's backend, and its
recall needs the parent. The same line on another host, or a stale
line after the merge or drop, meets the missing-store refusal: remove
the line.

## Scheduler controls

When the scheduler is stopped, memman is recall-only: every write
exits 1 with `Scheduler is stopped; cannot <verb>. Run 'memman
scheduler start' to enable.` The serve loop polls the state file every
iteration and mid-drain, so a pause takes effect even during a long
drain.

Drains never overlap: a lock on `<data dir>/drain.lock` gates entry to
the drain. A second drain started while one holds the lock logs
`drain: another drain is in progress, skipping` and exits 0. A
`scheduler trigger` during a running systemd drain answers `a
scheduled run is already in progress`.

- `memman scheduler serve [--interval N] [--once]` - long-running
  drain loop. `--interval 0` means continuous:
  drains run back-to-back, with a 100 ms idle backoff when the queue
  is empty.
- `memman scheduler status` - platform, interval, next run, state,
  last run, and the three worker-log paths.
- `memman scheduler start` - set state to STARTED (resume drains and
  writes).
- `memman scheduler stop` - set state to STOPPED (pause drains and
  reject writes).
- `memman scheduler interval --seconds N` - change cadence (min 60 s
  for systemd/launchd). In serve mode it only records the value; the
  serve loop reads `--interval`, then `MEMMAN_INTERVAL`, then 60 when
  it starts.
- `memman scheduler trigger` - dispatch a drain on systemd/launchd and
  return at once. It does not wait, so `dispatched` means the run is
  queued. `memman log worker` reports whether it ran and its outcome.
  Not applicable in serve mode.
- `memman log worker [--errors|--stack]` - tail one worker log target;
  the two flags are mutually exclusive. `--errors` reads `enrich.err`,
  the worker's own ERROR-level tracebacks. `--stack` reads the rotated
  `memman.log` and its backups, the only place that holds a traceback
  when the CLI error that reports it is one line. The `enrich` files
  always sit under `~/.memman/logs`; `memman.log` follows `--data-dir`,
  so under a non-default data dir they are in different directories
  and the error message names the exact command to run. `memman
  scheduler status` prints all three paths.

## Operator commands

| Command                                              | Purpose                            |
| ---------------------------------------------------- | ---------------------------------- |
| `memman log list [--since 7d --stats --text]`        | Operation audit log                |
| `memman scheduler queue list`                        | Inspect deferred-write queue       |
| `memman store list` / `use <name>` / `create <name>` | Multi-store management             |
| `memman config show`                                 | Effective settings (env + on-disk) |

## Guardrails

- Never store secrets, passwords, or tokens.
- `remember` and `replace` refuse text over 1,000 bytes, counted as
  UTF-8 bytes, and never truncate it. An oversized `remember` splits
  into several `remember` calls, one thought each. An oversized
  `replace` keeps the corrected claim and stores the other claims
  with `remember` (Corrections): a `remember` retires nothing, and a
  second `replace` of a target is refused while the first is queued.
  A long literal goes in
  a repo file, and the memory names the path. They also refuse text
  whose first word is the author's name (`author` carries that) or
  that names a line number (`auth.py:88`, `line 88`), which the next
  edit makes wrong: name the file and symbol instead. They refuse
  text spanning several lines, and text opening with a label of at
  most three words before a colon and a space (`Fix:`, `AWS
  gotcha:`).
