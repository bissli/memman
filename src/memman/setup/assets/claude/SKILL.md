---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, manage lifecycle.
---

# memman

`memman` is a CLI on PATH. Invoke commands directly via Bash. A memory
is one stored insight. A write goes to a queue. A background worker
stores and enriches it on its next drain, one worker run that
processes the queued writes.

## Storing memories

Store one thought per call, written as one paragraph that opens on
its subject, with no label prefix, list, or line break. A thought is
one thing that can become outdated on its own. If half could become
false while the rest stays true, that is two memories, and a
paragraph holding two independent thoughts is two memories. Do not
split what becomes false together: a decision and its reason, a rule
and the value it constrains, a constraint and its rationale stay in
the one paragraph, however many sentences it takes. Several calls per
turn is normal. Unrelated thoughts are not merged to look tidy, and
one thought is not padded to look substantial. When unsure, write the
smaller memory. A too-small memory stays retrievable and replaces
cleanly. A too-large one forces a rewrite and drops clauses.

```bash
memman remember "<thought>"
```

### When to write

A user directive - a stated preference, a decision, a correction, or
"remember this" - is stored at once, never deferred, even
mid-conversation. Deliberation that has reached no conclusion is
deferred: an intermediate conclusion that will shift with further
discussion wastes a write. The stability test for everything else:
would this be worth storing as-is if the exchange stopped here? If
yes, store it. If the next exchange might change it, defer.

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
- a topic explored, with its conclusion or current understanding, not
  just the questions
- a useful framing or analogy the user offered
- background context about the user's projects, tools, or setup

None of the above: stop.

Never stored, at any tier. The recoverability test: can this fact be
recovered from the project's code, config, IaC state, or cloud
account? If yes, do not store it.

- a bug or issue discovery: store the resolution, not the problem
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
recall. When that has happened, replace the outdated row with what it
should now say.

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
is self-contained: every "that", "this", and "it" is dereferenced into
its actual subject before the call. It never opens with a label such
as `Fix:` or `Decision:`, nor with who wrote it or when: `author`
and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

An event date, or the name of another person the fact is about, stays
in the text. The agent runs `memman remember` directly in the current
turn, never through a sub-agent.

A behavioral rule - universal language such as "never", "always", or
"mandatory", with no project-specific entity - goes to the project
CLAUDE.md under a `## Directives` section instead of `memman
remember`; the agent creates the section if absent. A directive needs
guaranteed recall, which CLAUDE.md gets by loading every turn and a
ranked recall page does not. The user prunes CLAUDE.md periodically,
so no confirmation is needed.

### The write pipeline

`memman remember` queues the write, then reads the store to list
`related`; when that read fails the reply carries `related_error`
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
a second write of the same text is a second row. Nothing a `remember`
does retires a stored memory; only `replace` does.

To correct a stored memory by id:

```bash
memman replace <id> "<new content>"
```

`<id>` is a current stored row's id or an unambiguous prefix of one,
or the `id` of a write still queued for the same store, so the agent
can replace its own write before the drain runs. `replace` refuses a
forgotten target. It refuses a target already replaced, and the
message names its successor, which is the row to replace instead. It
refuses a target, stored or queued, that a queued `replace` in the
same store already names, and the message quotes that replace's id
and full text, so an agent in another session or past a compaction
sees the first correction. The fix is to `replace` the queued
replace's id with text that keeps its correction and adds the second:
on the drain, a second replace of the same target retires the first,
and any claim only the first text held is lost.

On the drain the replacement is stored under the `id` that `replace`
printed, and the old row is replaced: it keeps its content behind
`replaced_by` and drops out of every recall and listing. A `replace`
waits behind its queued target and behind every earlier `replace` in
the store, so replacements are stored in queue order whatever retries
they take. `memman insights show <id> --history` lists the chain of
replacements through a row, old text included. A wrong correction is
itself an outdated row: replace it with the right text.

## Recall

Recall runs on every new user message and before each new task or
phase, unless all three hold: the message is a direct follow-up within
a topic already in context, it refers to no past session, decision, or
preference, and it depends on nothing outside the current
conversation. Recall always runs before:

- launching an explore, plan, or code agent - recall precedes
  delegation
- starting a new task or switching topics
- a web search, since stored context sharpens the query
- an architectural or design decision
- writing code that touches a pattern discussed in a past session

The query is focused and keyword-rich, never the raw user prompt.

Recall fuses keyword, vector, and recency anchors, blends keyword,
similarity, and the fused-anchor score, and reranks with a
cross-encoder. Every query ranks the same way. The reranker runs by
default on queries of three or more words, stopwords counted.

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
  timeline question sorts on this field rather than on row order,
  which is relevance order. Rows from one day tie. `memman insights
  show <id8>` carries the full timestamp when same-day order matters.
- `author`: who wrote the row - `MEMMAN_AUTHOR` from the directory's
  `.envrc`, else the OS username - or `-` when unset.
- `text`: the stored summary, else the first 200 characters of the
  content, with `...` where the cut dropped anything. Every whitespace
  run in either folds to one space. A summarized row carries no
  marker however much its summary left out.

The page is for choosing which row to open, not for reading the rows
themselves: `memman insights show <id8>` reads the rest of any row
worth more than a scan. `--limit` defaults to `MEMMAN_RECALL_LIMIT`,
or 20 when unset. A wide page costs little and carries more relevant
material than a narrow one, so scan the wide page and open the rows
worth reading. Rows come back in relevance order at every `--limit`,
so the first `n` of a page of `m` are exactly a page of `n`.

Recall prints rows even when nothing matches: a recency channel adds
the newest rows as anchors whatever the query. A scored page with no
line therefore means the store holds no memory, not that the query
failed. A full page is not evidence that anything on it is relevant.
A page that looks thin usually is not, because the store nearly
always holds something bearing on a query drawn from the same work.
Judge each row on its merits against the query and against its
siblings on the page. Report that nothing relevant is stored only
when no row bears on the query. If a paraphrase returns nothing that
bears on the query, re-ask in the store's own words before concluding
it is empty.

Rows assert; CLAUDE.md directs. A row recording a decision is history
with its rationale, and a rule to follow goes in CLAUDE.md. A row that
names a file path or a symbol is a claim about the code at the row's
`created_at`. Before acting on it, check the path's history since
that date with `git log --since=<created_at> -- <path>` from the
project directory. An empty result means the path did not change OR
the path is not in this repo, since a store can hold rows from
several repos; `git log -1 -- <path>` confirms the path exists here
before an empty result is taken to mean the row is current.

For a fast token-only lookup that skips vector search and reranking
(cheap: no query embed and no rerank; rows come back newest first):

```bash
memman recall "<keyword>" --basic
```

`--basic` computes no score, so it prints the same line without the
score field. It has no recency channel: an empty `--basic` page means
no row matched the keyword.

Read a single insight by ID:

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
`replace` retires without deleting: the old row keeps its content
behind `replaced_by`, and `memman insights show <id> --history`
reads it back. `forget` refuses a current row that replaced a row not
yet forgotten, because that row stays retired either way. Once every
row it replaced is forgotten, `forget` takes it.

## Inspecting the system

```bash
memman status                         # store, backend, insight counts, stale_insights, oplog size
memman doctor                         # health check (sqlite, queue, keys, scheduler, env_completeness)
```

Every memory verb refuses a store that does not exist and writes
nothing. Only `memman store create <name>` or `store fork` makes a
store.

## Experiment forks

A fork is a local copy of a store's current rows that takes one
thread's writes while the thread runs; the parent keeps serving every
other session. A fork starts only when the user asks for one, and
ends only when the user asks: `store merge` keeps the thread's
memories, `store drop` discards them.

```bash
memman store fork <parent> <label>    # <parent>: the `store` field of `memman status`
memman store merge <fork>
memman store drop <fork>
```

`store fork` prints an `instruction` line. Paste it into the notes the
thread's next session reads, and remove it after the merge or drop.
With no notes, instruction, or framework saying where the line goes,
ask the user how to carry it to future sessions, or whether future
sessions should know of the fork at all. While the line applies, every
memory verb (`recall`, `remember`, `replace`, `forget`, `insights
show`) takes `--store <fork>`, and every subagent brief carries the
line. A call without the flag reads and writes the parent.

The fork holds every row that was current in the parent at fork time.
For rows the parent gained since, run `memman recall --store <parent>`
and use only rows dated on or after the fork date in the line; a row
dated before that day is already in the fork, or was retired there on
purpose. A parent-only
row the thread finds wrong is corrected with `memman remember --store
<fork>`; after the merge, settle the pair in the parent with `replace`
or `forget`.

`store merge` copies the fork's own rows into the parent with their
dates and repeats each `replace` and `forget` the fork made on an
inherited row. Its `conflicts` list holds inherited rows the fork and
the parent retired differently. The parent keeps its own state for
each, so settle every entry in the parent with `replace` or `forget`.
Merge refuses while the fork has queued writes, and after an embed
model change on the parent until the fork is swapped to the same
model; each refusal names the fix.

`store drop` prints `dropped`, the fork's rows the parent lacks.
Re-save each claim unrelated to the thread with `memman remember
--store <parent>`, taking `<parent>` from the output, then write a
closing row the same way that says the thread was dropped and why.

A fork lives on the host that made it; the same line on another host
meets the missing-store refusal. So does a stale line after the merge
or drop: remove the line.

## Scheduler controls

memman has a single write path: every `remember` / `replace` enqueues,
and a worker drains the queue. The trigger varies by environment: a
systemd timer on Linux, a launchd agent on macOS, and a long-running
`memman scheduler serve` process inside containers.

When the scheduler is stopped, memman is recall-only: every write
exits 1 with `Scheduler is stopped; cannot <verb>. Run 'memman
scheduler start' to enable.` The serve loop polls the state file every
iteration and mid-drain, so a pause takes effect within seconds even
during a long drain.

Drains never overlap: a lock on `<data dir>/drain.lock` gates entry to
the drain. A manual `scheduler trigger` run while a timer-driven
drain is running logs `drain: another drain is in progress, skipping`
and exits 0.

- `memman scheduler serve [--interval N] [--once]` - long-running
  drain loop. `--interval 0` means continuous:
  drains run back-to-back, with a 100 ms idle backoff when the queue
  is empty.
- `memman scheduler status` - platform, interval, next run, state,
  last heartbeat, and the three worker-log paths.
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
  queued; `memman log worker` reports whether it ran and its outcome.
  Not applicable in serve mode.
- `memman log worker [--errors|--stack]` - tail one worker log target;
  the two flags are mutually exclusive. `--errors` reads `enrich.err`,
  the worker's own ERROR-level tracebacks. `--stack` reads the rotated
  `memman.log` and its backups, the only place a traceback survives
  when the CLI error that reports it is one line. The `enrich` files
  always sit under `~/.memman/logs`; `memman.log` follows `--data-dir`,
  so under a non-default data dir they are in different directories
  and the error message names the exact command to run. `memman
  scheduler status` prints all three paths.

## Operator commands

| Command                                              | Purpose                            |
| ---------------------------------------------------- | ---------------------------------- |
| `memman log list [--since 7d --stats --text]`        | Operation audit log                |
| `memman scheduler status`                            | Worker state, next run, log paths  |
| `memman scheduler queue list`                        | Inspect deferred-write queue       |
| `memman store list` / `use <name>` / `create <name>` | Multi-store management             |
| `memman config show`                                 | Effective settings (env + on-disk) |

## Guardrails

- Never store secrets, passwords, or tokens.
- `remember` and `replace` refuse text over 1,000 bytes, counted as
  UTF-8 bytes, and never truncate it. An oversized `remember` splits
  into several `remember` calls, one thought each. An oversized
  `replace` keeps the corrected claim in the replace and stores the
  other claims with `remember`, as the correction rule above says: a
  `remember` retires nothing, and a second `replace` of a target is
  refused while the first is queued, so a split replace leaves the
  outdated row current or meets that refusal. A long literal goes in
  a repo file, and the memory names the path. They also refuse text
  whose first word is the author's name (`author` carries that) or
  that names a line number (`auth.py:88`, `line 88`), which the next
  edit makes wrong: name the file and symbol instead. They refuse
  text spanning several lines: write one thought as one paragraph,
  and give each further thought its own call. They refuse text
  opening with a label of at most three words before a colon and a
  space (`Fix:`, `AWS gotcha:`): open on the subject and write the
  thought as a sentence.
- One thought per `remember` call. The worker stores each call as
  one memory, so a second unrelated subject is stored with the first
  and becomes outdated with it; give it its own call.
