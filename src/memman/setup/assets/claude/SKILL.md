---
name: memman
description: Persistent memory CLI for LLM agents. Store facts, recall past knowledge, manage lifecycle.
---

# memman

`memman` is a CLI on PATH. Invoke commands directly via Bash. Memory is
typed insights. A write goes to a queue and a background worker
enriches it.

## Storing what you learn

Store one thought per call, written as one paragraph that opens on
its subject, with no label prefix, list, or line break. A thought is
one thing that can go stale on its own. If half could become false
while the rest stays true, that is two memories, and a paragraph
holding two independent thoughts is two memories. Do not split what
shares a fate: a decision and its reason, a rule and the value it
constrains, a constraint and its rationale become false together, so
they stay together in the one paragraph, however many sentences it
takes. Several calls per turn is normal. Unrelated thoughts are not
merged to look tidy, and one thought is not padded to look
substantial. When unsure, go smaller. A too-small memory stays
retrievable and supersedes cleanly. A too-large one forces a rewrite
and drops clauses.

Pick the most accurate `--cat`.

```bash
memman remember "<thought>" --cat <category>
```

Categories: `preference`, `decision`, `fact`, `insight`, `context`.

### When to write

A user directive - a stated preference, a decision, a correction, or
"remember this" - is stored at once, never deferred, even
mid-conversation. Pure back-and-forth deliberation with no conclusion
yet is deferred: an intermediate conclusion that will shift with
further discussion wastes a write. The stability test for everything
else: would this be worth storing as-is if the exchange stopped here?
If yes, store it. If the next exchange might change it, defer.

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

Category mapping for `--cat`:

- a user-stated preference: `preference`
- an architectural or design decision with rationale: `decision`
- a discovered fact about a system, tool, or domain: `fact`
- a reasoning conclusion synthesized from several sources: `insight`
- background context (project setup, user role, environment):
  `context`

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

A correction of a memory the agent recalled goes through `memman
insights show <id>`, then `memman replace <id> "<new text>"`. Any
other correction names what is no longer true and what is true now,
and goes in with `memman remember`: the stale memory stays in recall
beside it. A settled open question is a correction of the row that
left it open. A memory recording a change names what it replaces. A
later write in the same session carries only the new claim, never an
earlier write restated plus the change.

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

`memman remember` is a fast queue-append. The full pipeline -
summary enrichment, then embedding - runs out-of-band in a
worker the scheduler fires on a timer (systemd on Linux, launchd
on macOS, `memman scheduler serve` in containers).
A newly stored memory is NOT visible to `memman recall` in the current
session; it lands for later sessions.

`memman graph rebuild` re-enriches every stored insight - summary
and vector - after a model or prompt change or to repair partial
enrichment.

The worker stores the text as written, as one memory; no model
rewords, splits, or judges it. Every write lands as its own row: a
second write of the same text is a second row. Nothing a `remember`
does retires a stored memory; only `replace` and `supersede` do.

To correct a stored insight by ID:

```bash
memman replace <id> "<new content>"
```

`replace` inherits the original's category unless `--cat` overrides
it.

The original is superseded, not deleted: it keeps its content behind
`superseded_by`, leaves every recall and listing, and `memman insights
show <id> --history` reads the chain back. When the correction was
stored as its own insight before the link was noticed, link the two
existing rows instead of writing a third:

```bash
memman supersede <old_id> <new_id>
```

## Recalling what you know

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
default on multi-token queries and skips 1-2 token queries.

```bash
memman recall "<query>"
```

The page is one plain-text line per row, best first, and nothing
else:

```
<id8> <score> <created_at> <author> <category> | <text>
```

- `id8`: the first eight characters of the id. Every id-taking
  command (`memman insights show <id8>`, `replace`, `forget`,
  `supersede`) resolves an unambiguous prefix.
- `score`: two decimals. Compare it only against the other scores on
  the same page, never against a fixed number and never across
  pages: the scale belongs to whichever reranker is configured.
- `created_at`: ISO UTC to the second. A timeline question sorts on
  this field rather than reading row order, which is relevance
  order.
- `author`: who wrote the row - `MEMMAN_AUTHOR` from the directory's
  `.envrc`, else the OS username - or `-` when unset.
- `text`: the stored summary, else the first 200 characters of the
  content, with `...` where the cut dropped anything. Every whitespace
  run in either folds to one space. A summarized row carries no
  marker however much its summary left out.

The page is for choosing which row to open, not for reading the rows
themselves: `memman insights show <id8>` reads the rest of any row
worth more than a scan. `--limit` defaults to 20. A wide page costs
little and carries several times the relevant material of a narrow
one, so scan wide and open what earns it. Rows come back in relevance
order at every `--limit`, so the first `n` of a page of `m` are
exactly a page of `n`.

Recall prints rows even when nothing matches: a recency channel seeds
the newest rows as anchors whatever the query. A scored page with no
line therefore means the store holds no memory, not that the query
failed. A full page
is not evidence that anything on it is relevant. A page that looks
thin usually is not, because the store nearly always holds something
bearing on a query drawn from the same work. Judge each row on its
merits against the query and against its siblings on the page.
Report that nothing relevant is stored only when no row bears on the
query. If a paraphrase returns nothing that bears on the query,
re-ask in the store's own words before concluding it is empty.

Rows assert; CLAUDE.md directs. A `decision` row is history with its
rationale, not an instruction to follow now. A row that names a file
path or a symbol is a claim about the code at the row's `created_at`.
Before acting on it, check the path's history since that date with
`git log --since=<created_at> -- <path>` from the project directory.
An empty result means the path did not change OR the path is not in
this repo, since a store can hold rows from several repos; `git log -1
-- <path>` confirms the path exists here before silence is read as
currency.

For a fast token-only lookup that skips vector search and reranking
(cheap, no network cost; rows come back newest first):

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

`remember` and `replace` return a `queue_uuid`. It is stamped on every
insight that write produces, so it answers "where did my write land"
once the scheduler has drained:

```bash
memman insights by-queue <queue_uuid>
```

`count: 0` has two causes: the write is still queued, or it went to a
different store -- the queue is global while this reads one store. A
row that fails every drain attempt stays queued with its text;
`memman doctor` warns on it, and `memman scheduler queue retry <id>`
requeues it.

## Forgetting

```bash
memman forget <id>                    # soft-delete
memman insights review                # scan for content quality issues
```

`insights review` only surfaces rows. It deletes nothing. Use
`forget <id>` to remove. Nothing else deletes: the store is
uncapped and a stored insight persists until someone forgets it.
Supersession (`replace`, `supersede`) hides without
deleting; `memman unsupersede <id>` brings a superseded row back once
its successor has been forgotten.

## Inspecting the system

```bash
memman status                         # insight count, store, scheduler state
memman doctor                         # health check (sqlite, queue, keys, scheduler, env_completeness)
```

## Scheduler controls

memman has a single write path: every `remember` / `replace` enqueues,
and a worker drains the queue. The trigger varies by environment: a
systemd timer on Linux, a launchd agent on macOS, and a long-running
`memman scheduler serve` process inside containers (set
`MEMMAN_SCHEDULER_KIND=serve` and run the command as PID 1).

When the scheduler is stopped, memman is recall-only: every write
exits 1 with `Scheduler is stopped; cannot <verb>. Run 'memman
scheduler start' to enable.` The serve loop polls the state file every
iteration and mid-drain, so a pause takes effect within seconds even
during a long drain.

Drains never overlap: a lock on `~/.memman/drain.lock` gates entry to
the drain. A manual `scheduler trigger` fired while a timer-driven
drain is running logs `drain: another drain is in progress, skipping`
and exits 0.

- `memman scheduler serve [--interval N] [--once]` - long-running
  drain loop (PID 1 in containers). `--interval 0` means continuous:
  drains run back-to-back, with a 100 ms idle backoff when the queue
  is empty.
- `memman scheduler status` - platform, interval, next run, state,
  last heartbeat, and the three worker-log paths.
- `memman scheduler start` - flip state to STARTED (resume drains and
  writes).
- `memman scheduler stop` - flip state to STOPPED (pause drains and
  reject writes).
- `memman scheduler interval --seconds N` - change cadence (min 60 s
  for systemd/launchd; serve mode accepts any value `>= 0`, with `0`
  meaning continuous).
- `memman scheduler trigger` - dispatch a drain on systemd/launchd and
  return at once. It does not wait for the drain, so a `dispatched`
  response means the run started, not that it finished; `memman log
  worker` reports the outcome. Not applicable in serve mode.
- `memman log worker [--errors|--stack]` - tail one worker log target;
  the two flags are mutually exclusive. `--errors` reads `enrich.err`,
  the worker's own ERROR-level tracebacks. `--stack` reads the rotated
  `memman.log` and its backups, the only place a traceback survives
  when the CLI error that reports it is one line. The `enrich` files
  always sit under `~/.memman/logs`; `memman.log` follows `--data-dir`,
  so under a non-default data dir they are in different directories
  and the error message names the exact command to run. `memman
  scheduler status` prints all three paths.

## Operator commands the agent rarely runs

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
  UTF-8 bytes, and never truncate it. Split the text into several
  calls, one thought each. A long literal goes in a repo file, and
  the memory names the path. They also refuse text whose first word
  is the author's name (`author` carries that) or that names a line
  number (`auth.py:88`, `line 88`), which goes stale on the next
  edit: name the file and symbol instead. They refuse text spanning
  several lines: write one thought as one paragraph, and give each
  further thought its own call. They refuse text opening with a
  label of at most three words before a colon and a space (`Fix:`,
  `AWS gotcha:`): open on the subject and write the thought as a
  sentence.
- One thought per `remember` call. The worker stores each call as
  one memory, so a second unrelated subject rides along and goes
  stale with the first; give it its own call.
