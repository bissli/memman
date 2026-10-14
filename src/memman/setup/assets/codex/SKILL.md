---
name: memman
description: Persistent memory for coding agents. Store lasting findings worth keeping, recall them in later sessions, and replace or forget outdated ones.
---

# memman

`memman` is a CLI on PATH, run through the shell. A memory is one
stored insight. A write goes to a queue. A background worker stores
and enriches it on its next drain, one worker run over the queued
writes.

## Codex

Codex has no memman hooks: nothing injects a guide at session start
or reminds the agent at a new user message or a delegation. The
recall and write rules below bind whenever this skill loads.

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
and the value it constrains, a constraint and its reason stay in the
one paragraph, however many sentences it takes. A turn that settles
several findings worth storing makes several calls. Never merge
unrelated thoughts to look tidy, or pad one to look substantial.
When unsure, write the smaller memory. A too-small memory is still
found and replaced cleanly. A too-large one forces a rewrite and
drops clauses.

```bash
memman remember "<thought>"
```

### Timing

A finding is worth storing when a later session without it would
redo work, repeat a mistake, or ask the user again. Anything else
stays out, however true. Store a user correction, or a claim the
user asks to keep, at once, even mid-conversation. A claim that is a
standing rule of behavior goes to the project AGENTS.md, never to
memman (see The text). Store anything else as soon as it is worth
storing. If the next exchange might change it, defer. After each
response, the agent weighs what is worth storing.

Use cases of memories worth storing:

- a user preference, only when it changes future work and no file
  that loads every session already holds it
- a decision about the work, with its reason
- a lasting fact about a system, project, or tool
- a root cause and its fix, stated as the resulting fact
- an experiment's result and what it means
- an approach ruled out, with the reason

Four pairs, one worth storing and one not:

- Worth storing: "Batching above 16 times out on the gateway, so the
  client caps batches at 16." Not worth storing: "Tried batch sizes
  8, 16 and 32 today."
- Worth storing: "The retry cap is three, because a fourth try only
  adds load." Not worth storing: "Raised the retry cap from two to
  three this session."
- Worth storing: "The staging database refuses connections from the
  CI runners, so integration tests run against a local Postgres
  container." Not worth storing: "The gateway reads its port from
  config/gateway.yaml." The second is true and lasting, but one
  search of the repo finds it.
- Worth storing: "The user wants plain-text error pages from the
  gateway, so every new page is plain text." Not worth storing: "The
  user liked the look of today's error page."

memman holds lasting findings only. The agent never stores any of
these, nor any fact the project's code, config, IaC state, or cloud
account can recover:

- session state: where a session left off, the next step or action,
  progress through steps, cycles, or phases, a step, migration, or
  rollout finishing, uncommitted or working-tree state, a commit or
  release receipt, an open question or a ruling waiting on someone,
  a plan and its steps, open TODOs, the order in which to finish
  work, or text that only says where notes or intermediate artifacts
  are kept or how to resume a session. A finding may carry a `--meta`
  pointer to its work folder. Store the decision, fact, or result
  the work reached.
- a user decision about how to run the session, such as which item
  to work on next or when to stop: store decisions about the work
- the steps that reached a result: store the result and what it
  means
- a symptom with no cause found yet: store the root cause once
  known, and the fix as the resulting fact. A known gotcha that
  someone would repeat is worth storing.
- a state snapshot: line numbers, line counts, file sizes, resource
  counts, instance IDs. Store the file path or symbol that locates
  the claim.
- a deployment or verification receipt ("all verified", "deployed
  via", "state clean"): store what the deployment changed, as the
  fact that holds
- a time-bound observation ("currently", "not yet", "TODO", "should
  be changed to"): store the claim that holds
- a finding that may still change before the task ends: store it
  once the task settles it

Mixed content: strip line numbers, counts, sizes, and other state
snapshots. Keep the file path or symbol that locates the claim, and
keep the reasoning and conclusions.

### Corrections

A correction goes through `memman replace <id> "<new text>"` on the
row it corrects. `remember` only adds a row and retires nothing, so a
correction written with `remember` leaves the outdated row current in
recall. When that has happened, replace the outdated row with the
current claim.

The id is one of:

- the row's id8 on a recall page this session
- the `id` an earlier `remember` or `replace` printed, which
  addresses the agent's own write even while it is still queued

With neither, recall the corrected topic first. When no row holds
the outdated claim, store the correction with `remember`.

The replacement restates every claim of the old row that is still
true, because the old row leaves recall whole. When an outdated row
holds more still-true claims than fit one 1,000-byte replacement, the
replacement carries the corrected claim and the other claims go in as
their own `remember` writes in the same turn.

Each of these is a correction:

- a stored claim that stops being true: the replacement states the
  claim that holds, never the event that changed it
- a later write in the same session that changes an earlier claim is
  a `replace` of that write. A later write that only adds a claim
  carries that claim alone.
- the work at hand (the code, a command's output, the user) showing
  a recalled row false corrects that row, with or without a write on
  that topic

`forget` removes a row that never belonged in the store, or an outdated
row whose every still-true claim a queued write already holds. Every
other correction goes through `replace`. `forget` refuses a queued
id, so a write still queued is forgotten after the drain.

### The `related` list

`remember` replies with `related`: up to three current rows of at
most 1,000 bytes, each as `<id8> <content>`, ranked by the words they
share with the new text. On equal overlap a short row outranks a long
one.

The agent acts on a related row only when one of its sentences is
outdated. A row the new text repeats, narrows, or extends stays. The
new row is already queued, so an outdated related row takes one of
two paths:

- `memman forget <id>` when the new row holds every claim the related
  row still has right
- `replace` with only the related row's still-true claims otherwise

Either way one row holds the new claim. A related row that replaced
a row not yet forgotten refuses `forget` (see Forgetting). The
refusal lists two routes: `replace` to correct the row, or forget
each row it replaced, then the row itself. For an outdated related
row the agent takes the forget route: the replaced rows are already
out of recall, so forgetting them loses nothing. When every related
row is outdated, the agent recalls the topic for the rest.

### The text

The text holds conclusions and enough context to understand them. It
is self-contained: the agent replaces every "that", "this", and "it"
with its subject before the call. It never opens with a label, nor
with who wrote it or when: `author` and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

An event date, or the name of another person the fact is about, stays
in the text. The agent runs `memman remember` directly in the current
turn, never through a sub-agent.

Every write, `remember` or `replace`, states the claim that holds,
never the event that changed it. A fix is stored as the resulting
fact with its reason, "the cache keys on tenant id, because the
per-host key served one tenant's rows to another", never as "the bug
was fixed by keying the cache on tenant id".

A behavioral rule goes to the project AGENTS.md under a
`## Directives` section, never to `memman remember`. Such a rule uses
universal language ("never", "always", "mandatory") and names no
project-specific entity. The agent creates the section if absent. A
directive must load every time, which AGENTS.md does by loading at
session start and a ranked recall page does not. The user prunes
AGENTS.md regularly, so the agent needs no confirmation.

### Meta pairs

`remember` and `replace` take `--meta key=value`, repeatable, for a
short pointer kept beside the text: the work folder a fact belongs
to, or a ticket. A key is lower-case letters and `_`. A value holds
no line break, control character, or format character, and an empty
value sets nothing. The pairs as recall prints them are at most 200
bytes. Over that, or malformed, the CLI refuses and nothing queues.

```bash
memman remember --meta doc=docs/auth.md --meta ticket=ABC-12 "<thought>"
```

A project may fix the key it uses, and its CLAUDE.md or AGENTS.md
names it.

The summary can omit a path written into the text, and the recall
line's cut can remove it. A pair prints whole at the end of the
recall line, after the cut text, and under `meta` in `memman insights
show <id>`. Enrichment, the embedding, and the reranker read the
content alone. The keyword channel of recall matches words in keys
and values, so a recall naming a ticket finds a row whose only
mention of it is the pair. `recall --basic` matches the stored meta
text as well as the content, so `memman recall --basic <path>` finds
a row by its pointer.

`replace` without `--meta` keeps the target's pairs. With any
`--meta`, the successor holds exactly the pairs given, so
`--meta doc=` clears them. A replacement whose target is gone when
the drain stores it is stored as a plain add and keeps no pairs.

### The write pipeline

`memman remember` queues the write, then reads the store to list
`related`. When that read fails the reply carries `related_error`
in its place and the write stays queued. The full pipeline, summary
enrichment and then embedding, runs in the background in a worker
the scheduler starts on a timer (systemd on Linux, launchd on macOS,
`memman scheduler serve` in containers). A write is visible to
`memman recall` once a drain stores it, in this session or a later
one.

`memman enrich` rebuilds the summary and vector of every stored
insight, after a model or prompt change or to repair partial
enrichment.

The worker stores the text as written, as one memory. No model
rewords, splits, or judges it. Every write is stored as its own row:
a second write of the same text is a second row.

To correct a stored memory by id:

```bash
memman replace <id> "<new content>"
```

`<id>` is a current stored row's id or an unambiguous prefix of one,
or the `id` of a write still queued for the same store. `replace`
refuses a forgotten target. It refuses a target already replaced, and
the message names the current row at the end of its chain, which is
the row to replace instead, or says the chain ends in a forgotten
row. It refuses a target, stored or queued, that a queued `replace`
in the same store already names, and the message quotes that
replace's id and full text, so an agent in another session sees the
first correction. The fix is to `replace` the queued replace's id
with text that keeps its correction and adds the second: on the
drain, a second replace of the same target retires the first, and any
claim only the first text held is lost.

On the drain the worker stores the replacement under the `id` that
`replace` printed and retires the old row, which keeps its content
behind `replaced_by` and drops out of every recall and listing. A
`replace` waits behind its queued target and behind every earlier
`replace` in the store, so the store holds replacements in queue
order whatever retries they take. `memman insights show <id>
--history` lists the chain of replacements through a row, old text
included. A wrong correction is itself an outdated row: replace it
with the right text.

## Recall

Recall runs on every new user message, unless all three hold: the
message is a direct follow-up within a topic already in context, it
refers to no past session, decision, or preference, and it depends on
nothing outside the current conversation. Recall always runs before:

- launching an explore, plan, or code agent
- starting a new task or switching topics
- a web search, since stored context sharpens the query
- an architectural or design decision
- writing code that touches a pattern discussed in a past session

The query is focused and keyword-rich, never the raw user prompt.

Recall fuses keyword, vector, and recency anchors and reranks with a
cross-encoder, which runs by default on queries of three or more
words, stopwords counted.

```bash
memman recall "<query>"
```

The page is one plain-text line per row, best first, and nothing
else:

```
<id8> <score> <created_at> <author> | <text> [<meta>]
```

- `id8`: the first eight characters of the id. Every id-taking
  command (`memman insights show <id8>`, `replace`, `forget`)
  resolves an unambiguous prefix.
- `score`: two decimals. Compare it only against the other scores on
  the same page, never against a fixed number and never across
  pages: the scale belongs to whichever reranker is configured.
- `created_at`: the UTC date, `YYYY-MM-DD`, with no time of day. A
  timeline question sorts on this field, since row order is relevance
  order. Rows from one day tie. `memman insights show <id8>` carries
  the full timestamp.
- `author`: `MEMMAN_AUTHOR` from the writer's environment at write
  time, else the OS username, or `-` when the row has none.
- `text`: the stored summary, else the content. Either folds every
  whitespace run to one space, then keeps its first 200 characters
  with `...` when longer. A summary that fits carries no marker
  however much of the content it left out.
- `[<meta>]`: the row's meta pairs as `key=value`, sorted by key,
  after the cut text, so the cut never takes a pair. A value holding
  whitespace, `"`, or `]` prints as a quoted JSON string. A row with no
  pairs ends at `text`.

The page is for choosing which row to open: `memman insights show
<id8>` reads the rest of any row worth more than a scan. `--limit`
defaults to `MEMMAN_RECALL_LIMIT`, or 20 when unset. A wide page costs
little and carries more relevant material than a narrow one. Rows
come back in relevance order at every `--limit`, so the first `n` of
a page of `m` are exactly a page of `n`.

Recall prints rows even when nothing matches: a recency channel adds
the newest rows as anchors whatever the query. A scored page with no
line therefore means the store holds no memory, not that the query
failed. A full page is no proof that anything on it is relevant.
Read a thin-looking page in full: rows from the same work often bear
on the query. Report that nothing relevant is stored only when no row
bears on the query. If a paraphrase returns nothing that bears on the
query, re-ask in the store's own words before concluding it is empty.

Rows assert, and AGENTS.md directs. A row recording a decision is
history with its reason, and a rule to follow goes in AGENTS.md. A
row that names a file path or a symbol is a claim about the code at
the row's `created_at`. Before acting on it, check the path's history
since that date with `git log --since=<created_at> -- <path>` from
the project directory. An empty result means the path did not change
OR the path is not in this repo, since a store can hold rows from
several repos. `git log -1 -- <path>` confirms the path exists here.
Only then does an empty result mean the row is current.

For a plain keyword match on every query word, with no query embed
and no rerank, rows newest first:

```bash
memman recall "<keyword>" --basic
```

`--basic` computes no score, so its line has no score field. It has
no recency channel: an empty `--basic` page means no row matched the
keyword.

Read one insight by id:

```bash
memman insights show <id>
```

`remember` and `replace` print the `id` the drain stores the row
under, and a `queue_id` that addresses the queue row only. `memman
insights show <id>` reads the row once a drain stores it, and until
then answers that the write is still queued. A write that fails every
drain attempt moves to queue status `failed`. For a failed write,
`insights show <id>` and `replace <id>` answer `insight <id> not
found`. A `replace` queued behind a target that then fails is stored
as a plain add with no link, and its result names the target under
`target_gone`. `memman doctor` warns on failed rows. `memman
scheduler queue retry <queue_id>` requeues one. Once the retry is
stored, the row sits under the printed `id`. When the retried target
of an unlinked `replace` is stored, the topic has two current rows,
so replace one of them.

## Forgetting

```bash
memman forget <id>                    # soft-delete
memman insights review                # scan for content quality issues
```

`insights review` lists rows and deletes nothing. Only `forget <id>`
deletes: the store is uncapped, and a row stays until someone forgets
it. `forget` refuses a current row that replaced a row not yet
forgotten, because that row stays retired either way. Once every row
it replaced is forgotten, `forget` takes it.

## Status and health

```bash
memman status                         # store, backend, insight counts, oplog size
memman doctor                         # health checks, each pass/warn/fail; exit 1 on a fail
```

Every memory verb refuses a store that does not exist and writes
nothing. Only `memman install`, `memman store create <name>` or
`store branch` makes a store.

## Branches

A branch is an empty local store layered over a parent. It takes the
same content under the same rules as the parent, and reads the parent
live. The parent gains only a token and keeps serving every other
session. A branch starts and ends only when the user asks: `store
merge` keeps the branch's memories, `store drop` discards them.

```bash
memman store branch <parent> <label>  # <parent>: the `store` field of `memman status`
memman store merge <branch>
memman store drop <branch>
```

The name is `<parent>__<label>_<4 hex>`, and the label takes no `__`.
`store branch` prints an `instruction` line. It routes later sessions
to the branch and stores nothing. Paste it into the notes later
sessions read, never into memman, and remove it after the merge or
drop. With no such notes, ask the user where it goes, or whether
later sessions need to know of the branch at all. Every memory verb
takes `--store <branch>` while the line applies, and every subagent
brief carries the line. A call without `--store <branch>` reads and
writes the parent. `store use` refuses a branch, since the active
file routes every session on the host. `store branch` refuses a
branch as parent.

Recall on the branch ranks its rows with the parent's current rows,
later parent rows included, so a session on the branch needs no
parent recall and no date filter. A `replace` or `forget` of a parent
row acts on a copy in the branch until merge. `forget` takes a row
the parent already replaced too, so a branch drops a topic as one
store does. `replace` of such a row refuses and names the current
row.

`store merge` copies the branch's rows into the parent with dates,
summaries and embeddings, and repeats each `replace` and `forget` made
on a parent row, in one parent transaction. `conflicts` holds parent
rows the two stores retired differently, and the parent keeps its
state. Each entry carries `branch_content`, `parent_content` and
`parent_head`, the parent's current row (null when the parent's chain
ends in a forgotten row). Settle each in the parent with `replace` or
`forget`. Merge refuses while the branch has queued writes and while
either store is in an embed swap or re-embed. Merge and branch recall
refuse when the parent does not hold the branch's token, as after a
parent recreated under the same name, pointed at another database,
or restored from an older backup. After an embed model change on the
parent, branch recall and merge refuse until the branch runs the
embed swap the refusal names. Every refusal names its fix. A merge
that stops part way says to re-run, which finishes it.

`store drop` discards the branch's rows. It prints `dropped`, one
`{id, content, replaces}` per current branch row, and `replaces`
names the parent row it corrects. After a token refusal, merge
cannot run, so the agent re-saves every listed claim worth storing
with `memman remember --store <parent>` (`<parent>` from the
output), or `memman replace --store <parent> <replaces>` where set.
Otherwise the user names the claims to keep, and the agent offers
each experiment result or ruled-out approach the branch reached.
Drop refuses while the branch has queued writes, and after a merge
that stopped part way, which only a re-run of merge ends.

A branch is local to its host, whatever the parent's backend, and its
recall needs the parent. The same line on another host, or a line
left after the merge or drop, gets the missing-store refusal: remove
the line.

## Scheduler controls

When the scheduler is stopped, memman is recall-only: every write
exits 1 with `Scheduler is stopped; cannot <verb>. Run 'memman
scheduler start' to enable.` The serve loop polls the state file every
iteration and during a drain, so a pause takes effect even during a
long drain.

Drains never overlap: `<data dir>/drain.lock` gates the drain, and a
run that finds it held logs `drain: another drain is in progress,
skipping` and exits 0.

- `memman scheduler serve [--interval N] [--once]` - long-running
  drain loop. `--interval 0` means continuous: drains run
  back-to-back, with a 100 ms pause after a drain that claimed
  nothing.
- `memman scheduler status` - platform, interval, next run, state,
  the last drain's summary as `last_run`, and the three worker-log
  paths.
- `memman scheduler start` - set state to STARTED (resume drains and
  writes).
- `memman scheduler stop` - set state to STOPPED (pause drains and
  reject writes).
- `memman scheduler interval --seconds N` - change the interval (min
  60 s for systemd/launchd). In serve mode it only records the value.
  The serve loop reads `--interval`, then `MEMMAN_INTERVAL`, then 60
  when it starts.
- `memman scheduler trigger` - dispatch a drain on systemd/launchd and
  return at once: `dispatched` means the run is queued, and `memman
  log worker` reports its outcome. On systemd, a start refused as
  already running answers `a scheduled run is already in progress`
  instead. Not applicable in serve mode.
- `memman log worker [--errors|--stack]` - tail one worker log. The
  two flags exclude each other. `--errors` reads `enrich.err`, the
  worker's own ERROR-level tracebacks. `--stack` reads the rotated
  `memman.log` and its backups, the only place that holds a traceback
  when the CLI error that reports it is one line. The `enrich` files
  always sit under `~/.memman/logs`. `memman.log` follows `--data-dir`,
  so a CLI error pointing at a stack names the exact command to run.

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
  into several `remember` calls, one thought each, and keeps only
  what is worth storing. An oversized `replace` keeps the corrected
  claim in the replace and stores the other claims with `remember`:
  a `remember` retires nothing, and a second `replace` of a target is
  refused while the first is queued, so a split replace leaves the
  outdated row current or meets that refusal. A long literal goes in
  a repo file, and the memory names the path. They also refuse text
  whose first word is the `MEMMAN_AUTHOR` name (`author` carries
  that) or that names a line number (`auth.py:88`, `line 88`), which
  the next edit makes wrong: name the file and symbol instead. They
  refuse text spanning several lines: write one thought as one
  paragraph, and give each further thought its own call. They refuse
  text opening with a label of at most three words before a colon
  and a space (`Fix:`, `AWS gotcha:`): open on the subject and state
  the lasting finding as a sentence, or store nothing when the text
  is session state.
- A write that passes may still carry `quality_warnings`, advisory
  and never blocking, on phrasing such as: "next step", a numbered
  step or phase finished or done, "cycle N", or "should be changed
  to". `insights review` lists the same warnings
  on stored rows. A warning asks the agent to reread the claim. A
  lasting fact that only matches a pattern stays as written. The
  agent replaces session state with the finding it reached, or
  forgets it when it holds none. `forget` takes a fresh write only
  after the drain.
