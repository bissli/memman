### Recall — before responding

**Default: recall on every new user message AND before each new task/phase**, unless ALL of these apply:
- Direct follow-up within a topic already fully in context
- No reference to past sessions, decisions, or preferences
- No knowledge dependency beyond the current conversation

**Always recall before**:
- Launching explore/plan/code agents — recall BEFORE delegation
- Starting a new task or switching topics
- Web search — stored context sharpens queries
- Making architectural or design decisions
- Writing code that touches patterns discussed in past sessions

To recall: `memman recall "<query>" --limit 5`.
Craft a focused, keyword-rich query — do not pass the raw user prompt.
The cross-encoder reranker runs by default on multi-token queries
(auto-skipped on 1-2 token queries).

On the scored path (no `--basic`) the recall response's `meta`
object carries:
- `hint`: intent-specific reasoning guidance (always present) — use it to
  frame your synthesis of the results
- `ordering`: how results are sorted — `causal_topological` (WHY),
  `chronological` (WHEN), or `score` (ENTITY/GENERAL)
- `reranked`: boolean — true when the cross-encoder rerank stage
  fired (false when query was too short or rerank is disabled for
  this store via `MEMMAN_RERANK_ENABLED_<store>=false`)
- `sparse`: boolean, present only on a low-confidence result set. It is
  set when the set is empty, when it holds fewer than `limit // 2` rows,
  or when no candidate matched a query token. Recall always returns rows
  (a recency channel seeds the newest insights as anchors whether or not
  they match), so a full page of `sparse` results means nothing relevant
  is stored, not that these are the answer. Broaden the query, or accept
  that the store does not hold it

Under `--basic` none of those keys exist. That envelope is
`{basic: true}`, plus `ignored` when a flag was inert:
- `ignored`: flag names, present only when non-empty. `--basic`
  returns before ranking, so `--intent` and `--expand` do nothing
  there and are named rather than obeyed. `--min-score` is not on
  this list -- it is rejected outright, because ignoring a filter
  would leave every returned row looking as though it had cleared a
  floor it never met
- There is no `sparse` under `--basic`. Its ABSENCE there is not
  confidence: `--basic` can return `results: []` and says nothing
  about how well anything matched

`--brief` cuts each row to `id`, `category`, `importance`, and
`summary`, on either path. Use it when scanning for which insight to
open rather than reading the insights themselves. A row with no
summary falls back to the first 200 characters of its content, so no
row comes back blank, and `truncated: true` marks that fallback when
the content ran past the cut. The marker's ABSENCE does not mean you
hold the whole row: a summarized row carries no marker however much
its summary left out. Read a row in full with
`memman insights show <id>`.

`--brief` also drops `created_at`. On the scored path a WHEN query
still reports `ordering: chronological` and orders rows newest first,
so brief rows keep their sequence but carry no dates.

### Phase awareness — when to write

**Store immediately** when the user states a preference, makes a decision, gives a
correction, or says "remember this." These are **user directives** — never defer them,
even mid-conversation.

**Defer** only when the exchange is pure back-and-forth deliberation with no conclusion
yet (e.g., "what are the pros and cons of X?" without a decision following). Intermediate
conclusions that will shift with further discussion waste writes.

**Stability test** for non-directive content: "Would I be comfortable storing this as-is
if we stopped here?" If yes, store it. If the conclusion might change in the next
exchange, defer.

### Remember — after responding

Run this decision tree after each response.
**Bias toward capturing**: when in doubt, store rather than defer.

**Step 1 — Does this exchange contain any of the following?**

Tier A (importance 4-5, always store):
- User directive — explicit preference, decision, correction, or "remember this"
- Reasoning conclusion — non-trivial judgment from multi-source synthesis
- Durable system/architectural fact discovered during this session
- User-specific context that no search engine can recover

Tier B (importance 2-3, store unless trivial):
- Casual preference revealed in passing ("I usually...", "I prefer...", "I don't like...")
- Topic explored, with conclusion or current understanding (not just questions)
- Useful framing or analogy the user offered
- Background context about the user's projects, tools, or setup

→ None of the above → STOP.

**Category mapping** (pass via `--cat`):
- User stated preference → `preference`
- Architectural/design decision with rationale → `decision`
- Discovered fact about a system, tool, or domain → `fact`
- Reasoning conclusion synthesized from multiple sources → `insight`
- Background context (project setup, user role, environment) → `context`

**Excluded — never store regardless of tier:**

Recoverability test: *Can this fact be recovered from the project's code,
config, IaC state, or cloud account?* If yes, do not store it.

- Bug/issue discoveries — store the *resolution*, not the problem
- State snapshots (line numbers, line counts, file sizes, resource counts, instance IDs)
- Deployment/verification receipts ("all verified", "deployed via", "state clean")
- Temporal observations ("currently", "not yet", "TODO", "should be changed to")
- Intermediate findings that will shift once the task completes

**Mixed content**: strip recoverable details (code paths, boot sequences),
keep only reasoning and conclusions.

**Step 2 — Does a highly overlapping memory already exist?**
→ Yes, incremental new info → UPDATE (merge into existing)
→ Yes, but contradicts/supersedes → REPLACE
→ No significant overlap → CREATE

**Step 3 — Importance calibration**
Use the full 2-5 scale intentionally:
- 5: Cross-session core fact, architectural decision, strong user preference
  NOT: deployment details, resource inventories, task completion receipts
- 4: Important context, significant finding, clear user preference
  NOT: facts recoverable from code/config, routine operational outcomes
- 3: Useful background, project context, topic of interest
- 2: Passing mention, soft preference, conversational color

Importance 2 is the floor — if imp=2 feels weak, reconsider storing at all.

**What to store**: conclusions AND sufficient context to understand them.
The text you pass must be **self-contained** — dereference anaphora
("that", "this", "it") into the actual subject before invoking the CLI.

**How to store**: run
`memman remember "<self-contained text>" --session $SESSION_ID`
directly via Bash in your current turn. No sub-agent delegation.
**Always pass `--session`** — it links this session's writes into one
temporal chain (WHEN recall walks it); a write without it joins no
chain. Use the session id shown above verbatim. A literal
`$SESSION_ID` placeholder means this host did not substitute one and
exports no session variable either: pass your own id, or omit the
flag and accept a write with no chain. Add
`--source agent` when storing your own conclusion rather than
relaying the user's words. The binary is a fast blob-append (~50 ms)
that queues the text; a background scheduler (systemd timer on Linux,
launchd on macOS, every 60 s) drains the queue and runs the
extraction/reconciliation/enrichment pipeline out-of-band. This means
**newly-stored memories are not recallable in the current session** —
they become available in later sessions.

### Behavioral rules — route to CLAUDE.md

When storing a memory that is a **behavioral rule** (importance >= 4, uses universal
language like "never"/"always"/"mandatory", and contains no project-specific entities),
write it to the project CLAUDE.md under a `## Directives` section instead of calling
`memman remember`. Create the section if absent. Directives need guaranteed recall
(CLAUDE.md is loaded every turn), not graph connectivity. The user prunes CLAUDE.md
periodically — no confirmation needed.

### Edge creation and enrichment

`memman remember` is a fast queue-append by default. The full pipeline
— fact extraction, reconciliation, enrichment, causal inference, edge
creation, re-embedding — runs out-of-band in a scheduler-driven worker
fired every 60 s by systemd (Linux) or launchd (macOS). Newly stored
memories are NOT visible to `memman recall` in the current session;
they land for future sessions.

`memman graph rebuild` re-enriches all already-stored insights through
the full LLM pipeline. Use it after model or prompt changes, or to
repair partial enrichment. Auto-created edges (semantic, entity,
temporal) are reindexed automatically on DB open when edge constants
change — no operator command for that.

### Scheduler controls

Memman has a single write path: every `remember` / `replace` enqueues,
and a worker drains the queue. The trigger varies by environment —
systemd timer on Linux hosts, launchd agent on macOS hosts, and a
long-running `memman scheduler serve` process inside containers (set
`MEMMAN_SCHEDULER_KIND=serve` and run the command as PID 1).

When the scheduler is **stopped**, memman is recall-only: every write
exits 1 with `Scheduler is stopped; cannot <verb>. Run 'memman
scheduler start' to enable.` The serve loop polls the state file every
iteration and mid-drain, so pause is observed within seconds even
during long drains.

Drains never overlap: an `fcntl.flock` on `~/.memman/drain.lock`
gates `_drain_queue` entry. If a manual `scheduler trigger` fires
while a timer-driven drain is running, the second invocation logs
`drain: another drain is in progress, skipping` and exits 0.

- `memman scheduler serve [--interval N] [--once]` — long-running drain loop (used as PID 1 in containers). `--interval 0` means continuous (drains back-to-back, with a 100 ms idle backoff when the queue is empty).
- `memman scheduler status` — platform, interval, next run, state, last heartbeat, and the three worker-log paths.
- `memman scheduler start` — flip state to STARTED (resume drains + writes).
- `memman scheduler stop` — flip state to STOPPED (pause drains + reject writes).
- `memman scheduler interval --seconds N` — change cadence (min 60s for systemd/launchd; serve mode accepts any value `>= 0`, with `0` meaning continuous).
- `memman scheduler trigger` — dispatch a drain on systemd/launchd and return at once. It does not wait for the drain, so a `dispatched` response means the run started, not that it finished; read `memman log worker` for the outcome. Not applicable in serve mode.
- `memman log worker [--errors|--stack]` - tail one worker log target; the two flags are mutually exclusive. `--errors` reads `enrich.err`, which carries the worker's own ERROR-level tracebacks. `--stack` reads the rotated `memman.log` and its backups, the only place a traceback survives when the CLI error that reports it is one line. The `enrich` files always sit under `~/.memman/logs`; `memman.log` follows `--data-dir`, so under a non-default data dir they are in different directories and the error message names the exact command to run. `memman scheduler status` prints all three paths.
