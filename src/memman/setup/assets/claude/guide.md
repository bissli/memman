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

To recall: `memman recall "<query>" --limit 5 --rerank`.
Craft a focused, keyword-rich query — do not pass the raw user prompt.
Always pass `--rerank` for the highest-quality top-K (the reranker
auto-skips on 1-2 token queries).

The recall response includes a `meta` object with:
- `hint`: intent-specific reasoning guidance (always present) — use it to
  frame your synthesis of the results
- `ordering`: how results are sorted — `causal_topological` (WHY),
  `chronological` (WHEN), or `score` (ENTITY/GENERAL)
- `reranked`: boolean — true when the cross-encoder rerank stage
  fired (false when query was too short or `--rerank` was omitted)
- `sparse`: boolean, present only when results are below half the requested
  limit — signals low-confidence retrieval; consider broadening the query

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

**How to store**: run `memman remember "<self-contained text>"` directly
via Bash in your current turn. No sub-agent delegation. The binary is a
fast blob-append (~50 ms) that queues the text; a background scheduler
(systemd timer on Linux, launchd on macOS, every 60 s) drains the queue
and runs the extraction/reconciliation/enrichment pipeline out-of-band.
This means **newly-stored memories are not recallable in the current
session** — they become available in later sessions.

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
- `memman scheduler status` — platform, interval, next run, state, last heartbeat.
- `memman scheduler start` — flip state to STARTED (resume drains + writes).
- `memman scheduler stop` — flip state to STOPPED (pause drains + reject writes).
- `memman scheduler interval --seconds N` — change cadence (min 60s for systemd/launchd; serve mode accepts any value `>= 0`, with `0` meaning continuous).
- `memman scheduler trigger` — run the drain now on systemd/launchd; not applicable in serve mode.
- `memman log worker [--errors]` — tail the enrichment worker logs.
