# memman 0.20.0 — breaking release

Six improvements shipping together because they share one schema
migration (`corroboration_count`). Breaking because of the schema
change; every store must be rebuilt.

Version note: the schema change and features first landed as the
v0.19.0 tag (published to PyPI, never deployed to a fleet); 0.20.0
adds the fixes from a second independent review and is the version
fleets migrate to. The store schema, payload version (3), and backup
format (3) are identical across 0.19.0 and 0.20.0.

## The features

- **F1 — per-stage LLM token accounting.** Every `complete()` call
  names its pipeline stage from a closed set; the provider's `usage`
  block is read per attempt inside the retry loop (an empty body
  retried twice is three billed completions) and accumulated behind a
  lock. Drains emit an `llm_usage_summary` trace event and an
  `llm_usage` key in their JSON output; `queue_done`/`queue_failed`
  trace events carry per-row deltas.
- **F2 — phase-level recall trace events.** `recall_anchors`
  (per-signal hits, fused pool, `vector_hits` vs `anchor_k`),
  `recall_traversal` (visited count, budget-capped anchors), and
  `recall_rerank` (positions moved, diffed by id). The debug gate is
  read once per recall.
- **F3 — exact-match dedup rung.** A fact byte-identical (modulo case
  and whitespace) to exactly one stored row skips reconciliation with
  no LLM call; two identical stored rows still escalate to the LLM.
  Never under `--no-reconcile`.
- **F4 — corroboration count.** The rung bumps the target's
  `corroboration_count` and writes a `reconcile-corroborate` oplog
  row. Observational only: never feeds retention immunity or
  effective importance.
- **F5 — per-string length caps.** LLM-proposed entities/keywords
  over 200 chars are dropped (never truncated) post-parse, before the
  count caps; user `--entities` stay uncapped. Measured against the
  fleet (longest legitimate string: 137 chars).
- **F6 — MMR diversity rerank, measured off.** The one-shot MMR
  mechanism is implemented and sweepable; the ablation harness was
  repaired (it could not run since the Backend Protocol landed) and
  the sweep measured `MMR_LAMBDA = 1.0` (disabled): under the default
  cross-encoder rerank, MMR contributes nothing at any lambda. The
  spec's alternative placement (MMR after the rerank, before the
  limit slice) was measured separately in 0.20.0 and does not change
  the verdict: it buys redundancy only by overriding the reranker's
  certified order. See `experiments/recall_ablation/README.md` for
  both sweep records.

## 0.20.0 review fixes

Behavior refinements from the second review, none of which changes
the schema or formats:

- The corroboration bump adopts the restating row's `queue_uuid`
  ONLY when the target carries none — a populated key (the creating
  row's replay guard) is never clobbered.
- A corroboration target soft-deleted between planning and apply
  degrades the skip to a plain add (previously the restated fact was
  silently dropped and the dead row bumped).
- One queue row bumps a given target at most once, however many
  identical facts its extraction emits.
- The exact-match skip result carries `target_id` naming the
  corroborated row.
- Usage ledger: non-2xx attempts land in a new `http_errors` counter
  instead of `calls` (a retried 429 storm no longer inflates the
  billed-call signal 3x); an HTTP-200 with a non-JSON body is booked
  and retried instead of escaping the ledger; `memman recall
  --expand` emits an `llm_usage_summary` trace event so the
  `query_expansion` bucket is observable.
- MMR tolerates mixed embedding dimensions (off-modal rows hold
  their positions instead of crashing the gram matrix).
- `scripts/rebuild_schema.py::verify_counts` also compares the
  summed corroboration counter, so the probe proves the new column
  round-trips even on an all-zero fleet.

## Breaking changes

| # | Break          | Consequence                                                                                                                       |
| --- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| A | Store schema   | A pre-0.19.0 store fails at open, naming `scripts/rebuild_schema.py`                                                              |
| B | Payload format | `PAYLOAD_VERSION` 2 -> 3; both `apply`s refuse stale payloads                                                                     |
| C | Backup bundles | `BACKUP_FORMAT_VERSION` 2 -> 3; restore refuses v2 bundles                                                                        |
| D | LLM client API | `MemmanLLMClient.complete()` requires a keyword-only `stage` from a closed set                                                    |
| E | Store Protocol | `NodeStore` gains `increment_corroboration`; `Insight` gains `corroboration_count`                                                |
| F | Write pipeline | Byte-identical re-statements no longer produce a reconcile LLM call (skipped + bump)                                              |
| G | JSON output    | Every serialized insight gains `corroboration_count`; drain JSON's `llm_usage` gains `http_errors`; skip results gain `target_id` |

**v2 bundles cannot be restored onto 0.19.0 or later.** Install a
0.18.x reader BEFORE upgrading if any v2 bundle is still needed
(`pipx install memman==0.18.1 --suffix=@0181` — the package spec is
required), exactly as `memman@0173` was pinned for v1 bundles.

## Rollout requirements

These are the invariants a rollout runbook must satisfy — each was
learned the hard way during the 0.18.0 cycle, and each is
phase-independent:

1. **The human operator runs it from a plain terminal**, never an agent inside a Claude Code
   session. The memman hooks make any session both a reader and a writer, and a hook-driven
   `recall` between `drop()` and `apply()` recreates an empty `data/<store>` via `open_db`'s
   `mkdir`, which `apply` then writes into.
2. **Quiesce means both timers.** `scheduler stop` never touches `memman-backup.timer`,
   which is armed daily at 03:00 with `Persistent=yes`; a firing mid-rebuild publishes a
   bundle recording any in-flight store as `status='failed'`. Verify with
   `systemctl --user is-active` / `is-enabled`, **not** `list-timers --all | grep`, which
   matches loaded-but-disarmed timers.
3. **Drain before stopping, and wait for it.** You cannot drain once stopped
   (`_require_started`), and `trigger` is fire-and-forget (`systemctl start --no-block`), so
   poll until the unit goes inactive before counting. Expect several rounds on a backlog.
4. **Gate on the queue via direct SQL.** `queue.stats()` seeds only
   `pending`/`done`/`failed` and filters `if status in result`, so **`stale` rows are
   invisible** through `scheduler queue list`. Read `select status, count(*) from queue
   group by status` and require every non-`done` count to be zero.
5. **Preserve the rollback path before anything destructive**: move existing bundles out of
   the prune window (`MEMMAN_BACKUP_KEEP=7` prunes the target root on every run), take a
   fresh bundle with `memman backup run` (bare `memman backup` only prints status), verify
   **every** store in the manifest reports `status: ok` (`build_bundle` publishes and exits 0
   even with failed stores, and `restore` then silently skips them), copy it outside Dropbox,
   and install a **0.18.x reader** via `pipx install memman==0.18.1 --suffix=@0181`
   (the package spec is REQUIRED — a bare `--suffix` invocation errors out, and
   mirroring the main install's local-path spec at rollout time would install the
   0.20.0 working tree as the "v2 reader", silently voiding the rollback path; pin the
   published 0.18.1, exactly as `memman@0173` pinned `memman==0.17.3`) — after the upgrade
   nothing else on the host can read v2 bundles.
6. **Binary identity is cwd-dependent** (direnv + editable install). Name absolute paths in
   every command. A bare `memman install` resolves to the editable binary and rewrites the
   **enrich** unit's `ExecStart` plus the 6 hook and 1 skill symlinks into the Dropbox
   working tree. It never touches `memman-backup.*` — that unit's `ExecStart` moves only
   via `memman backup schedule`, which is why moving both units takes a full pipx
   reinstall. Related hard-won caveat: a TEST used to run the real `uninstall_backup` and
   silently disarm the host's backup timer on every full suite run — fixed by stubbing it;
   if the timer ever reads `not-found` again, suspect an unstubbed test before suspecting
   systemd.
7. **Probe all 24 stores first, and hard-gate on count parity.** The schema change
   has no `default`-style FK hazard to exercise. `verify_counts` compares the summed
   corroboration counter alongside the row counts (added in 0.20.0 — on an all-zero
   fleet plain row parity would pass even if gather dropped the column), so the probe
   is a real proof that `gather`/`apply` round-trips the new column on every store.
8. **Do not restart the scheduler until the script exits 0.** A partial rebuild plus a live
   scheduler means hook writes for skipped stores fail five times and land as `failed` rows
   whose content exists nowhere else.
9. **Re-embedding is not required** and should be verified rather than assumed:
   `corroboration_count` is not an embedding input, and the embedded text is
   `content + [KEYWORDS: ...]`. Confirm per-store `count(*) where embedding is not null`
   matches on the probe.
10. **`memman graph rebuild` is not needed** unless a change touches one of the four
    prompts feeding `compute_prompt_version`. F5 deliberately applies its caps post-parse
    for exactly this reason — check before shipping, since a prompt change staleness-marks
    every row (the prompt hash is pinned unchanged by
    `test_prompt_version_unchanged_by_length_caps`).
11. **Redeploy pipx AFTER the last commit of the cycle.** The 0.18.0 runbook reinstalled
    pipx and then landed a follow-up commit, leaving the deployed binary one commit behind —
    its `MIGRATION_SCRIPT` named a file the repo no longer had, caught only by review and
    fixed in 0.18.1. If any commit lands after the service-restore reinstall, reinstall
    again (`pipx uninstall memman && pipx install <repo>` — `--force` is broken on this
    host's uv-backed pipx).
12. **Retention decision for `~/.memman/archive/` is unresolved and accretes.** Each
    rebuild renames every store's pre-migration directory into
    `archive/<store>/<date>_<NN>/` — the 0.18.0 generation is ~369 MB, larger than live
    `data/`, readable only by the pinned old-version install, and captured by NO backup
    bundle. This rebuild adds another generation. Decide before running: either delete the
    pre-0180 generation once this release's exit gate passes (bundles in `pre-0180-v1/` +
    offsite remain the rollback), or state a keep-N policy.
13. **Two pre-existing per-store doctor findings are accepted, not regressions:**
    `opendate` fails `orphan_insights` structurally (1 insight, 0 edges = 100%), and
    `default` warns at ~3% orphans; both predate 0.18.0 (verified against the archived
    pre-migration copies). `memman doctor` inspects only the ACTIVE store — a fleet
    check means sweeping `--store` over all 24.

## Notes

- The rebuild script (`scripts/rebuild_schema.py`) already carries the
  payload-v3 markers (`EXPECTED_PAYLOAD_VERSION = 3`,
  `corroboration_count` in `SCHEMA_COLUMNS`); its `repair_payload` is
  the structural orphan filter only — this cycle needs no data repair,
  the new column takes its `default 0` on every rebuilt row.
- **The rebuild script is SQLite-only.** A Postgres-routed store also
  fails at open (the same newest-column index tripwire, translated to
  the same diagnostic), but its rebuild path is a detour: migrate the
  store `--to sqlite` on the previous release, rebuild, then migrate
  back. No Postgres-routed store exists on this host today.
- History gets no retroactive corroboration counts; pre-migration
  restatements were reconciled by the LLM at write time and stay as
  they landed.
