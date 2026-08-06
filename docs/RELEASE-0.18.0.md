# memman 0.18.0 — breaking release

Three defect fixes shipped together because they share one schema
migration. Nothing here adds a feature; every change either corrects
behaviour that was provably wrong or repairs the data it produced.

## The defects

- **D1 — the temporal backbone chain was inert.** `source` carried
  three conflicting jobs (provenance, idempotency key, chain key) and
  the queue's synthetic `queue:N` value won, so backbone edges almost
  never formed. Now: `source` is provenance stored verbatim,
  idempotency rides on a `queue_uuid` minted at enqueue, and the
  chain keys on a new nullable `session_id` (`remember --session`,
  default `$MEMMAN_SESSION_ID`). No session, no backbone edge.
- **D2 — `recall --cat/--source` silently under-returned.** The CLI
  over-fetched `limit * 3`, post-filtered and truncated. Filters now
  run inside the anchor scans and after the weighted-sum sort (before
  rerank), so filtered recall fills to `--limit` whenever enough rows
  match. Unfiltered recall is byte-identical to 0.17.3.
- **D3 — an empty LLM response permanently degraded a row.** A
  transient empty body from a flaky endpoint (Ollama, llama.cpp) cost
  the row its extraction, enrichment or causal edges. Empty
  `choices`, empty/whitespace/null `content` are now retried
  immediately inside the client (no rate-limit backoff); structurally
  malformed responses still raise at once.

## Breaking changes

| # | Break           | Consequence                                                                                                                                                                          |
| --- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A | Store schema    | A pre-0.18.0 store fails at open, naming the rebuild script                                                                                                                          |
| B | Queue schema    | `queue.db` must be deleted and recreated (it gains `session_id` / `queue_uuid`)                                                                                                      |
| C | Data semantics  | `source` changes meaning; synthetic `queue:N` values are backfilled to `user` by the rebuild                                                                                         |
| D | Recall output   | Filtered recall changes result set and ranking                                                                                                                                       |
| E | Backup bundles  | `BACKUP_FORMAT_VERSION` 1 -> 2; restore refuses v1 bundles                                                                                                                           |
| F | Store Protocols | `has_active_with_source` / `get_latest_by_source` are replaced by `has_active_with_queue_uuid` / `get_latest_by_session`; `RecallSession.vector_anchors` gains `category` / `source` |

## Migrating

1. Quiesce: stop all writers, drain the queue fully (check `stale`
   rows via direct SQL — `queue.stats()` does not surface them), then
   stop the scheduler and disable the backup timer.
2. Preserve the rollback path: move existing bundles out of the prune
   window, take a fresh v1 bundle, verify every store in its manifest
   reports `status: ok`, copy it off-machine, and keep a 0.17.3
   install around (`pipx install --suffix=@0173 memman==0.17.3`) —
   nothing else can read v1 bundles afterwards.
3. Land 0.18.0, delete and recreate `queue.db` with a queue-only
   command.
4. Rebuild every store with the rebuild script — run `--probe` first
   (rebuilds throwaway copies and checks count parity), then the live
   run. **The 0.18.0-cycle script (`scripts/migrate_0180.py`, visible
   at the `v0.18.0` tag) performed the one-off data repairs**:
   converting within-window backbone edges to proximity edges,
   dropping out-of-window ones, and backfilling `queue:N` sources.
   The script retained on `master` afterwards,
   `scripts/rebuild_schema.py`, is the generic machinery with those
   one-off repairs removed (only the structural orphan filter
   remains) — a fresh migration from 0.17.3 must use the tagged
   version: `git show v0.18.0:scripts/migrate_0180.py`.
5. Reinstall the scheduler units from the new version, re-enable
   timers, take a fresh v2 bundle.

## Notes

- **v1 bundles cannot be restored onto 0.18.0.** Keep the pinned
  0.17.3 install for as long as any v1 bundle matters.
- **History gets no retroactive backbone chain.** Nothing records
  which historical writes shared a session; pre-migration rows have
  `session_id = null` permanently. Within-window backbone pairs are
  preserved as proximity edges.
- Agent integrations must pass `--session` on every `remember` /
  `replace` — the shipped hook guide and SKILL templates now carry
  the flag, with the live session id substituted into the injected
  guide.
