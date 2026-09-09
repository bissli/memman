# memman 0.35.0 - the staged reconcile

The reconciler judges one row per call. The batch call of 0.34.0 showed
the fact every shortlisted row at once and missed 25 of the 59 verified
contradictions in its own candidate set; shown one row at a time, the
same text finds 15 of those 25. Three stages replace the one call: a pairwise
screen, a verdict per kept row, and a merge per retiring target, with one
successor stored per retired row. No schema column moves; the store
format and the backup format are unchanged.

## What changes

- **Stage 1, the screen.** `screen_memory` shows the fact and ONE stored
  row per call, every shortlisted row of a fact in parallel on an executor
  sized to the shortlist. The model names the relation, `CONTRADICTS |
  REFINES | RESTATES | UNRELATED`, and quotes the clauses the fact
  overturns. An error, an unparsed body or a token outside the vocabulary
  reads `UNJUDGED` and is shown to stage 2 rather than dropped. The keep
  rule shows CONTRADICTS, RESTATES and UNJUDGED rows; the first REFINES
  row alone when nothing else is kept; an all-UNRELATED screen stores the
  fact with no further call. The call is booked under the new `screen`
  stage at `SCREEN_MAX_TOKENS` (2048).
- **Stage 2, the verdict, one row per call.** `judge_memory` shows each
  kept row alone under `[0]` with the verdict text: the 0.34.0 text with
  its merged-text paragraph removed and, since one row is shown, its
  several-memories tie-break removed. The first entry naming any non-null
  id decides the row, whatever id it names: the model numbers sections of
  the one row and judges a section, and every such verdict is about the
  row shown; DELETE reads as SUPERSEDE. An ADD there is `keep`; an entry
  with a null id or an unknown token is skipped and the next known entry
  decides; an error or an unparsed body is `keep`. `VERDICT_DISPOSITION`
  maps every token the text offers, and a test pins its keys to the
  text's `takes only` line. `assemble_verdicts` reads the
  kept rows in shortlist order: every `supersede` row is a target, a
  `none` row folds to `update` beside a supersede, the update slot takes
  the first `update` or folded row alone, else the first `none` row, else
  ADD. The call runs at `VERDICT_MAX_TOKENS` (2048).
- **Stage 3, the merge, one target per call.** `merge_successor` writes
  the successor text for one retiring target from a body that lists that
  target alone, the clauses stage 1 quoted for it (`(none)` for an update
  target), then the fact, under the new `merge` stage at
  `MERGE_MAX_TOKENS` (8192). No text back means the fact text is stored
  and that target's oplog row is marked `(unmerged)`.
- **One successor per retiring target.** The planner fans a linking
  result into one plan per target; `_apply_plan` runs once per plan and
  is otherwise unchanged. Each predecessor's edges, entity union and
  recall history move to its own successor. The drain's result lists one
  entry per successor.
- **The `(unmerged)` marker fires on update too.** Every reconcile
  relation retires its target the same way, so the marker and the
  `supersede_unmerged` trace event fire on an update stored without a
  merge text as on a supersede; a `replace` stores the caller's text by
  contract and is never marked.
- **The shortlist cap is 20.** `MAX_SIMILAR_FOR_RECONCILE` moves from 10
  to 20; the cap bounded one prompt, and the screen bounds per row
  instead. The cosine floor is unchanged.
- **The replay row records every stage.** The `reconcile-candidates`
  oplog row, written once per fact, carries per candidate its rung and
  score as before plus the stage-1 `relation`, `screened` (true when
  stage 2 saw it) and the stage-2 `verdict`.
- **A planned row is never an edge target before its insert.** Every
  planned row left the drain cache for the apply phase and re-enters it
  once inserted, with the vector it stores. At 0.34.0 a write of two
  facts whose vectors cleared the semantic threshold failed on the
  foreign key when the first row's semantic-edge step aimed at the
  second, not yet inserted; one successor per target made that the
  common case.
- **The stage texts are pinned.** `tests/test_reconcile_stages.py` pins
  the screen, verdict and merge texts by hash; an edit re-pins
  deliberately, after a measurement.

## Measured before shipping

Each stage shipped on the probe cell that decided it, on the
`slow_canonical` model over the 0.34.0 probe's replayed cases, three
reps each.

| line | 0.34.0 batch call | one row per call |
| --- | --- | --- |
| verified contradictions retired (59 confirmed targets) | 34 | 49 (McNemar b=16 c=1, p=0.0003) |
| strict protected rows retired (157 rows, 74 cases) | 47 | 36 |
| restatements answered NONE (11 synthetic) | 11 of 11 | 9 of 11; both losses applied the tie-break sentence the text no longer carries |
| merge texts dropping a true clause (24 headline cases) | 13 of 24, whole body | 7 of 24, one target per call (b=6 to 8, c=0) |
| reconcile wall time per fact, p50 / p90 | 15 s / 54 s, one call | screen p90 5.5 s with rows in parallel, then verdict 6.8 s / 14.7 s with rows in parallel, then merge p50 7.6 s per target in parallel |

The gate re-run of the whole procedure beside the batch arm, at cap 20
and three reps, is the ship condition and is recorded here when it runs.

## Rollout

- `pipx upgrade memman`; the scheduler's next drain runs the staged
  reconcile. No migration, no rebuild.
- Per-stage token accounting gains the `screen` and `merge` buckets; a
  fact costs up to 20 screen calls plus one verdict call per kept row
  plus one merge per retiring target, in place of one batch call.
