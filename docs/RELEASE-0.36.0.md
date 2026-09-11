# memman 0.36.0 - the rerank rung, the screen cut, and the fast reconcile tier

Four changes on 0.35.1's staged reconcile. No schema column moves; the
store format and the backup format are unchanged.

## What changes

- **The shortlist's third rung is the cross-encoder's order.** The
  reconciler's shortlist is up to 5 keyword hits, then up to 9 rows by
  cosine with no floor (positive cosines only; the 0.5 floor is gone),
  then the reranker's order over the top 100 rows by cosine until the
  cap of 20, each rung skipping rows already taken. The reranker is the
  read path's (`MEMMAN_RERANK_PROVIDER`), under the same
  `MEMMAN_RERANK_ENABLED_<store>` and `MEMMAN_RERANK_ENABLED` toggles; a
  failed, disabled or exhausted rerank leaves the cosine order to fill
  the list and never loses the fact. Each candidate in the
  `reconcile-candidates` oplog row carries its rung, `keyword`, `cosine`
  or `rerank`; the `reconcile_rerank` trace event carries `status`,
  `pool`, `taken` and `elapsed_ms`. The cosine quota is a measured
  plateau (every split from 6 to 12 slots reaches the same targets over
  64 frozen cases), and a swap of the embedding or rerank model
  re-measures it.
- **The screen sees the reranker's top ten scored rows plus every
  unscored row.** A keyword hit outside the top-100 pool has no rerank
  score and is always screened; when the rerank failed or was disabled
  no row has a score and every row is screened. A row the cut leaves
  unscreened carries no relation in the `reconcile-candidates` row; the
  `reconcile_screen` trace carries `shortlist`, `rows` (screened), `cut`
  and `kept`.
- **The three reconcile stages run on the fast model.** The screen,
  the verdict and the merge run on the new `fast_worker` role: the
  `MEMMAN_LLM_MODEL_FAST` model at the worker's token budget and read
  timeout (the `fast` role's own ten-second timeout is for interactive
  latency and would cut a merge short). Extraction stays on
  `slow_canonical`. A row's `model_id` names the model behind its
  content: the stage model for a merge text, the canonical model for an
  extracted fact.
- **The JSON reader accepts a literal newline inside a string.** A
  model that copies a memory's paragraph breaks into `merged_text`
  emits raw newlines; the strict decoder refused them and the merge
  fell back to the fact, stored `(unmerged)`. `parse_json_response`
  decodes with `strict=False` on both paths.

## Measured before shipping

Three measurements, each on the same 178 probe cases at three reps
(534 case-reps), read against the 0.34.0 batch call at cap 20 (BATCH,
one call per fact) by exact McNemar on the same cases or rows, Wilson
intervals at 95 percent. The pools are unit 1's ratified labels: 62
confirmed targets in 59 cases, 157 protected rows in 74 cases, 98
credited unlabeled rows, 11 synthetic restatements, 33 headline cases
for the merge line.

**The rerank rung on `anthropic/claude-sonnet-4.6`** (the shortlist
change alone, every stage on the 0.35.1 model):

| line | bar | BATCH | staged, rerank rung | pairing | read |
| --- | --- | --- | --- | --- | --- |
| confirmed contradictions retired, 59 cases | b > c, p < 0.05 | 35 | 50 | b=16 c=1, p=0.00027 | pass |
| protected rows retired, 157 rows | the intervals overlap | 42 = 0.268 [0.204, 0.342] | 29 = 0.185 [0.132, 0.253] | rows b=14 c=27, p=0.060 | pass |
| restatements answered NONE naming the row, 11 | at or above the batch's low bound 0.741 | 11 of 11 | 9 of 11 = 0.818 | b=0 c=2 | pass |
| merge texts dropping a true clause, case majority | not worse; decidable at 25 cases per arm | 14 of 20 = 0.70 | 4 of 26 = 0.15 | 17 paired cases, b=0 c=10, p=0.002 | pass |

Under the production reading (injected rows struck) the write path
retires the confirmed targets of 49 of the 59 cases.

**The screen cut, re-scored at zero calls** over that run: screening
the reranker's top ten scored rows plus every unscored row keeps every
line above at its uncut value (recall 50 of 59, protected rows 29 of
157, NONE 9 of 11) at 10.7 screen calls per fact against 20; no
confirmed target sat below rerank rank 10.

**The fast tier, `anthropic/claude-haiku-4.5` on the screen, the
verdict and the merge**, run live on the same shortlists, every row
screened so the cut re-scores at zero calls:

| line | BATCH | haiku tier | pairing | read |
| --- | --- | --- | --- | --- |
| confirmed contradictions retired, 59 cases | 35 | 46 | b=14 c=3, p=0.013 | pass |
| protected rows retired, 157 rows | 42 = 0.268 [0.204, 0.342] | 24 = 0.153 [0.105, 0.217]; under the cut 22 = 0.140 [0.094, 0.203] | rows b=18 c=36, p=0.020; under the cut b=16 c=36, p=0.008 | fewer false retires than the batch, the interval below the batch's |
| restatements answered NONE naming the row, 11 | 11 of 11 | 10 of 11 = 0.909 | b=0 c=1 | pass |
| merge texts dropping a true clause, case majority | 14 of 20 = 0.70 | 10 of 22 = 0.45 | 16 paired cases, b=1 c=6, p=0.125 | not worse, not separable |

Beside the sonnet-4.6 staged run on the same cases: recall 46 against
50 (b=1 c=5, p=0.22) and protected rows 24 against 29 (b=18 c=23,
p=0.53) are not significantly worse; the merge line is (dropped 9
against 2 of 21 shared cases, b=7 c=0, p=0.016): the haiku merge drops
a true clause in 0.45 of the headline cases against 0.15. The screen
agrees with sonnet-4.6's on 69 percent of the 9,828 row-reps and reads
CONTRADICTS on 2,055 rows against 1,508; two independent haiku screens
agree on 92 percent. Every number was re-derived by an independent
recompute from the raw rows.

| cost, per fact | 0.35.1 today (sonnet-4.6, 15.5-row lists) | sonnet-4.6 with the rung and the cut | haiku tier with the rung and the cut |
| --- | --- | --- | --- |
| screen, verdict, merge and one rerank call | about $0.20 | $0.16 | $0.057 |
| per 30 days at the fleet's 150 facts a day | about $850 | $710 | $258 |

## Rollout

- `pipx upgrade memman`; no migration, no rebuild. The scheduler's next
  drain builds the rerank-rule shortlist, screens the cut, and runs the
  three stages on `MEMMAN_LLM_MODEL_FAST`, which `memman install`
  already resolved for the `fast` role.
