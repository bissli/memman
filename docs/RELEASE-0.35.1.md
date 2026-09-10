# memman 0.35.1 - the merge guard and short ids

Three changes on 0.35.0's staged reconcile. No schema column moves; the
store format and the backup format are unchanged.

## What changes

- **The merge stage renders no clause copied from the fact.** Over the
  0.35.0 gate's 534 screened case-reps the screen handed the merge 4,239
  contradicted clauses. 46 of them, in 31 case-reps, were sentences of
  the new fact that do not occur in the target memory, and the merge,
  told to drop them, dropped the fact's own correction from the
  successor. `merge_successor` now drops a clause whose
  whitespace-collapsed, case-folded text occurs in the fact and not in
  the target's content before it renders the body. A clause found in the
  target renders whether or not the fact also carries it; a clause found
  in neither, a paraphrase, renders unchanged; an emptied list renders
  `(none)`. The merge text itself is unchanged. The `reconcile_merge`
  trace event carries `clauses`, the rendered count, and `suppressed`.
- **Every id-taking command accepts an unambiguous prefix.** `forget`,
  `replace`, `supersede`, `unsupersede`, `graph link`, `graph related`
  and `insights show` resolve each id argument through
  `NodeStore.resolve_id`: an exact id wins first; otherwise the one row
  whose id starts with the argument, over all rows including forgotten
  and superseded ones, so that `insights show` and `unsupersede` still
  reach them. An ambiguous prefix exits non-zero naming the number of
  rows it matches; an argument matching nothing reaches the command's
  own not-found path. `graph related` on an unknown id exits non-zero
  instead of printing an empty list. The same-row guards of `supersede`
  and `graph link` compare the resolved ids, so a prefix and the full id
  of one row are refused as the same insight.
- **The shared guide names no host tool.** `memman guide` emits one text
  to every host. The Claude SKILL names Bash and the OpenClaw SKILL names
  the `exec` tool, so the guide says "directly in your current turn" and
  each SKILL keeps its own tool name.

## The measurements behind the guard

- The stricter rule, rendering only clauses found in the target, was not
  taken: over the same run it would have filtered the 87 clauses the
  screen paraphrased (elisions and truncations, in 70 case-reps) to
  remove the 46 fact copies. The shipped rule fires on the fact copies
  alone and, by construction, never on a memory clause.
- A paired replay of the 37 affected merges with and without the guard,
  74 merge calls: with the guard the fact's sentence appears in 28 of
  the 46 successors, without it in 21. Two runs of the identical
  unguarded body differ on 12 of 46, so the successor-text effect is
  inside run-to-run variance at this size. The guard ships because the
  instruction it removes is wrong: a sentence of the fact is not a
  clause of the memory.
- The reconciler shortlist is unchanged in this release. Every rung over
  the stored vectors and claims fails the offline bar (more than 4 of
  the 8 missed targets entering the 20-row union with all 59 reached
  kept): the claim rung at best 4 of 8, reciprocal-rank fusion of five
  scorers 4 of 8 with one reached target lost, sentence-level chunking of
  the rows 4 of 8 with two lost. A cross-encoder rerank of the top-100
  cosine rows reaches 5 of 8 with all 59 kept and is the candidate for a
  later release, after the section 9 gate re-runs on that shortlist.

## Rollout

- `pipx upgrade memman`; no migration, no rebuild. The scheduler's next
  drain runs the guarded merge.
