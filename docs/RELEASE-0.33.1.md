# memman 0.33.1 - breaking release

Supersession: a corrected memory is superseded, never deleted. Breaking
because of a schema change (`insights.superseded_by`); every live store
gets the column by hand once, and the backup format moves to 4.

Version note: the schema change and features first landed as the
v0.33.0 tag, never published; 0.33.1 adds the fixes from an independent
review of that tree and is the version fleets migrate to. The store
schema and the backup format are identical across the two.

## What changes

- **`replace` and the reconciler keep the old row.** `_apply_plan`
  sets `superseded_by` on the predecessor instead of `deleted_at`. The
  predecessor keeps its content, leaves every recall, listing, count
  and edge build, and its edges move to the successor. For the same
  write history every read returns what it returned before.
- **Reconcile actions are ADD, UPDATE, SUPERSEDE, NONE.** UPDATE
  refines with compatible detail; SUPERSEDE contradicts, and its merged
  text keeps only the predecessor's still-true clauses. DELETE is gone:
  it fired five times fleet-wide and every time discarded the new fact.
  A stray DELETE from the model is read as SUPERSEDE.
- **Two facts on one predecessor** supersede it once; the second lands
  as a plain add instead of being dropped.
- **A row-addressing verdict with no resolvable target is an ADD.** An
  UPDATE, SUPERSEDE or NONE whose `target_id` is null or unmapped
  stores the fact as new information and carries neither the target
  nor the model's merged text; a NONE with a null target therefore
  stores the restatement instead of silently skipping it.
- **A queued `replace` follows the chain** to the current head when its
  target was superseded between enqueue and drain, and reports
  `redirected_from`.
- **New verbs.** `memman insights show <id> --history` walks the chain
  oldest first, forgotten rows included. `memman supersede <old> <new>`
  links two rows that both already exist. `memman unsupersede <id>`
  reverses a link once the successor is forgotten, re-embeds the row
  and rebuilds its entity and semantic edges.
- **Doctor.** `supersession_integrity` (dangling pointer, superseded
  row with edges, self-pointer, and a pointer cycle that reaches no
  current head; a successor with two predecessors is a join, not a
  defect) and
  `partial_index_predicates` (a partial index kept from the previous
  schema by name, or the retired listing index left in place).
  `schema_columns` expects `superseded_by`.
- **`status`** reports `total_insights` (current), `superseded_insights`
  and `deleted_insights`; the three sum to the table.
- **Indexes.** `idx_insights_current_listing` on `(deleted_at,
  superseded_by, importance, created_at)` replaces
  `idx_insights_deleted_importance_created`; `idx_insights_pending_link`
  becomes `(linked_at, created_at)` under the three-clause predicate.
  Every partial index carries `superseded_by is null`.
- **Deleted.** `scripts/rebuild_schema.py` and the migrator probes for
  pre-schema stores; `min_importance` on `nodes.query`;
  `tolerate_missing` on `soft_delete`.

## Migration

Stop the scheduler first and keep it stopped until every store has been
opened once on 0.33.0: the previous release's `replace` still deletes.

SQLite, per store, with no memman process open on it:

    alter table insights add column superseded_by text;
    drop index if exists idx_insights_deleted_importance_created;
    drop index if exists idx_insights_pending_link;

Postgres, per store schema `s`:

    alter table s.insights add column superseded_by text;
    drop index if exists s.idx_insights_pending_link_s;
    drop index if exists s.idx_insights_kw_tokens_s;
    drop index if exists s.idx_insights_hnsw_s;

`create index if not exists` matches by name, so an index whose
definition changed keeps its old shape until dropped; the next open runs
the baseline and recreates each one. `memman doctor` then passes
`schema_columns` and `partial_index_predicates` on every store, and the
latter names a partial index left with the old predicate or the retired
listing index left in place. A store opened before its DDL fails at
open naming the missing column.

Backfilling history is a separate, optional operation over the oplog:
every `replace` / `reconcile-update` event names a predecessor and, in
its detail, the successor. Setting `superseded_by` on the predecessor
and clearing its `deleted_at` restores it as superseded, so
`insights show --history` walks pre-0.33 chains. The scope rule: only
rows whose deletion came from such an event; a row with a `forget`
event keeps `deleted_at`, since `--history` prints the content of a
superseded row and `unsupersede` can restore one. A successor absent
from the table is skipped. Run `memman doctor` afterwards;
`supersession_integrity` must pass on every store.

Backups written by 0.32.x carry format 3 and are refused by 0.33.0
`restore`; read them with the pinned earlier install.
