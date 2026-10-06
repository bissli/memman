# 4. Lifecycle and embedding

[Previous: pipelines](03-pipelines.md) | [Design overview](../DESIGN.md) | [Next: Claude Code integration](05-integration.md)

## 4.1 Retention

Memories have no expiry date or automatic size cap. A memory stays current until an agent or operator replaces or forgets it.

| Action                          | Stored change                                                                                                    | Effect on recall                                                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `replace <id> "<text>"`         | Create a successor and set the target's `replaced_by` when the worker commits.                                   | Return the successor; exclude the old version.                                                                                            |
| `forget <id>`                   | Set `deleted_at`; keep the row.                                                                                  | Exclude the forgotten memory.                                                                                                             |
| `store remove <name>`           | Delete the store and its queued writes. Refuse a branch, or a store that holds a branch token.                   | Remove the entire collection.                                                                                                             |
| `store branch <parent> <label>` | Create an empty SQLite store over the live parent, marked with `branch_parent`.                                  | `--store <branch>` recall ranks the branch's rows with the parent's current rows. Parent recall is unaffected.                            |
| `store merge <branch>`          | Insert the branch's own rows into the parent, repeat its replaces and forgets on copied rows, delete the branch. | The parent returns the branch's own current rows and applies the branch's retirements. A conflicting retirement keeps the parent's state. |
| `store drop <branch>`           | Delete the branch and list its current rows.                                                                     | Remove the branch. Parent recall is unaffected.                                                                                           |

No command undoes a forget or makes a replaced row current again. Replacing the successor corrects a wrong correction. `insights show <id> --history` displays the chain; forgotten entries omit their content.

`forget` refuses a current memory whose predecessor has not been forgotten. The error directs the caller to `replace` so the correction history remains intact. Both commands require a started scheduler.

The operation log has its own retention rules. Maintenance trims entries older than 180 days (`OPLOG_RETENTION_DAYS`) in stores where a drain completed work. A 5,000-entry cap (`MAX_OPLOG_ENTRIES`) applies when the enrichment pass that follows has work. Neither limit affects a stored memory ([drain maintenance](03-pipelines.md#maintenance-after-each-drain)).

## 4.2 Inspecting memories

| Command                        | Purpose                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| `insights show <id>`           | Inspect a current or replaced memory in full.                       |
| `insights show <id> --history` | Inspect its replacement chain, including forgotten entries.         |
| `insights review`              | Find current memories containing potentially temporary information. |

Review uses the same pattern checks as write-time `quality_warnings`: instance IDs, counts, the word `currently`, dated observations, and similar wording. Warnings leave the decision to the agent or operator. The separate CLI validation rules reject line references and other invalid input before queueing.

[Inspection commands](../USAGE.md#insights) lists the limits and output.

## 4.3 Embedding support

An embedding is a numeric representation of memory content used for semantic search. Vectors from different models cannot be treated as interchangeable, even when their dimensions match.

Each store records an **embedding fingerprint** in `meta.embed_fingerprint`: model and vector dimension. Recall, the worker, and re-enrichment build their client from this fingerprint through `bound_embedder`, so stores in one process can use different embedding models.

The global `MEMMAN_EMBED_MODEL` setting serves three purposes:

| Role                 | Effect                                                                                                                                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| New store            | Supplies the fingerprint when `store create` makes a store. A branch copies its parent's fingerprint instead. Recall and `store merge` refuse a branch whose fingerprint differs from the parent's. |
| Model change         | Selects the target for `embed reembed`. Changing it alone does not convert existing vectors.                                                                                                             |
| Store-opening checks | Normal store sessions also construct this model's client. A missing `MEMMAN_API_KEY` can stop the command before retrieval.                                                                              |

The diagnostic and maintenance paths for `doctor`, `embed status`, `embed swap`, `migrate`, and `backup` bypass the normal fingerprint initialization check. The related-memory read in `remember` also avoids model clients.

If the shared endpoint lacks credentials, recall falls back to keyword and recency ranking, and the worker fails queued writes that need those credentials. This fallback cannot recover a command that already failed while building the global model's client.

`embed status` reports the fingerprint, any swap in progress, and whether the endpoint has credentials. The `embed_fingerprint` check in `doctor` fails when that key is missing, and on a store that holds memories but has no fingerprint. The drain refuses such a store.

### Embedding endpoint

Every embedding call goes to `<MEMMAN_ENDPOINT>/embeddings` with `MEMMAN_API_KEY`. The shipped default model is `voyageai/voyage-4-lite`, which returns 1024-dimension vectors. A store binds its client by the fingerprint's model on the shared endpoint. A model other than the default requires an initial embedding probe to determine its dimension. [Provider setup](../USAGE.md#provider-setup) lists the endpoint settings.

### Vector storage

| Backend  | Representation                                        | Search                                                         |
| -------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| SQLite   | Little-endian float64 BLOB, eight bytes per dimension | Matrix product over vectors matching the query width.          |
| Postgres | pgvector `vector(N)`, stored as float32               | HNSW approximate-nearest-neighbor index for vector candidates. |

A new Postgres store sizes its vector column from the embedding client. On SQLite, a vector with a different width contributes zero similarity.

### Text sent for embedding

| Operation          | Model selection       | Text                    |
| ------------------ | --------------------- | ----------------------- |
| Drain and `enrich` | Store fingerprint     | Original memory content |
| Recall             | Store fingerprint     | Query                   |
| `embed swap`       | Explicit target       | Original memory content |
| `embed reembed`    | Global embed settings | Original memory content |

The worker attempts embedding after enrichment. A handled HTTP or provider runtime failure leaves the memory without a vector; missing credentials fail the queued write. A later enrichment pass can repair incomplete memories, as described in [failure and retry](03-pipelines.md#failure-and-retry).

### Changing the embedding model

A configuration change alone leaves an existing store's fingerprint unchanged. One of these operations changes it, with the scheduler stopped:

| Command         | Scope                                                                  | During the operation                                            |
| --------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------- |
| `embed swap`    | One SQLite or Postgres store                                           | Recall uses old vectors until an atomic switch.                 |
| `embed reembed` | SQLite stores in the data directory, Postgres-parent branches excepted | Vectors are rewritten in place, so recall may see mixed models. |

The [embedding command reference](../USAGE.md#embedding-operations) gives complete stop, change, and restart examples.

### Embedding swap

A swap writes target-model vectors to `embedding_pending`, then switches the store to them. The swap records its progress in `embed_swap_*` metadata; the implementation is [embed/swap.py](../../src/memman/embed/swap.py).

| State            | Meaning                                                            | Recovery after interruption                    |
| ---------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| `backfilling`    | Fill pending vectors in batches, saving a cursor after each batch. | `--resume` continues from the cursor.          |
| `cutover`        | Backfill is complete; the final transaction is next.               | `--resume` retries the switch.                 |
| No swap metadata | No swap is in progress.                                            | `embed status` reports the active fingerprint. |

On SQLite, cutover copies pending vectors into `embedding` and clears the pending column. On Postgres, it replaces the old column and index with the pending ones. The cutover commits in its own transaction. A second transaction then writes the target fingerprint and deletes every `embed_swap_*` key, so a store with no such key has no swap in progress.

Postgres builds the pending HNSW index concurrently before backfill and requires Postgres 12 or newer. `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` limits that index build; zero means no timeout. `MEMMAN_EMBED_SWAP_BATCH_SIZE` controls batch size, defaulting to 200. Both are process-environment settings.

Only current memories receive new vectors. At cutover, Postgres clears vectors on retired memories; SQLite keeps their old vectors. A swap to the current fingerprint does nothing. Returning to an earlier model requires another swap.

`--abort` discards pending vectors and swap metadata and does not require a stopped scheduler. It refuses a swap in the `cutover` state, because the cutover may have committed before a crash, and clearing the metadata then would leave the old fingerprint recorded against the new vectors. `--resume` finishes such a swap: on Postgres, a schema with no `embedding_pending` column counts as cut over, and on SQLite the copy touches only rows whose pending vector is set. `doctor` reports leftover swap metadata through `no_stale_swap_meta` until the swap completes or is aborted.

### In-place re-embedding

`embed reembed` visits all SQLite stores, except a branch whose parent is not a local SQLite store, and rewrites current memories whose model or vector width differs from the target, or whose vector is missing. It skips matching vectors and retired memories, saves a cursor to support resumption, and writes each store's fingerprint after completion.

It refuses to start with a Postgres store selected and otherwise skips Postgres stores. `--dry-run` counts the changes without writing and does not require stopping the scheduler.
