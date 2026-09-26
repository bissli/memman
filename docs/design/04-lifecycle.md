# 4. Lifecycle & Embedding

[< Back to Design Overview](../DESIGN.md)

---

No memory expires. A memory stays in recall until a `forget`, `replace` or `supersede` call takes it out.

## 4.1 Retention

A store has no size cap and no retention score. Deletion is always an operator or agent action. Three calls take a memory out of recall:

| Call                           | Effect                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `memman forget <id>`           | Soft delete: sets `deleted_at`. The row stays in the table, and no command restores it.              |
| `memman replace <id> "<text>"` | Queues a successor. The drain stores it and sets the target's `superseded_by` to the successor's id. |
| `memman supersede <old> <new>` | Sets `superseded_by` on one current memory to point at another current memory. Neither text changes. |

A superseded memory keeps its content but leaves every recall and listing. `memman unsupersede <id>` returns it to recall once its successor is forgotten, and re-embeds it with the store's embedding model. `forget`, `replace`, `supersede` and `unsupersede` are writes, so none can run while the scheduler is stopped.

**Rationale.**

- **No size cap.** A store becomes more useful as it accumulates memories. A cap would force memman to delete true claims to make room.
- **Supersession keeps content.** `replace` never deletes. The old row keeps its text and records its successor in `superseded_by`. `memman insights show <id> --history` shows the chain of replacements.
- **The oplog is bounded.** The oplog records changes to memories. After each drain, memman deletes oplog rows older than 180 days (`OPLOG_RETENTION_DAYS`) in every store where the drain finished a row. The 5,000-row cap (`MAX_OPLOG_ENTRIES`) runs only when that store still has a current memory without `linked_at`.

## 4.2 Inspecting memories

The `memman insights` commands inspect memories:

```bash
# Read a single insight by ID (a superseded row shows its successor)
memman insights show <id>

# Walk the supersession chain through an id, oldest first
memman insights show <id> --history

# List the insights one queued write produced
memman insights by-queue <queue_uuid>

# Scan stored insights for transient content
memman insights review
```

`insights review` scans current memories, newest first, for the temporary information flagged by `quality_warnings`, including an AWS instance id, a resource count, a line count, a state observation ("state is clean"), the word "currently" and an "as of <date>" statement. It stops at `--limit` flagged memories (default 20) and returns each with its `quality_warnings`. The user or agent decides what to forget. `remember` and `replace` run the same scan at write time and store the text anyway, so `review` finds temporary information that was stored. Both commands reject text that names a line number.

The [USAGE guide](../USAGE.md#insights) lists the output of each command.

---

## 4.3 Embedding support

Recall uses embeddings for vector search. Each store is bound to one embedding model. The store's `meta.embed_fingerprint` row holds that model as JSON: provider, model and vector dimension. This record is the store's **fingerprint**. A store changes model only through an explicit `memman embed swap` or `memman embed reembed` ([4.3.5](#435-changing-the-embedding-model)).

**Model selection.** The fingerprint determines the embedding client for every reader and writer of the store: the background worker, recall, `graph rebuild` and `unsupersede`. Each resolves the client through `bound_embedder`, which reads the fingerprint and builds the client for that provider and model. One process can open stores that use different providers. The [USAGE guide](../USAGE.md#embedding-operations) gives a worked example.

**What `MEMMAN_EMBED_PROVIDER` controls.**

| Role                | Effect                                                                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| New stores          | When a store has no fingerprint and no rows, the first open writes this provider's fingerprint. `memman install` configures a new SQLite default store this way.                               |
| Target              | It names the target of `memman embed reembed` and the default provider of `memman embed swap`.                                                                                                 |
| Startup requirement | Every command that reads a store first builds this provider's client. A missing Voyage or OpenRouter key causes the command to fail. `doctor`, `embed status` and `embed swap` skip this step. |

**Missing credentials for the bound provider.** Recall logs a warning and ranks with the keyword and recency channels only. The background worker marks the queued write as failed after retries are exhausted. `memman unsupersede` refuses to run.

`memman embed status` reports the stored fingerprint, any swap in progress, and whether credentials for the fingerprint's provider are available. The `embed_fingerprint` check in `memman doctor` passes when the store has a fingerprint and its provider's key is present. It fails when the key is missing, and it fails on a store that holds memories but has no fingerprint.

### 4.3.1 Supported providers

| Provider     | Default model             | Key                                        | Offered by install |
| ------------ | ------------------------- | ------------------------------------------ | ------------------ |
| `voyage`     | `voyage-3-lite` (512-dim) | `MEMMAN_VOYAGE_API_KEY`                    | Yes (default)      |
| `openai`     | `text-embedding-3-small`  | `MEMMAN_OPENAI_EMBED_API_KEY`              | Yes                |
| `openrouter` | `baai/bge-m3`             | `MEMMAN_OPENROUTER_API_KEY`                | Yes                |
| `ollama`     | `nomic-embed-text`        | None. Local server at `MEMMAN_OLLAMA_HOST` | No                 |

- `openai` accepts any OpenAI-compatible endpoint through `MEMMAN_OPENAI_EMBED_ENDPOINT` (default `https://api.openai.com`).
- `openrouter` reads `MEMMAN_OPENROUTER_ENDPOINT` (default `https://openrouter.ai/api/v1`). On an OpenRouter LLM endpoint, install copies `MEMMAN_OPENROUTER_API_KEY` into `MEMMAN_LLM_API_KEY` when the LLM key is unset, so one secret serves both.
- `MEMMAN_OLLAMA_HOST` defaults to `http://localhost:11434`.
- Each provider's model is a setting: `MEMMAN_VOYAGE_EMBED_MODEL`, `MEMMAN_OPENAI_EMBED_MODEL`, `MEMMAN_OPENROUTER_EMBED_MODEL`, `MEMMAN_OLLAMA_EMBED_MODEL`. For every model except the Voyage default, memman determines the vector dimension from an initial test embedding.

### 4.3.2 Vector storage

The store's backend decides the vector format:

| Backend  | Column                           | Format                                       | Search                                                    |
| -------- | -------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| SQLite   | `insights.embedding` BLOB        | Little-endian float64, 8 bytes per dimension | One matrix product over every vector of the query's width |
| Postgres | `insights.embedding` `vector(N)` | pgvector, stored as float32                  | HNSW index on current rows, cosine distance               |

HNSW (hierarchical navigable small world) is an index for approximate nearest-neighbor search. A fresh Postgres store sizes its `vector(N)` column using the dimension of the initial embedding client. On SQLite, a vector of another width scores 0 against the query.

### 4.3.3 Embedding in the pipeline

| Step                      | Client                  | Text embedded      |
| ------------------------- | ----------------------- | ------------------ |
| Drain (remember, replace) | The store's fingerprint | Content alone      |
| `memman graph rebuild`    | The store's fingerprint | Content alone      |
| Recall                    | The store's fingerprint | The query as given |
| `memman unsupersede`      | The store's fingerprint | Content alone      |
| `memman embed swap`       | The target model        | Content alone      |
| `memman embed reembed`    | `MEMMAN_EMBED_PROVIDER` | Content alone      |

The drain embeds each row once, after LLM enrichment. When the embedding call fails with an HTTP or provider error, the drain stores the row without a vector. Missing credentials cause the row to fail.

### 4.3.4 Recovery

`memman graph rebuild` re-enriches every current memory through the full LLM pipeline and re-embeds it. This adds vectors to rows stored without them. The maintenance step after a later drain that finishes a row in the same store also retries up to 3 such rows ([chapter 3](03-pipelines.md#maintenance-after-each-drain)). It requires `memman scheduler stop` first, except with `--dry-run`. `--stale-only` limits the pass to rows whose enrichment prompt or LLM model changed.

### 4.3.5 Changing the embedding model

Two commands replace existing vectors with vectors from another model. Both require `memman scheduler stop` first, so memman is recall-only while they run. The [USAGE guide](../USAGE.md#embedding-operations) gives the command syntax.

| Command                | Scope                                       | Target                                                       | Recall during the run                        |
| ---------------------- | ------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| `memman embed swap`    | One store, SQLite or Postgres               | `--to <model>`, with `--provider` or `MEMMAN_EMBED_PROVIDER` | Reads the old vectors until the final switch |
| `memman embed reembed` | Every SQLite store under the data directory | The `MEMMAN_EMBED_PROVIDER` client                           | Mixes vectors from both models               |

### 4.3.6 Embedding swap

`memman embed swap` fills a separate column, `embedding_pending`, with vectors from the target model, then switches the store to those vectors. The code in `src/memman/embed/swap.py` records its progress in the store's `embed_swap_*` meta keys:

```
(no swap) --swap--> backfilling --backfill done--> cutover --commit--> (no swap)

A swap that stops early keeps its state and cursor:
  swap --resume   continues from the recorded state
  swap --abort    discards the pending vectors -> (no swap)
```

| State         | Meaning                                                                                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backfilling` | Each batch embeds the next `MEMMAN_EMBED_SWAP_BATCH_SIZE` current memories (default 200) into `embedding_pending`. The cursor advances per batch, so `--resume` continues from the last batch. |
| `cutover`     | Set just before the cutover transaction. `--resume` from this state runs the cutover again.                                                                                                    |
| (no swap)     | No `embed_swap_*` key exists. `memman embed status` shows the target fingerprint.                                                                                                              |

**Switching to the new vectors (cutover).**

- **Postgres.** Before filling the new column, memman adds `embedding_pending vector(N)` and builds its HNSW index concurrently. `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` caps that build in seconds (default 0, no limit). The cutover runs in one transaction and replaces the old column and index with the pending ones. On Postgres older than 12, memman refuses the swap before the backfill starts.
- **SQLite.** The cutover copies `embedding_pending` into `embedding` and clears the pending column in one transaction.
- **Both.** One transaction writes the target fingerprint and deletes every `embed_swap_*` key. When these keys are absent, no swap is in progress. Returning to the old model requires another full swap. A swap to the model the fingerprint already names does nothing.
- **Superseded and forgotten memories.** Only current memories receive new vectors. At cutover, a forgotten or superseded memory loses its vector on Postgres and keeps its old vector on SQLite. `memman unsupersede` re-embeds a memory when it returns to recall.

**Abort.** `--abort` discards the pending vectors and deletes the swap meta keys. It does not need a stopped scheduler. The `no_stale_swap_meta` check in `memman doctor` warns while any `embed_swap_*` key remains, which includes a swap that stopped and waits for `--resume` or `--abort`.

memman reads `MEMMAN_EMBED_SWAP_BATCH_SIZE` and `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` directly from the process environment.

### 4.3.7 Offline re-embed

`memman embed reembed` rewrites vectors in place, store by store, with the `MEMMAN_EMBED_PROVIDER` client. To change providers for all SQLite stores, run `memman config set MEMMAN_EMBED_PROVIDER <name>` followed by `memman embed reembed`. For a Postgres store, use `embed swap`.

- It refuses to run when the active store uses Postgres. Otherwise, it skips any Postgres stores.
- It re-embeds a current memory whose model or vector width differs from the target, or that has no vector, and skips the rest. A superseded or forgotten memory keeps its old vector. `unsupersede` re-embeds a memory when it returns to recall.
- It keeps a per-store cursor, so a second run resumes where the first stopped.
- At the end of each store it writes the fingerprint.
- `--dry-run` scans the current memories and writes nothing. It does not need a stopped scheduler.
