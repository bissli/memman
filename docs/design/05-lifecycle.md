# 5. Lifecycle & Embedding

[< Back to Design Overview](../DESIGN.md)

---

memman is not append-only, but nothing expires on its own: a stored memory persists until an operator removes it.

## 5.1 Retention

A store is **uncapped** and nothing deletes automatically. Deletion is always an operator action: `memman forget <id>`.

**Rationale.**

- **No count cap.** A memory store's value grows with what it holds, so a capacity limit is the wrong shape for it. The former `MAX_INSIGHTS = 1000` also drove an `auto_prune` that soft-deleted real memories with only an oplog trace, and its predicate (`importance < 4 and access_count < 3`) made the deletions unpredictable rather than gentle: on a store of mostly high-importance rows it pruned nothing and the cap silently failed to bound anything, while on a store of low-importance rows it deleted freely.
- **No retention score either.** A stored `effective_importance` once combined base importance, access frequency, and a 30-day half-life into one number, and an `importance >= 4 or access_count >= 3` predicate exempted rows from the report it fed. Nothing read the column: both backends recomputed the score on every call and wrote the result back, and no query filtered or ordered by it. The column, the predicate, the `insights candidates` report and the `insights protect` command that existed to keep a row off it are all gone.
- **Supersession keeps content.** `replace` never deletes: the corrected row keeps its content behind `superseded_by`, leaves recall and every listing. `memman supersede <old> <new>` links two rows that both already exist; `memman unsupersede <id>` reverses a link once the successor is forgotten; `memman insights show <id> --history` walks the chain. Deletion stays operator-only.
- **Scan cost is not a reason to cap.** Bounded recall latency is the storage layer's problem, not the operator's; see [04-pipelines.md § Smart recall](04-pipelines.md).
- **`MAX_OPLOG_ENTRIES = 5000`**: the oplog is an audit trail, not memory, and a bounded trail is the point. Roughly five operations per insight at the scale where the value was chosen; retained without unbounded growth.

## 5.2 Insights group

Manual inspection lives under the `memman insights` group:

```bash
# Read a single insight by ID (a superseded row shows its successor)
memman insights show <id>

# Walk the supersession chain through an id, oldest first
memman insights show <id> --history

# Resolve a write to the insights it produced
memman insights by-queue <queue_uuid>

# Review stored insights for content quality issues
memman insights review
```

`insights review` scans all active insights against transient content patterns (AWS instance IDs, resource counts, verification receipts, deployment receipts, state observations, line number references) and returns flagged entries sorted by warning count. Since the remember pipeline rejects content with 2+ quality warnings at write time, `insights review` primarily catches insights stored before the hard gate was introduced, or single-warning content that accumulated additional transient characteristics over time.

---

## 5.3 Embedding support

Embeddings power semantic search. Vector dimensionality is provider-defined and recorded in a per-store `meta.embed_fingerprint` (provider, model, dim). Switching a store's embedder is explicit - online via `memman embed swap` (resumable shadow-column backfill) or offline via `memman embed reembed`.

**Per-store embedder sovereignty.** Each store's stored fingerprint is the runtime authority over which embedder client serves that store. Every consumer - drain worker (`_StoreContext`), recall (`bound_embedder(backend)` → the query embedding), graph rebuild, `run_remember(ec=...)` - binds via `embed.fingerprint.bound_embedder(backend)`, which resolves `meta.embed_fingerprint` and dispatches to `embed.registry.get_for(provider, model)`. One process can sequentially open two stores fingerprinted to different providers without env mutation. The operator-facing worked example lives in [USAGE.md § Embedding Operations](../USAGE.md#embedding-operations).

`MEMMAN_EMBED_PROVIDER`'s runtime role narrows to two cases:

1. **Seeding a fresh store** - when a store has no stored fingerprint yet, `seed_if_fresh(backend, get_client())` writes the env-active client's fingerprint into `meta.embed_fingerprint`. After that write, the env var no longer drives runtime selection for that store.
2. **Carrying credentials** - providers read `MEMMAN_VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, etc. from the env file. A store fingerprinted to a provider whose credentials are absent fails at the embed call site (recall warns and degrades to keyword-only; drain marks the row failed via `EmbedCredentialError`).

`memman embed status` reports the store's stored fingerprint and whether credentials for that fingerprint's provider are available. `memman doctor` (`check_embed_fingerprint`) follows the same shape: pass on stored + creds-available, fail on stored-but-missing-creds, fail on populated-store-without-fingerprint (corruption).

### 5.3.1 Supported providers

| Provider     | Default model             | Notes                                                                                                                         |
| ------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `voyage`     | `voyage-3-lite` (512-dim) | Requires `MEMMAN_VOYAGE_API_KEY`.                                                                                             |
| `openai`     | `text-embedding-3-small`  | Requires `MEMMAN_OPENAI_EMBED_API_KEY` + `MEMMAN_OPENAI_EMBED_ENDPOINT`. Any OpenAI-compatible endpoint (vLLM, LiteLLM, ...). |
| `openrouter` | `baai/bge-m3`             | Reuses `MEMMAN_OPENROUTER_API_KEY` + `MEMMAN_OPENROUTER_ENDPOINT`; no separate secret needed.                                 |
| `ollama`     | `nomic-embed-text`        | Local Ollama at `MEMMAN_OLLAMA_HOST` (default `http://localhost:11434`).                                                      |

The wizard ships defaulted to `voyage`; any of the four is selectable.

### 5.3.2 Vector storage

Vector serialization depends on the active storage backend for the store (`MEMMAN_BACKEND_<store>`, falling back to `MEMMAN_DEFAULT_BACKEND`):

- **SQLite** - little-endian float64 BLOB stored in `insights.embedding`; bytes per row = 8 × provider dim (e.g., a 512-dim model writes 4096 bytes).
- **Postgres** - `pgvector` `vector(N)` typed column, persisted as float32 (HNSW-indexed). The migrate path (`PostgresMigrator` in `src/memman/store/postgres.py`) casts SQLite float64 BLOBs to `numpy.float32` before binding to avoid silent rounding by psycopg.

### 5.3.3 Embedding in the pipeline

- **Embed (remember, post-enrichment)**: the row is embedded once, after LLM enrichment - keyword-enriched text when enrichment returned keywords, else the content alone.
- **Recovery (`graph rebuild`)**: re-enriches all insights through the full LLM pipeline and updates embeddings.
- **Recall**: the query string is embedded as given for vector search anchors and reranking.

### 5.3.4 Recovery

`memman graph rebuild` re-enriches all insights through the full LLM pipeline and updates embeddings. The worker owns the embedding lifecycle (post-enrichment, rebuild).

### 5.3.5 Online embedding swap

`memman embed swap` performs a per-store provider/model change without going recall-only. The orchestrator (`src/memman/embed/swap.py`) drives a state machine recorded in per-store meta keys:

```
(idle)  ──swap──▶  backfilling  ──last batch──▶  cutover  ──commit──▶  (idle)
                       ▲   │                          │
                       │   └── --resume               │
                       │                              ▼
                       └────── (continues)         (--abort)
```

| State         | Meaning                                                                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backfilling` | Each batch embeds the next `MEMMAN_EMBED_SWAP_BATCH_SIZE` rows under the new provider into a shadow column. Recall keeps using the live column. The cursor advances per-batch so a crash resumes from where it stopped. |
| `cutover`     | Set immediately before the atomic cutover transaction. See "Cutover details" below.                                                                                                                                     |
| (cleared)     | All `embed_swap_*` meta keys absent; `embed status` shows the new fingerprint.                                                                                                                                          |

**Cutover details.** Postgres uses `CREATE INDEX CONCURRENTLY` (timeout `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT`, default unlimited). SQLite copies the shadow column over the live column. The new fingerprint is written and the swap meta keys are **deleted** - absence is the canonical "no swap in flight" signal, not zeroed sentinel values.

`--abort` drops `embedding_pending` (and any uncommitted side column) and clears the swap meta. `memman doctor`'s `no_stale_swap_meta` check warns if any `embed_swap_*` key remains on a store that is not actively swapping.

`embed reembed` is the offline alternative: it rewrites every store in place with the active provider, requires `memman scheduler stop` first, and is intended for one-shot rewrites (not provider migrations).
