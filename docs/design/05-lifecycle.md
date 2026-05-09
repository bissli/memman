# 5. Lifecycle & Embedding

[< Back to Design Overview](../DESIGN.md)

---

MemMan is not an append-only system. Effective memory management requires important memories to persist while outdated ones naturally decay.

![Lifecycle & Retention](../diagrams/06-lifecycle-retention.drawio.png)

## 5.1 Effective Importance (EI)

EI combines base importance, access frequency, time decay, and graph connectivity:

```
EI = base_weight(importance) × access_factor × decay_factor × edge_factor

base_weight:   imp 5 → 1.0,  4 → 0.8,  3 → 0.5,  2 → 0.3,  1 → 0.15
access_factor: max(1.0, log(1 + access_count))
decay_factor:  0.5 ^ (days_since_access / 30)     // half-life of 30 days
edge_factor:   1.0 + 0.1 × min(edge_count, 5)     // up to +0.5
```

Interpretation:
- **High importance** -> higher base score
- **Frequent access** -> logarithmic growth bonus
- **Long period without access** -> exponential decay (halves every 30 days)
- **Rich graph connections** -> indicates relevance to other knowledge, bonus applied

**Rationale:**

- **`base_weight` (1.0, 0.8, 0.5, 0.3, 0.15)**: Non-linear spacing creates a 6.7:1 ratio between importance 5 and 1. The largest gap (0.3→0.15) falls between importance 2→1, while the 0.8→0.5 gap between importance 4→3 reinforces the immunity boundary — importance 4+ is the "protected" tier.
- **`HALF_LIFE_DAYS = 30`**: One calendar month. Balances retention vs decay across typical project lifecycles: at 30 days EI halves, at 60 days quarters, at 90 days ~12.5%. Inspired by Ebbinghaus forgetting curve research but not derived from a specific paper — chosen as a round-number approximation for monthly project cadence. Not from MAGMA (the paper has no decay mechanism).
- **`edge_factor`: cap at 5 edges, +0.1 per edge (max +50%)**: Prevents highly-connected hub nodes from becoming permanently immune through connectivity alone.

## 5.2 Immunity Rules

The following insights are exempt from automatic cleanup:
- `importance >= 4` (high-value memories)
- `access_count >= 3` (frequently retrieved)

**Rationale:**

- **`importance >= 4`**: Follows directly from the importance scale definition — importance 4 = "immune to auto-pruning" (Section 2.1).
- **`access_count >= 3`**: Three independent retrievals provide statistical evidence of genuine utility, not coincidental access. The threshold is deliberately low — in a personal memory system, even two recalls suggest real value, but three provides a safety margin.

## 5.3 Auto-Pruning

Triggered when the total number of active insights exceeds **1000**:

1. Compute EI for all insights
2. Exclude immune insights
3. Take the lowest EI entries in ascending order (up to 10 per batch)
4. Soft-delete (set `deleted_at`)
5. Cascade-delete related edges

**Rationale:**

- **`MAX_INSIGHTS = 1000`**: Practical capacity for a single-user CLI memory system. Keeps SQLite scan cost bounded. Not from MAGMA (the paper specifies no storage capacity limit; its `Max Nodes: 200` is a per-query traversal budget, a different concept).
- **`PRUNE_BATCH_SIZE = 10`**: ~1% of MAX_INSIGHTS. Limits write amplification per `remember` call — a single insert never cascades into mass deletion.
- **`MAX_OPLOG_ENTRIES = 5000`**: 5× MAX_INSIGHTS; retains approximately five operations per insight on average. Sufficient audit trail without unbounded growth.

## 5.4 Insights Group

Manual lifecycle management lives under the `memman insights` group:

```bash
# View low-retention candidates (read-only — does NOT delete)
memman insights candidates --threshold 0.5

# Retain a specific insight (increases access_count by +3)
memman insights protect <id>

# Review stored insights for content quality issues
memman insights review

# Read a single insight by ID
memman insights show <id>
```

`insights review` scans all active insights against transient content patterns (AWS instance IDs, resource counts, verification receipts, deployment receipts, state observations, line number references). Returns flagged entries sorted by warning count. Note: since the remember pipeline now **rejects** content with 2+ quality warnings at write time, `insights review` primarily catches insights stored before the hard gate was introduced, or single-warning content that accumulated additional transient characteristics over time.

**Rationale:**

- **`boost_retention +3`**: Deliberately matches the immunity threshold (`access_count >= 3`). A single `insights protect` guarantees immunity regardless of prior access count — the insight crosses the threshold immediately.

---

## 5.5 Embedding Support

Embeddings power semantic search and graph connectivity. Vector dimensionality is provider-defined and recorded in a per-store `meta.embed_fingerprint` (provider, model, dim). Switching a store's embedder is always explicit -- online via `memman embed swap` (resumable shadow-column backfill) or offline via `memman embed reembed`. There is never a silent migration.

**Per-store embedder sovereignty.** Each store's stored fingerprint is the runtime authority over which embedder client serves that store. Every consumer -- drain worker (`_StoreContext`), recall (`bound_embedder(backend)` -> `intent_aware_recall(fingerprint=...)`), snapshot writes/reads, graph rebuild, and `run_remember(ec=...)` -- binds via `embed.fingerprint.bound_embedder(backend)`, which resolves `meta.embed_fingerprint` and dispatches to `embed.registry.get_for(provider, model)`. As a result, one process can sequentially open store A on Voyage and store B on OpenAI without env mutation; `MEMMAN_EMBED_PROVIDER=voyage memman --store openai_store recall ...` succeeds.

`MEMMAN_EMBED_PROVIDER`'s runtime role narrows to two cases:

1. **Seeding a fresh store** -- when a store has no stored fingerprint yet, `seed_if_fresh(backend, get_client())` writes the env-active client's fingerprint into `meta.embed_fingerprint`. After that one write, the env var no longer drives runtime selection for that store.
2. **Carrying credentials** -- providers read `VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, etc. from process env. A store fingerprinted to a provider whose credentials are absent in the current process will fail at the embed call site (recall warns and degrades to keyword-only; drain marks the row failed via `EmbedCredentialError`).

`memman embed status` reports the store's stored fingerprint and whether credentials for that fingerprint's provider are available; it does not compare against an "env-active" fingerprint (that comparison is malformed under sovereignty). `memman doctor` (`check_embed_fingerprint`) follows the same shape: pass on stored + creds-available, fail on stored-but-missing-creds, fail on populated-store-without-fingerprint (corruption).

### 5.5.1 Supported providers

| Provider           | Default model             | Notes                                                                                                                         |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `voyage` (default) | `voyage-3-lite` (512-dim) | Requires `VOYAGE_API_KEY`. Default; tuned thresholds (e.g., `AUTO_SEMANTIC_THRESHOLD = 0.62`) target this model.              |
| `openai`           | `text-embedding-3-small`  | Requires `MEMMAN_OPENAI_EMBED_API_KEY` + `MEMMAN_OPENAI_EMBED_ENDPOINT`. Any OpenAI-compatible endpoint (vLLM, LiteLLM, ...). |
| `openrouter`       | `baai/bge-m3`             | Reuses `OPENROUTER_API_KEY` + `MEMMAN_OPENROUTER_ENDPOINT`; no separate secret needed.                                        |
| `ollama`           | `nomic-embed-text`        | Local Ollama at `MEMMAN_OLLAMA_HOST` (default `http://localhost:11434`).                                                      |

### 5.5.2 Vector Storage

Vector serialization depends on the active storage backend for the store (`MEMMAN_BACKEND_<store>`, falling back to `MEMMAN_DEFAULT_BACKEND`):

- **SQLite** — little-endian float64 BLOB stored in `insights.embedding` (e.g., 512 × 8 = 4096 bytes for `voyage-3-lite`).
- **Postgres** — `pgvector` `vector(N)` typed column, persisted as float32 (HNSW-indexed). The migrate path (`PostgresMigrator` in `src/memman/store/postgres.py`) explicitly casts SQLite float64 BLOBs to `numpy.float32` before binding to avoid silent rounding by psycopg.

> **Threshold recalibration.** The semantic-edge auto-link threshold (`AUTO_SEMANTIC_THRESHOLD = 0.62`) is calibrated for `voyage-3-lite`. Different providers and dimensionalities produce different similarity distributions; if you switch provider, recalibrate from observed pairwise distributions.

### 5.5.3 Embedding in the Pipeline

- **Initial (remember — sequential)**: Each fact is embedded immediately after extraction
- **Merged (remember — sequential)**: If reconciliation merges facts, the merged text is re-embedded
- **Enriched (remember — parallel)**: After LLM enrichment extracts keywords, the insight is re-embedded with enriched text (content + keywords)
- **Recovery (`graph rebuild`)**: Re-enriches all insights through the full LLM pipeline and updates embeddings
- **Recall**: Expanded query is embedded for vector search anchors and reranking

### 5.5.4 Recovery

`memman graph rebuild` re-enriches all insights through the full LLM pipeline and updates embeddings. There is no separate operator command for embedding maintenance — the worker owns the embedding lifecycle (initial, merged, enriched, rebuild).

### 5.5.5 Online Embedding Swap

`memman embed swap` performs a per-store provider/model change without going recall-only. The orchestrator (`src/memman/embed/swap.py`) drives a small state machine recorded in per-store meta keys:

```
(idle)  ──swap──▶  backfilling  ──last batch──▶  cutover  ──commit──▶  (idle)
                       ▲   │                          │
                       │   └── --resume               │
                       │                              ▼
                       └────── (continues)         (--abort)
```

| State         | Meaning                                                                                                                                                                                                                                                                                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backfilling` | Each batch embeds the next `MEMMAN_EMBED_SWAP_BATCH_SIZE` rows under the new provider into a shadow column (`embedding_pending` on SQLite, side column on Postgres). Recall keeps using the live column. The cursor advances per-batch so a crash resumes from where it stopped.                                                                                  |
| `cutover`     | Set immediately before the atomic cutover transaction. Postgres uses `CREATE INDEX CONCURRENTLY` (timeout `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT`, default unlimited); SQLite copies the shadow column over the live column. The new fingerprint is written and the swap meta keys are **deleted** (not zeroed -- absence is the canonical "no swap in flight" signal). |
| (cleared)     | All `embed_swap_*` meta keys absent; `embed status` shows the new fingerprint.                                                                                                                                                                                                                                                                                    |

`--abort` drops `embedding_pending` (and any uncommitted side column) and clears the swap meta. `memman doctor`'s `no_stale_swap_meta` check warns if any `embed_swap_*` key remains on a store that is not actively swapping -- this guards against future cutover regressions leaking sentinel rows into the meta table.

`embed reembed` is the offline alternative: it rewrites every store in place with the active provider, requires `memman scheduler stop` first, and is intended for one-shot rewrites (not provider migrations).
