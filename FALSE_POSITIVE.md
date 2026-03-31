# Audit — False Positives

Findings invalidated by validation swarms. Documented for future audit
reference.

## 1. Dry-run semantic edge count overcount

**Claim**: `create_semantic_edges` increments `count` twice per pair in
dry-run mode (forward + reverse), reporting 2× actual edges.

**Why false**: The code increments `count` once per pair, then inserts two
rows (forward + reverse) inside the same block. Dry-run skips inserts but
the count is already correct — it counts *pairs*, not *rows*.

## 2. `sim_cache` threshold inconsistency

**Claim**: `sim_cache` populated with `s > 0` but vector anchor threshold
is `s > 0.10`, inflating scores for barely-related content (0.01–0.10).

**Why false**: `sim_cache` is intentionally permissive. It feeds the graph
walk reranker, which applies its own weighting. Filtering at 0.10 would
discard nodes reachable via graph edges that legitimately boost relevance.
The two thresholds serve different purposes.

## 3. DOT output edge_label not escaped

**Claim**: `edge_label` from `e.metadata.get('sub_type', '')` is not escaped,
so quotes or newlines break DOT syntax.

**Why false**: `sub_type` values are constrained to a fixed set of ASCII
keywords (`causes`, `enables`, `prevents`, `proximity`, etc.) — they never
contain quotes, newlines, or special characters. No user-controlled input
reaches this label.

## 4. `_scan_insight` missing `last_accessed_at`

**Claim**: `_scan_insight` never populates `last_accessed_at`, so any code
using `insight.last_accessed_at` after a scan gets `None`.

**Why false**: `last_accessed_at` is populated by `get_insight_by_id` (the
primary retrieval path) which runs its own SELECT. `_scan_insight` is used
for bulk listing where `last_accessed_at` is not needed. No caller depends
on this field from `_scan_insight`.

## 5. `parse_timestamp` naive datetime for SQLite `datetime('now')`

**Claim**: `datetime('now')` in migration SQL produces `YYYY-MM-DD HH:MM:SS`
(no timezone). `parse_timestamp` returns a naive datetime that causes
`TypeError` on arithmetic with UTC-aware datetimes.

**Why false**: `datetime('now')` is only used in the narrative-edge migration
(`db.py:239`) which sets `deleted_at` on rows permanently excluded by
`WHERE deleted_at IS NULL`. Those values are never read back through
`parse_timestamp`. All production timestamps use `format_timestamp` (RFC3339+Z).

## 6. `diff()` returns up to 2x limit matches

**Claim**: keyword_search returns up to `limit`, then cosine pass adds up to
another `limit`. The merged `matches` list is never capped.

**Why false**: Callers only use `matches[best_match_idx]` (a single element by
index). No caller iterates the full list or treats its length as meaningful.
The uncapped length has no observable effect.

## 7. `link_pending` missing `update_entities` call

**Claim**: `link_pending` calls `_prepare_entities(insight)` but never
calls `update_entities()`, leaving the entities column stale.

**Why false**: Entities are already persisted at remember-time via
`cli.py:304-305` (`update_entities(db, insight.id, insight.entities)` inside
`tx_body`). The `_prepare_entities` call in linking re-derives the same
result for in-memory use by `create_semantic_edges`.

## 8. Entity IDF off-by-one drops ubiquitous entities

**Claim**: `doc_freq = count + 1` makes an entity in all N docs get
`doc_freq = N+1 > total_docs = N`, returning weight 0.0 from
`entity_idf_weight`.

**Why false**: The `+ 1` reconstructs true doc frequency from the
exclude-filtered count (`count_insights_with_entity` excludes the current
insight). `doc_freq >= total_docs` returning 0.0 is standard IDF
(`log(N/N) = 0`) — ubiquitous entities are correctly suppressed as they
provide zero discriminative power for linking.

## 9. Antonym check should be gated behind similarity threshold

**Claim**: Both the antonym check and the negation check in
`classify_suggestion` should be moved after the `similarity < 0.55` gate
to prevent false CONFLICT on unrelated texts.

**Why false**: Antonym pairs require matching contradictory terms in BOTH
texts (e.g., "single-threaded" in one and "multi-threaded" in the other).
This is a strong enough signal to fire CONFLICT regardless of token
similarity — two texts mentioning opposite terms about the same concept
are genuinely contradictory even when their overall token overlap is low.
The e2e test `test_contradicting_facts_flagged_on_store` confirms this:
"Redis is single-threaded" vs "Redis supports multi-threaded IO" has low
token similarity but is a real contradiction. Only the negation check
(single-word "not"/"never"/etc. in either text) needed gating.

## 10. Edge count inflation on `insert_edge` failure

**Claim**: `count += 1` in `create_entity_edges` and `create_semantic_edges`
runs outside the `try` block, so count increments even when `insert_edge`
raises. This inflates reported edge counts and causes `MAX_TOTAL_ENTITY_EDGES`
to fire prematurely.

**Why false**: `insert_edge` uses UPSERT (`ON CONFLICT DO UPDATE SET weight =
MAX(weight, excluded.weight)`) — duplicate keys are absorbed without raising.
The only failure modes are catastrophic (disk full, corruption) where count
accuracy is irrelevant and the surrounding system would also be failing.
Additionally, `temporal.py` was wrongly flagged — its `count += 1` is already
inside the `try` block.

## 11. Unbounded `--depth` in `related` CLI command

**Claim**: No upper limit on `--depth` could cause resource exhaustion with
large values via BFS traversal.

**Why false**: BFS loads all insights and all edges into memory upfront
(fixed cost regardless of depth) and uses a `visited` set ensuring each node
is processed at most once — computation is O(V+E) regardless of depth.
`MAX_INSIGHTS = 1000` caps graph size system-wide. A depth of 2,000,000 on a
1,000-node graph does no more work than a depth of 999. The depth parameter
is cosmetic beyond the graph's natural diameter.
