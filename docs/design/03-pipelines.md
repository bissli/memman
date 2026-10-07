# 3. Read and write pipelines

[Previous: core concepts](02-concepts.md) | [Design overview](../DESIGN.md) | [Next: lifecycle and embedding](04-lifecycle.md)

## 3.1 The turn and the background worker

The agent runs CLI commands during its turn and waits for each to return. A separate worker processes queued writes. A **drain** is one worker run, scheduled every 60 seconds by default.

| Operation                               | Runs in      | Model work                               |
| --------------------------------------- | ------------ | ---------------------------------------- |
| Validate and queue `remember`           | Agent's turn | None                                     |
| Enrich, embed, and store a queued write | Worker       | Summary generation and content embedding |
| `replace` and `forget`                  | Agent's turn | None                                     |
| Scored `recall`                         | Agent's turn | Query embedding and optional reranking   |
| `recall --basic`                        | Agent's turn | No query embedding or reranking          |

`replace`, `forget`, and recall open normal store sessions. Their store-opening checks can require embedding credentials or send a probe, separate from the model work listed above. Summary generation and content embedding for a new write always run in the worker.

A queued memory becomes searchable only after the worker stores it. Recall reads the store on each call, so it can see a memory saved earlier in the same session.

Stopping the scheduler disables `remember`, `replace`, `forget`, and manual drain triggers. Recall remains available. Maintenance commands that rebuild generated fields require this stopped state ([scheduler controls](../USAGE.md#scheduler)).

## 3.2 Write pipeline: remember

![Write submission, background processing, and storage](../diagrams/04-remember-pipeline.drawio.png)

### Write queuing

`remember` performs these steps before returning:

1. Check that writes are enabled, then validate the text against the [input rules](../USAGE.md#rejected-input).
2. Identify potentially temporary information and report it as advisory `quality_warnings`.
3. Check that the selected store exists, then append a pending entry to `queue.db`, including a new UUID, the selected store, and the caller's author identity. The check also refuses a branch whose merge started. On SQLite the check and the append share one `begin immediate` transaction. `store merge` and `store drop` remove a branch in the same kind of transaction. So a write either reaches the queue before the removal, and the removal then sees it and refuses, or the write meets the missing-store refusal. On Postgres the check is one schema query with `PGCONNECT_TIMEOUT` defaulting to three seconds. A connection error lets the write queue.
4. Look for up to three related current memories. This uses word overlap and calls no model.
5. Return JSON with `action: queued`, `id`, `queue_id`, `store`, `quality_warnings`, and `related`.

Related memories must be at most 1,000 bytes. Their score is shared-word count divided by the square root of the memory's distinct-word count. This favors focused matches. A failed read returns `related_error`; the write remains queued and the command succeeds.

The UUID returned as `id` becomes the memory's persistent ID. The numeric `queue_id` identifies the queue entry, which maintenance can delete after processing.

`replace` also queues a write, with a `replaced_id`. It accepts a current memory or a queued write in the same store. It rejects forgotten or replaced targets and targets with a replacement already queued. Its response includes `replaced_id` instead of `related`.

### Write processing

A systemd timer or launchd agent runs the hidden `scheduler drain` command, and the serve loop runs the same drain inside its own process. An exclusive file lock on `<data dir>/drain.lock` prevents overlapping drains; the operating system releases it if the process exits. A drain that cannot acquire the lock reports `skipped`.

Each drain processes up to 100 entries by default, stopping when it reaches its limit or timeout, empties the queue, or sees the stopped state. Under `scheduler serve`, SIGTERM or SIGINT also stops it.

1. **Claim an entry.** An atomic update claims the oldest eligible pending write and increments its attempt count. A claim older than 600 seconds (`STALE_CLAIM_SECONDS`) can be claimed again, so a crashed drain loses no entry. A replacement waits for its pending target and for earlier replacements in the same store, so the worker stores replacements in queue order. A write that fails or goes stale no longer blocks the entries behind it.
2. **Open the store.** Resolve its backend and embedding fingerprint. Before each write, check that the fingerprint still matches the cached client, because a swap that finishes mid-drain would leave that client writing vectors of the wrong size. An error while opening the store, from the fingerprint check, or from planning (a missing LLM setting or embedding credentials) takes the same backoff retry as an apply error.
3. **Check for a completed attempt.** If any memory already carries the entry's `queue_uuid`, mark the entry done without inserting again. Retired memories count too. The UUID identifies the write across retries, since a backup restore can reset the queue's row id counter.
4. **Resolve a replacement.** Follow an existing replacement chain to its current successor when necessary, recording `redirected_from`.
5. **Generate a summary and embedding.** Enrichment requests a one-sentence summary, with one additional request if no JSON object parses. A summary at least 85% as long as the original content is discarded. The embedder processes the original content.
6. **Commit one transaction.** Retire the replacement target by linking it to the new ID when it is still current, then insert the new memory, save generated fields and markers, and record operations.
7. **Finish the queue entry.** Mark it done, or record an error for retry.

A replacement always creates a new memory with its own content, author, timestamps, summary, and vector. If the target is no longer current at commit time, the worker stores the new memory without a replacement link and records `target_gone` in the result and `target-gone` in the operation log.

### Failure and retry

| Failure                                                                                                                | Outcome                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store cannot open; required configuration or embedding credentials are missing; fingerprint changes; transaction fails | Retry the queued write.                                                                                                                                                                              |
| LLM request or a handled embedding HTTP/runtime error persists after client retries                                    | Save the memory with incomplete generated fields.                                                                                                                                                    |
| Neither enrichment response contains a JSON object                                                                     | Save an empty summary. With a saved vector, this counts as completed enrichment, so later drains do not repeat the paid call. A summary at least 85% as long as the content is dropped the same way. |

Queue retries wait 60, 120, 240, and 480 seconds. After five failed attempts, the entry stays `failed` until an explicit retry ([queue commands](../USAGE.md#queue)).

`enrich_attempted_at` records the attempt. `enriched_at` is set only when both enrichment and a vector were saved. A memory with an attempt but no completion is **stranded** and can be retried through maintenance or `enrich --stranded-only`.

### Maintenance after each drain

Maintenance runs when at least 30 seconds remain in the drain's timeout:

- Delete completed queue entries older than 60 seconds and drain history older than seven days.
- For each store where this drain completed an entry, trim operation logs older than 180 days (`OPLOG_RETENTION_DAYS`), make up to three stranded memories eligible for enrichment (`MAINTENANCE_REENRICH_MAX`), and enrich up to three pending memories (`MAINTENANCE_ENRICH_PENDING_MAX`).
- Unless the time budget is spent, end each store's pass with the operation-log cap, which retains the newest 5,000 entries (`MAX_OPLOG_ENTRIES`), and on SQLite one incremental-vacuum step.

A store with no completed entry in the drain gets no store maintenance. Its stranded memories wait for a later write or an explicit [re-enrichment](../USAGE.md#re-enrichment).

The daily model check runs outside maintenance, so it runs even when maintenance is skipped for lack of time.

### Scheduler implementations

| Host                      | Mechanism                         | Drain timeout in seconds                            |
| ------------------------- | --------------------------------- | --------------------------------------------------- |
| Linux                     | systemd user timer                | `max(60, interval - 20)`                            |
| macOS                     | launchd agent                     | `max(60, interval - 20)`                            |
| Other hosts or containers | Foreground `scheduler serve` loop | `max(10, interval - 10)`; 300 when interval is zero |

The systemd/launchd interval is written into the installed unit. The serve loop takes `--interval`, then the installed `MEMMAN_INTERVAL`, then 60. It checks stopped state while waiting and exits when stopped. SIGTERM or SIGINT lets it finish the current memory before exiting successfully.

## 3.3 LLM calls

### LLM routing

`MemmanLLMClient` handles enrichment and the `doctor` connectivity probe. It posts to `<MEMMAN_ENDPOINT>/chat/completions` using `MEMMAN_LLM_MODEL` and `MEMMAN_API_KEY`.

Each request allows up to 4,096 output tokens and has a 60-second timeout. The client makes up to three attempts. Retriable HTTP responses (429, 500, 502, 503, 504, 529) use one- and two-second waits; empty replies use a 0.1-second wait. The enrichment parser's extra request is separate from these transport retries.

On OpenRouter, the client adds attribution headers. Other endpoints receive none.

The OpenRouter install default is `qwen/qwen3-235b-a22b-2507`. Other endpoints require an explicit model ID, because the installed default is an OpenRouter model ID that another endpoint rejects. memman never changes the selected model on its own ([provider setup](../USAGE.md#provider-setup)).

### Daily model check

For OpenRouter, installation and the worker check public catalogs without an API key or an LLM request:

- `/endpoints/zdr` must list a zero-data-retention endpoint for the exact model id, on any vendor.
- `/models` must not list an expiration date for it.

The worker writes `{model, checked_at, notice}` to `<data dir>/model.state` and skips the check while that record names the configured model and is less than 24 hours old (`CHECK_INTERVAL_SECONDS`). A model change therefore triggers a check on the next drain. A failed fetch keeps the existing notice and restarts the interval. `memman prime` prints the notice when it concerns the configured model. A catalog outage does not stop installation.

The check never selects a replacement model. If enrichment requests fail, the worker still stores the memories without summaries, and re-enrichment supplies the summaries once a working model is set.

### Token accounting

LLM calls identify their stage as `enrichment` or `probe`. Usage is counted for each attempt, including an HTTP 200 response that is empty and subsequently retried. Missing usage blocks and HTTP errors have separate counters; tokens reported in an error response still count.

The drain reports totals in `llm_usage`. Debug events include per-entry usage in `queue_done` and `queue_failed`, plus a drain-level `llm_usage_summary`.

## 3.4 Read pipeline: recall

![Keyword, vector, and recency retrieval followed by reranking](../diagrams/05-recall-pipeline.drawio.png)

Recall considers current memories only. The [command reference](../USAGE.md#recall) describes its output and limits.

### Basic matching

`--basic` bypasses scoring. Every whitespace-separated query word must appear as a substring of the content. Matching is case-insensitive, limited to ASCII case folding on SQLite. Results are newest first.

This path skips query embedding and reranking, but normal store-opening checks still run, so the command can still require an API key and a network call.

### Candidate selection

Scored recall embeds the query with the store's bound model. If embedding fails, it logs a warning and proceeds with keyword and recency retrieval.

| Channel | Ranking                                       | Candidate count          |
| ------- | --------------------------------------------- | ------------------------ |
| Keyword | Number of distinct query terms in the content | `ANCHOR_TOP_K` = 30      |
| Vector  | Positive cosine similarity                    | `RERANK_SHORTLIST` = 100 |
| Recency | Creation time, newest first                   | `ANCHOR_TOP_K` = 30      |

The union of these lists forms the candidate set, with no further cap. The vector channel takes 100 so the reranker sees a full shortlist of the query's nearest memories.

Keyword tokenization lowercases text, splits on any character that is not a Unicode letter or number, and removes stopwords. SQLite uses an FTS5 probe per term; Postgres counts intersections with `kw_tokens`. Both count precomposed text alike. A decomposed accent or a Turkish dotted capital I still splits differently in FTS5, so such a word scores no keyword hit on SQLite.

SQLite computes vector similarities in a matrix product. Postgres uses pgvector, including HNSW for vector candidates. The vector candidate list excludes zero and negative cosines and applies no other floor. A fixed cosine threshold means different things under different embedding models, while the sign boundary means the same under every model. `tests/test_vector_anchor_floor.py` fails if an absolute floor is reintroduced.

### Combined scoring

Reciprocal Rank Fusion (RRF) gives a candidate one contribution per channel that found it:

```text
rrf = sum(1 / (RRF_K + rank))    # RRF_K = 60, ranks start at 1
```

The fused score is normalized across the candidate set and combined with keyword overlap and cosine similarity:

```text
keyword = matched distinct query terms / distinct query terms
anchor  = (rrf - minimum rrf) / (maximum rrf - minimum rrf)
score   = (0.25 * keyword + 0.45 * similarity + 0.15 * anchor) / 0.85
```

The raw weights are `_RERANK_WEIGHTS_RAW`, divided by their sum so the used weights sum to 1. The division rescales the printed score and changes no ranking. The raw values are hand-chosen defaults. No labeled evaluation set is large enough to fit or certify other values.

With no query terms, the keyword term is zero. With equal RRF scores, the anchor term is zero. Missing vectors and nonpositive cosines contribute zero similarity. Candidates sort by the combined score.

Recency contributes only through the anchor term, so a recent memory can appear without a keyword or vector match. The agent judges each row's relevance.

### Reranking and limits

A cross-encoder reads the query and one memory together and scores their relevance directly, so it catches matches that cosine and word overlap miss. Reranking runs when enabled, the query has more than two whitespace-separated words (`MIN_RERANK_TOKENS`), and at least two candidates exist. It scores the query against the original content of the top 100 candidates (`RERANK_SHORTLIST`), then replaces their scores and order. A failed request, including a missing key, preserves the combined ranking and logs a warning.

The client posts `{model, query, documents, top_n}` to `<MEMMAN_ENDPOINT>/rerank`. The default model is `MEMMAN_RERANK_MODEL`, `voyageai/rerank-3-lite`. `MEMMAN_RERANK_ENABLED_<store>` overrides the global `MEMMAN_RERANK_ENABLED` setting. Reranking uses `MEMMAN_API_KEY`. Recall has no rerank flag, so the agent never makes this choice.

Reranking changes what the blend weights decide:

- With more than 100 candidates, the weights decide which ones reach the reranker. In a store holding at least 100 memories with a positive cosine to the query, the vector channel alone fills 100 slots, and keyword and recency hits outside it push the pool past 100.
- With 100 or fewer candidates, the reranker rescores all of them, and the weights have no effect on the final order.
- When reranking does not run (a query of two or fewer words, reranking disabled for the store, or a failed request), the weights set the final order.

The limit is `--limit`, then `MEMMAN_RECALL_LIMIT`, then 20. A positive `--limit` applies last, with no further sort. A larger limit keeps the earlier rows in place. Results stay in relevance order, because a date sort would present them as a timeline. Each line includes `created_at`, so dates remain available. Reranked candidates precede any remaining candidates, which keep their combined scores. A limit over 100, or `--limit 0`, can expose both groups; their scores are not comparable. Scores also cannot be compared across queries. On the basic path, `--limit 0` returns no rows; on the scored path it means no limit.

### Recall trace events

With tracing enabled, `recall_anchors` reports each channel's hits and the candidate union; `recall_rerank` reports shortlist size and changed positions. These events help distinguish poor candidate selection from an ineffective rerank.

Only the drain and serve loop attach the trace file handler. A standalone recall process does not write these events to `~/.memman/logs/debug.log`.

## 3.5 Handling model changes

Prompts, models, and providers change over time. memman does not aim for identical output across versions. It records the models behind each summary and vector, so an operator can compare outcomes by model with SQL. It applies no fixed similarity cutoff, because a cutoff would tie the code to one model's behavior.

| Record                                      | Detects                                 | Check or recovery                                              |
| ------------------------------------------- | --------------------------------------- | -------------------------------------------------------------- |
| `summary_model`                             | LLM model that wrote a summary          | SQL comparison; no command reads it                            |
| `enrich_attempted_at` without `enriched_at` | Incomplete enrichment or missing vector | `enrichment_coverage`; maintenance or `enrich --stranded-only` |
| `embed_fingerprint`                         | Store's bound embedding model           | `embed status`; model swap or re-embed                         |
| `embedding_model`                           | Model recorded for a memory's vector    | `embed reembed` checks model and vector width                  |
| `embed_swap_*`                              | Unfinished model swap                   | `no_stale_swap_meta`; resume or abort                          |

`summary_model` stays null when the enrichment call failed. An empty summary, from an undecodable reply or the summary-length filter, still records the model. After a prompt or model change, `memman enrich` re-runs every current memory. [Chapter 4](04-lifecycle.md) covers embedding model changes, and [re-enrichment](../USAGE.md#re-enrichment) covers the commands.

## 3.6 Store branches

[Chapter 2](02-concepts.md#store-branches) defines a branch and its identity. A branch is an empty SQLite store whose `OverlayBackend` (`store/overlay.py`) reads the parent live. The three verbs run in the agent's turn, hold no row copy, and make no LLM or embedding call. `merge` and `drop` hold the drain lock. Each verb refuses before it writes anything.

### Branch

`store branch <parent> <label>` refuses a label that fails the store-name rule or holds `__`, a missing parent, a parent that is itself a branch, a parent with no fingerprint or a corrupt one, and a parent with an embed swap or re-embed in progress. It takes no drain lock. It then:

1. Reads the parent's meta through the parent's own backend, SQLite or Postgres.
2. Draws a name, drawing again while a store of that name exists, and creates the branch directory.
3. Writes `MEMMAN_BACKEND_<branch>=sqlite`, and the parent's rerank key when it has one.
4. Creates the branch store with no rows and a meta of `embed_fingerprint` (the parent's), `branch_parent`, `branch_created_at`, and `branch_token`. Writes the same token to the parent's meta as `branch_token:<branch>`.

A failure after the directory exists removes the directory and the env keys before raising. The output carries the instruction line that routes a session to the branch: "Use memman store <branch> for this thread: pass --store <branch> to every memman recall, remember, replace, forget and insights show call."

### Reads and writes

The overlay opens the parent read-only on first use, so a plain `remember` and the drain's enrich work while the parent is unreachable. Every id the branch holds, in any state, hides the parent row with that id.

- Recall ranks the branch's rows and the parent's current rows in one pass. It refuses when the parent's embed fingerprint differs from the branch's, and the message names the branch's embed swap. It also refuses when the parent's `branch_token:<branch>` is missing or differs from the branch's token, as after a parent recreated under the same name, pointed at another database, or restored from a backup older than the branch. The message names the fix: repair a parent that points at the wrong database, else `store drop <branch>`, which lists the branch rows to remember again in the parent.
- Opening a branch never compares fingerprints, so the drain, `doctor`, an embed swap, and `drop` work across a parent swap.
- `count_active` and the `status` total count the rows recall sees.
- `remember`, `replace`, and `forget` write only the branch. A `replace` or `forget` of a current parent row copies the row raw into the branch and retires the copy there (copy-on-write). A `replace` or `forget` of a parent row the parent already retired fails and names the current head of its chain.
- Enrich and embed write branch rows only. The drain's chain-follow on a `replace` follows only a successor the branch holds.

### Merge

`store merge <branch>` takes its target from `branch_parent` alone, never from `--store`, `MEMMAN_STORE`, or the active-store file. It refuses a store without `branch_parent`, the active branch, a missing parent, a fingerprint that differs from the parent's (the message names the swap command for the branch), a swap or re-embed in progress in either store, and a parent whose `branch_token:<branch>` is missing or differs from the branch's token. A token mismatch means the parent is a recreated store, points at another database, or came from a backup older than the branch.

1. **Mark.** Write `branch_merging` into the branch's meta. From then the overlay refuses every node write to the branch, and `remember` and `replace` refuse to queue for it. After the parent commit of step 3, `drop` refuses it too, so only another merge finishes it.
2. **Check the queue.** In one `begin immediate` transaction on `queue.db`, look for `pending` or `failed` queue rows for the branch and for a `pending` or `failed` parent `replace` whose target's chain in the parent holds a row the branch retired. A failed one counts because a retry would follow the chain onto the branch's successor and retire it. A refusal here clears the flag.
3. **Apply.** Read the branch, then in one parent transaction: re-read the parent's fingerprint, insert every branch-only row raw (`insert or ignore` on SQLite, `on conflict (id) do nothing` on Postgres, keeping timestamps, summaries, authors, and embeddings), repeat each retirement of a copied row by the table below, and remove the parent's token. A retired branch-only row whose vector width differs from the parent's is inserted with no vector: an embed swap re-embeds current rows only, so such a row keeps the old width. Each applied retirement writes a parent oplog row whose detail is `merged from <branch>`.
4. **Remove.** In one `begin immediate` transaction on `queue.db`: re-check for pending or failed branch rows, delete the branch directory, purge the branch's queue rows, commit. Then delete the branch's per-store env keys.

Ids are random UUIDs, so rows written on either side never collide. Merge leaves a row only the parent holds as it is. The retirement step reads each copied row's state in both stores:

| Branch state | Parent state                                | Result                                    |
| ------------ | ------------------------------------------- | ----------------------------------------- |
| replaced     | current, with or without a live predecessor | `mark_replaced` to the branch's successor |
| replaced     | replaced by the same successor              | done                                      |
| replaced     | replaced by another row, or deleted         | conflict                                  |
| deleted      | current                                     | `soft_delete_current`                     |
| deleted      | current with a live predecessor             | conflict, as `forget` itself would refuse |
| deleted      | deleted                                     | done                                      |
| deleted      | replaced                                    | conflict                                  |

A branch row with `replaced_by` set counts as replaced whatever its `deleted_at` holds. Each write checks that the parent row is still current. A failed check re-reads the row and takes the table again, which covers a row another host's drain changed meanwhile. A conflict leaves the parent row as it is and adds `{id, branch_state, parent_state, branch_successor, parent_head, branch_content, parent_content}` to the output. `parent_head` is the current row that ends the parent's chain from `id`: `id` itself while it is current, and null when the chain ends in a forgotten row. The two contents are the texts of `branch_successor` and `parent_head`. In a `replaced` conflict, step 3 inserted the branch's successor as current, so the parent holds both it and its own state. A `deleted` conflict leaves the parent row current. Merge deletes the branch even when conflicts exist, and the agent settles each entry in the parent with `replace` or `forget`. The output also carries `copied` and `retired`.

A failure in the checks before the parent transaction clears the flag, so the branch stays writable. When the parent transaction fails, it rolls back everything it wrote. Merge leaves the flag set and says to re-run. When a re-run finds the flag set, the parent's token gone, and every branch id already in the parent, it writes nothing to the parent. It takes the retirement table against the parent as it stands, reports each conflict, then runs the removal. Merge discards the branch's own oplog, and a merge-time soft delete carries the merge time.

### Drop

`store drop <branch>` refuses a store without `branch_parent`, a branch that a re-run of merge would finish (the state above), the active branch, a running drain, and a pending or failed queue row for the branch. It reads the branch's current rows, then asks the parent which rows they correct. `replaces` names the parent row at the root of the row's chain in the branch, and is null for a branch-only chain and for every row when the parent cannot answer. The parent read uses `PGCONNECT_TIMEOUT` defaulting to three seconds. A missing parent does not stop the drop. An unreachable parent stops it only when the flag is set, since drop then cannot tell whether a merge wrote the parent. Drop then removes the branch as merge does. After that it removes the parent's token, when the parent holds it. When the token removal fails, drop logs a warning and still returns the listing. When the queue re-check refuses the drop, the token stays, so a later merge still accepts the branch. The output lists each current row as `{id, content, replaces}`, so the agent can re-save a claim unrelated to the thread with `remember --store <parent>` and then write a closing row.
