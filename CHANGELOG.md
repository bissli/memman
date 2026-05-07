# Changelog

## 0.14.0

### Per-store backend routing

The `MEMMAN_BACKEND` / `MEMMAN_PG_DSN` global keys are gone. Each store
picks its own backend via `MEMMAN_BACKEND_<store>` (with
`MEMMAN_DEFAULT_BACKEND` as the fallback) and, when on Postgres, its own
DSN via `MEMMAN_PG_DSN_<store>` (with `MEMMAN_DEFAULT_PG_DSN` as the
fallback). A `work` store can sit on Postgres while a `default` store
stays on SQLite under the same data dir.

`memman migrate <store>` now writes `MEMMAN_BACKEND_<store>=postgres`
for the migrated store only — other stores are unaffected. Revert
per-store with `memman config set MEMMAN_BACKEND_<store> sqlite` (or
unset the key to fall back to `MEMMAN_DEFAULT_BACKEND`).

The Postgres backend's drain heartbeat moved from the cluster-global
`queue.worker_runs` table to a per-store `store_<name>.worker_runs`
table. The deferred-write queue is always SQLite at
`<data_dir>/queue.db` regardless of any store's backend choice.

The `Insight` dataclass now carries `linked_at` and `enriched_at`
lifecycle stamps. The recall snapshot writer/reader round-trips them;
older snapshots without the keys still load (defaulted to None).

### Validator tighten

`MEMMAN_PG_DSN` (bare canonical) is no longer accepted. The
`PostgresBackendConfig` validator rejects it with a hint pointing at
`MEMMAN_PG_DSN_<store>`. Use the per-store form or the
`MEMMAN_DEFAULT_PG_DSN` cross-store fallback.

### Performance

`embed.registry.get_for(provider, model)` now caches its result for
the process lifetime, so the per-provider `factory()` + `prepare()`
network probe runs once per pair per process instead of once per
`_StoreContext` open.

`queue.mark_failed` reschedules `claimed_at` into the past so the
existing stale-claim arithmetic unlocks a retried row after an
exponential backoff (60 s, 120 s, 240 s, 480 s, capped at the
`STALE_CLAIM_SECONDS=600` ceiling). Permanent-failure rows (bad creds,
429 storms) get a gentle retry curve instead of hammering upstream
every drain tick.

### Removed

- `memman.store.backend.Cluster` Protocol and the `cluster.open()`
  call site (use `factory.open_backend(store, data_dir)` instead).
- `memman.store.backend.QueueBackend` Protocol and
  `PostgresQueueBackend` (the queue is always SQLite).
- `memman.setup.per_store_bootstrap` (one-shot legacy converter; no
  shim layer survives the cutover).
- `tests/test_per_store_keys_bootstrap.py`,
  `tests/test_queue_heartbeat.py`,
  `tests/e2e/test_postgres_round_trip.py` (covered by per-store
  dispatch tests in `tests/test_mixed_backends.py`).
