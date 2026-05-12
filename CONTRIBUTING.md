# Contributing to memman

memman is a single-user CLI memory daemon. SQLite is the default backend; Postgres is available via the `memman[postgres]` extra.

## Development setup

```bash
make dev           # poetry install (editable)
make test          # unit/integration suite (pytest)
make e2e           # end-to-end suite against real DBs
```

The project uses Poetry; run commands via `poetry run <cmd>` or inside `poetry shell`.

## Configuration

The operator-facing model (env file location, install precedence, override path) lives in [USAGE.md § Configuration](docs/USAGE.md#configuration). The contributor-side facts:

- Defaults live in `config.INSTALL_DEFAULTS` and are written to `<MEMMAN_DATA_DIR>/env` by `memman install` only — there is no code-default fallback at runtime. If a key is missing from the env file, the resolver returns `None` and the caller raises `ConfigError` with `run memman install` guidance.
- Process-control variables (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`) are read directly from `os.environ` and excluded from the env-file model.
- `memman doctor` has an `env_completeness` check that warns when a new `INSTALLABLE_KEYS` entry is missing, and an `optional_extras` check that reports which `memman[extras]` install groups resolve at runtime.

### Variable reference

The `Type` column distinguishes how each variable is sourced:

- `required` — must be present in the env file before any command runs (`memman install` prompts for these in TTY mode and fails otherwise).
- `installed` — optional INSTALLABLE_KEYS; seeded by `memman install` from defaults or a one-time shell pull, then read from the env file. Override later with `memman config set KEY VALUE`.
- `process` — never persisted; read directly from `os.environ` by the component that owns them.

| Variable                          | Type      | Purpose                                                                                                                                                                                                                             |
| --------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_OPENROUTER_API_KEY`       | installed | OpenRouter API key. Required when `MEMMAN_EMBED_PROVIDER=openrouter`, and reused as the LLM bearer token when `MEMMAN_LLM_ENDPOINT` points at OpenRouter (the install wizard auto-fills `MEMMAN_LLM_API_KEY` from it in that case). |
| `MEMMAN_VOYAGE_API_KEY`           | installed | Voyage AI embeddings (512-dim). Required when `MEMMAN_EMBED_PROVIDER=voyage` (the default).                                                                                                                                         |
| `MEMMAN_LLM_ENDPOINT`             | installed | OpenAI-compatible `/chat/completions` endpoint URL (default `https://openrouter.ai/api/v1`). Set to any vendor's OpenAI-compat URL to switch providers.                                                                             |
| `MEMMAN_LLM_API_KEY`              | installed | Bearer token for the configured LLM endpoint. Required for any non-loopback endpoint; loopback endpoints (Ollama, local vLLM/LiteLLM) may leave it blank.                                                                           |
| `MEMMAN_LLM_MODEL_FAST`           | installed | Hot-path model id. OpenRouter endpoints resolve the latest at install time; other endpoints are prompted interactively.                                                                                                             |
| `MEMMAN_LLM_MODEL_SLOW_CANONICAL` | installed | Worker model for canonical content (fact extraction, reconciliation). OpenRouter endpoints resolve the latest at install time; other endpoints are prompted interactively.                                                          |
| `MEMMAN_LLM_MODEL_SLOW_METADATA`  | installed | Worker model for derived metadata (enrichment summaries/keywords, causal-edge inference). OpenRouter endpoints resolve the latest at install time; other endpoints are prompted interactively.                                      |
| `MEMMAN_EMBED_PROVIDER`           | installed | `voyage` (default), `openai`, `openrouter`, or `ollama`.                                                                                                                                                                            |
| `MEMMAN_OPENROUTER_ENDPOINT`      | installed | OpenRouter base URL (default `https://openrouter.ai/api/v1`).                                                                                                                                                                       |
| `MEMMAN_LOG_LEVEL`                | installed | Logger level when neither `--verbose` nor `--debug` is passed (default `WARNING`).                                                                                                                                                  |
| `MEMMAN_OPENAI_EMBED_API_KEY`     | installed | Secret for the OpenAI-compatible embed provider (`MEMMAN_EMBED_PROVIDER=openai`).                                                                                                                                                   |
| `MEMMAN_OPENAI_EMBED_ENDPOINT`    | installed | Endpoint for the OpenAI-compatible embed provider.                                                                                                                                                                                  |
| `MEMMAN_OPENAI_EMBED_MODEL`       | installed | Model name for the OpenAI-compatible embed provider.                                                                                                                                                                                |
| `MEMMAN_OPENROUTER_EMBED_MODEL`   | installed | Model id for the OpenRouter embed provider (`MEMMAN_EMBED_PROVIDER=openrouter`). Default `baai/bge-m3`. Reuses `MEMMAN_OPENROUTER_API_KEY` and `MEMMAN_OPENROUTER_ENDPOINT`; no separate secret needed.                             |
| `MEMMAN_OLLAMA_HOST`              | installed | Ollama host URL (default `http://localhost:11434`).                                                                                                                                                                                 |
| `MEMMAN_OLLAMA_EMBED_MODEL`       | installed | Ollama embedding model name (default `nomic-embed-text`).                                                                                                                                                                           |
| `MEMMAN_DEFAULT_BACKEND`          | installed | Fallback storage backend for stores without an explicit per-store override (`sqlite` default; `postgres` requires the `memman[postgres]` extra).                                                                                    |
| `MEMMAN_DEFAULT_POSTGRES_DSN`     | installed | Fallback Postgres DSN (`postgresql://...`); secret. Stripped on `memman uninstall`.                                                                                                                                                 |
| `MEMMAN_BACKEND_<store>`          | installed | Per-store backend override (e.g., `MEMMAN_BACKEND_work=postgres`). Not seeded by `install`; set via `memman config set` or written by `memman migrate <store>`.                                                                     |
| `MEMMAN_POSTGRES_DSN_<store>`     | installed | Per-store Postgres DSN override; secret. Stripped on `memman uninstall`.                                                                                                                                                            |
| `MEMMAN_RERANK_PROVIDER`          | installed | Cross-encoder rerank provider (default `voyage`). Used when rerank is enabled for the active store.                                                                                                                                 |
| `MEMMAN_VOYAGE_RERANK_MODEL`      | installed | Voyage rerank model id (default `rerank-2.5-lite`).                                                                                                                                                                                 |
| `MEMMAN_RERANK_ENABLED`           | installed | Global default for whether `memman recall` runs the cross-encoder rerank stage (default `true`). Per-store override via `MEMMAN_RERANK_ENABLED_<store>`.                                                                            |
| `MEMMAN_RERANK_ENABLED_<store>`   | installed | Per-store override of the rerank-enabled toggle (e.g., `MEMMAN_RERANK_ENABLED_work=false`). Set via `memman config set`.                                                                                                            |
| `MEMMAN_DATA_DIR`                 | process   | Locates the env file itself; persisting it inside the file is circular.                                                                                                                                                             |
| `MEMMAN_STORE`                    | process   | Per-invocation override of the active store; not a global default.                                                                                                                                                                  |
| `MEMMAN_WORKER`                   | process   | Set to `1` by the systemd/launchd unit; enables the rotating worker log.                                                                                                                                                            |
| `MEMMAN_SCHEDULER_KIND`           | process   | Deployment directive (set by container entrypoint or auto-detected).                                                                                                                                                                |
| `MEMMAN_DEBUG`                    | process   | Runtime toggle; persistent state lives in `~/.memman/debug.state` instead.                                                                                                                                                          |

## Conventions

### Schema sources of truth

Both backends use one schema source of truth per backend, additive-only: column additions and new indexes only — never `DROP COLUMN`, `RENAME`, `DROP TABLE`, `TRUNCATE`, or column-type/nullability changes.

**SQLite** — `_BASELINE_SCHEMA` in `src/memman/store/db.py` (per-store DB) and `src/memman/queue.py` (queue DB). No `PRAGMA user_version` ladder. Fresh databases are created via `CREATE TABLE IF NOT EXISTS`; existing stores get one-off `ALTER TABLE` invocations against `~/.memman/data/*/memman.db` and `~/.memman/queue.db`.

**Postgres** — `PG_BASELINE_SCHEMA` in `src/memman/store/postgres.py` creates per-store schemas (`store_<name>`), each carrying its own `worker_runs` table for drain heartbeats. The deferred-write queue is always SQLite (`<data_dir>/queue.db`); Postgres has no shared queue schema. Existing stores receive one-off `ALTER TABLE` invocations against the live Postgres server when a column is added.

When a schema change is needed:

1. Update the relevant baseline (`_BASELINE_SCHEMA` for SQLite, `PG_BASELINE_SCHEMA` for Postgres). Fresh databases pick the change up automatically.
2. For existing stores, run a one-off `ALTER TABLE` against every `~/.memman/data/*/memman.db` (SQLite) or every `store_<name>` schema (Postgres).
3. Commit the schema change and the evidence (test asserting the column is present) in the same change.

Do not add a SQLite or Postgres migration ladder; only additive ALTERs are permitted.

### Migrating between SQLite and Postgres

Operational details:

- A preflight verifies `select 1`, the `pgvector` extension, and `CREATE` privilege on the Postgres side.
- The shared `~/.memman/drain.lock` is held for the duration so a scheduler-fired drain cannot race the source reader.
- Per-store work runs inside one transaction (`autocommit=False` on Postgres); any failure rolls back.
- Per-store schemas are inspected up front and classified ABSENT / EMPTY / POPULATED. The plan is echoed (with the DSN password redacted) and confirmed; `--yes` skips the prompt. `--dry-run` is supported only with `--to postgres`.
- For `--to sqlite`, the Postgres `store_<name>` schema is dumped under `<data_dir>/archive/` before being dropped, so the source remains recoverable.
- On success `MEMMAN_BACKEND_<store>` is flipped to the target backend. Revert without re-migrating data via `memman config set MEMMAN_BACKEND_<store> <backend>` (or unset the key to fall back to `MEMMAN_DEFAULT_BACKEND`).

The two migrators (`SqliteMigrator`, `PostgresMigrator`) extend a common `Migrator` ABC and dispatch from a `BACKENDS` registry; adding a new RDBMS backend means implementing the ABC, not forking the CLI.

### LLM dispatch

`src/memman/llm/client.py` defines a single concrete `MemmanLLMClient` that posts to whatever URL `MEMMAN_LLM_ENDPOINT` resolves to (default `https://openrouter.ai/api/v1`). All reachable frontier vendors expose an OpenAI-compatible `/chat/completions` shim, so one wire protocol covers OpenRouter, OpenAI, Anthropic (`https://api.anthropic.com/v1`), Google (`https://generativelanguage.googleapis.com/v1beta/openai`), Ollama, vLLM, LiteLLM, and any other OpenAI-compat endpoint. There is no provider registry and no per-vendor subclass — switching providers is an `MEMMAN_LLM_ENDPOINT` / `MEMMAN_LLM_API_KEY` edit.

The client adds OpenRouter attribution headers (`HTTP-Referer`, `X-Title`) when `config.is_openrouter_endpoint(endpoint)` matches; for any other host the request goes through plain. Loopback endpoints (`localhost`, `127.0.0.1`, `::1`) may omit `MEMMAN_LLM_API_KEY` — the client drops the `Authorization` header when the key is blank.

Per-role model selection comes from `MEMMAN_LLM_MODEL_FAST` / `_SLOW_CANONICAL` / `_SLOW_METADATA`. For OpenRouter endpoints these are resolved against `/v1/models` at install time; for non-OpenRouter endpoints the install wizard prompts interactively for each slug.

Retry policy, timeouts, and JSON parsing helpers live in `src/memman/_http.py` and `src/memman/llm/shared.py` respectively; embed providers (`voyage`, `openai`, `openrouter`, `ollama`) are a separate subsystem under `src/memman/embed/` keyed by `MEMMAN_EMBED_PROVIDER`.

### Tests

- Mocking: `tests/conftest.py` patches `memman.llm.client.MemmanLLMClient.complete` and the Voyage HTTP path via an autouse fixture. Tests that need to exercise the real HTTP layer should mark themselves `@pytest.mark.no_mock_llm`.
- `MEMMAN_DEBUG=0` is the safe way to run the suite when `~/.memman/debug.state` is `on` (tracing adds log lines to stderr that break tests parsing stdout).

## Filing changes

- No deprecated code or backward-compat shims. When a name changes, delete the old reader in the same commit.
- Keep `src/memman/config.py` as the canonical list of env vars — no scattered `os.environ.get('NEW_VAR')` call sites.
- Match existing CLI output conventions: most subcommands emit flat JSON via `_json_out(...)`. Recall wraps in `{results, meta}`.
