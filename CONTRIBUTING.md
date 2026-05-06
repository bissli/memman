# Contributing to memman

memman is a single-user CLI memory daemon. The default storage backend is SQLite; Postgres is supported as an optional backend (`memman[postgres]` extra). This file covers the short list of conventions that aren't obvious from the code.

## Development setup

```bash
make dev           # poetry install (editable)
make test          # run the unit/integration suite (pytest)
make e2e           # run the end-to-end pytest suite against real DBs
```

The project uses Poetry; run commands via `poetry run <cmd>` or inside `poetry shell`.

## Configuration

memman resolves env vars from a single canonical source at runtime: `<MEMMAN_DATA_DIR>/env`, a `KEY=VALUE` file at mode 0600 (default `~/.memman/env`). Shell environment variables are NOT consulted at runtime for installable settings, so a stale shell export cannot silently override the file. There is no code-default fallback at runtime. Defaults live in `config.INSTALL_DEFAULTS` and are consumed only by `memman install`, which writes them to the env file when the file lacks a value. If a key is missing from the file, the resolver returns `None` and the caller raises a `ConfigError` with "run `memman install`" guidance.

`memman install` performs a one-time pull from the current shell into the env file. Precedence per key: existing file value > wizard prompt (TTY only) > `os.environ` > OpenRouter `/models` resolver (FAST/SLOW only) > `INSTALL_DEFAULTS`. Existing file values are sticky -- a later shell export never overrides them on reinstall, so there is no override path through install. Mandatory secrets (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`) must be present in the file, the shell, or be supplied via the wizard prompt; install fails loud otherwise. After install, interactive `memman recall`, `memman doctor`, and the scheduler-driven worker all read from the file; the keys never need to be re-exported in subsequent shells.

Process-control variables (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`) are read directly from `os.environ` by their owners -- they are deliberately excluded from the env-file model.

The install wizard adds three flags: `--backend [sqlite|postgres]` selects the storage backend (sqlite default; postgres hidden until the `memman[postgres]` extra and the `memman.backend.postgres` module are both available); `--pg-dsn URL` provides the Postgres DSN non-interactively; `--no-wizard` disables prompts so flags + defaults drive the install. Conflicts between an `INSTALLABLE_KEYS` flag and an existing env-file value are rejected loudly with the exact `memman config set ...` command to run -- install never silently swallows a flag.

`memman config set KEY VALUE` is the explicit override path. Use it to change an `INSTALLABLE_KEYS` value after initial install (switching backends, rotating an API key, updating a DSN). The install command stays sticky-seed by design; `config set` is the unambiguous "I'm changing my mind" verb.

`memman uninstall` strips the secret keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, `MEMMAN_PG_DSN`) from the env file but keeps non-secret settings (including `MEMMAN_BACKEND`), so a later `memman install` resurrects model/provider/backend preferences without re-export.

`memman doctor` includes an `env_completeness` check that warns when a new `INSTALLABLE_KEYS` entry is missing, plus an `optional_extras` check that reports which `memman[extras]` install groups (e.g., `postgres`) resolve at runtime.

### Required keys

| Variable             | Purpose                         |
| -------------------- | ------------------------------- |
| `OPENROUTER_API_KEY` | LLM inference via OpenRouter.   |
| `VOYAGE_API_KEY`     | Voyage AI embeddings (512-dim). |

### Persisted at install (`INSTALLABLE_KEYS`)

Set any of these in your shell before `memman install` and they land in the env file. `memman doctor` shows the resolved value and which layer it came from.

| Variable                          | Purpose                                                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MEMMAN_LLM_PROVIDER`             | Registered provider name (default `openrouter`).                                                                                                                                                 |
| `MEMMAN_LLM_MODEL_FAST`           | Hot-path model id. OpenRouter resolves the latest at install time.                                                                                                                               |
| `MEMMAN_LLM_MODEL_SLOW_CANONICAL` | Worker model for canonical content (fact extraction, reconciliation). OpenRouter resolves the latest at install time.                                                                            |
| `MEMMAN_LLM_MODEL_SLOW_METADATA`  | Worker model for derived metadata (enrichment summaries/keywords, causal-edge inference). OpenRouter resolves the latest at install time.                                                        |
| `MEMMAN_EMBED_PROVIDER`           | `voyage` (default), `openai`, `openrouter`, or `ollama`.                                                                                                                                         |
| `MEMMAN_OPENROUTER_ENDPOINT`      | OpenRouter base URL (default `https://openrouter.ai/api/v1`).                                                                                                                                    |
| `MEMMAN_LOG_LEVEL`                | Logger level when neither `--verbose` nor `--debug` is passed (default `WARNING`).                                                                                                               |
| `MEMMAN_OPENAI_EMBED_API_KEY`     | Secret for the OpenAI-compatible embed provider (`MEMMAN_EMBED_PROVIDER=openai`).                                                                                                                |
| `MEMMAN_OPENAI_EMBED_ENDPOINT`    | Endpoint for the OpenAI-compatible embed provider.                                                                                                                                               |
| `MEMMAN_OPENAI_EMBED_MODEL`       | Model name for the OpenAI-compatible embed provider.                                                                                                                                             |
| `MEMMAN_OPENROUTER_EMBED_MODEL`   | Model id for the OpenRouter embed provider (`MEMMAN_EMBED_PROVIDER=openrouter`). Default `baai/bge-m3`. Reuses `OPENROUTER_API_KEY` and `MEMMAN_OPENROUTER_ENDPOINT`; no separate secret needed. |
| `MEMMAN_OLLAMA_HOST`              | Ollama host URL (default `http://localhost:11434`).                                                                                                                                              |
| `MEMMAN_OLLAMA_EMBED_MODEL`       | Ollama embedding model name (default `nomic-embed-text`).                                                                                                                                        |
| `MEMMAN_BACKEND`                  | Storage backend (`sqlite` default, `postgres` requires the `memman[postgres]` extra).                                                                                                            |
| `MEMMAN_PG_DSN`                   | Postgres DSN (`postgresql://...`); secret. Stripped from the env file on `memman uninstall`.                                                                                                     |
| `MEMMAN_RERANK_PROVIDER`          | Cross-encoder rerank provider (default `voyage`). Used when callers pass `memman recall --rerank`.                                                                                               |
| `MEMMAN_VOYAGE_RERANK_MODEL`      | Voyage rerank model id (default `rerank-2.5-lite`).                                                                                                                                              |

### Process-control vars (NOT persisted in the env file)

These bypass the resolver and are read directly from `os.environ`. Persisting them would either be circular, override per-invocation choices, or leak deployment specifics across hosts.

| Variable                | Why it stays direct                                                        |
| ----------------------- | -------------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`       | Locates the env file itself; persisting it inside the file is circular.    |
| `MEMMAN_STORE`          | Per-invocation override of the active store; not a global default.         |
| `MEMMAN_WORKER`         | Set to `1` by the systemd/launchd unit; enables the rotating worker log.   |
| `MEMMAN_SCHEDULER_KIND` | Deployment directive (set by container entrypoint or auto-detected).       |
| `MEMMAN_DEBUG`          | Runtime toggle; persistent state lives in `~/.memman/debug.state` instead. |

Run `memman doctor` to probe both providers with cheap calls (the `llm_probe` and `embed_probe` checks).

## Conventions

### Schema sources of truth

memman has one schema source of truth per backend. Both are additive-only: column additions and new indexes only -- never `DROP COLUMN`, `RENAME`, `DROP TABLE`, `TRUNCATE`, or column-type/nullability changes.

**SQLite** -- `_BASELINE_SCHEMA` in `src/memman/store/db.py` (per-store DB) and `src/memman/queue.py` (queue DB). There is no `PRAGMA user_version` ladder. Fresh databases are created via `CREATE TABLE IF NOT EXISTS`; existing stores get one-off `ALTER TABLE` invocations against `~/.memman/data/*/memman.db` and `~/.memman/queue.db` if the change is in queue schema.

**Postgres** -- `_PG_BASELINE_SCHEMA` and `_PG_QUEUE_SCHEMA` in `src/memman/store/postgres.py` create per-store schemas (`store_<name>`) and a shared `queue` schema. There is no migration ladder: the baseline is the only schema source. Existing stores receive one-off `ALTER TABLE` invocations against the live cluster when a column is added.

When a schema change is needed:

1. Update the relevant baseline (`_BASELINE_SCHEMA` for SQLite, `_PG_BASELINE_SCHEMA` for Postgres). Fresh databases pick the change up automatically.
2. For existing stores, run a one-off `ALTER TABLE` against every `~/.memman/data/*/memman.db` (SQLite) or every `store_<name>` schema (Postgres).
3. Commit the schema change and the evidence (test asserting the column is present) in the same change.

Do not add a SQLite or Postgres migration ladder; only additive ALTERs are permitted.

### SQLite -> Postgres migration

`memman migrate` (in `src/memman/migrate.py`, wrapping `scripts/import_sqlite_to_postgres.py`) copies a store from SQLite into the configured Postgres backend. It is copy-only -- the SQLite source is never modified -- and idempotent (`ON CONFLICT (id) DO NOTHING` on the `insights` insert path so an interrupted run is safely re-runnable).

Operationally:

- A DSN preflight verifies `select 1`, the `pgvector` extension, and `CREATE` privilege.
- The shared `~/.memman/drain.lock` is held for the duration so a scheduler-fired drain cannot race the SQLite reader.
- Each store runs inside one Postgres transaction with `autocommit=False`; any failure rolls back.
- Per-store schemas are inspected up front and classified ABSENT / EMPTY / POPULATED. The plan is echoed (with the DSN password redacted) and the user must confirm; `--yes` skips the prompt. EMPTY and POPULATED schemas are dropped and recreated; ABSENT schemas are created.
- On success `MEMMAN_BACKEND` is flipped to `postgres` in the env file so the next drain routes to the new database. Revert with `memman config set MEMMAN_BACKEND sqlite` if needed.

Postgres -> SQLite is not implemented; restore from the preserved SQLite source if needed.

### Adding an LLM provider

`src/memman/llm/client.py` defines `LLMProvider` (a `typing.Protocol` declaring `complete(system: str, user: str) -> str`) and a module-level `PROVIDERS: dict[str, Callable[[], LLMProvider]]` registry. Only `openrouter` is registered today.

To add a provider:

1. Create `src/memman/llm/<name>_client.py` with a class that implements `complete(...)`. Use helpers from `memman.llm.shared` (JSON parsing, retry constants, `safe_json`).
2. Add `PROVIDERS['<name>'] = _<name>_factory` in `client.py`, where the factory reads provider-specific env vars and returns a new client instance.
3. Wire provider-specific env vars into `src/memman/config.py` so `enumerate_effective_config()` picks them up.

Voyage is embedding-only and is deliberately NOT part of `LLMProvider`.

### Tests

- Mocking: `tests/conftest.py` patches `OpenRouterClient.complete` and the Voyage HTTP path via an autouse fixture. Tests that need to exercise the real HTTP layer should mark themselves `@pytest.mark.no_mock_llm`.
- `MEMMAN_DEBUG=0` is the safe way to run the suite when `~/.memman/debug.state` is `on` (tracing adds log lines to stderr that break tests parsing stdout).

## Filing changes

- No deprecated code / backward-compat shims. When a name changes, delete the old reader in the same commit.
- Keep `src/memman/config.py` as the canonical list of env vars — do not introduce scattered `os.environ.get('NEW_VAR')` call sites.
- Match existing CLI output conventions: most subcommands emit flat JSON via `_json_out(...)`. Recall wraps in `{results, meta}`.
