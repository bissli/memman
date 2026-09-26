# Contributing to memman

memman is a single-user command-line memory store for Claude Code. A scheduled worker enriches each memory and creates its embedding. SQLite is the default backend. The `memman[postgres]` extra adds Postgres.

## Development setup

```bash
make dev         # poetry install: editable, with dev dependencies
make test        # unit and integration tests (pytest, skips tests/e2e)
make test-live   # the same tests against the real provider APIs
make e2e         # end-to-end tests in tests/e2e
make diagrams    # render each docs/diagrams/*.drawio to .drawio.png
```

The project uses Poetry. `poetry run <cmd>` runs a command inside the project environment.

- `make test-live` reads the API keys from the shell, then from `~/.memman/env`.
- The Postgres tests and the container end-to-end tests start containers through testcontainers, so they need Docker.
- `make diagrams` needs `/snap/bin/drawio` and `xvfb-run`.

## Configuration

[USAGE.md](docs/USAGE.md#configuration) describes the env file, the order in which installation uses configuration sources, and `memman config set`. [Variable reference](#variable-reference) lists every variable. Contributors need to know the following:

- `src/memman/config.py` names every variable as a module constant. `INSTALLABLE_KEYS` lists the settings `memman install` writes to `<data dir>/env`, and `INSTALL_DEFAULTS` holds their defaults.
- A new setting is one `INSTALLABLE_KEYS` entry, plus an `INSTALL_DEFAULTS` row when it has a default. A secret also goes in `SECRET_VARS`, which `memman uninstall` strips from the env file and `memman config show` redacts.
- `config.get` has no code default. It reads the env file only and returns `None` for a missing key. `config.require` raises `ConfigError`, which tells the user to run `memman install`.
- Installation copies a missing key from the shell once. It reads the `MEMMAN_` name first, then the vendor name for three keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, `OPENAI_API_KEY`). A value already in the file takes precedence.
- Process variables (`MEMMAN_DATA_DIR`, `MEMMAN_STORE`, `MEMMAN_WORKER`, `MEMMAN_DEBUG`, `MEMMAN_SCHEDULER_KIND`, `MEMMAN_AUTHOR`) and tuning variables (`MEMMAN_EMBED_SWAP_BATCH_SIZE`, `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT`, `MEMMAN_REINDEX_TIMEOUT`) come from `os.environ` and are never written to the env file.
- `memman doctor` runs `env_completeness`, which warns when the env file lacks an `INSTALLABLE_KEYS` entry, and `optional_extras`, which reports which `memman[...]` extras can be imported. `env_completeness` skips `MEMMAN_LLM_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`, the three backup settings, and `MEMMAN_DEFAULT_POSTGRES_DSN` unless the default backend is `postgres`.

### Variable reference

The `Type` column says where each variable comes from:

- `installed` - an `INSTALLABLE_KEYS` entry. `memman install` sets it from a flag, a wizard answer, the shell, or `INSTALL_DEFAULTS`, and memman then reads it from the env file. `memman config set KEY VALUE` changes it.
- `per store` - an env-file key named for one store. Installation sets it only for the `default` store, when the wizard or `--backend` picks the backend. `memman store remove` deletes it with the store.
- `process` - read from `os.environ` by the component that uses it and never written to the env file.

A secret is stripped by `memman uninstall`, left out of backups, and redacted by `memman config show`.

| Variable                          | Type      | Default                              | Purpose                                                                                                                                                                                         |
| --------------------------------- | --------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MEMMAN_LLM_ENDPOINT`             | installed | `https://openrouter.ai/api/v1`       | OpenAI-compatible `/chat/completions` endpoint for enrichment.                                                                                                                                  |
| `MEMMAN_LLM_API_KEY`              | installed | none                                 | Secret. Bearer token for the endpoint. The wizard accepts a blank key only on a loopback endpoint. On OpenRouter, install copies `MEMMAN_OPENROUTER_API_KEY` into it.                           |
| `MEMMAN_LLM_MODEL`                | installed | `qwen/qwen3-235b-a22b-2507`          | Enrichment model. Installation sets the default only on OpenRouter. On any other endpoint the wizard asks for a model ID, and an install without the wizard refuses to finish when none is set. |
| `MEMMAN_LLM_PROVIDER_ONLY`        | installed | `amazon-bedrock,azure,google-vertex` | OpenRouter only. Comma-separated vendors allowed to serve the model. Empty means any vendor.                                                                                                    |
| `MEMMAN_LLM_DATA_COLLECTION`      | installed | `deny`                               | OpenRouter only. The `data_collection` routing value.                                                                                                                                           |
| `MEMMAN_LLM_ZDR`                  | installed | `true`                               | OpenRouter only. Route only to zero-data-retention endpoints.                                                                                                                                   |
| `MEMMAN_EMBED_PROVIDER`           | installed | `voyage`                             | `voyage`, `openai`, or `openrouter` from the wizard. `ollama` is set only with `memman config set`. Sets the provider for new stores and the target of `embed reembed`.                         |
| `MEMMAN_VOYAGE_API_KEY`           | installed | none                                 | Secret. Required for the `voyage` embedding provider and for reranking.                                                                                                                         |
| `MEMMAN_VOYAGE_EMBED_MODEL`       | installed | `voyage-3-lite`                      | Voyage embedding model (512 dimensions).                                                                                                                                                        |
| `MEMMAN_OPENAI_EMBED_API_KEY`     | installed | none                                 | Secret. Required for the `openai` embedding provider.                                                                                                                                           |
| `MEMMAN_OPENAI_EMBED_ENDPOINT`    | installed | `https://api.openai.com`             | Base URL of an OpenAI-compatible embeddings API.                                                                                                                                                |
| `MEMMAN_OPENAI_EMBED_MODEL`       | installed | `text-embedding-3-small`             | Model for the `openai` provider.                                                                                                                                                                |
| `MEMMAN_OPENROUTER_API_KEY`       | installed | none                                 | Secret. Required for the `openrouter` embedding provider.                                                                                                                                       |
| `MEMMAN_OPENROUTER_ENDPOINT`      | installed | `https://openrouter.ai/api/v1`       | Base URL for the `openrouter` embedding provider.                                                                                                                                               |
| `MEMMAN_OPENROUTER_EMBED_MODEL`   | installed | `baai/bge-m3`                        | Model for the `openrouter` provider.                                                                                                                                                            |
| `MEMMAN_OLLAMA_HOST`              | installed | `http://localhost:11434`             | Ollama host.                                                                                                                                                                                    |
| `MEMMAN_OLLAMA_EMBED_MODEL`       | installed | `nomic-embed-text`                   | Ollama embedding model.                                                                                                                                                                         |
| `MEMMAN_OLLAMA_MAX_INPUT_CHARS`   | installed | `1500`                               | Ollama input is cut to this many characters before embedding.                                                                                                                                   |
| `MEMMAN_RERANK_PROVIDER`          | installed | `voyage`                             | Rerank provider. `voyage` is the only one.                                                                                                                                                      |
| `MEMMAN_VOYAGE_RERANK_MODEL`      | installed | `rerank-3-lite`                      | Voyage rerank model.                                                                                                                                                                            |
| `MEMMAN_RERANK_ENABLED`           | installed | `true`                               | Enables or disables reranking for every store.                                                                                                                                                  |
| `MEMMAN_LOG_LEVEL`                | installed | `WARNING`                            | Stderr log level when neither `--verbose` nor `--debug` is given.                                                                                                                               |
| `MEMMAN_DEFAULT_BACKEND`          | installed | `sqlite`                             | Backend for a store with no `MEMMAN_BACKEND_<store>`. `postgres` needs the `memman[postgres]` extra.                                                                                            |
| `MEMMAN_DEFAULT_POSTGRES_DSN`     | installed | none                                 | Secret. DSN for a Postgres store with no `MEMMAN_POSTGRES_DSN_<store>`.                                                                                                                         |
| `MEMMAN_INTERVAL`                 | installed | `60`                                 | Seconds between drains for `memman scheduler serve` without `--interval`.                                                                                                                       |
| `MEMMAN_BACKUP_CRON`              | installed | none                                 | Backup schedule, written by `memman backup schedule`.                                                                                                                                           |
| `MEMMAN_BACKUP_TARGET`            | installed | none                                 | Backup directory, written by `memman backup schedule`.                                                                                                                                          |
| `MEMMAN_BACKUP_KEEP`              | installed | `7`                                  | Bundles to keep, written by `memman backup schedule --keep`.                                                                                                                                    |
| `MEMMAN_BACKEND_<store>`          | per store | none                                 | The store's backend. Written by install for `default`, by the first drain, by `memman migrate`, or by `memman config set`.                                                                                                |
| `MEMMAN_POSTGRES_DSN_<store>`     | per store | none                                 | The store's DSN. Redacted by `config show` and left out of backups. `memman uninstall` keeps it.                                                                                                |
| `MEMMAN_RERANK_ENABLED_<store>`   | per store | none                                 | Enables or disables reranking for one store. Overrides `MEMMAN_RERANK_ENABLED`.                                                                                                                 |
| `MEMMAN_DATA_DIR`                 | process   | `~/.memman`                          | Data directory containing the env file. `--data-dir` takes precedence.                                                                                                                          |
| `MEMMAN_STORE`                    | process   | none                                 | Store for this process. `--store` takes precedence.                                                                                                                                             |
| `MEMMAN_AUTHOR`                   | process   | login name                           | Author recorded when `remember` or `replace` queues a write, preserving the identity of the user who submitted it.                                                                 |
| `MEMMAN_DEBUG`                    | process   | none                                 | A true value (`1`, `true`, `yes`, `on`) turns the trace on, and any other value turns it off. When unset, the `memman scheduler debug` state in `~/.memman/debug.state` decides.                |
| `MEMMAN_SCHEDULER_KIND`           | process   | none                                 | `serve` selects serve mode instead of systemd or launchd.                                                                                                                                       |
| `MEMMAN_WORKER`                   | process   | none                                 | The scheduler unit and `memman scheduler serve` set `1`, which adds the rotating `<data dir>/logs/memman.log`.                                                                                  |
| `MEMMAN_REINDEX_TIMEOUT`          | process   | `180`                                | Seconds allowed for the Postgres HNSW index build when a store opens. A build that times out is dropped and rebuilt on the next open.                                                           |
| `MEMMAN_EMBED_SWAP_BATCH_SIZE`    | process   | `200`                                | Memories per batch in `memman embed swap`.                                                                                                                                                      |
| `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` | process   | `0` (no limit)                       | Seconds allowed for the Postgres HNSW index build at the start of `memman embed swap`.                                                                                                          |

## Conventions

### Baseline schemas

Each database has one baseline schema that defines its current structure:

| Database                        | Baseline                                               |
| ------------------------------- | ------------------------------------------------------ |
| SQLite store (`memman.db`)      | `_BASELINE_SCHEMA` in `src/memman/store/db.py`         |
| Queue (`<data dir>/queue.db`)   | `_BASELINE_SCHEMA` in `src/memman/queue.py`            |
| Postgres store (`store_<name>`) | `PG_BASELINE_SCHEMA` in `src/memman/store/postgres.py` |

`create table if not exists` builds a fresh database from the baseline. The SQLite keyword index (the FTS5 table `insights_fts` and its triggers) is the one exception: `_migrate` in `store/db.py` creates and fills it in one transaction. The queue is always SQLite, whatever the store backend. Each Postgres store schema has its own `worker_runs` table.

There are no incremental migrations, rebuild scripts, or checks for columns missing from older schemas. To change a schema:

1. Update the baseline.
2. Index the newest column in the baseline. `create table if not exists` skips an existing table, so that index makes an older store fail at open with the error `store <name> predates the current schema`.
3. Apply the change manually to each live store. Drop each index whose definition changed so the baseline recreates it when the store next opens. The queue database is wiped and recreated instead.
4. Keep the one-off SQL with the rollout notes in the plan's evidence directory.
5. Update `EXPECTED_INSIGHT_COLUMNS` in `src/memman/doctor.py` for a new `insights` column, and `RETIRED_INSIGHT_INDEXES` for a dropped index. The `schema_columns` and `partial_index_predicates` checks read them.
6. Increment `PAYLOAD_VERSION` in `src/memman/migrate/__init__.py` when the migration payload changes. Increment `BACKUP_FORMAT_VERSION` in `src/memman/backup/__init__.py` when a backed-up schema changes. Restore copies the data directly and rejects bundles with a different version.
7. Include a test that checks the new schema.

### Migrating between SQLite and Postgres

[USAGE.md](docs/USAGE.md#migrating-between-sqlite-and-postgres) covers the command. Implementation details:

- `memman migrate` needs `pg_dump` on the PATH in both directions.
- Before migration, the command runs `select 1` against Postgres and checks for the `vector` extension and the `CREATE` privilege on the database.
- The command holds `<data dir>/drain.lock` throughout, so a drain cannot change the source while migration reads it.
- Before any write, the command classifies each target schema as ABSENT, EMPTY, or POPULATED. It prints the plan with the DSN password masked and asks to proceed. `--yes` skips the question. `--dry-run` works only with `--to postgres`. An EMPTY or POPULATED target schema is dropped and recreated.
- `--to postgres` copies the SQLite store into `store_<name>` in one transaction (`autocommit=False`). It checks the `insights`, `oplog`, and `meta` row counts, writes `MEMMAN_BACKEND_<store>=postgres` and `MEMMAN_POSTGRES_DSN_<store>`, and then moves `data/<store>/` to `archive/<store>/<YYYYMMDD>_<NN>/`.
- `--to sqlite` builds the SQLite store in a temporary directory and moves it into `data/<store>/`. It dumps the Postgres schema with `pg_dump -Fc` to `archive/<store>/<YYYYMMDD>_<NN>/dump.pgdump`, writes `MEMMAN_BACKEND_<store>=sqlite`, removes `MEMMAN_POSTGRES_DSN_<store>`, and then drops the schema.
- `SqliteMigrator` (`store/sqlite.py`) and `PostgresMigrator` (`store/postgres.py`) implement the `Migrator` base class in `migrate/__init__.py`. Each descriptor in `BACKENDS` (`store/factory.py`) names its migrator class. The `migrate` command imports both classes and runs each direction in its own branch, so a third backend needs a new branch as well as a new migrator.

### LLM dispatch

`src/memman/llm/client.py` defines one concrete client, `MemmanLLMClient`. It posts to `<MEMMAN_LLM_ENDPOINT>/chat/completions` using the OpenAI-compatible protocol supported by OpenRouter, OpenAI, Anthropic, Google, Ollama, vLLM, and LiteLLM. There is no provider registry and no subclass per vendor. A change of vendor edits `MEMMAN_LLM_ENDPOINT`, `MEMMAN_LLM_API_KEY`, and `MEMMAN_LLM_MODEL`.

- On an OpenRouter endpoint (`config.is_openrouter_endpoint`), the client adds the attribution headers `HTTP-Referer` and `X-Title`. It also sends a `provider` block built from `MEMMAN_LLM_PROVIDER_ONLY`, `MEMMAN_LLM_DATA_COLLECTION`, and `MEMMAN_LLM_ZDR`. Other endpoints receive neither.
- A blank `MEMMAN_LLM_API_KEY` drops the `Authorization` header. The wizard accepts a blank key only for a loopback endpoint (`localhost`, `127.0.0.1`, `::1`, or a `*.localhost` host).
- On an OpenRouter endpoint, `src/memman/llm/openrouter_models.py` checks the model against the public `/endpoints/zdr` and `/models` catalogs. A zero-data-retention endpoint on a vendor in `MEMMAN_LLM_PROVIDER_ONLY` must serve it, and OpenRouter must list no retirement date for it. Installation runs the check immediately. The drain runs it once a day and records the result in `<data dir>/model.state`, which `memman prime` reads ([LLM routing](docs/design/03-pipelines.md#llm-routing)).
- `src/memman/_http.py` holds the retry policy and timeouts. `src/memman/llm/shared.py` holds the JSON parsing helpers, and `src/memman/llm/usage.py` counts tokens per stage.
- The embedding providers are defined under `src/memman/embed/`, selected by `MEMMAN_EMBED_PROVIDER`. The reranker is defined under `src/memman/rerank/`.

### Tests

- The autouse fixture `_mock_apis` in `tests/conftest.py` patches `MemmanLLMClient.complete`, the Voyage embed calls, the Voyage rerank call, and the OpenRouter catalog fetch. The markers `no_mock_llm`, `no_mock_rerank`, and `no_mock_catalog` each disable the corresponding mock. `--live` turns off every mock.
- The autouse fixture `_isolate_env` points `MEMMAN_DATA_DIR` at a temporary directory, fills its env file from `INSTALL_DEFAULTS`, and clears the process variables and API keys from the environment. The `no_default_env` marker skips this step.
- The autouse fixture `_scheduler_started` reports the scheduler as started, so writes succeed. The `scheduler_stopped` marker reports it as stopped, for commands such as `graph rebuild` that need a stopped scheduler.
- The end-to-end tests under `tests/e2e/` skip these fixtures and run the installed binary with the inherited environment.
- `_isolate_env` clears `MEMMAN_DEBUG`, so trace mode in a unit test follows `~/.memman/debug.state`. `memman scheduler debug off` clears that state.
- `pyproject.toml` lists every marker.

## Submitting changes

- No deprecated code and no backward-compatibility shims. A rename deletes the old reader in the same commit.
- A new variable gets a constant in `src/memman/config.py`. Call sites import the constant and never repeat the name as a literal.
- Most commands print JSON through `_json_out`, indented two spaces with sorted keys. `recall` prints one plain line per memory.
