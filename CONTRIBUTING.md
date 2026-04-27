# Contributing to memman

memman is a single-user CLI memory daemon backed by SQLite. This file covers the short list of conventions that aren't obvious from the code.

## Development setup

```bash
make dev           # poetry install (editable)
make test          # run the unit/integration suite (pytest)
make e2e           # run the end-to-end pytest suite against real DBs
```

The project uses Poetry; run commands via `poetry run <cmd>` or inside `poetry shell`.

## Required environment variables

The scheduler-driven enrichment worker needs two API keys. Set them in `~/.memman/env` (mode 0600):

| Variable             | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `OPENROUTER_API_KEY` | LLM inference via OpenRouter (ZDR-enforced routing). |
| `VOYAGE_API_KEY`     | Voyage AI embeddings (512-dim).                      |

Optional:

| Variable                     | Purpose                                                                    |
| ---------------------------- | -------------------------------------------------------------------------- |
| `MEMMAN_DATA_DIR`            | Override `~/.memman` as the data root.                                     |
| `MEMMAN_STORE`               | Override the active store without editing `~/.memman/active`.              |
| `MEMMAN_LLM_PROVIDER`        | Registered provider name (default `openrouter`).                           |
| `MEMMAN_LLM_MODEL_FAST`      | Override the auto-picked Haiku for the recall hot path (query expansion).  |
| `MEMMAN_LLM_MODEL_SLOW`      | Override the auto-picked Haiku for the scheduler worker (extraction etc.). |
| `MEMMAN_OPENROUTER_ENDPOINT` | Override the OpenRouter base URL.                                          |
| `MEMMAN_CACHE_DIR`           | Override the ZDR endpoint-list cache location.                             |
| `MEMMAN_DEBUG`               | Truthy value enables JSONL tracing to `~/.memman/logs/debug.log`.          |
| `MEMMAN_WORKER`              | `1` inside the scheduler-triggered worker; enables the rotating log.       |
| `MEMMAN_LOG_LEVEL`           | Override logger level when neither `--verbose` nor `--debug` is passed.    |

Run `memman doctor` to probe both providers with cheap calls (the `llm_probe` and `embed_probe` checks).

## Conventions

### Canonical-schema-only migrations

memman is a single-user tool. There is no `PRAGMA user_version` ladder, no `_MIGRATIONS` registry, and no rolling migration code. `_BASELINE_SCHEMA` in `src/memman/store/db.py` and `src/memman/queue.py` is the single source of truth.

When a schema change is needed:

1. Update `_BASELINE_SCHEMA` (adds columns to fresh databases via `CREATE TABLE IF NOT EXISTS`).
2. Perform a one-off `ALTER TABLE` against every existing store database (`~/.memman/data/*/memman.db`) and against `~/.memman/queue.db` if the change is in queue schema. A short Python snippet invoked with `poetry run python` is sufficient.
3. Commit the schema change and the migration evidence (test asserting the column is present) in the same change.

Do not add new migration-ladder code.

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
