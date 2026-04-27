# Contributing to memman

memman is a single-user CLI memory daemon backed by SQLite. This file covers the short list of conventions that aren't obvious from the code.

## Development setup

```bash
make dev           # poetry install (editable)
make test          # run the unit/integration suite (pytest)
make e2e           # run the end-to-end pytest suite against real DBs
```

The project uses Poetry; run commands via `poetry run <cmd>` or inside `poetry shell`.

## Configuration

memman resolves env vars in two layers at runtime:

1. `os.environ` — wins when set (shell rc, direnv export, transient overrides).
2. `<MEMMAN_DATA_DIR>/env` — `KEY=VALUE` file at mode 0600, default `~/.memman/env`.

There is no third-tier code default at runtime. Defaults live in `config.INSTALL_DEFAULTS` and are consumed only by `memman install`, which writes them (or your exported overrides) to the env file. If a key is missing from both env and file, the resolver returns `None` and the caller raises a `ConfigError` with "run `memman install`" guidance.

`memman install` snapshots every `INSTALLABLE_KEYS` value: prefer `os.environ`, else `INSTALL_DEFAULTS`. For OpenRouter, it queries `/models` once to pick the actual current latest haiku/sonnet rather than persisting a version that ages. After install, the keys do not need to be exported in every shell — interactive `memman recall`, `memman doctor`, and the scheduler-driven worker all read from the file. Per-project overrides are via direnv (or any tool that exports `KEY=VALUE` before invoking memman); memman itself does not parse a project-local file.

`memman uninstall` strips the secret keys (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, `MEMMAN_OPENAI_EMBED_API_KEY`) from the env file but keeps non-secret settings, so a later `memman install` resurrects model/provider preferences without re-export.

`memman doctor` includes an `env_completeness` check: when `pipx upgrade memman` adds a new `INSTALLABLE_KEYS` entry, the check warns with the missing-keys list and "run `memman install`" guidance.

### Required keys

| Variable             | Purpose                         |
| -------------------- | ------------------------------- |
| `OPENROUTER_API_KEY` | LLM inference via OpenRouter.   |
| `VOYAGE_API_KEY`     | Voyage AI embeddings (512-dim). |

### Persisted at install (`INSTALLABLE_KEYS`)

Set any of these in your shell before `memman install` and they land in the env file. `memman doctor` shows the resolved value and which layer it came from.

| Variable                        | Purpose                                                                                                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MEMMAN_LLM_PROVIDER`           | Registered provider name (default `openrouter`).                                                                                                                                                 |
| `MEMMAN_LLM_MODEL_FAST`         | Hot-path model id. OpenRouter resolves the latest at install time.                                                                                                                               |
| `MEMMAN_LLM_MODEL_SLOW`         | Worker-pipeline model id. OpenRouter resolves the latest at install time.                                                                                                                        |
| `MEMMAN_EMBED_PROVIDER`         | `voyage` (default), `openai`, `openrouter`, or `ollama`.                                                                                                                                         |
| `MEMMAN_OPENROUTER_ENDPOINT`    | OpenRouter base URL (default `https://openrouter.ai/api/v1`).                                                                                                                                    |
| `MEMMAN_LOG_LEVEL`              | Logger level when neither `--verbose` nor `--debug` is passed (default `WARNING`).                                                                                                               |
| `MEMMAN_OPENAI_EMBED_API_KEY`   | Secret for the OpenAI-compatible embed provider (`MEMMAN_EMBED_PROVIDER=openai`).                                                                                                                |
| `MEMMAN_OPENAI_EMBED_ENDPOINT`  | Endpoint for the OpenAI-compatible embed provider.                                                                                                                                               |
| `MEMMAN_OPENAI_EMBED_MODEL`     | Model name for the OpenAI-compatible embed provider.                                                                                                                                             |
| `MEMMAN_OPENROUTER_EMBED_MODEL` | Model id for the OpenRouter embed provider (`MEMMAN_EMBED_PROVIDER=openrouter`). Default `baai/bge-m3`. Reuses `OPENROUTER_API_KEY` and `MEMMAN_OPENROUTER_ENDPOINT`; no separate secret needed. |
| `MEMMAN_OLLAMA_HOST`            | Ollama host URL (default `http://localhost:11434`).                                                                                                                                              |
| `MEMMAN_OLLAMA_EMBED_MODEL`     | Ollama embedding model name (default `nomic-embed-text`).                                                                                                                                        |

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
