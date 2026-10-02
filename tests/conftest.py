"""Shared fixtures for memman tests.

Dual-mode API mocking: mocked by default, real APIs with --live flag.

    pytest                    # fast, mocked LLM + embeddings
    pytest --live             # real LLM, embed, and rerank APIs (needs keys)

Mock mode patches `MemmanLLMClient.complete` and `embed.client.Client.embed`
at the HTTP layer, so all enrichment logic still runs with realistic
canned responses. This exercises the real code paths.
"""

import hashlib
import json
import logging
import os
import struct
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click.testing
import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import _LAST_HEARTBEAT_AT, cli
from memman.embed import fingerprint as fp_mod
from memman.embed import get_client
from memman.embed import registry as _embed_registry
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.llm import client as llm_client_mod
from memman.queue import open_queue_db, queue_db
from memman.setup import scheduler as sched_mod
from memman.store.db import open_db, open_read_only, read_active, store_dir
from memman.store.factory import drop_store, resolve_store_backend
from memman.store.factory import resolve_store_pg_dsn
from memman.store.model import Insight, format_timestamp
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteBackend, drop_sqlite_store
from memman.store.sqlite import open_sqlite_backend

try:
    import psycopg  # noqa: F401
    import testcontainers.postgres  # noqa: F401
    pytest_plugins = ['tests.fixtures.postgres']
    _POSTGRES_AVAILABLE = True
except ImportError:
    pytest_plugins = ()
    _POSTGRES_AVAILABLE = False

EMBEDDING_DIM = 512


def pytest_collection_modifyitems(
        config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.postgres tests when psycopg is not installed.
    """
    if _POSTGRES_AVAILABLE:
        return
    skip_pg = pytest.mark.skip(reason='postgres extras not installed')
    for item in items:
        if 'postgres' in item.keywords:
            item.add_marker(skip_pg)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --live flag for real API calls.
    """
    parser.addoption(
        '--live', action='store_true', default=False,
        help='Use real Haiku LLM and Voyage embedding APIs')


@pytest.fixture(autouse=True)
def logger_state() -> Iterator[logging.Logger]:
    """Restore the process-wide `memman` logger after every test.

    `_configure_logging` mutates a module-level logger and is written
    to run once per process, so any CLI invocation leaks handlers and
    a level into every later test in the session.
    """
    log = logging.getLogger('memman')
    saved_handlers = [(h, h.level) for h in log.handlers]
    saved_level = log.level
    yield log
    for handler in log.handlers:
        if handler not in [h for h, _ in saved_handlers]:
            handler.close()
    log.handlers[:] = [h for h, _ in saved_handlers]
    for handler, level in saved_handlers:
        handler.setLevel(level)
    log.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 request: pytest.FixtureRequest):
    """Pin MEMMAN_DATA_DIR and HOME to tmp and seed the env file.

    Prevents the user's real `~/.memman/env` from leaking into the
    config resolver during unit tests. The home redirect keeps a live
    `~/.memman/debug.state` or scheduler state out of the test, and the
    trace log out of the live `~/.memman/logs/`. By default, seeds a fresh env
    file with `INSTALL_DEFAULTS` so runtime call sites resolve cleanly
    (no code-default fallback exists at runtime). Tests that need to
    assert "absent key" behavior mark themselves
    `@pytest.mark.no_default_env` and the seed step is skipped.

    Skipped entirely for e2e tests, which run real binaries with the
    inherited environment.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        yield
        return
    live_mode = request.config.getoption('--live')
    real_secrets = {}
    if live_mode:
        live_keys = ('MEMMAN_API_KEY', 'MEMMAN_ENDPOINT')
        for key in live_keys:
            val = os.environ.get(key)
            if val:
                real_secrets[key] = val
        home_env = Path.home() / '.memman' / config.ENV_FILENAME
        if home_env.exists():
            home_values = config.parse_env_file(home_env)
            for key in live_keys:
                if key in real_secrets:
                    continue
                val = home_values.get(key)
                if val:
                    real_secrets[key] = val
    data_dir = tmp_path / 'memman'
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
    home_dir = tmp_path / 'isolated-home'
    home_dir.mkdir()
    monkeypatch.setenv('HOME', str(home_dir))
    monkeypatch.delenv('CODEX_HOME', raising=False)
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    monkeypatch.delenv('MEMMAN_WORKER', raising=False)
    monkeypatch.delenv('MEMMAN_SCHEDULER_KIND', raising=False)
    monkeypatch.delenv('MEMMAN_AUTHOR', raising=False)
    monkeypatch.delenv('MEMMAN_API_KEY', raising=False)
    monkeypatch.delenv('MEMMAN_ENDPOINT', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    # `remember` sets a PGCONNECT_TIMEOUT default in-process. The set
    # records the prior state, so teardown clears what a test leaves.
    monkeypatch.setenv('PGCONNECT_TIMEOUT', '3')
    monkeypatch.delenv('PGCONNECT_TIMEOUT')
    if live_mode and real_secrets:
        for key, val in real_secrets.items():
            monkeypatch.setenv(key, val)
    if 'no_default_env' not in request.keywords:
        _write_default_env_file(data_dir, real_secrets=real_secrets or None)
    config.reset_file_cache()
    _embed_registry.reset_for_tests()
    yield
    config.reset_file_cache()
    _embed_registry.reset_for_tests()


_TEST_MOCK_SECRETS = {
    'MEMMAN_API_KEY': 'mock-api-key-for-testing',
    }


def _set_env_file_value(key: str, value: str | None) -> None:
    """Write or remove a key in the active test env file.

    Installable keys live in the env file because the runtime resolver
    does not read `os.environ`. Pass `value=None` to remove the key.
    """
    data_dir = os.environ.get(config.DATA_DIR)
    if not data_dir:
        raise RuntimeError(
            '_set_env_file_value requires MEMMAN_DATA_DIR;'
            ' invoke from a test that uses the _isolate_env fixture')
    path = config.env_file_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = config.parse_env_file(path) if path.exists() else {}
    if value is None:
        rows.pop(key, None)
    else:
        rows[key] = value
    contents = '\n'.join(f'{k}={v}' for k, v in rows.items()) + '\n'
    path.write_text(contents)
    config.reset_file_cache()


@pytest.fixture
def env_file():
    """Yield a callable that writes/removes keys in the test env file.

    Usage: `env_file('MEMMAN_LLM_MODEL', 'foo')` writes the row;
    `env_file('MEMMAN_LLM_MODEL', None)` removes it. Cache is
    auto-reset; the autouse `_isolate_env` fixture handles cleanup.
    """
    return _set_env_file_value


def _write_default_env_file(
        data_dir: Path, real_secrets: dict[str, str] | None = None) -> None:
    """Seed `<data_dir>/env` with `INSTALL_DEFAULTS` for tests.

    Mirrors a post-install state so runtime call sites (which use
    `config.require`) resolve cleanly. By default seeds mock API key
    values, because the runtime resolver does not consult `os.environ`.
    Pass `real_secrets={...}` (from `--live` mode) to seed real
    credentials captured from the shell instead. Tests that need the
    broken state opt out via `@pytest.mark.no_default_env`.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / config.ENV_FILENAME
    secrets = dict(_TEST_MOCK_SECRETS)
    if real_secrets:
        secrets.update(real_secrets)
    rows = list(config.INSTALL_DEFAULTS.items()) + list(secrets.items())
    contents = '\n'.join(f'{k}={v}' for k, v in rows) + '\n'
    path.write_text(contents)
    path.chmod(0o600)
    config.reset_file_cache()


@pytest.fixture(autouse=True)
def _reset_heartbeat_state():
    """Clear the module-level heartbeat dict between tests.

    `cli._LAST_HEARTBEAT_AT` is process-global; in-process CliRunner
    tests share it. Reset before AND after each test to prevent
    cross-test contamination if a future fixture reuses a data_dir.
    """
    _LAST_HEARTBEAT_AT.clear()
    yield
    _LAST_HEARTBEAT_AT.clear()


@pytest.fixture(autouse=True)
def _scheduler_started(request: pytest.FixtureRequest,
                       monkeypatch: pytest.MonkeyPatch):
    """Force scheduler state to STARTED so writes are accepted in tests.

    cli.py's `_require_started` rejects writes when `read_state()`
    returns STATE_STOPPED. The autouse fixture monkeypatches it to
    STARTED for all non-e2e tests.

    Inline-mode auto-drain is NOT injected here -- write commands
    enqueue and return immediately, exactly as production. Tests that
    need to read what was just written go through the
    `MemmanCliRunner` (default `runner` fixture) which auto-drains
    after `remember`/`replace`. Tests that intentionally inspect a
    pre-drain queue should use the `no_auto_drain` mark.

    The `scheduler_stopped` mark forces STOPPED instead, for a command
    such as `enrich` that requires the drain to be down. It has
    to patch rather than merely stand aside, because `read_state()`
    otherwise reads the developer machine's own scheduler state and a
    started timer there fails the test.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if request.node.fspath.basename == 'test_scheduler_setup.py':
        return
    if 'scheduler_stopped' in request.keywords:
        monkeypatch.setattr(sched_mod, 'read_state',
                            lambda: sched_mod.STATE_STOPPED)
        return
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    original_invoke = click.testing.CliRunner.invoke

    def _wrapped_invoke(
            self: CliRunner, cli_obj: Any, args: list[str] | None = None,
            **kwargs: Any) -> click.testing.Result:
        result = original_invoke(self, cli_obj, args, **kwargs)
        if (result.exit_code == 0
                and 'no_auto_drain' not in request.keywords
                and _args_target_write(args)):
            data_dir = _args_data_dir(args)
            if data_dir is not None:
                _force_drain_with(self.__class__, data_dir, original_invoke)
        return result

    monkeypatch.setattr(
        click.testing.CliRunner, 'invoke', _wrapped_invoke)


_AUTO_DRAIN_TRIGGERS = ('remember', 'replace')


def _args_target_write(args: list[str] | None) -> bool:
    """True when the CLI args name `remember` or `replace`.
    """
    if not args:
        return False
    for arg in args:
        if isinstance(arg, str) and arg in _AUTO_DRAIN_TRIGGERS:
            return True
    return False


def _args_data_dir(args: list[str] | None) -> str | None:
    """The value after `--data-dir` in the CLI args, or None.
    """
    if not args:
        return None
    seq = list(args)
    for i, arg in enumerate(seq):
        if arg == '--data-dir' and i + 1 < len(seq):
            return seq[i + 1]
    return None


def _force_drain_with(runner_cls: type[CliRunner], data_dir: str,
                      original_invoke: Any) -> None:
    """Run `scheduler drain` via the underlying click invoke.

    Bypasses the autouse-wrapped `invoke` to avoid re-triggering the
    auto-drain path on the drain command itself.
    """
    instance = runner_cls()
    result = original_invoke(
        instance, cli,
        ['--data-dir', data_dir, 'scheduler', 'drain'])
    assert result.exit_code == 0, (
        f'force_drain failed: exit={result.exit_code} '
        f'output={result.output} exc={result.exception}')


def force_drain(data_dir: str) -> None:
    """Synchronously drain the queue for the given data dir.

    Tests that follow `remember`/`replace` with a read assertion call
    this to flush pending work through the worker before reading. Uses
    the same `scheduler drain` code path the OS timer fires.
    """
    instance = CliRunner()
    result = instance.invoke(
        cli, ['--data-dir', data_dir,
              'scheduler', 'drain'])
    assert result.exit_code == 0, (
        f'force_drain failed: exit={result.exit_code} '
        f'output={result.output} exc={result.exception}')


@pytest.fixture(autouse=True)
def _autoseed_fingerprint(request: pytest.FixtureRequest,
                          monkeypatch: pytest.MonkeyPatch):
    """Auto-seed `meta.embed_fingerprint` on `bound_embedder`.

    `tmp_db` already writes a fingerprint via `write_fingerprint`; this
    fixture additionally backstops tests that hand-build backends and
    then call `bound_embedder(backend)` -- it seeds the env-active
    fingerprint on first lookup so those tests don't need
    `seed_if_fresh` boilerplate.

    Tests that exercise the strict production behavior (raw missing-
    fingerprint error from `bound_embedder`) should mark themselves
    `@pytest.mark.no_autoseed_fingerprint`.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if 'no_autoseed_fingerprint' in request.keywords:
        return

    real_bound = fp_mod.bound_embedder

    def seed_then_bound(backend: Any) -> Any:
        if fp_mod.stored_fingerprint(backend) is None:
            fp_mod.write_fingerprint(
                backend, fp_mod.Fingerprint.from_client(get_client()))
        return real_bound(backend)

    monkeypatch.setattr(fp_mod, 'bound_embedder', seed_then_bound)


@pytest.fixture(autouse=True)
def _mock_apis(request: pytest.FixtureRequest,
               monkeypatch: pytest.MonkeyPatch):
    """Mock LLM and embedding HTTP calls unless --live is set.

    Patches at the method layer: MemmanLLMClient.complete returns
    realistic JSON that the real enrichment code parses.
    Voyage embed returns a deterministic content-hash vector.

    Tests that exercise the real MemmanLLMClient.complete method
    should mark themselves with `@pytest.mark.no_mock_llm` to skip
    the method-level patch while keeping the embedding stubs in
    place.
    Tests that exercise the real `rerank.client.Client.rerank` method
    mark themselves `@pytest.mark.no_mock_rerank` the same way.
    The OpenRouter catalog fetch behind the model check returns a clean
    verdict, so no drain or install GETs openrouter.ai; a test driving
    the real fetch marks itself `@pytest.mark.no_mock_catalog`.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if request.config.getoption('--live'):
        return

    if 'no_mock_llm' not in request.keywords:
        monkeypatch.setattr(
            'memman.llm.client.MemmanLLMClient.complete',
            _mock_llm_complete)

    if 'no_mock_embed' not in request.keywords:
        monkeypatch.setattr(
            'memman.embed.client.Client.embed', _mock_embed)
        monkeypatch.setattr(
            'memman.embed.client.Client.embed_batch', _mock_embed_batch)
        monkeypatch.setattr(
            'memman.embed.client.Client.available', _mock_available)
    if 'no_mock_rerank' not in request.keywords:
        monkeypatch.setattr(
            'memman.rerank.client.Client.rerank', _mock_rerank)
    if 'no_mock_catalog' not in request.keywords:
        monkeypatch.setattr(
            'memman.llm.openrouter_models.fetch_model_notice',
            lambda endpoint, *, model, vendors: '')
    config.reset_file_cache()
    llm_client_mod.reset_client_cache()


def _mock_llm_complete(self: Any, system: str, user: str, *,
                       stage: str) -> str:
    """Route a `complete` call to the mock for its system text.

    Parameters
    ----------
    self : object
        The patched client instance, unused.
    system : str
        The system text; a marker unique to each pipeline stage picks
        the mock.
    user : str
        The user body the mock parses.
    stage : str
        Accepted and ignored. The signature mirrors
        `MemmanLLMClient.complete`, so a call site passing a keyword
        the real client lacks fails here as it would in production.

    Returns
    -------
    str
        The canned JSON response for that stage; a neutral non-empty
        reply (the doctor probe needs one) when no marker matches.
    """
    if 'summary' in system.lower():
        return _mock_enrichment(user)
    return json.dumps({'ok': True})


def _mock_enrichment(content: str) -> str:
    """Canned enrichment JSON whose summary is the first 100 characters.
    """
    return json.dumps({'summary': content[:100]})


def _mock_rerank(self: Any, query: str, documents: list[str],
                 top_n: int | None = None) -> list[tuple[int, float]]:
    """Passthrough reranker: input order preserved, scores descending.

    The write path's shortlist reranks its cosine pool on every fact, so
    without this stub every pipeline test would post to the endpoint. Keeping
    the input order means a test that plants rows by cosine sees the
    rerank slots filled in that same order unless it installs its own
    stub.
    """
    return [(i, 1.0 - i / max(1, len(documents))) for i in range(len(documents))]


def _mock_available(self: Any) -> bool:
    """Report the endpoint reachable and learn `dim` as a real probe does.
    """
    self.dim = self.dim or EMBEDDING_DIM
    return True


def _mock_embed_batch(
        self: Any, texts: list[str]) -> list[list[float]]:
    """Batch variant of `_mock_embed`. One vector per input.
    """
    return [_mock_embed(self, t) for t in texts]


def _mock_embed(self: Any, text: str) -> list[float]:
    """Deterministic embedding from content hash.

    Reads target dimension from `self.dim` when available, falling
    back to `EMBEDDING_DIM`. Produces a unit vector seeded by content,
    so identical text gives identical vectors. Different text gives
    different vectors with low cosine similarity. Values are derived
    as int32-mapped uniforms so the float32 cast (used by pgvector)
    never produces NaN or Inf.
    """
    dim = getattr(self, 'dim', 0) or EMBEDDING_DIM
    digest = hashlib.sha256(text.encode()).digest()
    ints = list(struct.unpack(
        f'<{len(digest) // 4}i', digest))
    while len(ints) < dim:
        extra = hashlib.sha256(
            digest + len(ints).to_bytes(4, 'little')).digest()
        ints.extend(struct.unpack(f'<{len(extra) // 4}i', extra))
    ints = ints[:dim]
    floats = [x / (1 << 31) for x in ints]
    norm = sum(x * x for x in floats) ** 0.5
    if norm > 0:
        floats = [x / norm for x in floats]
    return floats


def _vec(*prefix: float, dim: int = EMBEDDING_DIM) -> list[float]:
    """Build a fixed-dim vector for cross-backend embedding tests.

    SQLite stores embeddings as BLOB and accepts any dimension;
    Postgres uses `vector(dim)` and rejects shorter vectors. Padding
    with zeros preserves cosine similarity for the math the call
    sites assert on.
    """
    return list(prefix) + [0.0] * (dim - len(prefix))


@pytest.fixture
def tmp_db(request: pytest.FixtureRequest, tmp_path: Path):
    """Fresh SQLite database in temp directory.

    Seeds `meta.embed_fingerprint` to match the active client by
    default, mirroring `setup.claude._init_default_store`. Tests
    exercising unseeded behavior should use the
    `no_autoseed_fingerprint` mark.
    """
    db = open_db(str(tmp_path))
    if 'no_autoseed_fingerprint' not in request.keywords:
        write_fingerprint(
            SqliteBackend(db), seed_default_fingerprint())
    yield db
    db.close()


@pytest.fixture
def tmp_backend(tmp_db: Any) -> SqliteBackend:
    """Wrap `tmp_db` in a SqliteBackend.

    Pipeline / search / graph entry points take `Backend`. Tests that
    drive those entry points against a fresh store use this fixture;
    the underlying DB and SqliteBackend share the same connection so
    free-function and verb-surface calls see one transaction.
    """
    return SqliteBackend(tmp_db)


def _backend_params() -> list:
    """Parametrize slots for the cross-backend `backend` fixture.

    SQLite is always present. Postgres only emits when `psycopg` and
    `testcontainers.postgres` are importable, and its slot carries
    `pytest.mark.postgres` so `pytest -m "not postgres"` skips it.
    """
    params = [pytest.param('sqlite', id='sqlite')]
    try:
        import psycopg  # noqa: F401
        import testcontainers.postgres  # noqa: F401
        params.append(pytest.param(
            'postgres', id='postgres',
            marks=pytest.mark.postgres))
    except ImportError:
        pass
    return params


@pytest.fixture(params=_backend_params())
def backend_kind(request) -> str:
    """The backend identifier for this parametrization slot.
    """
    return request.param


@pytest.fixture(params=_backend_params())
def runner_kind(request) -> str:
    """Backend identifier for CliRunner-driven cross-backend tests.

    Pairs with the `cross_backend_runner` fixture to flip the
    per-store `MEMMAN_BACKEND_<store>` between sqlite and postgres
    for each test invocation.
    """
    return request.param


@pytest.fixture
def cross_backend_runner(
        request: pytest.FixtureRequest, runner_kind: str, tmp_path: Path,
        env_file: Any, monkeypatch: pytest.MonkeyPatch):
    """CliRunner whose env writes per-store keys for `<runner_kind>`.

    For postgres mode writes `MEMMAN_BACKEND_<store>` and
    `MEMMAN_POSTGRES_DSN_<store>` from the session container DSN and
    registers a teardown that drops the per-test schema. For sqlite
    mode writes `MEMMAN_DEFAULT_BACKEND=sqlite`. Returns the same
    `(runner, data_dir)` tuple shape as the legacy `runner` fixture
    in `test_memory_system.py` so a test can swap one for the other
    transparently.
    """
    r = CliRunner()
    env_data_dir = os.environ.get('MEMMAN_DATA_DIR')
    data_dir = env_data_dir or str(tmp_path / 'memman_data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    env_file('MEMMAN_DEFAULT_BACKEND', runner_kind)
    if runner_kind == 'postgres':
        pg_dsn = request.getfixturevalue('pg_dsn')
        store_name = _safe_store_name(request.node.name)
        env_file(f'MEMMAN_BACKEND_{store_name}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store_name}', pg_dsn)
        env_file('MEMMAN_DEFAULT_POSTGRES_DSN', pg_dsn)
        monkeypatch.setenv('MEMMAN_STORE', store_name)

        def _drop_postgres_schema() -> None:
            drop_store(store_name, data_dir)
        request.addfinalizer(_drop_postgres_schema)
    return r, data_dir


@pytest.fixture
def backend(request: pytest.FixtureRequest, backend_kind: str,
            tmp_path: Path):
    """Cross-backend Backend fixture for pipeline tests.

    Parametrizes over `{sqlite, postgres}` (postgres slot active only
    when extras are importable). Yields a fully-isolated Backend with
    `meta.embed_fingerprint` pre-seeded so pipeline tests that touch
    embeddings do not trip the fingerprint refusal. Postgres tests
    get a unique store name per test so schemas don't collide; the
    schema is dropped on teardown.

    Pipeline / search / graph tests should use this fixture instead
    of `tmp_backend` to gain Postgres parity.
    """
    pg_dsn = None
    if backend_kind == 'sqlite':
        data_dir = str(tmp_path / 'memman')
        store_name = 'test'
        b = open_sqlite_backend(store_name, data_dir)
    else:
        pg_dsn = request.getfixturevalue('pg_dsn')
        from memman.store.postgres import drop_postgres_store
        from memman.store.postgres import open_postgres_backend
        store_name = _safe_store_name(request.node.name)
        drop_postgres_store(store_name, pg_dsn)
        b = open_postgres_backend(store_name, pg_dsn)
    b.meta.set(META_KEY, seed_default_fingerprint().to_json())
    try:
        yield b
    finally:
        b.close()
        if backend_kind == 'postgres':
            drop_postgres_store(store_name, pg_dsn)
        else:
            drop_sqlite_store(store_name, str(tmp_path / 'memman'))


def _safe_store_name(test_id: str) -> str:
    """Derive a postgres-schema-safe store name from a test node id.

    Postgres identifiers must match `[a-z][a-z0-9_]*`; pytest test
    node ids contain `[`, `]`, `-`, `.`, etc. Replace non-alnum with
    underscores, lowercase, truncate to fit `_check_identifier`.
    """
    safe = ''.join(c if c.isalnum() else '_' for c in test_id).lower()
    if safe and not safe[0].isalpha():
        safe = 'p_' + safe
    return safe[:40] or 'p_test'


def set_created_at(backend: Any, insight_id: str, when: datetime) -> None:
    """Test-only: directly update `created_at` on a stored insight.

    The Backend Protocol's `nodes.insert` ignores caller-passed
    `Insight.created_at` (server-side timestamps). Tests that
    exercise temporal logic against pre-existing rows with
    controlled timestamps call this helper after
    `backend.nodes.insert` to override the server-stamped value.
    Bypasses the Protocol intentionally; do NOT use outside test
    code.
    """
    when_str = format_timestamp(when)
    if isinstance(backend, SqliteBackend):
        backend._db._exec(
            'update insights set created_at = ? where id = ?',
            (when_str, insight_id))
    else:
        with backend._conn.cursor() as cur:
            cur.execute(
                f'update {backend._schema}.insights'
                ' set created_at = %s where id = %s',
                (when, insight_id))
        backend._conn.commit()


def make_insight(**overrides: Any) -> Insight:
    """Factory for test Insight instances.
    """
    now = datetime.now(timezone.utc)
    defaults = {
        'id': 'test-id',
        'content': 'test content',
        'created_at': now,
        'updated_at': now,
        'deleted_at': None,
        }
    defaults.update(overrides)
    return Insight(**defaults)


def insert_pending(db: Any, insight_id: str, content: str = 'test content',
                   **kw: Any) -> None:
    """Insert an insight with enrich_attempted_at = NULL.

    Helper for enrichment tests that need pending insights as fixtures.
    Forwards extra kwargs to `make_insight`.
    """
    insert_insight(db, make_insight(id=insight_id, content=content, **kw))
    db._conn.execute(
        'update insights set enrich_attempted_at = null where id = ?',
        (insight_id,))


@pytest.fixture
def queue_conn(tmp_path: Path):
    """Fresh queue.db connection for direct queue helper tests.
    """
    conn = open_queue_db(str(tmp_path))
    yield conn
    conn.close()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME and `Path.home` to a tmp_path.

    Used by setup-adjacent tests that touch `~/.memman` directly.
    Does not pin `MEMMAN_DATA_DIR` (the autouse `_isolate_env` already
    handles env scoping for unit tests). Tests that need both the
    redirect and an explicit data-dir under the fake home should
    construct it locally as `fake_home / 'memman'`.
    """
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    return tmp_path


def install_env_factory(data_dir: str | Path, **keys: str | None) -> None:
    """Seed an env file at `data_dir` with selected keys.

    Pass key=value to write a row; pass key=None to omit it.
    Recognized convenience alias: ``api_key`` -> MEMMAN_API_KEY. Other
    kwargs are written as-is.
    """
    aliases = {'api_key': config.API_KEY}
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, v in keys.items():
        if v is None:
            continue
        real_key = aliases.get(k, k)
        rows.append(f'{real_key}={v}')
    if rows:
        path = p / config.ENV_FILENAME
        path.write_text('\n'.join(rows) + '\n')
        path.chmod(0o600)
    config.reset_file_cache()


def fake_subprocess(monkeypatch: pytest.MonkeyPatch, target_module: Any,
                    active: bool = True) -> None:
    """Stub `subprocess` on `target_module` so tests don't shell out.

    `target_module` is the module under test that imports
    `subprocess` (typically `memman.setup.scheduler`). When `active`
    is True, the fake `run` returns a `_FakeResult` with returncode 0
    and stdout 'active'; when False, returncode 3 and stdout 'inactive'.
    `_record_subprocess` in `test_scheduler_setup.py` is a richer
    variant that captures call arguments; this helper covers the common
    case where the test only needs subprocess to be quiet.
    """
    class _FakeResult:
        returncode = 0 if active else 3
        stdout = 'active' if active else 'inactive'
        stderr = ''

    fake = type('S', (), {
        'run': staticmethod(lambda *a, **k: _FakeResult()),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(target_module, 'subprocess', fake)


def make_cli_runner(tmp_path: Path, *, subdir: str = 'mm') -> tuple:
    """Build a `(CliRunner, data_dir)` tuple.

    The data_dir matches `MEMMAN_DATA_DIR` set by the autouse
    `_isolate_env` fixture so that env-file reads keyed off the CLI
    `--data-dir` arg find the seeded keys (per-store routing reads
    `<data_dir>/env` directly).
    """
    r = CliRunner()
    env_data_dir = os.environ.get('MEMMAN_DATA_DIR')
    data_dir = env_data_dir or str(tmp_path / subdir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return r, data_dir


@pytest.fixture
def mm_runner(tmp_path: Path) -> tuple:
    """Default `(CliRunner, data_dir)` tuple for sqlite-only CLI tests.

    Tests that need cross-backend parity use `cross_backend_runner`
    instead. A file-level `runner` fixture can delegate here:
    `def runner(mm_runner): return mm_runner`.
    """
    return make_cli_runner(tmp_path)


def invoke(runner_tuple: tuple, args: list[str]) -> click.testing.Result:
    """Invoke memman CLI with `--data-dir` prepended.

    Shared replacement for the per-file `invoke` helpers.
    """
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def queued_contents(data_dir: str) -> list[str]:
    """Return the content of every queue row, in insert order.
    """
    with queue_db(data_dir) as conn:
        return [r[0] for r in conn.execute(
            'select content from queue order by id').fetchall()]


def parse_remember(result: click.testing.Result,
                   runner_tuple: tuple | None = None) -> dict:
    """Parse remember/replace output, returning a fact-shaped dict.

    `remember`/`replace` output is `{action: queued, queue_id, store}`.
    The autouse drain runs the worker after the invocation, so the new
    insight sits in the store DB carrying the queue row's `queue_uuid`.
    This helper reads the uuid off the queue row (`purge_done` retains
    done rows long enough for a test) and looks the insight up by it.
    The lookup query switches when the per-store
    `MEMMAN_BACKEND_<store>=postgres` resolves.
    """
    raw = json.loads(result.output)
    if runner_tuple is None:
        return raw
    queue_id = raw.get('queue_id')
    if queue_id is None:
        return raw
    _, data_dir = runner_tuple
    with queue_db(data_dir) as qconn:
        qrow = qconn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()
    if qrow is None:
        return raw
    queue_uuid = qrow[0]
    name = raw.get('store') or read_active(data_dir) or 'default'
    backend_kind = resolve_store_backend(name, data_dir)
    if backend_kind == 'postgres':
        import psycopg
        from memman.store.postgres import _store_schema
        schema = _store_schema(name)
        sql = f"""
select id, content
from {schema}.insights
where queue_uuid = %s
  and deleted_at is null
order by created_at
"""
        dsn = resolve_store_pg_dsn(name, data_dir)
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (queue_uuid,))
            rows = cur.fetchall()
    else:
        sdir = store_dir(data_dir, name)
        db = open_read_only(sdir)
        sql = """
select id, content
from insights
where queue_uuid = ?
  and deleted_at is null
order by created_at
"""
        try:
            rows = db._query(sql, (queue_uuid,)).fetchall()
        finally:
            db.close()
    if not rows:
        return raw
    action = 'replace' if raw.get('replaced_id') else 'add'
    fact = {
        'id': rows[0][0],
        'content': rows[0][1],
        'action': action,
        'replaced_id': raw.get('replaced_id'),
        '_raw': raw,
        }
    return fact
