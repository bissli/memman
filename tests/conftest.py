"""Shared fixtures for memman tests.

Dual-mode API mocking: mocked by default, real APIs with --live flag.

    pytest                    # fast, mocked LLM + embeddings
    pytest --live             # real Haiku + Voyage APIs (slow, needs keys)

Mock mode patches `OpenRouterClient.complete` and `voyage.Client.embed`
at the HTTP layer, so all extraction/reconciliation/expansion logic
still runs with realistic canned responses. This exercises the real
code paths.
"""

import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest
from memman.store.model import Edge, Insight

try:
    import psycopg  # noqa: F401
    import testcontainers.postgres  # noqa: F401
    pytest_plugins = ['tests.fixtures.postgres']
except ImportError:
    pytest_plugins = ()

EMBEDDING_DIM = 512


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --live flag for real API calls."""
    parser.addoption(
        '--live', action='store_true', default=False,
        help='Use real Haiku LLM and Voyage embedding APIs')


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch, request):
    """Pin MEMMAN_DATA_DIR to tmp and seed the env file.

    Prevents the user's real `~/.memman/env` from leaking into the
    config resolver during unit tests. By default, seeds a fresh env
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
    from memman import config
    data_dir = tmp_path / 'memman'
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    monkeypatch.delenv('MEMMAN_WORKER', raising=False)
    monkeypatch.delenv('MEMMAN_SCHEDULER_KIND', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    monkeypatch.delenv('MEMMAN_OPENAI_EMBED_API_KEY', raising=False)
    if 'no_default_env' not in request.keywords:
        _write_default_env_file(data_dir)
    config.reset_file_cache()
    yield
    config.reset_file_cache()


_TEST_MOCK_SECRETS = {
    'OPENROUTER_API_KEY': 'mock-key-for-testing',
    'VOYAGE_API_KEY': 'mock-voyage-key-for-testing',
    }


def _set_env_file_value(key: str, value: str | None) -> None:
    """Write or remove a key in the active test env file.

    Replacement for `monkeypatch.setenv` for installable keys -- the
    runtime resolver no longer reads `os.environ`, so tests must mutate
    the env file directly. Pass `value=None` to remove the key.
    """
    import os

    from memman import config
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

    Usage: `env_file('MEMMAN_LLM_MODEL_FAST', 'foo')` writes the row;
    `env_file('MEMMAN_LLM_MODEL_FAST', None)` removes it. Cache is
    auto-reset; the autouse `_isolate_env` fixture handles cleanup.
    """
    return _set_env_file_value


def _write_default_env_file(data_dir):
    """Seed `<data_dir>/env` with `INSTALL_DEFAULTS` for tests.

    Mirrors a post-install state so runtime call sites (which use
    `config.require`) resolve cleanly. Also seeds mock API key values
    for `OPENROUTER_API_KEY` and `VOYAGE_API_KEY` since the runtime
    resolver no longer consults `os.environ` -- the keys must live in
    the env file. Tests that need the broken state opt out via
    `@pytest.mark.no_default_env`.
    """
    from memman import config
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / config.ENV_FILENAME
    rows = list(config.INSTALL_DEFAULTS.items()) + list(
        _TEST_MOCK_SECRETS.items())
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
    from memman.cli import _reset_heartbeat_state as _reset
    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _scheduler_started(request, monkeypatch):
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
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if request.node.fspath.basename == 'test_scheduler_setup.py':
        return
    if 'no_scheduler_started_mock' in request.keywords:
        return
    from memman.setup import scheduler as sched_mod
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    import click.testing
    original_invoke = click.testing.CliRunner.invoke

    def _wrapped_invoke(self, cli_obj, args=None, **kwargs):
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


def _args_target_write(args) -> bool:
    if not args:
        return False
    for arg in args:
        if isinstance(arg, str) and arg in _AUTO_DRAIN_TRIGGERS:
            return True
    return False


def _args_data_dir(args) -> str | None:
    if not args:
        return None
    seq = list(args)
    for i, arg in enumerate(seq):
        if arg == '--data-dir' and i + 1 < len(seq):
            return seq[i + 1]
    return None


def _force_drain_with(runner_cls, data_dir, original_invoke) -> None:
    """Run `scheduler drain --pending` via the underlying click invoke.

    Bypasses the autouse-wrapped `invoke` to avoid re-triggering the
    auto-drain path on the drain command itself.
    """
    from memman.cli import cli
    instance = runner_cls()
    result = original_invoke(
        instance, cli,
        ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])
    assert result.exit_code == 0, (
        f'force_drain failed: exit={result.exit_code} '
        f'output={result.output} exc={result.exception}')


def force_drain(data_dir: str) -> None:
    """Synchronously drain the queue for the given data dir.

    Tests that follow `remember`/`replace` with a read assertion call
    this to flush pending work through the worker before reading. Uses
    the same `scheduler drain --pending` code path the OS timer fires.
    """
    import click.testing
    from memman.cli import cli
    instance = click.testing.CliRunner()
    result = instance.invoke(
        cli, ['--data-dir', data_dir,
              'scheduler', 'drain', '--pending'])
    assert result.exit_code == 0, (
        f'force_drain failed: exit={result.exit_code} '
        f'output={result.output} exc={result.exception}')


@pytest.fixture(autouse=True)
def _autoseed_fingerprint(request, monkeypatch):
    """Auto-seed `meta.embed_fingerprint` whenever assert_consistent runs.

    Production seeding (in `_open_db`, `_StoreContext`, `store_create`,
    `_init_default_store`) routes through `seed_if_fresh`, which
    declines to seed a DB that already has insights without a
    fingerprint (corruption). This fixture is intentionally more
    lenient: when a test hand-builds a DB by calling `open_db` and
    `insert_insight` directly, the fixture seeds on first
    `assert_consistent` regardless of insights count, so tests that
    don't care about fingerprint mechanics don't need boilerplate.

    Tests that exercise the strict production behavior (corruption
    detection or the raw missing-fingerprint error) should mark
    themselves `@pytest.mark.no_autoseed_fingerprint`.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if 'no_autoseed_fingerprint' in request.keywords:
        return

    from memman.embed import fingerprint as fp_mod
    real_assert = fp_mod.assert_consistent

    def seed_then_assert(backend):
        if fp_mod.stored_fingerprint(backend) is None:
            fp_mod.write_fingerprint(
                backend, fp_mod.active_fingerprint())
        real_assert(backend)

    monkeypatch.setattr(fp_mod, 'assert_consistent', seed_then_assert)


@pytest.fixture(autouse=True)
def _mock_apis(request, monkeypatch):
    """Mock LLM and embedding HTTP calls unless --live is set.

    Patches at the method layer: OpenRouterClient.complete returns
    realistic JSON that the real extract/reconcile/expand code parses.
    Voyage embed returns a deterministic content-hash vector.
    `openrouter_models.resolve_latest_in_family` is stubbed to a fixed
    id so install-path tests never hit the network.

    Tests that exercise the real OpenRouterClient.complete method
    should mark themselves with `@pytest.mark.no_mock_llm` to skip
    the method-level patch while keeping the resolver and embedding
    stubs in place.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if request.config.getoption('--live'):
        return

    if 'no_mock_llm' not in request.keywords:
        monkeypatch.setattr(
            'memman.llm.openrouter_client.OpenRouterClient.complete',
            _mock_llm_complete)
    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_in_family',
        lambda api_key, endpoint, family: f'anthropic/claude-{family}-4.5')
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _mock_embed)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed_batch', _mock_embed_batch)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.available', lambda self: True)
    from memman import config
    config.reset_file_cache()
    from memman.llm import client as llm_client_mod
    from memman.llm import extract as llm_extract_mod
    llm_client_mod.reset_role_cache()
    llm_extract_mod.reset_expand_cache()


def _mock_llm_complete(self: object, system: str, user: str) -> str:
    """Route to appropriate mock based on system prompt content."""
    if 'Extract atomic facts' in system:
        return _mock_fact_extraction(user)
    if 'Compare new facts' in system:
        return _mock_reconciliation(user)
    if 'Expand a search query' in system:
        return _mock_query_expansion(user)
    if 'keyword' in system.lower() and 'enrichment' in system.lower():
        return _mock_enrichment(user)
    if 'causal' in system.lower():
        return _mock_causal(user)
    return json.dumps({'facts': [{'text': user, 'category': 'fact',
                                  'importance': 3, 'entities': []}]})


def _mock_fact_extraction(content: str) -> str:
    """Generate realistic fact extraction response.

    Extracts entities by finding capitalized words. Preserves
    content as-is in a single fact — mimics Haiku behavior for
    substantive content.
    """
    entities = _extract_mock_entities(content)
    category = _infer_category(content)
    importance = _infer_importance(content)
    return json.dumps({
        'facts': [{
            'text': content,
            'category': category,
            'importance': importance,
            'entities': entities,
            }],
        'skip_reason': None,
        })


def _mock_reconciliation(prompt: str) -> str:
    """Generate realistic reconciliation response.

    Parses the structured prompt to find existing memories and new
    facts. If a new fact is very similar to an existing memory,
    returns UPDATE; otherwise ADD.
    """
    existing = {}
    new_facts = []
    in_existing = False
    in_new = False
    for line in prompt.split('\n'):
        if line.startswith('EXISTING MEMORIES:'):
            in_existing = True
            in_new = False
            continue
        if line.startswith('NEW FACTS:'):
            in_new = True
            in_existing = False
            continue
        if in_existing:
            m = re.match(r'\[(\d+)\]\s+(.*)', line)
            if m:
                existing[m.group(1)] = m.group(2)
        if in_new and line.startswith('- '):
            new_facts.append(line[2:])

    actions = []
    for fact in new_facts:
        fact_lower = fact.lower()
        best_id = None
        best_overlap = 0
        for eid, econtent in existing.items():
            words_f = set(fact_lower.split())
            words_e = set(econtent.lower().split())
            overlap = len(words_f & words_e) / max(len(words_f | words_e), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = eid

        if best_overlap > 0.6:
            actions.append({
                'fact': fact,
                'action': 'UPDATE',
                'target_id': best_id,
                'merged_text': fact,
                'reason': 'similar content, updating',
                })
        elif best_overlap > 0.4:
            actions.append({
                'fact': fact,
                'action': 'NONE',
                'target_id': best_id,
                'merged_text': None,
                'reason': 'already captured',
                })
        else:
            actions.append({
                'fact': fact,
                'action': 'ADD',
                'target_id': None,
                'merged_text': None,
                'reason': 'new information',
                })

    return json.dumps({'actions': actions})


def _mock_query_expansion(query: str) -> str:
    """Generate realistic query expansion response."""
    words = query.split()
    entities = [w for w in words if w[0:1].isupper()]
    return json.dumps({
        'expanded_query': query,
        'keywords': words,
        'entities': entities,
        'intent': 'GENERAL',
        })


def _mock_enrichment(content: str) -> str:
    """Generate realistic enrichment response."""
    return json.dumps({
        'keywords': content.lower().split()[:5],
        'summary': content[:100],
        'entities': _extract_mock_entities(content),
        })


def _mock_causal(content: str) -> str:
    """Generate realistic causal analysis response."""
    return json.dumps({'causal_links': []})


def _extract_mock_entities(text: str) -> list[str]:
    """Extract entities from text using capitalized words heuristic."""
    stopwords = {'The', 'A', 'An', 'In', 'On', 'For', 'With', 'And',
                 'Or', 'Is', 'Are', 'Was', 'Were', 'To', 'From', 'By',
                 'At', 'Of', 'But', 'Not', 'It', 'This', 'That', 'If',
                 'As', 'So', 'We', 'I', 'My', 'No', 'Yes', 'Chose',
                 'Switched', 'Store', 'Production', 'Configured',
                 'Infrastructure', 'Deployed', 'Critical', 'Uses',
                 'Using', 'Based', 'After', 'Before'}
    entities = []
    for word in re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', text):
        if word not in stopwords and word not in entities:
            entities.append(word)
    return entities[:5]


def _infer_category(text: str) -> str:
    """Infer category from content keywords."""
    lower = text.lower()
    if any(w in lower for w in ['chose', 'decided', 'picked', 'switched',
                                'migrated', 'prefer']):
        return 'decision'
    if any(w in lower for w in ['like', 'prefer', 'always', 'never',
                                'love', 'hate']):
        return 'preference'
    return 'fact'


def _infer_importance(text: str) -> int:
    """Infer importance from content keywords."""
    lower = text.lower()
    if any(w in lower for w in ['critical', 'outage', 'production',
                                'architecture']):
        return 4
    return 3


def _mock_embed_batch(
        self: object, texts: list[str]) -> list[list[float]]:
    """Batch variant of `_mock_embed`. One vector per input."""
    return [_mock_embed(self, t) for t in texts]


def _mock_embed(self: object, text: str) -> list[float]:
    """Deterministic embedding from content hash.

    Produces a 512-dim unit vector seeded by content, so identical
    text gives identical vectors. Different text gives different
    vectors with low cosine similarity.
    """
    digest = hashlib.sha256(text.encode()).digest()
    floats = list(struct.unpack(
        f'<{len(digest) // 4}f', digest))
    while len(floats) < EMBEDDING_DIM:
        extra = hashlib.sha256(
            digest + len(floats).to_bytes(4, 'little')).digest()
        floats.extend(struct.unpack(f'<{len(extra) // 4}f', extra))
    floats = floats[:EMBEDDING_DIM]
    norm = sum(x * x for x in floats) ** 0.5
    if norm > 0:
        floats = [x / norm for x in floats]
    return floats


@pytest.fixture
def tmp_db(request, tmp_path):
    """Fresh SQLite database in temp directory.

    Seeds `meta.embed_fingerprint` to match the active client by
    default, mirroring `setup.claude._init_default_store`. Tests
    exercising unseeded behavior should use the
    `no_autoseed_fingerprint` mark.
    """
    from memman.store.db import open_db
    from memman.store.sqlite import SqliteBackend
    db = open_db(str(tmp_path))
    if 'no_autoseed_fingerprint' not in request.keywords:
        from memman.embed.fingerprint import active_fingerprint
        from memman.embed.fingerprint import write_fingerprint
        write_fingerprint(SqliteBackend(db), active_fingerprint())
    yield db
    db.close()


@pytest.fixture
def tmp_backend(tmp_db):
    """Wrap `tmp_db` in a SqliteBackend.

    Pipeline / search / graph entry points take `Backend`. Tests that
    drive those entry points against a fresh store use this fixture;
    the underlying DB and SqliteBackend share the same connection so
    free-function and verb-surface calls see one transaction.
    """
    from memman.store.sqlite import SqliteBackend
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
    """The backend identifier for this parametrization slot."""
    return request.param


@pytest.fixture(params=_backend_params())
def runner_kind(request) -> str:
    """Backend identifier for CliRunner-driven cross-backend tests.

    Pairs with the `cross_backend_runner` fixture below to flip
    `MEMMAN_BACKEND` between sqlite and postgres for each test
    invocation. Phase 4b adds this fixture so the deferred
    `test_memory_system.py` 53 CliRunner tests can be parametrized
    over both backends.
    """
    return request.param


@pytest.fixture
def cross_backend_runner(request, runner_kind, tmp_path, env_file):
    """CliRunner whose env writes `MEMMAN_BACKEND=<runner_kind>` first.

    For postgres mode also writes `MEMMAN_PG_DSN` from the session
    container and registers a teardown that drops the per-test
    schema. Returns the same `(runner, data_dir)` tuple shape as
    the legacy `runner` fixture in `test_memory_system.py` so a
    test can swap one for the other transparently.
    """
    from click.testing import CliRunner
    r = CliRunner()
    data_dir = str(tmp_path / 'memman_data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    env_file('MEMMAN_BACKEND', runner_kind)
    if runner_kind == 'postgres':
        pg_dsn = request.getfixturevalue('pg_dsn')
        env_file('MEMMAN_PG_DSN', pg_dsn)
        store_name = _safe_store_name(request.node.name)
        env_file('MEMMAN_STORE', store_name)

        def _drop_postgres_schema() -> None:
            try:
                from memman.store.postgres import PostgresCluster
                cluster = PostgresCluster(dsn=pg_dsn)
                cluster.drop_store(store=store_name, data_dir='')
            except Exception:
                pass
        request.addfinalizer(_drop_postgres_schema)
    return r, data_dir


@pytest.fixture
def backend(request, backend_kind, tmp_path):
    """Cross-backend Backend fixture for Phase 3 pipeline tests.

    Parametrizes over `{sqlite, postgres}` (postgres slot active only
    when extras are importable). Yields a fully-isolated Backend with
    `meta.embed_fingerprint` pre-seeded so pipeline tests that touch
    embeddings do not trip the fingerprint refusal. Postgres tests
    get a unique store name per test so schemas don't collide; the
    schema is dropped on teardown.

    Pipeline / search / graph tests should use this fixture instead
    of `tmp_backend` to gain Postgres parity.
    """
    from memman.embed.fingerprint import META_KEY, active_fingerprint
    if backend_kind == 'sqlite':
        from memman.store.sqlite import SqliteCluster
        cluster = SqliteCluster()
        data_dir = str(tmp_path / 'memman')
        b = cluster.open(store='test', data_dir=data_dir)
        store_name = 'test'
    else:
        pg_dsn = request.getfixturevalue('pg_dsn')
        from memman.store.postgres import PostgresCluster
        cluster = PostgresCluster(dsn=pg_dsn)
        store_name = _safe_store_name(request.node.name)
        try:
            cluster.drop_store(store=store_name, data_dir='')
        except Exception:
            pass
        b = cluster.open(store=store_name, data_dir='')
    b.meta.set(META_KEY, active_fingerprint().to_json())
    try:
        yield b
    finally:
        try:
            b.close()
        except Exception:
            pass
        if backend_kind == 'postgres':
            try:
                cluster.drop_store(store=store_name, data_dir='')
            except Exception:
                pass
        try:
            cluster.close()
        except Exception:
            pass


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


def set_created_at(backend, insight_id: str, when: datetime) -> None:
    """Test-only: directly UPDATE `created_at` on a stored insight.

    The Backend Protocol's `nodes.insert` ignores caller-passed
    `Insight.created_at` per Phase 1a Decision #1 (server-side
    timestamps). Tests that exercise temporal logic against
    pre-existing rows with controlled timestamps call this helper
    after `backend.nodes.insert` to override the server-stamped
    value. Bypasses the Protocol intentionally; do NOT use outside
    test code.
    """
    from memman.store.model import format_timestamp
    from memman.store.sqlite import SqliteBackend
    when_str = format_timestamp(when)
    if isinstance(backend, SqliteBackend):
        backend._db._exec(
            'UPDATE insights SET created_at = ? WHERE id = ?',
            (when_str, insight_id))
    else:
        with backend._conn.cursor() as cur:
            cur.execute(
                f'UPDATE {backend._schema}.insights'
                ' SET created_at = %s WHERE id = %s',
                (when, insight_id))
        backend._conn.commit()


@pytest.fixture
def populated_db(tmp_db):
    """DB pre-loaded with 5 insights for query/graph tests."""
    from memman.store.node import insert_insight
    insights = [
        make_insight(id='pop-1', content='Go uses SQLite for storage',
                     importance=3,
                     entities=['Go', 'SQLite']),
        make_insight(id='pop-2', content='Python web framework comparison',
                     importance=2, category='decision'),
        make_insight(id='pop-3', content='Graph traversal algorithm for knowledge',
                     importance=4,
                     entities=['MAGMA']),
        make_insight(id='pop-4', content='Docker deployment strategy',
                     importance=5, category='preference',
                     entities=['Docker']),
        make_insight(id='pop-5', content='Go concurrency patterns',
                     importance=3, entities=['Go']),
        ]
    for ins in insights:
        insert_insight(tmp_db, ins)
    return tmp_db


def make_insight(**overrides) -> Insight:
    """Factory for test Insight instances."""
    now = datetime.now(timezone.utc)
    defaults = {
        'id': 'test-id',
        'content': 'test content',
        'category': 'fact',
        'importance': 3,
        'entities': [],
        'source': 'test',
        'access_count': 0,
        'created_at': now,
        'updated_at': now,
        'deleted_at': None,
        'last_accessed_at': None,
        'effective_importance': 0.0,
        }
    defaults.update(overrides)
    if 'entities' in overrides and overrides['entities'] is None:
        defaults['entities'] = []
    return Insight(**defaults)


def make_edge(**overrides) -> Edge:
    """Factory for test Edge instances."""
    now = datetime.now(timezone.utc)
    defaults = {
        'source_id': 'src',
        'target_id': 'tgt',
        'edge_type': 'semantic',
        'weight': 0.5,
        'metadata': {},
        'created_at': now,
        }
    defaults.update(overrides)
    return Edge(**defaults)
