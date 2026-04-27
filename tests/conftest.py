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

import pytest
from memman.model import Edge, Insight

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


def _write_default_env_file(data_dir):
    """Seed `<data_dir>/env` with `INSTALL_DEFAULTS` for tests.

    Mirrors a post-install state so runtime call sites (which use
    `config.require`) resolve cleanly. Tests that need the broken
    state opt out via `@pytest.mark.no_default_env`.
    """
    from memman import config
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / config.ENV_FILENAME
    contents = '\n'.join(
        f'{k}={v}' for k, v in config.INSTALL_DEFAULTS.items()) + '\n'
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

    Production: `setup.claude._init_default_store` seeds the meta row at
    install time. Tests don't run install; instead, this fixture wraps
    `assert_consistent` to seed-if-missing then assert. Net effect:
    every fresh DB used in CLI tests behaves as if `memman install`
    had pre-seeded it.

    Tests that exercise unseeded behavior (e.g. the assert itself)
    should mark themselves `@pytest.mark.no_autoseed_fingerprint`.
    """
    if 'tests/e2e/' in str(request.node.fspath):
        return
    if 'no_autoseed_fingerprint' in request.keywords:
        return

    from memman.embed import fingerprint as fp_mod
    real_assert = fp_mod.assert_consistent

    def seed_then_assert(db):
        if fp_mod.stored_fingerprint(db) is None:
            fp_mod.write_fingerprint(db, fp_mod.active_fingerprint())
        real_assert(db)

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
    monkeypatch.setenv('OPENROUTER_API_KEY', 'mock-key-for-testing')
    monkeypatch.setenv('VOYAGE_API_KEY', 'mock-voyage-key-for-testing')
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
    db = open_db(str(tmp_path))
    if 'no_autoseed_fingerprint' not in request.keywords:
        from memman.embed.fingerprint import active_fingerprint
        from memman.embed.fingerprint import write_fingerprint
        write_fingerprint(db, active_fingerprint())
    yield db
    db.close()


@pytest.fixture
def populated_db(tmp_db):
    """DB pre-loaded with 5 insights for query/graph tests."""
    from memman.store.node import insert_insight
    insights = [
        make_insight(id='pop-1', content='Go uses SQLite for storage',
                     importance=3, tags=['go', 'sqlite'],
                     entities=['Go', 'SQLite']),
        make_insight(id='pop-2', content='Python web framework comparison',
                     importance=2, category='decision'),
        make_insight(id='pop-3', content='Graph traversal algorithm for knowledge',
                     importance=4, tags=['graph'],
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
        'tags': [],
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
    if 'tags' in overrides and overrides['tags'] is None:
        defaults['tags'] = []
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
