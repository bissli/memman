"""Unit tests for the OpenRouter install-time model resolver."""

import pytest
from memman.exceptions import ConfigError
from memman.llm import openrouter_models as om
from memman.llm.client import MemmanLLMClient

_REAL_RESOLVE = om.resolve_latest_for_role

SAMPLE_MODELS = [
    {'id': 'qwen/qwen3-235b-a22b-2507'},
    {'id': 'qwen/qwen3-235b-a22b-2412'},
    {'id': 'qwen/qwen3-235b-a22b-thinking-2507'},
    {'id': 'qwen/qwen3-235b-a22b'},
    {'id': 'qwen/qwen3-30b-a3b-instruct-2507'},
    {'id': 'anthropic/claude-haiku-4.5'},
    {'id': 'anthropic/claude-sonnet-4.6'},
    {'id': 'meta-llama/llama-4-maverick'},
    ]


@pytest.fixture(autouse=True)
def _undo_global_resolver_mock(monkeypatch):
    """Restore the real `resolve_latest_for_role` for these tests."""
    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_for_role',
        _REAL_RESOLVE)
    om.clear_cache()
    yield
    om.clear_cache()


def test_fast_picks_the_newest_snapshot_of_the_pinned_line(monkeypatch):
    """The resolver picks the newest dated snapshot, not an older one.

    Mutation: the version sort reverses, or the pattern loses its
    date anchor, so install pins a stale snapshot of the same model
    line and every later release has to be set by hand.
    Oracle: the catalog above, which holds a newer and an older
    dated snapshot of one model line.
    """
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_for_role('fast') == 'qwen/qwen3-235b-a22b-2507'


def test_resolver_never_picks_a_reasoning_variant(monkeypatch):
    """No role resolves to a `-thinking-` snapshot of the pinned line.

    Mutation: the role pattern admits the reasoning variant, so a
    fresh install pins a model that bills reasoning tokens on every
    reconcile call and can return an empty body at the stage's token
    ceiling.
    Oracle: the catalog above, which carries the plain and the
    thinking snapshot of the same line at the same date.
    """
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    for role in ('fast', 'slow'):
        resolved = om.resolve_latest_for_role(role)
        assert 'thinking' not in resolved, f'{role} resolved {resolved!r}'


def test_unknown_role_returns_none(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: SAMPLE_MODELS)
    assert om.resolve_latest_for_role('bogus') is None


def test_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(om, '_fetch_models', lambda *a, **k: [
        {'id': 'meta-llama/llama-4-maverick'}])
    assert om.resolve_latest_for_role('fast') is None


def test_returns_none_on_network_failure(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError('no route')

    monkeypatch.setattr(om, '_fetch_models', boom)
    assert om.resolve_latest_for_role('fast') is None


def test_caches_within_session(monkeypatch):
    calls = {'n': 0}

    def counting(*a, **k):
        calls['n'] += 1
        return SAMPLE_MODELS

    monkeypatch.setattr(om, '_fetch_models', counting)
    om.resolve_latest_for_role('fast')
    om.resolve_latest_for_role('fast')
    assert calls['n'] == 1


def test_fetch_models_sends_no_authorization(monkeypatch):
    """OR's /models is public; memman must not send an Authorization header."""
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'data': SAMPLE_MODELS}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            captured['url'] = url
            captured['headers'] = kwargs.get('headers')
            return _Resp()

    monkeypatch.setattr('memman.llm.openrouter_models.httpx.Client', _Client)
    om._fetch_models('https://openrouter.ai/api/v1')
    assert captured['headers'] is None or 'Authorization' not in (
        captured['headers'] or {})


def test_version_sort_key_orders_correctly():
    assert om._version_sort_key('anthropic/claude-haiku-10.0') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_version_sort_key_suffix_outranks_base():
    assert om._version_sort_key('anthropic/claude-haiku-4.5-v2') > \
        om._version_sort_key('anthropic/claude-haiku-4.5')


def test_llm_client_requires_model():
    with pytest.raises(ConfigError, match='model is empty'):
        MemmanLLMClient(
            'https://openrouter.ai/api/v1',
            'sk-or-test',
            model='')


def test_llm_client_accepts_model():
    client = MemmanLLMClient(
        'https://openrouter.ai/api/v1',
        'sk-or-test',
        model='anthropic/claude-haiku-4.5')
    assert client.model == 'anthropic/claude-haiku-4.5'
    assert client.endpoint == 'https://openrouter.ai/api/v1'


def test_install_resolver_puts_every_role_on_one_tier():
    """Both LLM roles resolve to the same model family at install.

    Mutation: the slow-role pattern names a pricier family than the
    fast role, so a fresh install bills fact extraction and enrichment
    at a rate the cost model never covered.
    Oracle: the fast role's resolved slug over the same catalog.
    """
    fast = om.resolve_latest_for_role('fast')
    slow = om.resolve_latest_for_role('slow')
    fast_family = fast.split('/', 1)[-1].rsplit('-', 1)[0]
    slow_family = slow.split('/', 1)[-1].rsplit('-', 1)[0]
    assert slow_family == fast_family, f'{slow!r} is not {fast_family!r}'


def _capture_post(monkeypatch):
    """Stub the shared session and return the dict the body lands in."""
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'choices': [{'message': {'content': '{"ok": true}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                }

    class _Session:
        def post(self, url, **kwargs):
            captured['url'] = url
            captured['body'] = kwargs.get('json')
            return _Resp()

    monkeypatch.setattr('memman.llm.client.get_session', lambda _n: _Session())
    return captured


@pytest.mark.no_mock_llm
def test_openrouter_request_carries_the_operator_provider_routing(monkeypatch):
    """A client given provider routing sends it as the body's `provider`.

    Mutation: the routing block is dropped from the request body, so
    every call routes to whichever provider OpenRouter picks and the
    operator's jurisdiction and retention choice goes unenforced with
    no error raised.
    Oracle: the request body captured from a stubbed session.
    """
    captured = _capture_post(monkeypatch)
    routing = {'only': ['amazon-bedrock'], 'data_collection': 'deny'}
    client = MemmanLLMClient(
        'https://openrouter.ai/api/v1', 'sk-or-test',
        model='anthropic/claude-haiku-4.5', provider_routing=routing)
    client.complete('sys', 'user', stage='enrichment')
    assert captured['body']['provider'] == routing


@pytest.mark.no_mock_llm
def test_client_without_routing_sends_no_provider_key(monkeypatch):
    """A vendor-neutral endpoint gets no OpenRouter-only field.

    Mutation: the provider block is attached unconditionally, so an
    Ollama or vLLM shim receives an unknown top-level field it may
    reject.
    Oracle: the request body captured from a stubbed session.
    """
    captured = _capture_post(monkeypatch)
    client = MemmanLLMClient(
        'http://localhost:11434/v1', '', model='llama3')
    client.complete('sys', 'user', stage='enrichment')
    assert 'provider' not in captured['body']
