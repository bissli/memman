"""Tests for the daily check that the configured OpenRouter model routes."""

import json
import os
import time
from pathlib import Path

import httpx
import pytest
from memman import config
from memman.cli import cli
from memman.llm import openrouter_models as om
from memman.setup import claude

PINNED_VENDORS = frozenset({'amazon-bedrock', 'azure', 'google-vertex'})
MODEL = 'qwen/qwen3-235b-a22b-2507'
ENDPOINT = 'https://openrouter.ai/api/v1'


def _model(model_id, expiration_date=None):
    """Build a `/models` entry; the live catalog carries a null date.
    """
    return {'id': model_id, 'expiration_date': expiration_date}


def _zdr(model_id, tag='google-vertex/us-south1'):
    """Build a `/endpoints/zdr` row.
    """
    return {'model_id': model_id, 'tag': tag}


def _notice(models, zdr_rows, vendors=PINNED_VENDORS):
    """Check MODEL against the two catalogs under `vendors`.
    """
    return om.model_notice(models, zdr_rows, model=MODEL, vendors=vendors)


def _unroutable():
    """The notice for MODEL finding no pinned ZDR endpoint.
    """
    return om.UNROUTABLE_NOTICE.format(model=MODEL)


def _data_dir():
    """The test data dir the autouse fixture seeded with the defaults.
    """
    return os.environ[config.DATA_DIR]


def _write_state(model, age_seconds, notice):
    """Record a check of `model` made `age_seconds` ago.
    """
    state = {
        'model': model,
        'checked_at': int(time.time()) - age_seconds,
        'notice': notice,
        }
    (Path(_data_dir()) / 'model.state').write_text(json.dumps(state))


def _read_state():
    """The recorded check as the data dir holds it.
    """
    return json.loads((Path(_data_dir()) / 'model.state').read_text())


@pytest.fixture
def fetches(monkeypatch):
    """Replace the catalog fetch with a spy returning a fixed notice.
    """
    calls = []

    def _fetch(endpoint, *, model, vendors):
        calls.append((endpoint, model, vendors))
        return 'fresh notice'

    monkeypatch.setattr(
        'memman.llm.openrouter_models.fetch_model_notice', _fetch)
    return calls


def test_a_pinned_zdr_endpoint_routes_the_model():
    """A ZDR row on a pinned vendor leaves no notice.

    Mutation: the vendor read from the whole tag, so
        `google-vertex/us-south1` misses the pin.
    Oracle: the model's real Vertex ZDR tag.
    """
    assert _notice([_model(MODEL)], [_zdr(MODEL)]) == ''


def test_an_off_pin_endpoint_alone_leaves_the_model_unroutable():
    """A ZDR row on a vendor outside the pin does not route the model.

    Mutation: the vendor filter dropped, so a model only DeepInfra
        serves reads as routed.
    Oracle: the model's real DeepInfra ZDR tag, off the shipped pin.
    """
    zdr_rows = [_zdr(MODEL, tag='deepinfra/fp8')]
    assert _notice([_model(MODEL)], zdr_rows) == _unroutable()


def test_a_longer_id_sharing_the_prefix_does_not_route_the_model():
    """Only an exact model id counts as the configured model.

    Mutation: a prefix match, so the `-thinking` sibling routes the
        configured id.
    Oracle: catalogs that list only the sibling, on a pinned vendor.
    """
    sibling = MODEL + '-thinking'
    assert _notice([_model(sibling)], [_zdr(sibling)]) == _unroutable()


def test_a_retirement_date_yields_the_retiring_notice():
    """A routed model with an expiration date gets the retiring notice.

    Mutation: the date never read, or read from the first catalog entry
        that carries one.
    Oracle: a hand-set date on the model, a different one on an earlier
        entry.
    """
    models = [
        _model('qwen/qwen3-max-thinking', expiration_date='2026-10-01'),
        _model(MODEL, expiration_date='2026-10-09'),
        ]
    expected = om.RETIRING_NOTICE.format(model=MODEL, date='2026-10-09')
    assert _notice(models, [_zdr(MODEL)]) == expected


def test_an_empty_pin_routes_through_any_zdr_vendor():
    """With no vendor pin, any ZDR endpoint routes the model.

    Mutation: an empty pin matches nothing, so clearing
        MEMMAN_LLM_PROVIDER_ONLY reports every model unroutable.
    Oracle: the runtime client sends no `only` list for an empty pin.
    """
    zdr_rows = [_zdr(MODEL, tag='deepinfra/fp8')]
    assert _notice([_model(MODEL)], zdr_rows, vendors=frozenset()) == ''


@pytest.mark.no_mock_catalog
def test_fetch_reads_only_the_two_public_catalogs(monkeypatch):
    """Fetching makes two keyless GETs and no chat call.

    Mutation: a live chat probe added to confirm the model, or an
        Authorization header sent to the public endpoints.
    Oracle: the URLs and headers a stubbed HTTP client records, and a
        spy on the chat call.
    """
    payloads = {
        f'{ENDPOINT}/endpoints/zdr': {'data': [_zdr(MODEL)]},
        f'{ENDPOINT}/models': {'data': [_model(MODEL)]},
        }
    requests = []
    chat_calls = []

    class _Resp:
        status_code = 200

        def __init__(self, url):
            self.url = url

        def raise_for_status(self):
            return None

        def json(self):
            return payloads[self.url]

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            requests.append((url, kwargs.get('headers') or {}))
            return _Resp(url)

    monkeypatch.setattr('memman.llm.openrouter_models.httpx.Client', _Client)
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        lambda *a, **k: chat_calls.append(a))
    notice = om.fetch_model_notice(
        ENDPOINT, model=MODEL, vendors=PINNED_VENDORS)
    assert sorted(url for url, _ in requests) == sorted(payloads)
    assert all('Authorization' not in headers for _, headers in requests)
    assert chat_calls == []
    assert notice == ''


def test_a_state_inside_the_interval_skips_the_fetch(fetches):
    """A same-model check younger than the interval is not repeated.

    Mutation: the age test dropped or inverted, so every drain GETs
        both catalogs.
    Oracle: a state one minute short of the interval.
    """
    _write_state(MODEL, om.CHECK_INTERVAL_SECONDS - 60, '')
    assert om.refresh_model_state(_data_dir(), force=False) is None
    assert fetches == []


def test_a_state_past_the_interval_refetches(fetches):
    """A same-model check older than the interval is repeated and recorded.

    Mutation: the age compared in milliseconds, or the new notice
        returned but never written.
    Oracle: a state one minute past the interval, and the seeded
        endpoint, model and pin.
    """
    _write_state(MODEL, om.CHECK_INTERVAL_SECONDS + 60, '')
    assert om.refresh_model_state(_data_dir(), force=False) == 'fresh notice'
    assert fetches == [(ENDPOINT, MODEL, PINNED_VENDORS)]
    assert _read_state()['notice'] == 'fresh notice'


def test_a_state_naming_another_model_refetches_at_once(fetches):
    """A model change is checked on the next drain, whatever the age.

    Mutation: freshness judged on age alone, so a new model waits a day
        for its first check.
    Oracle: a state one minute old naming the previous model.
    """
    _write_state('qwen/qwen3-old', 60, '')
    om.refresh_model_state(_data_dir(), force=False)
    assert len(fetches) == 1
    assert _read_state()['model'] == MODEL


def test_force_refetches_a_fresh_state(fetches):
    """`force=True` checks even when a fresh same-model state stands.

    Mutation: `force` ignored, so an install after a recent drain
        reports nothing.
    Oracle: a state one minute old naming the configured model.
    """
    _write_state(MODEL, 60, '')
    om.refresh_model_state(_data_dir(), force=True)
    assert len(fetches) == 1


def test_a_failed_fetch_keeps_the_notice_and_reraises(monkeypatch):
    """An outage keeps the standing notice, restarts the clock, and raises.

    Mutation: a failure blanks the notice, or leaves `checked_at`
        unchanged so every drain in an outage refetches.
    Oracle: a standing notice two days old and a fetch raising
        `httpx.ConnectError`.
    """
    def _unreachable(endpoint, *, model, vendors):
        raise httpx.ConnectError('no route')

    monkeypatch.setattr(
        'memman.llm.openrouter_models.fetch_model_notice', _unreachable)
    _write_state(MODEL, 2 * om.CHECK_INTERVAL_SECONDS, 'standing notice')
    started = int(time.time())
    with pytest.raises(httpx.ConnectError):
        om.refresh_model_state(_data_dir(), force=False)
    state = _read_state()
    assert state['notice'] == 'standing notice'
    assert state['checked_at'] >= started


def test_a_loopback_endpoint_is_never_checked(env_file, fetches):
    """A non-OpenRouter endpoint has no catalog, so no check runs.

    Mutation: the endpoint guard dropped, so an Ollama install GETs
        openrouter.ai.
    Oracle: a loopback Ollama endpoint and a forced refresh.
    """
    env_file(config.LLM_ENDPOINT, 'http://localhost:11434/v1')
    assert om.refresh_model_state(_data_dir(), force=True) is None
    assert fetches == []


def test_read_model_notice_drops_a_notice_for_another_model():
    """A notice recorded for a replaced model is not shown.

    Mutation: the model comparison dropped, so a stale notice outlives
        the fix it names.
    Oracle: the same notice recorded under the old and the configured
        model.
    """
    _write_state('qwen/qwen3-old', 60, 'old notice')
    assert om.read_model_notice(_data_dir()) == ''
    _write_state(MODEL, 60, 'old notice')
    assert om.read_model_notice(_data_dir()) == 'old notice'


def test_back_to_back_drains_check_the_catalog_once(mm_runner, fetches):
    """The scheduler drain checks the model once per interval.

    Mutation: the drain never checks, or checks with `force=True`.
    Oracle: two drains a moment apart and a spy on the fetch.
    """
    runner, data_dir = mm_runner
    for _ in range(2):
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
        assert result.exit_code == 0, result.output
    assert len(fetches) == 1


def _run_install_flow(monkeypatch, refresh):
    """Run the install flow with the scheduler stubbed and `refresh` patched.
    """
    monkeypatch.setattr(
        'memman.setup.claude.install_scheduler',
        lambda data_dir, knobs: {'platform': 'systemd', 'actions': []})
    monkeypatch.setattr(
        'memman.llm.openrouter_models.refresh_model_state', refresh)
    env = {
        'detected': False,
        'display': 'Claude Code',
        'version': '',
        'config_dir': '',
        }
    claude._run_install_flow(env, target='', data_dir=_data_dir(), knobs={})


def test_install_forces_the_check_and_prints_the_notice(monkeypatch, capsys):
    """The install checks the model at once and shows what it found.

    Mutation: the install skips the check, calls it without `force`, or
        drops the notice it returns.
    Oracle: a refresh spy returning a fixed notice.
    """
    seen = []

    def _refresh(data_dir, *, force):
        seen.append(force)
        return 'the model retires soon'

    _run_install_flow(monkeypatch, _refresh)
    assert seen == [True]
    assert 'the model retires soon' in capsys.readouterr().out


def test_install_finishes_when_the_catalog_is_unreachable(
        monkeypatch, capsys):
    """An unreachable catalog prints an error and the install completes.

    Mutation: the network error propagates and aborts the install.
    Oracle: a refresh raising `httpx.ConnectError`.
    """
    def _unreachable(data_dir, *, force):
        raise httpx.ConnectError('no route')

    _run_install_flow(monkeypatch, _unreachable)
    printed = capsys.readouterr().out
    assert 'cannot read the OpenRouter catalogs' in printed
    assert 'no route' in printed


@pytest.mark.no_mock_catalog
@pytest.mark.parametrize(
    'body', ['<html>maintenance</html>', '[]'], ids=['html', 'json-list'])
def test_a_drain_survives_a_catalog_reply_that_is_not_a_json_object(
        mm_runner, monkeypatch, body):
    """A 200 catalog reply that is not a JSON object ends no drain early.

    Mutation: the reply parsed outside the RuntimeError contract, so a
        JSONDecodeError or AttributeError escapes the drain's catch and
        skips its lock release and run record.
    Oracle: a stubbed 200 reply carrying an HTML page, and one carrying
        a JSON list.
    """
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return json.loads(body)

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            return _Resp()

    monkeypatch.setattr('memman.llm.openrouter_models.httpx.Client', _Client)
    runner, data_dir = mm_runner
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
    assert result.exit_code == 0, repr(result.exception)
    assert _read_state()['model'] == MODEL
