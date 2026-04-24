"""Tests for memman.trace -- structured debug trace mode.

Trace mode is default-off. When MEMMAN_DEBUG=1 (or equivalent truthy
value) and `trace.setup(data_dir)` is called, a RotatingFileHandler
is attached to the 'memman' logger at DEBUG level and one JSON line
per call to `trace.event(...)` is written to
<data_dir>/logs/debug.log. The file is chmod 600. Header redaction
strips secret values while keeping bodies verbatim.
"""

import json
import logging
import os
import stat
from pathlib import Path

import httpx
import pytest
from memman import trace
from memman.llm.openrouter_client import OpenRouterClient


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect HOME + Path.home to a tmp_path (mirrors test_scheduler_setup)."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    return tmp_path


@pytest.fixture
def debug_on(monkeypatch):
    """Turn trace mode on for the duration of the test."""
    monkeypatch.setenv('MEMMAN_DEBUG', '1')


@pytest.fixture(autouse=True)
def _reset_trace_state():
    """Remove any trace handlers the previous test left on the memman logger."""
    yield
    logger = logging.getLogger('memman')
    for h in list(logger.handlers):
        if getattr(h, '_memman_trace', False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_is_enabled_reads_env_var(monkeypatch):
    """is_enabled() is True when MEMMAN_DEBUG is truthy, False otherwise.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    assert trace.is_enabled() is False
    monkeypatch.setenv('MEMMAN_DEBUG', '1')
    assert trace.is_enabled() is True
    monkeypatch.setenv('MEMMAN_DEBUG', 'true')
    assert trace.is_enabled() is True
    monkeypatch.setenv('MEMMAN_DEBUG', '0')
    assert trace.is_enabled() is False


def test_setup_is_noop_when_disabled(fake_home, monkeypatch):
    """setup() creates no log file when MEMMAN_DEBUG is unset.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    trace.setup()
    logs_dir = fake_home / '.memman' / 'logs'
    assert not logs_dir.exists() or not any(logs_dir.iterdir())


def test_event_is_noop_when_disabled(fake_home, monkeypatch):
    """event() writes nothing when MEMMAN_DEBUG is unset.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    trace.setup()
    trace.event('some_event', foo='bar')
    logs_dir = fake_home / '.memman' / 'logs'
    assert not logs_dir.exists() or not any(logs_dir.iterdir())


def test_setup_creates_mode_600_file_when_enabled(fake_home, debug_on):
    """setup() creates ~/.memman/logs/debug.log at mode 600.
    """
    trace.setup()
    trace.event('probe')
    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    assert log_path.exists()
    mode = stat.S_IMODE(os.stat(log_path).st_mode)
    assert mode == 0o600


def test_event_writes_one_jsonl_line(fake_home, debug_on):
    """event() emits exactly one JSON line per call with the expected keys.
    """
    trace.setup()
    trace.event('probe', foo='bar', count=3)
    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed['event'] == 'probe'
    assert parsed['foo'] == 'bar'
    assert parsed['count'] == 3
    assert 'ts' in parsed


def test_event_writes_multiple_lines_in_order(fake_home, debug_on):
    """Multiple event() calls produce one line each, in order.
    """
    trace.setup()
    trace.event('first')
    trace.event('second')
    trace.event('third')
    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    events = [json.loads(ln)['event']
              for ln in log_path.read_text().strip().splitlines()]
    assert events == ['first', 'second', 'third']


def test_setup_is_idempotent(fake_home, debug_on):
    """Repeated setup() calls do not attach duplicate handlers.
    """
    trace.setup()
    trace.setup()
    trace.setup()
    logger = logging.getLogger('memman')
    trace_handlers = [h for h in logger.handlers
                      if getattr(h, '_memman_trace', False)]
    assert len(trace_handlers) == 1


def test_redact_headers_strips_authorization():
    """redact_headers() replaces Authorization values with ***REDACTED***.
    """
    out = trace.redact_headers({
        'Authorization': 'Bearer sk-very-secret',
        'Content-Type': 'application/json',
        })
    assert out['Authorization'] == '***REDACTED***'
    assert out['Content-Type'] == 'application/json'


def test_redact_headers_strips_x_api_key():
    """redact_headers() replaces x-api-key values.
    """
    out = trace.redact_headers({
        'x-api-key': 'sk-ant-secret',
        'User-Agent': 'memman',
        })
    assert out['x-api-key'] == '***REDACTED***'
    assert out['User-Agent'] == 'memman'


def test_redact_headers_is_case_insensitive():
    """redact_headers() matches header names case-insensitively.
    """
    out = trace.redact_headers({
        'AUTHORIZATION': 'Bearer x',
        'X-API-KEY': 'y',
        'Api-Key': 'z',
        })
    assert out['AUTHORIZATION'] == '***REDACTED***'
    assert out['X-API-KEY'] == '***REDACTED***'
    assert out['Api-Key'] == '***REDACTED***'


def test_redact_headers_does_not_mutate_input():
    """redact_headers() returns a new dict; input dict is unchanged.
    """
    original = {'Authorization': 'Bearer secret'}
    out = trace.redact_headers(original)
    assert original['Authorization'] == 'Bearer secret'
    assert out is not original


def test_openrouter_complete_emits_request_and_response(
        fake_home, debug_on, monkeypatch):
    """OpenRouterClient.complete emits llm_request and llm_response in order.
    """
    from memman.llm import openrouter_cache as cache_mod
    monkeypatch.setattr(cache_mod, '_fetch',
                        lambda: [{'model_id': 'anthropic/claude-haiku-4.5'}])
    monkeypatch.setenv('MEMMAN_CACHE_DIR', str(fake_home))

    def _fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(
            200,
            request=httpx.Request('POST', url),
            json={'choices': [{'message': {'content': 'hi'}}]})

    monkeypatch.setattr(httpx, 'post', _fake_post)
    trace.setup()
    client = OpenRouterClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-secret-key')
    out = client.complete('sys', 'user')
    assert out == 'hi'

    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    events = [json.loads(ln)
              for ln in log_path.read_text().strip().splitlines()]
    names = [e['event'] for e in events]
    assert 'llm_request' in names
    assert 'llm_response' in names
    req_idx = names.index('llm_request')
    resp_idx = names.index('llm_response')
    assert req_idx < resp_idx

    req = events[req_idx]
    assert req['provider'] == 'openrouter'
    assert req['headers']['Authorization'] == '***REDACTED***'
    assert req['body']['model'] == 'anthropic/claude-haiku-4.5'

    resp = events[resp_idx]
    assert resp['provider'] == 'openrouter'
    assert resp['status'] == 200
    assert resp['body']['choices'][0]['message']['content'] == 'hi'
