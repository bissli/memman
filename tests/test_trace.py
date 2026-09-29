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
import pwd
import stat
from pathlib import Path

import httpx
import pytest
from memman import _http, trace
from memman.llm import client as llm_client_mod
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient


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


def test_is_enabled_reads_env_var(fake_home, monkeypatch):
    """Verify is_enabled() follows MEMMAN_DEBUG when it is set.

    Mutation: treating any non-empty value as on, so '0' enables tracing, or
        ignoring the variable.
    Oracle: hand-picked values: unset, '1', 'true' and '0'.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    assert trace.is_enabled() is False
    monkeypatch.setenv('MEMMAN_DEBUG', '1')
    assert trace.is_enabled() is True
    monkeypatch.setenv('MEMMAN_DEBUG', 'true')
    assert trace.is_enabled() is True
    monkeypatch.setenv('MEMMAN_DEBUG', '0')
    assert trace.is_enabled() is False


def test_is_enabled_reads_state_file_when_env_unset(fake_home, monkeypatch):
    """Verify is_enabled() falls back to debug.state when the env var is unset.

    Mutation: ignoring debug.state, or reading 'off' as on.
    Oracle: a state file written by hand with 'on', then 'off'.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    state_path = fake_home / '.memman' / 'debug.state'
    state_path.parent.mkdir(parents=True, exist_ok=True)

    assert trace.is_enabled() is False
    state_path.write_text('on\n')
    assert trace.is_enabled() is True
    state_path.write_text('off\n')
    assert trace.is_enabled() is False


def test_env_var_overrides_state_file(fake_home, monkeypatch):
    """Verify a truthy MEMMAN_DEBUG wins over an 'off' state file.

    Mutation: reading debug.state before the env var, so an 'off' file disables
        MEMMAN_DEBUG=1.
    Oracle: state file 'off' with MEMMAN_DEBUG=1 exported.
    """
    state_path = fake_home / '.memman' / 'debug.state'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('off\n')
    monkeypatch.setenv('MEMMAN_DEBUG', '1')
    assert trace.is_enabled() is True


def test_setup_is_noop_when_disabled(fake_home, monkeypatch):
    """Verify setup() creates no log file when MEMMAN_DEBUG is unset.

    Mutation: attaching the file handler without the is_enabled() check. The
        handler opens debug.log at once (delay=False), so the file appears with
        tracing off.
    Oracle: the logs directory stays absent or empty.
    """
    monkeypatch.delenv('MEMMAN_DEBUG', raising=False)
    trace.setup()
    logs_dir = fake_home / '.memman' / 'logs'
    assert not logs_dir.exists() or not any(logs_dir.iterdir())


def test_event_is_noop_when_disabled(fake_home, monkeypatch):
    """Verify event() writes nothing once tracing is switched off.

    Mutation: dropping the is_enabled() guard in event(), so a handler
        attached earlier keeps receiving lines with tracing off.
    Oracle: debug.log stays empty after event() runs with a handler
        attached and MEMMAN_DEBUG set to 0.
    """
    monkeypatch.setenv('MEMMAN_DEBUG', '1')
    trace.setup()
    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    assert log_path.exists()
    monkeypatch.setenv('MEMMAN_DEBUG', '0')
    trace.event('some_event', foo='bar')
    assert log_path.read_text() == ''


def test_setup_creates_mode_600_file_when_enabled(fake_home, debug_on):
    """Verify setup() creates ~/.memman/logs/debug.log at mode 600.

    Mutation: dropping the chmod, which leaves the umask default (0644) on a
        file that holds raw memory content.
    Oracle: os.stat mode compared with the literal 0o600.
    """
    trace.setup()
    trace.event('probe')
    log_path = fake_home / '.memman' / 'logs' / 'debug.log'
    assert log_path.exists()
    mode = stat.S_IMODE(os.stat(log_path).st_mode)
    assert mode == 0o600


def test_event_writes_one_jsonl_line(fake_home, debug_on):
    """Verify event() emits exactly one JSON line per call with its fields.

    Mutation: indenting the JSON over several lines, or dropping the event
        name, ts or the passed fields.
    Oracle: the single log line parsed with json.loads against the values
        passed in.
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
    """Verify successive event() calls append one line each, in call order.

    Mutation: reopening the log in write mode per event so only the last
        survives, or reordering events.
    Oracle: the hand-listed order first, second, third.
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
    """Verify repeated setup() calls attach one handler.

    Mutation: dropping the _memman_trace check in setup(), so each call adds a
        handler and duplicates every line.
    Oracle: the count of tagged handlers on the memman logger equals 1.
    """
    trace.setup()
    trace.setup()
    trace.setup()
    logger = logging.getLogger('memman')
    trace_handlers = [h for h in logger.handlers
                      if getattr(h, '_memman_trace', False)]
    assert len(trace_handlers) == 1


def test_autouse_isolation_keeps_trace_log_off_the_real_home():
    """With no fake_home, the trace log still resolves outside the real home.

    Mutation: _isolate_env leaving Path.home() at the real home, so a
      live debug.state of 'on' sends test log lines into the developer's
      ~/.memman/logs/debug.log.
    Oracle: the password-database home directory, which no fixture
      patches.
    """
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert not trace._trace_path().is_relative_to(real_home)


class TestRedaction:
    """redact_headers and redact_dsn strip secrets from trace output."""

    def test_redact_headers_strips_authorization(self):
        """Verify redact_headers() masks Authorization and keeps other headers.

        Mutation: leaving authorization out of REDACT_HEADER_NAMES, or masking
            every header.
        Oracle: the literal '***REDACTED***' for Authorization and the original
            Content-Type value.
        """
        out = trace.redact_headers({
            'Authorization': 'Bearer sk-very-secret',
            'Content-Type': 'application/json',
            })
        assert out['Authorization'] == '***REDACTED***'
        assert out['Content-Type'] == 'application/json'

    def test_redact_headers_strips_x_api_key(self):
        """Verify redact_headers() masks x-api-key and keeps other headers.

        Mutation: leaving x-api-key out of REDACT_HEADER_NAMES, or masking
            every header.
        Oracle: the literal '***REDACTED***' for x-api-key and the original
            User-Agent value.
        """
        out = trace.redact_headers({
            'x-api-key': 'sk-ant-secret',
            'User-Agent': 'memman',
            })
        assert out['x-api-key'] == '***REDACTED***'
        assert out['User-Agent'] == 'memman'

    def test_redact_headers_is_case_insensitive(self):
        """Verify redact_headers() matches header names regardless of case.

        Mutation: comparing names without lower(), so an upper-case
            AUTHORIZATION leaks.
        Oracle: three upper- or mixed-case names, each read back as
            '***REDACTED***'.
        """
        out = trace.redact_headers({
            'AUTHORIZATION': 'Bearer x',
            'X-API-KEY': 'y',
            'Api-Key': 'z',
            })
        assert out['AUTHORIZATION'] == '***REDACTED***'
        assert out['X-API-KEY'] == '***REDACTED***'
        assert out['Api-Key'] == '***REDACTED***'

    def test_redact_headers_does_not_mutate_input(self):
        """Verify redact_headers() returns a new dict and leaves the input alone.

        Mutation: masking values in place on the caller's dict, which corrupts
            the live request headers.
        Oracle: the input dict keeps its secret, and the result is a different
            object.
        """
        original = {'Authorization': 'Bearer secret'}
        out = trace.redact_headers(original)
        assert original['Authorization'] == 'Bearer secret'
        assert out is not original

    def test_masks_inline_password(self):
        """Verify redact_dsn() masks the password of user:password@host.

        Mutation: a pattern that drops the password group or masks the user
            name instead.
        Oracle: the hand-written expected DSN with '***' in the password slot.
        """
        assert trace.redact_dsn(
            'postgresql://alice:s3cret@db.example.com:5432/memman'
            ) == 'postgresql://alice:***@db.example.com:5432/memman'

    def test_passthrough_when_no_password(self):
        """Verify redact_dsn() returns a passwordless DSN unchanged.

        Mutation: masking the user name of a DSN that carries no password.
        Oracle: output equals input.
        """
        assert trace.redact_dsn(
            'postgresql://alice@db.example.com:5432/memman'
            ) == 'postgresql://alice@db.example.com:5432/memman'

    def test_passthrough_for_non_dsn_string(self):
        """Verify redact_dsn() returns text without a DSN shape unchanged.

        Mutation: making the scheme:// prefix optional, so 14:18@noon reads
            as user:password@host, or a rewrite that raises on the empty
            string.
        Oracle: output equals input for each string, including ones that
            carry ':' and '@' but no scheme://.
        """
        for text in (
                'not a connection string', '', 'host:5432',
                'user@example.com', '14:18@noon'):
            assert trace.redact_dsn(text) == text

    def test_handles_alternate_schemes(self):
        """Verify redact_dsn() masks the password under any scheme.

        Mutation: hardcoding postgresql:// in the pattern, so a postgres:// DSN
            leaks its password.
        Oracle: the hand-written expected 'postgres://u:***@h/db'.
        """
        assert trace.redact_dsn(
            'postgres://u:p@h/db') == 'postgres://u:***@h/db'

    def test_masks_whole_password_containing_literal_at_sign(self):
        """Verify redact_dsn() masks a password that holds a literal '@'.

        Mutation: a password class that stops at the first '@', so the
            tail of 'p@ss' stays in the output.
        Oracle: hand-written expected DSNs; userinfo runs to the last '@'
            before the host, and a later '@' after the path is untouched.
        """
        assert trace.redact_dsn(
            'postgres://u:p@ss@h/db') == 'postgres://u:***@h/db'
        assert trace.redact_dsn(
            'postgres://u:a@b@c@h:5432/db?x=y@z'
            ) == 'postgres://u:***@h:5432/db?x=y@z'

    def test_masks_password_with_percent_encoded_at_sign(self):
        """Verify redact_dsn() masks a password whose '@' is percent-encoded.

        Mutation: a password class that stops at '%', so 'p%40ss' is cut
            short and the tail 'ss' leaks.
        Oracle: the hand-written expected DSN with '***' in the password
            slot.
        """
        assert trace.redact_dsn(
            'postgres://u:p%40ss@h/db') == 'postgres://u:***@h/db'


@pytest.mark.no_mock_llm
def test_llm_complete_emits_request_and_response(
        fake_home, debug_on, monkeypatch):
    """Verify complete() traces a redacted request, then the response.

    Mutation: dropping either trace event, emitting the response first, or
        logging the Authorization header unmasked.
    Oracle: the log lines parsed and compared with the endpoint, model, status
        and reply body the fake server and client were given.
    """
    def _fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(
            200,
            request=httpx.Request('POST', url),
            json={'choices': [{'message': {'content': 'hi'}}]})

    monkeypatch.setitem(
        _http._SESSIONS, llm_client_mod.__name__,
        type('FakeClient', (), {'post': staticmethod(_fake_post)})())
    trace.setup()
    client = MemmanLLMClient(
        endpoint='https://openrouter.ai/api/v1',
        api_key='fake-secret-key',
        model='anthropic/claude-haiku-4.5')
    out = client.complete(
        'sys', 'user', stage=llm_usage.STAGE_PROBE)
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
    assert req['endpoint'] == 'https://openrouter.ai/api/v1'
    assert req['headers']['Authorization'] == '***REDACTED***'
    assert req['body']['model'] == 'anthropic/claude-haiku-4.5'

    resp = events[resp_idx]
    assert resp['endpoint'] == 'https://openrouter.ai/api/v1'
    assert resp['status'] == 200
    assert resp['body']['choices'][0]['message']['content'] == 'hi'
