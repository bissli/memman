"""CliRunner coverage for the new memman command groups.

Tests exercise `memman enrich`, `memman queue`, and `memman scheduler`
at the Click layer (not the module layer) so regressions in argument
wiring and JSON output shape are caught.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.setup import scheduler as sch


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to a fresh temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'mm')
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return r, data_dir


def invoke(runner_tuple, args):
    """Invoke the CLI with --data-dir prepended."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def _fake_run_success(*a, **kw):
    """subprocess.run stub that pretends every systemctl/launchctl call succeeds."""
    class _Result:
        returncode = 0
        stdout = 'active'
        stderr = ''
    return _Result()


def _patch_no_subprocess(monkeypatch, *, active: bool = True):
    """Block real subprocess calls from scheduler.py."""
    class _FakeResult:
        returncode = 0 if active else 3
        stdout = 'active' if active else 'inactive'
        stderr = ''
    fake = type('S', (), {
        'run': staticmethod(lambda *a, **kw: _FakeResult()),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(sch, 'subprocess', fake)


def test_enrich_requires_pending(runner):
    """`memman enrich` without --pending errors out.
    """
    result = invoke(runner, ['enrich'])
    assert result.exit_code != 0
    assert 'pending' in result.output.lower()


def test_enrich_pending_empty_queue(runner):
    """`memman enrich --pending` on an empty queue returns processed=0.
    """
    result = invoke(runner, ['enrich', '--pending', '--limit', '5',
                             '--timeout', '5'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['processed'] == 0
    assert data['failed'] == 0
    assert data['remaining']['pending'] == 0


def test_queue_list_returns_stats_and_rows(runner):
    """`memman queue list` wraps rows in a {stats, rows} envelope.
    """
    invoke(runner, ['remember', '--defer', 'hello queue'])
    result = invoke(runner, ['queue', 'list'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert 'stats' in data
    assert 'rows' in data
    assert len(data['rows']) == 1
    assert data['rows'][0]['content_preview'].startswith('hello queue')


def test_queue_list_failed_same_shape(runner):
    """`memman queue list-failed` returns the same envelope as `queue list`.
    """
    result = invoke(runner, ['queue', 'list-failed'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert 'stats' in data
    assert 'rows' in data


def test_queue_cat_missing_errors(runner):
    """`memman queue cat <missing-id>` surfaces a clean error.
    """
    result = invoke(runner, ['queue', 'cat', '999'])
    assert result.exit_code != 0
    assert 'not found' in result.output.lower()


def test_queue_purge_requires_done(runner):
    """`memman queue purge` without --done errors out.
    """
    result = invoke(runner, ['queue', 'purge'])
    assert result.exit_code != 0
    assert '--done' in result.output


def test_queue_retry_noop_on_unknown(runner):
    """`memman queue retry <missing-id>` surfaces a clean error.
    """
    result = invoke(runner, ['queue', 'retry', '999'])
    assert result.exit_code != 0


def test_scheduler_bare_shows_help(runner):
    """`memman scheduler` with no subcommand prints the help listing.
    """
    result = invoke(runner, ['scheduler'])
    assert 'Commands:' in result.output
    assert 'trigger' in result.output
    assert 'status' in result.output


def test_scheduler_status_reports_not_installed(runner, monkeypatch):
    """`memman scheduler status` returns installed=false on a clean system.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['installed'] is False


def test_scheduler_enable_fails_when_not_installed(runner, monkeypatch):
    """`memman scheduler enable` errors when unit files are missing.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'enable'])
    assert result.exit_code != 0
    assert 'not installed' in result.output.lower()


def test_scheduler_disable_is_noop_when_not_installed(runner, monkeypatch):
    """`memman scheduler disable` returns cleanly when unit files are absent.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'disable'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data.get('note') == 'not installed'


def test_scheduler_interval_show_when_not_installed(runner, monkeypatch):
    """`memman scheduler interval` without --seconds shows current state.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'interval'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['installed'] is False
    assert data['interval_seconds'] is None


def test_scheduler_interval_rejects_too_short(runner, monkeypatch):
    """`memman scheduler interval --seconds 30` rejects sub-60s values.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'interval', '--seconds', '30'])
    assert result.exit_code != 0
    assert 'too short' in result.output.lower()


def test_scheduler_trigger_cli_happy_path(runner, monkeypatch):
    """`memman scheduler trigger` returns the dict from sch.trigger().
    """
    monkeypatch.setattr(
        sch, 'trigger',
        lambda: {'platform': 'systemd', 'actions': ['x'], 'note': 'n'})
    result = invoke(runner, ['scheduler', 'trigger'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['platform'] == 'systemd'
    assert data['note'] == 'n'


def test_scheduler_trigger_cli_fails_when_not_installed(
        runner, monkeypatch):
    """`memman scheduler trigger` surfaces FileNotFoundError cleanly.
    """
    def _raise():
        raise FileNotFoundError(
            "scheduler unit not installed at /x; run 'memman setup' first")
    monkeypatch.setattr(sch, 'trigger', _raise)
    result = invoke(runner, ['scheduler', 'trigger'])
    assert result.exit_code != 0
    assert 'not installed' in result.output.lower()
