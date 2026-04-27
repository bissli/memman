"""Tests for `memman scheduler serve` long-running drain command.

Covers --once mode (single drain pass), state-file stop polling, the
SIGTERM clean-exit contract, and the persisted serve-interval file
that doctor reads to compute the heartbeat threshold.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman.cli import cli


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Fresh CliRunner with isolated data + home dirs."""
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    return CliRunner(), str(tmp_path / 'data')


def test_serve_once_drains_and_exits(runner, monkeypatch):
    """`--once` runs a single drain pass and returns clean.
    """
    from memman.setup import scheduler as sched_mod
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    r, data_dir = runner
    add_result = r.invoke(
        cli, ['--data-dir', data_dir, 'remember', 'hello world'])
    assert add_result.exit_code == 0, add_result.output

    serve_result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve',
         '--interval', '0', '--once'])
    assert serve_result.exit_code == 0, serve_result.output

    queue_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'queue', 'list'])
    assert queue_result.exit_code == 0, queue_result.output


def test_serve_writes_interval_file(runner, monkeypatch):
    """Serve startup writes ~/.memman/scheduler.serve_interval (mode 600).

    The file is removed on clean exit. Doctor reads this for the
    heartbeat threshold when running under serve mode.
    """
    from memman.setup import scheduler as sched_mod
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    interval_path = Path(os.environ['HOME']) / '.memman' / 'scheduler.serve_interval'
    captured: dict = {}

    real_drain = sched_mod  # placeholder

    def _capture_then_stop(*args, **kwargs):
        captured['interval'] = sched_mod.read_serve_interval()
        captured['exists'] = interval_path.exists()

    import memman.cli as cli_mod
    monkeypatch.setattr(cli_mod, '_drain_queue', _capture_then_stop)

    r, data_dir = runner
    result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve',
         '--interval', '42', '--once'])
    assert result.exit_code == 0, result.output

    assert captured.get('interval') == 42
    assert captured.get('exists') is True
    assert not interval_path.exists(), (
        'serve should remove the interval file on clean exit')


def test_serve_stops_when_state_file_says_stopped(runner, monkeypatch):
    """Loop exits when read_state() returns STATE_STOPPED.

    Used for `memman scheduler stop` semantics in serve mode: the stop
    command flips the state file and the running serve loop notices on
    its next iteration.
    """
    from memman.setup import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STOPPED)

    drain_calls = {'count': 0}

    def _count_drains(*args, **kwargs):
        drain_calls['count'] += 1

    import memman.cli as cli_mod
    monkeypatch.setattr(cli_mod, '_drain_queue', _count_drains)

    r, data_dir = runner
    result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve', '--interval', '60'])
    assert result.exit_code == 0, result.output
    assert drain_calls['count'] == 0, (
        'serve should not drain when state=STOPPED')


@pytest.mark.no_mock_llm
def test_serve_handles_sigterm_cleanly(tmp_path):
    """A real subprocess running serve exits 0 on SIGTERM.

    Uses subprocess (not CliRunner) because POSIX signals do not
    propagate to in-process click invocations.
    """
    home = tmp_path / 'home'
    home.mkdir()
    data_dir = tmp_path / 'data'

    env = {
        **os.environ,
        'HOME': str(home),
        'OPENROUTER_API_KEY': 'mock',
        'MEMMAN_DATA_DIR': str(data_dir),
        }
    proc = subprocess.Popen(
        [sys.executable, '-m', 'memman.cli', 'scheduler', 'serve',
         '--interval', '60'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:
        time.sleep(2.0)
        proc.send_signal(signal.SIGTERM)
        ret = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    assert ret == 0, (
        f'serve exited with {ret};'
        f' stdout={proc.stdout.read()!r} stderr={proc.stderr.read()!r}')
