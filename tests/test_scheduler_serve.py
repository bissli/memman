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

import memman.cli as cli_mod
import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.setup import scheduler as sched_mod


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Fresh CliRunner with isolated data + home dirs.
    """
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    return CliRunner(), str(tmp_path / 'memman')


def test_serve_once_drains_and_exits(runner, monkeypatch):
    """`--once` runs a single drain pass and returns clean.

    Mutation: `--once` returning before the drain pass, or a drain
        that claims the row but never marks it done.
    Oracle: the queued row's status reads 'done' after the drain.
    """
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
    assert '"status": "done"' in queue_result.output, queue_result.output


def test_serve_writes_interval_file(runner, monkeypatch):
    """Verify serve writes scheduler.serve_interval and removes it on exit.

    Mutation: skipping write_serve_interval at startup, writing a different
        value, or leaving the file behind, which gives doctor a stale heartbeat
        threshold.
    Oracle: the value read through read_serve_interval() from inside the drain,
        then the file's absence after exit.
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    interval_path = Path(os.environ['HOME']) / '.memman' / 'scheduler.serve_interval'
    captured: dict = {}

    def _capture_then_stop(*args, **kwargs):
        captured['interval'] = sched_mod.read_serve_interval()
        captured['exists'] = interval_path.exists()

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
    """Verify the serve loop exits without draining when the state is STOPPED.

    Mutation: draining before the read_state() check, so a stopped scheduler
        still drains.
    Oracle: a drain counter that stays 0.
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STOPPED)

    drain_calls = {'count': 0}

    def _count_drains(*args, **kwargs):
        drain_calls['count'] += 1

    monkeypatch.setattr(cli_mod, '_drain_queue', _count_drains)

    r, data_dir = runner
    result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve', '--interval', '60'])
    assert result.exit_code == 0, result.output
    assert drain_calls['count'] == 0, (
        'serve should not drain when state=STOPPED')


def test_serve_interval_zero_loops_until_signaled(runner, monkeypatch):
    """Verify interval=0 keeps looping until a stop is requested.

    Mutation: breaking out of the loop after one drain when interval is 0.
    Oracle: a stub drain that requests stop on its 5th call, so 5 or more calls
        prove the loop kept iterating.
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    drain_calls = {'count': 0}

    def _counting_drain(*args, **kwargs):
        drain_calls['count'] += 1
        if drain_calls['count'] >= 5:
            cli_mod._request_stop()
        return {'claimed': 0, 'processed': 0, 'failed': 0}

    monkeypatch.setattr(cli_mod, '_drain_queue', _counting_drain)

    r, data_dir = runner
    result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve', '--interval', '0'])
    assert result.exit_code == 0, result.output
    assert drain_calls['count'] >= 5, (
        f'expected 5+ drain calls (continuous loop); got '
        f'{drain_calls["count"]} - interval=0 exited too early')


def test_serve_interval_zero_idle_backoff(runner, monkeypatch):
    """Verify empty drains at interval=0 sleep 100ms between iterations.

    Mutation: dropping the 0.1s sleep after an empty drain, so interval=0 spins
        the CPU and the SQLite WAL.
    Oracle: wall clock of at least 0.2s across 3 empty drains (two sleeps).
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    drain_calls = {'count': 0}

    def _empty_drain(*args, **kwargs):
        drain_calls['count'] += 1
        if drain_calls['count'] >= 3:
            cli_mod._request_stop()
        return {'claimed': 0, 'processed': 0, 'failed': 0}

    monkeypatch.setattr(cli_mod, '_drain_queue', _empty_drain)

    r, data_dir = runner
    t0 = time.monotonic()
    result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve', '--interval', '0'])
    elapsed = time.monotonic() - t0

    assert result.exit_code == 0, result.output
    assert drain_calls['count'] >= 3
    assert elapsed >= 0.2, (
        f'expected >=200ms wall (2x100ms backoff between 3 drains);'
        f' got {elapsed:.3f}s - backoff missing')


@pytest.mark.no_auto_drain
def test_a_drain_after_a_stopped_serve_stores_its_row(runner, monkeypatch):
    """Verify serve leaves no stop behind for a later drain in its process.

    Mutation: serve returning with the stop flag still set, so every
        later in-process drain exits before its first claim.
    Oracle: the queued row's status reads 'done' after the drain.
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    def _stopping_drain(*args, **kwargs):
        cli_mod._request_stop()
        return {'claimed': 0, 'processed': 0, 'failed': 0}

    r, data_dir = runner
    with monkeypatch.context() as serve_patch:
        serve_patch.setattr(cli_mod, '_drain_queue', _stopping_drain)
        serve_result = r.invoke(
            cli,
            ['--data-dir', data_dir, 'scheduler', 'serve', '--interval', '0'])
    assert serve_result.exit_code == 0, serve_result.output

    r.invoke(cli, ['--data-dir', data_dir, 'remember', 'hello world'])
    r.invoke(cli, ['--data-dir', data_dir, 'scheduler', 'drain'])

    queue_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'queue', 'list'])
    assert '"status": "done"' in queue_result.output, queue_result.output


def test_serve_default_interval_runs_one_drain(runner, monkeypatch):
    """Verify interval=60 with --once drains once and exits 0.

    Mutation: the interval > 0 setup (per-drain timeout, interval file) raising
        under --once, or --once waiting out the interval instead of exiting.
    Oracle: exit code 0 from a real drain of an empty queue.
    """
    monkeypatch.setattr(sched_mod, 'read_state',
                        lambda: sched_mod.STATE_STARTED)

    r, data_dir = runner
    serve_result = r.invoke(
        cli,
        ['--data-dir', data_dir, 'scheduler', 'serve',
         '--interval', '60', '--once'])
    assert serve_result.exit_code == 0, serve_result.output


@pytest.mark.no_mock_llm
def test_serve_handles_sigterm_cleanly(tmp_path):
    """Verify a real serve process exits 0 on SIGTERM.

    A subprocess replaces CliRunner because POSIX signals do not reach in-
    process click invocations.

    Mutation: installing no SIGTERM handler, so the process dies from the
        signal (return code -15).
    Oracle: the subprocess return code 0 within 10 seconds.
    """
    home = tmp_path / 'home'
    home.mkdir()
    data_dir = tmp_path / 'data'

    env = {
        **os.environ,
        'HOME': str(home),
        'MEMMAN_API_KEY': 'mock',
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
