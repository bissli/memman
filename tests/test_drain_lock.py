"""Tests for `memman.drain_lock` and `_drain_queue`'s flock guard."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import drain_lock
from memman.cli import cli


def test_acquire_succeeds_when_unheld(tmp_path):
    """First acquirer gets the lock and creates the lock file."""
    fd = drain_lock.acquire(str(tmp_path))
    try:
        assert (tmp_path / 'drain.lock').exists()
    finally:
        drain_lock.release(fd)


def test_acquire_raises_when_already_held(tmp_path):
    """Second acquirer in same process raises DrainLockBusy without blocking.

    fcntl.flock on Linux is per-file-descriptor: two separate open()
    calls in the same process get distinct fds; the second LOCK_EX|LOCK_NB
    contends with the first.
    """
    fd1 = drain_lock.acquire(str(tmp_path))
    try:
        with pytest.raises(drain_lock.DrainLockBusy):
            drain_lock.acquire(str(tmp_path))
    finally:
        drain_lock.release(fd1)


def test_release_allows_reacquire(tmp_path):
    """After release, the lock can be acquired again."""
    fd1 = drain_lock.acquire(str(tmp_path))
    drain_lock.release(fd1)
    fd2 = drain_lock.acquire(str(tmp_path))
    drain_lock.release(fd2)


def test_lock_releases_on_subprocess_exit(tmp_path):
    """Subprocess holds lock; lock is released after the process exits.

    Verifies kernel-level auto-release on process death — no manual
    release call needed.
    """
    script = (
        'import sys, time;'
        ' from memman.drain_lock import acquire;'
        f' fd = acquire({str(tmp_path)!r});'
        ' print("acquired", flush=True);'
        ' time.sleep(60)'
        )
    proc = subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        first_line = proc.stdout.readline().strip()
        assert first_line == b'acquired'
        with pytest.raises(drain_lock.DrainLockBusy):
            drain_lock.acquire(str(tmp_path))
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            fd = drain_lock.acquire(str(tmp_path))
            drain_lock.release(fd)
            break
        except drain_lock.DrainLockBusy:
            time.sleep(0.05)
    else:
        pytest.fail('lock not released after subprocess exit')


def test_drain_skips_when_locked(tmp_path, monkeypatch):
    """If the lock is held, `_drain_queue` returns the skip JSON.

    Holds the lock in-process and runs `scheduler drain --pending`
    via CliRunner. Same-process contention works because fcntl.flock
    on Linux is per-file-descriptor.
    """
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    data_dir = str(tmp_path / 'data')
    Path(data_dir).mkdir()

    fd = drain_lock.acquire(data_dir)
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        assert out['skipped'] == 'another drain in progress'
        assert out['processed'] == 0
        assert out['failed'] == 0
    finally:
        drain_lock.release(fd)
