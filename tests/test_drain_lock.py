"""Tests for `memman.drain_lock` and `_drain_queue`'s flock guard.
"""

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
    """Verify the first acquirer gets the lock and creates the lock file.

    Mutation: acquire returning without creating drain.lock.
    Oracle: the file's existence in the data dir.
    """
    fd = drain_lock.acquire(str(tmp_path))
    try:
        assert (tmp_path / 'drain.lock').exists()
    finally:
        drain_lock.release(fd)


def test_acquire_raises_when_already_held(tmp_path):
    """Verify a second acquire in the same process raises DrainLockBusy.

    Mutation: blocking, or succeeding, when the lock is already held.
    Oracle: pytest.raises(DrainLockBusy) while the first fd is open.
    """
    fd1 = drain_lock.acquire(str(tmp_path))
    try:
        with pytest.raises(drain_lock.DrainLockBusy):
            drain_lock.acquire(str(tmp_path))
    finally:
        drain_lock.release(fd1)


def test_release_allows_reacquire(tmp_path):
    """Verify the lock can be acquired again after release.

    Mutation: release leaving the lock held, so the second acquire raises.
    Oracle: the second acquire completing without an exception.
    """
    fd1 = drain_lock.acquire(str(tmp_path))
    drain_lock.release(fd1)
    fd2 = drain_lock.acquire(str(tmp_path))
    drain_lock.release(fd2)


def test_lock_releases_on_subprocess_exit(tmp_path):
    """Verify the kernel frees the lock when the holder process dies.

    Mutation: a lock kept in a marker file that outlives its holder.
    Oracle: a SIGKILLed subprocess that held the lock; a later acquire
        succeeds within two seconds.
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


def test_drain_lock_released_on_setup_failure(tmp_path, monkeypatch):
    """Verify the lock is released when drain setup raises after acquiring.

    Mutation: acquiring the lock outside the try/finally, so a failing
        open_queue_db leaks the fd for the process lifetime.
    Oracle: a fresh acquire on the same data dir succeeds after the failed
        drain.
    """
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    data_dir = str(tmp_path / 'data')
    Path(data_dir).mkdir()

    def _boom(*_a, **_kw):
        raise RuntimeError('synthetic setup failure')

    monkeypatch.setattr('memman.queue.open_queue_db', _boom)

    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
    assert result.exit_code != 0

    fd = drain_lock.acquire(data_dir)
    drain_lock.release(fd)


def test_drain_skips_when_locked(tmp_path, monkeypatch):
    """Verify `scheduler drain` returns the skip JSON while the lock is held.

    Mutation: draining anyway, or exiting nonzero, when another drain holds
        the lock.
    Oracle: the JSON skipped reason with processed and failed at 0.
    """
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    data_dir = str(tmp_path / 'data')
    Path(data_dir).mkdir()

    fd = drain_lock.acquire(data_dir)
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        assert out['skipped'] == 'another drain in progress'
        assert out['processed'] == 0
        assert out['failed'] == 0
    finally:
        drain_lock.release(fd)
