"""CliRunner coverage for the new memman command groups.

Tests exercise `memman scheduler drain`, `memman scheduler queue`, and
`memman scheduler` at the Click layer (not the module layer) so
regressions in argument wiring and JSON output shape are caught.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from memman.queue import open_queue_db
from memman.setup import scheduler as sch
from tests.conftest import fake_subprocess, invoke


@pytest.fixture
def runner(mm_runner):
    return mm_runner


def _patch_no_subprocess(monkeypatch, *, active: bool = True):
    """Thin wrapper for shared `fake_subprocess` keyed on `sch`.
    """
    fake_subprocess(monkeypatch, sch, active=active)


def test_drain_empty_queue(runner):
    """Verify `scheduler drain` on an empty queue reports zero work.

    Mutation: counting a phantom row as processed or failed, or dropping
        the `remaining.pending` key from the drain summary.
    Oracle: hand-set zeros for an empty queue.
    """
    result = invoke(runner, ['scheduler', 'drain', '--limit', '5',
                             '--timeout', '5'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['processed'] == 0
    assert data['failed'] == 0
    assert data['remaining']['pending'] == 0


def test_queue_list_returns_stats_and_rows(runner, monkeypatch):
    """Verify `scheduler queue list` wraps rows in a {stats, rows} envelope.

    Mutation: returning a bare row list, or truncating the content
        preview so it no longer starts with the queued text.
    Oracle: the one row queued by `remember`, checked by key and prefix.
    """

    invoke(runner, ['remember', 'hello queue'])
    result = invoke(runner, ['scheduler', 'queue', 'list'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert 'stats' in data
    assert 'rows' in data
    assert len(data['rows']) == 1
    assert data['rows'][0]['content_preview'].startswith('hello queue')


def test_queue_list_failed_same_shape(runner):
    """Verify `scheduler queue failed` returns the same envelope as `list`.

    Mutation: the failed subcommand emitting a bare list, or omitting
        `stats`.
    Oracle: envelope keys of `queue list`, checked by hand.
    """
    result = invoke(runner, ['scheduler', 'queue', 'failed'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert 'stats' in data
    assert 'rows' in data


def test_queue_cat_missing_errors(runner):
    """Verify `scheduler queue show` on an unknown id exits non-zero.

    Mutation: printing an empty result and exiting 0 for a missing row.
    Oracle: exit code and the "not found" message for id 999.
    """
    result = invoke(runner, ['scheduler', 'queue', 'show', '999'])
    assert result.exit_code != 0
    assert 'not found' in result.output.lower()


def test_queue_purge_requires_flag(runner):
    """Verify `scheduler queue purge` needs --done.

    Mutation: defaulting to a purge of every done row when no flag is
        given.
    Oracle: non-zero exit with the flag name in the message.
    """
    result = invoke(runner, ['scheduler', 'queue', 'purge'])
    assert result.exit_code != 0
    assert '--done' in result.output


def test_queue_retry_noop_on_unknown(runner):
    """Verify `scheduler queue retry` on an unknown id exits non-zero.

    Mutation: reporting success for a row id that does not exist.
    Oracle: exit code for id 999 in an empty queue.
    """
    result = invoke(runner, ['scheduler', 'queue', 'retry', '999'])
    assert result.exit_code != 0


def _seed_row(data_dir: str, status: str) -> int:
    """Insert one queue row with the given status.

    Parameters
    ----------
    data_dir : str
        Data directory that holds the queue database.
    status : str
        Status stored on the new row.

    Returns
    -------
    int
        Id of the new row.
    """
    conn = open_queue_db(data_dir)
    try:
        cur = conn.execute(
            "insert into queue"
            ' (store, content, status, queue_uuid, queued_at)'
            " values (?, ?, ?, ?, strftime('%s','now'))",
            ('default', f'{status}-row', status, str(uuid.uuid4())))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def test_scheduler_status_text_output(runner, monkeypatch):
    """Verify `scheduler status --text` prints key:value lines.

    Mutation: ignoring --text and emitting JSON.
    Oracle: the literal `installed:` and `platform:` keys in the output.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home', lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'status', '--text'])
    assert result.exit_code == 0, result.output
    assert 'installed:' in result.output
    assert 'platform:' in result.output


def test_scheduler_status_reports_the_rotated_worker_log(
        runner, monkeypatch, tmp_path):
    """`scheduler status` names <data-dir>/logs/memman.log.

    Mutation: resolving the rotated log under ~/.memman/logs like the
        two enrich redirects, or omitting the key, either of which
        leaves an operator with no route to a preserved stack.
    Oracle: the absolute path built from the runner's own data dir,
        compared for equality, with the fake home held distinct so a
        conftest change collapsing the two could not hide the
        mutation.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    fake_home = tmp_path / 'fake_home'
    fake_home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake_home)
    assert Path(runner[1]) != fake_home
    result = invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['stack_path'] == str(
        Path(runner[1]) / 'logs' / 'memman.log')


def test_scheduler_status_times_the_stack_file_it_names(
        runner, monkeypatch, tmp_path):
    """`scheduler status` reports each log's OWN mtime.

    Mutation: pairing `stack_mtime` with `err_path` or `log_path` in
        the mtime loop, which tells an operator asking whether a stack
        is fresh about a different file entirely. Dropping the entry is
        caught by the same assertion.
    Oracle: the three files are stamped to three known, distinct
        epochs, so each key can only match its own file.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    fake_home = tmp_path / 'fake_home'
    home_logs = fake_home / '.memman' / 'logs'
    home_logs.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: fake_home)
    data_logs = Path(runner[1]) / 'logs'
    data_logs.mkdir(parents=True, exist_ok=True)
    stamps = {
        home_logs / 'enrich.log': 1000000000,
        home_logs / 'enrich.err': 1200000000,
        data_logs / 'memman.log': 1400000000,
        }
    for path, when in stamps.items():
        path.write_text('x\n')
        os.utime(path, (when, when))

    result = invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    expected = {
        'log_mtime': 1000000000,
        'err_mtime': 1200000000,
        'stack_mtime': 1400000000,
        }
    for key, when in expected.items():
        got = datetime.fromisoformat(data[key]).timestamp()
        assert got == when, f'{key}: {data[key]}'


def test_scheduler_bare_shows_help(runner):
    """Verify bare `memman scheduler` prints the help listing.

    Mutation: unregistering a subcommand, or running an action when no
        subcommand is given.
    Oracle: the literal `Commands:` header and two subcommand names.
    """
    result = invoke(runner, ['scheduler'])
    assert 'Commands:' in result.output
    assert 'trigger' in result.output
    assert 'status' in result.output


def test_scheduler_status_reports_not_installed(runner, monkeypatch):
    """Verify `scheduler status` reports installed=false with no unit files.

    Mutation: reporting installed=true from platform detection alone.
    Oracle: an empty fake home holding no unit files.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['installed'] is False


def test_scheduler_start_fails_when_not_installed(runner, monkeypatch):
    """Verify `scheduler start` errors when unit files are missing.

    Mutation: invoking the service manager anyway and exiting 0.
    Oracle: non-zero exit and the "not installed" message.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'start'])
    assert result.exit_code != 0
    assert 'not installed' in result.output.lower()


def test_scheduler_interval_show_when_not_installed(runner, monkeypatch):
    """Verify `scheduler interval` without --seconds shows the current state.

    Mutation: writing a default interval, or reporting a number when no
        unit is installed.
    Oracle: installed=false and interval_seconds=None for an empty home.
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
    """Verify `scheduler interval --seconds 30` rejects sub-60s values.

    Mutation: a flipped or removed lower-bound comparison.
    Oracle: non-zero exit and the "too short" message for 30 seconds.
    """
    _patch_no_subprocess(monkeypatch)
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(Path, 'home',
                        lambda: Path(runner[1]))
    result = invoke(runner, ['scheduler', 'interval', '--seconds', '30'])
    assert result.exit_code != 0
    assert 'too short' in result.output.lower()


def test_scheduler_trigger_cli_happy_path(runner, monkeypatch):
    """Verify `scheduler trigger` prints the dict `sch.trigger()` returns.

    Mutation: dropping or renaming a key of the trigger result on its
        way to the output.
    Oracle: a stub trigger returning known values.
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
    """Verify `scheduler trigger` turns FileNotFoundError into a clean error.

    Mutation: letting the exception escape as a traceback with the
        wrong exit code, or swallowing it with exit 0.
    Oracle: non-zero exit and the "not installed" message from a stub
        that raises.
    """
    def _raise():
        raise FileNotFoundError(
            "scheduler unit not installed at /x; run 'memman setup' first")
    monkeypatch.setattr(sch, 'trigger', _raise)
    result = invoke(runner, ['scheduler', 'trigger'])
    assert result.exit_code != 0
    assert 'not installed' in result.output.lower()


def test_scheduler_start_leaves_an_old_untried_write_pending(
        runner, monkeypatch):
    """Verify `scheduler start` keeps a week-old untried write pending.

    Mutation: the start path moving long-pending rows to a status the
        drain never claims, so the write is never stored.
    Oracle: the row's status read back from the queue database, for a
        row queued 8 days ago with no attempt.
    """
    _, data_dir = runner
    monkeypatch.setattr(sch, 'start', lambda: {'status': 'started'})
    row_id = _seed_row(data_dir, 'pending')
    conn = open_queue_db(data_dir)
    try:
        conn.execute(
            'update queue set queued_at = queued_at - ? where id = ?',
            (8 * 24 * 3600, row_id))
        conn.commit()
    finally:
        conn.close()

    result = invoke(runner, ['scheduler', 'start'])
    assert result.exit_code == 0, result.output

    conn = open_queue_db(data_dir)
    try:
        status = conn.execute(
            'select status from queue where id = ?', (row_id,)).fetchone()[0]
    finally:
        conn.close()
    assert status == 'pending'
