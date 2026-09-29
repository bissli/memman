"""CliRunner coverage for the new memman command groups.

Tests exercise `memman scheduler drain`, `memman scheduler queue`, and
`memman scheduler` at the Click layer (not the module layer) so
regressions in argument wiring and JSON output shape are caught.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from memman.maintenance import run_maintenance
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
    """Verify `scheduler queue purge` needs --done or --stale.

    Mutation: defaulting to a purge of every row when no flag is given.
    Oracle: non-zero exit with both flag names in the message.
    """
    result = invoke(runner, ['scheduler', 'queue', 'purge'])
    assert result.exit_code != 0
    assert '--done' in result.output
    assert '--stale' in result.output


def test_queue_purge_rejects_conflicting_flags(runner):
    """`queue purge` takes exactly one of its two target flags.

    Mutation: dropping the mutual-exclusion guard, so a pair of flags
        silently purges only whichever branch happens to run first.
    Oracle: a conflicting pair exits non-zero naming both flags.
    """
    result = invoke(
        runner, ['scheduler', 'queue', 'purge', '--done', '--stale'])
    assert result.exit_code != 0
    assert 'exactly one of --done, --stale' in result.output


def test_queue_retry_noop_on_unknown(runner):
    """Verify `scheduler queue retry` on an unknown id exits non-zero.

    Mutation: reporting success for a row id that does not exist.
    Oracle: exit code for id 999 in an empty queue.
    """
    result = invoke(runner, ['scheduler', 'queue', 'retry', '999'])
    assert result.exit_code != 0


def test_queue_retry_requires_arg_or_flag(runner):
    """Verify `scheduler queue retry` needs a row id or --all-stale.

    Mutation: dropping the missing-argument guard, so a bare `retry`
        requeues nothing and exits 0.
    Oracle: non-zero exit naming --all-stale.
    """
    result = invoke(runner, ['scheduler', 'queue', 'retry'])
    assert result.exit_code != 0
    assert '--all-stale' in result.output


def test_queue_retry_rejects_id_with_all_stale(runner):
    """Verify `scheduler queue retry 5 --all-stale` is rejected.

    Mutation: dropping the exclusion guard, so the row id silently wins
        over --all-stale or the reverse.
    Oracle: non-zero exit with the "not both" message.
    """
    result = invoke(
        runner, ['scheduler', 'queue', 'retry', '5', '--all-stale'])
    assert result.exit_code != 0
    assert 'not both' in result.output


def _seed_row(data_dir: str, status: str = 'stale') -> int:
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
            ' (store, content, category, status, queue_uuid, queued_at)'
            " values (?, ?, ?, ?, ?, strftime('%s','now'))",
            ('default', f'{status}-row', 'fact', status, str(uuid.uuid4())))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _seed_stale_row(data_dir: str) -> int:
    """Insert one stale queue row and return its id.
    """
    return _seed_row(data_dir, 'stale')


def test_queue_retry_all_stale_requeues(runner):
    """Verify `queue retry --all-stale` flips a stale row to pending.

    Mutation: reporting the requeue in JSON without updating the row's
        status, or updating a different status.
    Oracle: the status column read back from the queue database.
    """
    _, data_dir = runner
    row_id = _seed_stale_row(data_dir)
    result = invoke(
        runner, ['scheduler', 'queue', 'retry', '--all-stale'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['action'] == 'requeued'
    assert data['count'] >= 1

    conn = open_queue_db(data_dir)
    try:
        status = conn.execute(
            'select status from queue where id = ?', (row_id,)).fetchone()[0]
        assert status == 'pending'
    finally:
        conn.close()


def test_queue_purge_stale_deletes_only_stale(runner):
    """Verify `queue purge --stale` deletes stale rows and keeps failed ones.

    Mutation: a purge that matches every non-pending status, or one that
        deletes nothing.
    Oracle: rows read back by id from the queue database.
    """
    _, data_dir = runner
    stale_id = _seed_row(data_dir, 'stale')
    failed_id = _seed_row(data_dir, 'failed')

    result = invoke(runner, ['scheduler', 'queue', 'purge', '--stale'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['deleted'] >= 1

    conn = open_queue_db(data_dir)
    try:
        gone = conn.execute(
            'select id from queue where id = ?', (stale_id,)).fetchone()
        assert gone is None
        survived = conn.execute(
            'select status from queue where id = ?',
            (failed_id,)).fetchone()
        assert survived is not None
        assert survived[0] == 'failed'
    finally:
        conn.close()


def test_maintenance_retries_stale_rows(runner):
    """Verify `run_maintenance` requeues a stale row.

    Mutation: dropping the stale-retry step from the maintenance phase.
    Oracle: the row's status read back from the queue database.
    """
    _, data_dir = runner
    row_id = _seed_stale_row(data_dir)

    conn = open_queue_db(data_dir)
    try:
        run_maintenance(
            queue_conn=conn,
            touched_stores=set(),
            store_contexts={},
            deadline_monotonic=time.monotonic() + 60)
        status = conn.execute(
            'select status from queue where id = ?', (row_id,)).fetchone()[0]
        assert status == 'pending'
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
