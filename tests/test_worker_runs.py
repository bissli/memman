"""Tests for B6 worker_runs observability table.

Covers the start/finish helpers, last_worker_run lookup, drain-path
instrumentation via the enrich CLI, and the scheduler-status
integration.
"""

import json
import time

import pytest
from memman.cli import cli
from memman.queue import finish_worker_run, last_worker_run, open_queue_db
from memman.queue import start_worker_run


def test_start_finish_round_trip(queue_conn):
    """start/finish writes and updates exactly one row with counts.
    """
    run_id = start_worker_run(queue_conn, worker_pid=4242)
    assert isinstance(run_id, int)
    assert run_id > 0
    finish_worker_run(
        queue_conn, run_id,
        rows_claimed=5, rows_done=3, rows_failed=2, error=None)
    row = queue_conn.execute(
        'SELECT worker_pid, rows_claimed, rows_done, rows_failed,'
        ' started_at, finished_at, duration_ms, error'
        ' FROM worker_runs WHERE id = ?',
        (run_id,)).fetchone()
    assert row[0] == 4242
    assert row[1:4] == (5, 3, 2)
    assert row[4] is not None
    assert row[5] is not None
    assert row[6] is not None
    assert row[6] >= 0
    assert row[7] is None


def test_finish_records_error(queue_conn):
    """finish_worker_run persists the error string when provided.
    """
    run_id = start_worker_run(queue_conn, worker_pid=1)
    finish_worker_run(
        queue_conn, run_id,
        rows_claimed=0, rows_done=0, rows_failed=0,
        error='RuntimeError: boom')
    row = queue_conn.execute(
        'SELECT error FROM worker_runs WHERE id = ?',
        (run_id,)).fetchone()
    assert row[0] == 'RuntimeError: boom'


def test_last_worker_run_returns_most_recent(queue_conn):
    """last_worker_run returns the highest started_at.
    """
    assert last_worker_run(queue_conn) is None
    first = start_worker_run(queue_conn, worker_pid=1)
    finish_worker_run(queue_conn, first, 1, 1, 0)
    time.sleep(1.1)
    second = start_worker_run(queue_conn, worker_pid=2)
    finish_worker_run(queue_conn, second, 2, 2, 0)
    out = last_worker_run(queue_conn)
    assert out is not None
    assert out['id'] == second
    assert out['worker_pid'] == 2
    assert out['rows_claimed'] == 2


@pytest.fixture
def runner(mm_runner):
    return mm_runner


def _invoke(runner_tuple, args):
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def test_drain_records_worker_run(runner):
    """`memman scheduler drain --pending` on an empty queue still writes a row.
    """
    result = _invoke(runner, ['scheduler', 'drain', '--pending',
                              '--limit', '5', '--timeout', '5'])
    assert result.exit_code == 0, result.output

    conn = open_queue_db(runner[1])
    try:
        row = last_worker_run(conn)
    finally:
        conn.close()
    assert row is not None
    assert row['rows_claimed'] == 0
    assert row['rows_done'] == 0
    assert row['rows_failed'] == 0
    assert row['error'] is None
    assert row['finished_at'] is not None


def test_scheduler_status_includes_last_run(runner):
    """`memman scheduler status` surfaces the most recent worker_runs row.
    """
    _invoke(runner, ['scheduler', 'drain', '--pending',
                     '--limit', '1', '--timeout', '5'])
    result = _invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert 'last_run' in payload
    assert payload['last_run'] is not None
    assert payload['last_run']['rows_claimed'] == 0


def test_scheduler_status_last_run_null_before_any_drain(runner):
    """When no drain has fired yet, last_run is null.
    """
    result = _invoke(runner, ['scheduler', 'status'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['last_run'] is None
