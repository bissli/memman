"""End-to-end tests: memman remember --defer → enrich --pending → recall.

Verifies the deferred-write contract:
- `remember --defer` queues without writing to the store DB.
- `enrich --pending` drains the queue and lands insights in the store.
- The same queue row cannot be processed twice (idempotency).
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman.cli import cli


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


def test_defer_enqueues_and_drain_commits(runner):
    """Remember --defer + enrich --pending lands one insight in the store.
    """
    r = invoke(runner, ['remember', '--defer',
                        'Paris is the capital of France'])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data['action'] == 'queued'

    r = invoke(runner, ['status'])
    before = json.loads(r.output)
    assert before.get('total_insights', 0) == 0

    r = invoke(runner, ['enrich', '--pending',
                        '--limit', '5', '--timeout', '30'])
    assert r.exit_code == 0, r.output
    drain = json.loads(r.output)
    assert drain['processed'] == 1
    assert drain['failed'] == 0

    r = invoke(runner, ['status'])
    after = json.loads(r.output)
    assert after['total_insights'] >= 1


def test_idempotency_requeue_does_not_duplicate(runner):
    """Re-processing a queue row after the pipeline already committed
    does not create duplicate insights — the queue:<id> source tag
    short-circuits re-extraction.
    """
    import sqlite3

    r = invoke(runner, ['remember', '--defer',
                        'The Eiffel Tower is in Paris'])
    assert r.exit_code == 0, r.output
    data_dir = runner[1]

    r = invoke(runner, ['enrich', '--pending',
                        '--limit', '5', '--timeout', '30'])
    assert r.exit_code == 0, r.output
    first = json.loads(r.output)
    assert first['processed'] == 1

    r = invoke(runner, ['status'])
    count_after_first = json.loads(r.output)['total_insights']
    assert count_after_first >= 1

    queue_db = Path(data_dir) / 'queue.db'
    with sqlite3.connect(str(queue_db)) as conn:
        conn.execute(
            "UPDATE queue SET status='pending', claimed_at=NULL,"
            ' worker_pid=NULL, attempts=0')
        conn.commit()

    r = invoke(runner, ['enrich', '--pending',
                        '--limit', '5', '--timeout', '30'])
    assert r.exit_code == 0, r.output

    r = invoke(runner, ['status'])
    count_after_second = json.loads(r.output)['total_insights']
    assert count_after_second == count_after_first, (
        f'expected no new insights on re-process,'
        f' got {count_after_first} -> {count_after_second}')


def test_multiple_defers_single_drain(runner):
    """Three defer calls followed by one drain process all three.
    """
    for text in ['first fact', 'second fact', 'third fact']:
        r = invoke(runner, ['remember', '--defer', text])
        assert r.exit_code == 0, r.output

    r = invoke(runner, ['enrich', '--pending',
                        '--limit', '10', '--timeout', '60'])
    assert r.exit_code == 0, r.output
    drain = json.loads(r.output)
    assert drain['processed'] == 3
    assert drain['failed'] == 0
