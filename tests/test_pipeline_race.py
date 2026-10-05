"""Tests for race conditions between synchronous mutations and queued writes.

The forget+queued-replace race: a `replace` enqueues with
replaced_id; a synchronous `forget` runs against the same id
before the worker drains. The worker degrades the replace to a plain
add when the target is already gone. Raising from soft_delete_insight
would roll back the row's transaction and land the row as `failed`
with the user's content lost.
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import cli
from tests.conftest import _create_seeded_store


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Fresh CliRunner and an isolated data dir.
    """
    data_dir = str(tmp_path / 'memman')
    _create_seeded_store('default', data_dir)
    return CliRunner(), data_dir


def _invoke(r, data_dir, *args):
    """Run a memman subcommand, asserting clean exit and JSON output.
    """
    result = r.invoke(cli, ['--data-dir', data_dir, *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output) if result.output.strip() else {}


@pytest.mark.no_auto_drain
def test_forget_then_replace_race(runner):
    """Verify a queued replace whose target was forgotten lands as an add.

    Mutation: the worker raising on the missing target, which rolls back
        the row and leaves it in the failed queue with the new content
        lost.
    Oracle: an empty failed queue, and recall showing the replacement
        text and not the original.
    """
    r, data_dir = runner

    original_content = (
        'Postgres VACUUM ANALYZE runs nightly at 03:00 UTC via pg_cron')
    replacement_content = (
        'Postgres VACUUM ANALYZE moved to weekly Sunday 02:00 UTC via pg_cron')
    add_result = r.invoke(
        cli, ['--data-dir', data_dir, 'remember', original_content])
    assert add_result.exit_code == 0, add_result.output
    drain_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
    assert drain_result.exit_code == 0, drain_result.output

    recall_pre = r.invoke(
        cli, ['--data-dir', data_dir, 'recall', 'VACUUM ANALYZE', '--basic'])
    assert recall_pre.exit_code == 0, recall_pre.output
    pre_lines = recall_pre.output.splitlines()
    assert pre_lines, 'remember + drain failed to land insight'
    original_id = pre_lines[0].split(' ', 1)[0]

    _invoke(r, data_dir, 'replace', original_id, replacement_content)
    _invoke(r, data_dir, 'forget', original_id)

    drain_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain'])
    assert drain_result.exit_code == 0, drain_result.output

    failed_out = _invoke(r, data_dir, 'scheduler', 'queue', 'failed')
    assert failed_out['rows'] == [], (
        f'queue rows failed unexpectedly: {failed_out!r}')

    recall_post = r.invoke(
        cli, ['--data-dir', data_dir, 'recall', 'VACUUM ANALYZE', '--basic'])
    assert recall_post.exit_code == 0, recall_post.output
    contents = [line.split(' | ', 1)[1]
                for line in recall_post.output.splitlines()]
    assert any('Sunday' in c or 'weekly' in c for c in contents), contents
    assert all('03:00 UTC' not in c for c in contents), contents
