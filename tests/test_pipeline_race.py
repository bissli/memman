"""Tests for race conditions between synchronous mutations and queued writes.

The forget+queued-replace race: a `replace` enqueues with
hint_replaced_id; a synchronous `forget` runs against the same id
before the worker drains. Without the fix the worker raises ValueError
from soft_delete_insight, the row's transaction rolls back, and
eventually the row lands as `failed` with the user's content lost.
The fix degrades to a plain add when the target is already gone.
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import cli


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Fresh CliRunner with an isolated data dir; inline drain disabled.

    The race window only opens when `replace` enqueues without an
    immediate inline drain. Override the autouse-fixture's
    `is_inline_trigger=True` to False so the queue actually buffers.
    """
    return CliRunner(), str(tmp_path)


def _invoke(r, data_dir, *args):
    """Run a memman subcommand, asserting clean exit + JSON parse."""
    result = r.invoke(cli, ['--data-dir', data_dir, *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output) if result.output.strip() else {}


@pytest.mark.no_auto_drain
def test_forget_then_replace_race(runner):
    """Replace queued + forget on target + drain = add (target gone).

    The queued replace must not crash the row's transaction. The new
    insight must commit; queue list --status failed must be empty.
    """
    r, data_dir = runner

    add_result = r.invoke(
        cli, ['--data-dir', data_dir, 'remember', 'original content here'])
    assert add_result.exit_code == 0, add_result.output
    drain_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])
    assert drain_result.exit_code == 0, drain_result.output

    recall_pre = _invoke(
        r, data_dir, 'recall', 'original content here', '--basic')
    assert recall_pre['results'], 'remember + drain failed to land insight'
    original_id = recall_pre['results'][0]['id']

    _invoke(r, data_dir, 'replace', original_id, 'replacement content')
    _invoke(r, data_dir, 'forget', original_id)

    drain_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])
    assert drain_result.exit_code == 0, drain_result.output

    failed_out = _invoke(r, data_dir, 'scheduler', 'queue', 'failed')
    assert failed_out['rows'] == [], (
        f'queue rows failed unexpectedly: {failed_out!r}')

    recall_post = _invoke(
        r, data_dir, 'recall', 'replacement content', '--basic')
    contents = [hit['content'] for hit in recall_post.get('results', [])]
    assert any('replacement content' in c for c in contents), contents
    assert all('original content here' not in c for c in contents), contents
