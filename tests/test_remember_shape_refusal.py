"""`remember` and `replace` must refuse text not shaped as one paragraph.

A memory holds one thought as one paragraph that opens on its subject.
The CLI refuses a line break and a short leading label before enqueue
and never rewrites the text, so the agent restates the thought in its
own words.
"""

import pytest
from tests.conftest import invoke, parse_remember, queued_contents


@pytest.mark.parametrize('separator', ['\n', '\r', '\u2028'])
def test_remember_refuses_a_line_break(mm_runner, separator):
    r"""Verify text spanning two lines fails the write and enqueues nothing.

    Mutation: the line-break check dropped, or a check for `\\n` alone
        that lets a bare carriage return or a Unicode line separator
        through.
    Oracle: the CLI exit code, the error naming the one-paragraph shape,
        and a direct read of the queue, which must stay empty.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember',
        f'The cache holds one row per store.{separator}The drain reads it.'])

    assert result.exit_code != 0
    assert 'one paragraph' in result.output
    assert queued_contents(data_dir) == []


@pytest.mark.parametrize(('text', 'label'), [
    ('Fix: reload all_insights from the database on every pass', 'Fix:'),
    ('AWS gotcha: the role needs iam:PassRole', 'AWS gotcha:'),
    ('User decision 2026-09-17: the cap stays at three',
     'User decision 2026-09-17:'),
    ('## User decision 2026-09-17: the cap stays at three',
     '## User decision 2026-09-17:'),
    ('**Note**: the cache is shared across stores', '**Note**:'),
    ('**Fix:** reload all_insights on every pass', '**Fix:**'),
    ('The rule is simple: never log a token value', None),
    ('localhost:6379 serves Redis inside the VPC', None),
    ('At 14:18 the backup job starts', None),
    ('See https://example.com/docs for the API', None),
    ('`fatal: not a git repository` means the hook ran outside the repo',
     None),
])
def test_remember_refuses_only_a_short_leading_label(mm_runner, text, label):
    """Verify a label of up to three words refuses and a sentence passes.

    Mutation: the label check dropped; a word window wider than three,
        which refuses a sentence such as "The rule is simple:"; a
        leading markdown marker no longer skipped, which pushes
        `## User decision 2026-09-17:` past three words; a bold
        marker after the colon read as text, which passes `**Fix:**`;
        a colon matched without the space after it, which refuses a
        host port, a clock time or a URL; or a quote read as a word,
        which refuses a quoted error message.
    Oracle: hand-labeled rows; each refusing row names the label the
        message must quote and leaves the queue empty, each passing row
        enqueues byte for byte.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['remember', text])

    if label is None:
        assert result.exit_code == 0, result.output
        assert queued_contents(data_dir) == [text]
    else:
        assert result.exit_code != 0
        assert f'({label!r})' in result.output
        assert queued_contents(data_dir) == []


def test_replace_refuses_a_label_and_a_line_break(mm_runner):
    """Verify `replace` shares both shape refusals, not just `remember`.

    Mutation: the shape checks wired into `remember` alone.
    Oracle: the CLI exit codes, plus the queue holding only the row the
        initial `remember` wrote.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note that will be replaced'])
    old = parse_remember(first, mm_runner)
    before = queued_contents(data_dir)

    labeled = invoke(mm_runner, [
        'replace', old['id'], 'Fix: the poller retries three times'])
    two_lines = invoke(mm_runner, [
        'replace', old['id'], 'The poller retries.\nIt backs off.'])

    assert labeled.exit_code != 0
    assert "('Fix:')" in labeled.output
    assert two_lines.exit_code != 0
    assert 'one paragraph' in two_lines.output
    assert queued_contents(data_dir) == before
