"""`remember` and `replace` must refuse text over the size cap.

The cap is counted in UTF-8 bytes and enforced at the CLI before
enqueue, so an oversized write never reaches the drain. The CLI never
truncates: the agent splits the text into one memory per thought.
"""

from tests.conftest import invoke, parse_remember, queued_contents


def test_remember_refuses_one_byte_over_the_cap(mm_runner):
    """Verify a 1,001-byte remember fails and writes no queue row.

    Mutation: the cap left at 8,000, or the check dropped, so the
        oversized text enqueues and reports success.
    Oracle: the CLI exit code, a direct read of the queue, which must
        stay empty, and the error naming the split the agent owes.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['remember', 'x' * 1001])

    assert result.exit_code != 0
    assert 'content too long (1001 bytes, max 1000)' in result.output
    assert 'one thought each' in result.output
    assert queued_contents(data_dir) == []


def test_remember_accepts_exactly_the_cap(mm_runner):
    """Verify a write of exactly 1,000 bytes enqueues whole.

    Mutation: `>=` in place of `>`, which refuses a write that sits on
        the cap, or a truncating path that stores fewer bytes.
    Oracle: the enqueued row's content compared byte for byte with the
        hand-built 1,000-byte input.
    """
    _, data_dir = mm_runner
    text = 'x' * 1000

    result = invoke(mm_runner, ['remember', text])

    assert result.exit_code == 0, result.output
    assert queued_contents(data_dir) == [text]


def test_cap_counts_utf8_bytes_not_characters(mm_runner):
    """Verify 501 two-byte characters, 1,002 bytes, are refused.

    Mutation: `len(content_str)` in place of the encoded length, which
        counts 501 characters and lets the write through.
    Oracle: the hand-computed byte count of 501 copies of U+00E9,
        whose UTF-8 form is two bytes.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['remember', '\u00e9' * 501])

    assert result.exit_code != 0
    assert '1002 bytes' in result.output
    assert queued_contents(data_dir) == []


def test_replace_refuses_text_over_the_cap(mm_runner):
    """Verify `replace` shares the cap, not just `remember`.

    Mutation: `replace` not routed through `_content_refusal_message`,
        so its text skips the cap.
    Oracle: the CLI exit code, plus the queue holding only the row the
        initial `remember` wrote.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note that will be replaced'])
    old = parse_remember(first, mm_runner)
    before = queued_contents(data_dir)

    result = invoke(mm_runner, [
        'replace', old['id'], 'x' * 1001])

    assert result.exit_code != 0
    assert 'content too long' in result.output
    assert queued_contents(data_dir) == before
