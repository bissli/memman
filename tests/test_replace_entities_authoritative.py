"""A `replace` stores its content verbatim and rejects an unknown flag.
"""

from memman.store.db import read_active
from memman.store.factory import open_backend
from tests.conftest import invoke, parse_remember


def _seed(mm_runner, data_dir):
    """Store one row and return its id and store name."""
    first = invoke(mm_runner, [
        'remember', 'the broker is kombu'])
    old = parse_remember(first, mm_runner)
    name = read_active(data_dir) or 'default'
    return old['id'], name


def test_replace_offers_no_reconcile_flag(mm_runner):
    """Verify `replace` has no `--reconcile` option, having no such step.

    Mutation: adding a `--reconcile` option that the command silently
        ignores, so the flag reads as live and is not.
    Oracle: the CLI's own exit code and usage error for an unknown
        option.
    """
    _, data_dir = mm_runner
    old_id, _name = _seed(mm_runner, data_dir)

    result = invoke(mm_runner, [
        'replace', old_id, 'the broker is rabbit now', '--reconcile'])

    assert result.exit_code != 0
    assert 'no such option' in result.output.lower()


def test_a_replace_stores_its_content_verbatim(mm_runner):
    """Verify a replace stores its text as one row, exactly as typed.

    Mutation: the drain splitting the text into several facts or
        rewording it, so the successor holds a fragment or a
        paraphrase.
    Oracle: the multi-clause text as typed, against the successor's
        stored content.
    """
    _, data_dir = mm_runner
    old_id, name = _seed(mm_runner, data_dir)
    text = ('the broker is rabbit now and the cache is redis and the'
            ' queue drains every minute')

    result = invoke(mm_runner, ['replace', old_id, text])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    assert open_backend(name, data_dir).nodes.get(
        successor['id']).content == text
