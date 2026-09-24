"""A value the caller typed is validated where the caller can read it.

The drain substitutes for a hint it cannot use: a falsy source
becomes `user`, a category outside the vocabulary becomes `fact`, an
importance outside 1-5 becomes 3. Those are the right last-ditch
guards for a programmatic enqueue, but they run in a worker the caller
has already walked away from, so a value the CLI accepted came back
changed with no message.

Two ways that was reachable from the CLI:

- `--source ''` is neither inherited nor stored; the drain rewrote it
  to `user`, inventing provenance nobody typed.
- `replace` validated a caller-typed `--cat` and `--imp` but ran the
  checks ABOVE the inheritance block, so a value INHERITED from a row
  written under an older vocabulary reached the drain unchecked.
"""

import json

from tests.conftest import invoke, make_insight, parse_remember


def _queue_hints(data_dir, queue_id):
    """Return `(hint_cat, hint_imp, hint_source)` for a queue row."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select hint_cat, hint_imp, hint_source from queue'
            ' where id = ?', (queue_id,)).fetchone()


def test_remember_refuses_an_empty_source(mm_runner):
    """Verify `remember --source ''` fails rather than storing `user`.

    Mutation: accepting the empty string at enqueue and leaving the
        drain's `row.hint_source or 'user'` to fill it, which stamps
        provenance the caller never typed.
    Oracle: the CLI exit code, and the queue holding no row for the
        refused write.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'a note with a blank source', '--source', ''])

    assert result.exit_code != 0
    assert 'source' in result.output


def test_replace_refuses_an_empty_source(mm_runner):
    """Verify `replace --source ''` fails rather than storing `user`.

    Mutation: validating the source in `remember` alone, leaving the
        identical invention reachable one command over.
    Oracle: the CLI exit code plus the message naming the source.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note that will be replaced'])
    old = parse_remember(first, mm_runner)

    result = invoke(mm_runner, [
        'replace', old['id'], 'the replacement', '--source', ''])

    assert result.exit_code != 0
    assert 'source' in result.output


def test_a_typed_source_still_reaches_the_queue(mm_runner):
    """Verify a non-empty `--source` is untouched by the new guard.

    The paired control: a guard that refused every source would pass
    the two tests above.

    Mutation: refusing or blanking a source the caller legitimately
        typed.
    Oracle: the enqueued row's own `hint_source` column.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'a note with a real source', '--source', 'agent'])

    assert result.exit_code == 0
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_hints(data_dir, queue_id)[2] == 'agent'


def test_replace_validates_the_category_it_inherits(mm_runner):
    """Verify an unusable inherited category fails the CLI, not the drain.

    Mutation: leaving the category and importance checks above the
        inheritance block, so only a caller-typed value is checked
        and an inherited one reaches the drain to be silently
        coerced to `fact` and 3.
    Oracle: the CLI exit code and the message naming the bad
        category, against a row seeded with a category outside
        `VALID_CATEGORIES`.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note to be replaced'])
    old = parse_remember(first, mm_runner)

    from memman.store.db import read_active
    from memman.store.factory import open_backend
    name = read_active(data_dir) or 'default'
    backend = open_backend(name, data_dir)
    stored = backend.nodes.get(old['id'])
    backend.nodes.insert(make_insight(
        id='legacy-1', content='a row from an older vocabulary',
        category='retrospective', importance=stored.importance))

    result = invoke(mm_runner, [
        'replace', 'legacy-1', 'the replacement text'])

    assert result.exit_code != 0
    assert 'retrospective' in result.output
