"""A value the caller typed is validated where the caller can read it.

`replace` validated a caller-typed `--cat` but ran the check ABOVE the
inheritance block, so a category INHERITED from a row written under an
older vocabulary reached the drain unchecked, where the drain's
last-ditch guard silently coerced it to `fact` instead of failing loud
in a worker the caller has already walked away from.
"""

from tests.conftest import invoke, make_insight, parse_remember


def test_replace_validates_the_category_it_inherits(mm_runner):
    """Verify an unusable inherited category fails the CLI, not the drain.

    Mutation: leaving the category check above the inheritance block,
        so only a caller-typed value is checked and an inherited one
        reaches the drain to be silently coerced to `fact`.
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
    backend.nodes.insert(make_insight(
        id='legacy-1', content='a row from an older vocabulary',
        category='retrospective'))

    result = invoke(mm_runner, [
        'replace', 'legacy-1', 'the replacement text'])

    assert result.exit_code != 0
    assert 'retrospective' in result.output
