"""`--cat`/`--imp` always reach the queue as a hint, typed or default.

`remember` enqueues the parsed `cat`/`imp` value whether or not the
caller typed the flag, since no model downstream picks a category or
an importance for the stored row: the parent insight's own value is
what `_plan_fact` carries onto every planned row.
"""

import json

from tests.conftest import invoke, parse_remember


def _hints(data_dir, queue_id):
    """Return (hint_cat, hint_imp) for a queue row."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select hint_cat, hint_imp from queue where id = ?',
            (queue_id,)).fetchone()


def test_explicit_default_category_reaches_queue_as_hint(mm_runner):
    """Verify `--cat fact` pins the category instead of reading as unset.

    Mutation: deciding explicitness with `cat if cat != 'fact' else
        None`, which drops an explicitly typed default and hands the
        category back to the LLM extractor.
    Oracle: the queue row's `hint_cat` column, read directly.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, [
        'remember', 'an explicitly fact categorized note',
        '--cat', 'fact'])
    queue_id = json.loads(result.output)['queue_id']

    assert _hints(data_dir, queue_id)[0] == 'fact'


def test_explicit_default_importance_reaches_queue_as_hint(mm_runner):
    """Verify `--imp 3` pins the importance instead of reading as unset.

    Mutation: deciding explicitness with `imp if imp != 3 else None`,
        which drops an explicitly typed default and hands the
        importance back to the LLM extractor.
    Oracle: the queue row's `hint_imp` column, read directly.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, [
        'remember', 'an explicitly middling importance note',
        '--imp', '3'])
    queue_id = json.loads(result.output)['queue_id']

    assert _hints(data_dir, queue_id)[1] == 3


def test_omitted_category_defaults_to_fact_and_importance_is_always_set(
        mm_runner):
    """Verify an omitted `--cat`/`--imp` reach the queue and the row as `fact`/3.

    Mutation: the CLI deferring an omitted `--cat` to a None hint
        again, which hands the stored row's category back to a model
        no write path calls any more.
    Oracle: `('fact', 3)` on the queue row when neither flag is
        passed, and the stored row's own category equal to the hint.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, [
        'remember', 'a note with no category or importance flag'])
    queue_id = json.loads(result.output)['queue_id']

    assert _hints(data_dir, queue_id) == ('fact', 3)
    assert parse_remember(result, mm_runner)['category'] == 'fact'
