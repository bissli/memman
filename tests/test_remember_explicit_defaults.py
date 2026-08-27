"""An explicitly typed `--cat`/`--imp` must reach the queue as a hint.

`remember` decides whether the caller pinned a category or an
importance so `_plan_fact` knows whether to keep the caller's value or
let the extractor's per-fact guess win. Comparing the parsed value
against the default cannot make that call: a caller who types the
default value looks identical to one who typed nothing.

`replace` already resolves this with `ctx.get_parameter_source`.
"""

import json

from tests.conftest import invoke


def _hints(data_dir, queue_id):
    """Return (hint_cat, hint_imp) for a queue row."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select hint_cat, hint_imp from queue where id = ?',
            (queue_id,)).fetchone()


def test_explicit_default_category_reaches_queue_as_hint(mm_runner):
    """Verify `--cat general` pins the category instead of reading as unset.

    Mutation: deciding explicitness with `cat if cat != 'general' else
        None`, which drops an explicitly typed default and hands the
        category back to the LLM extractor.
    Oracle: the queue row's `hint_cat` column, read directly.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, [
        'remember', 'an explicitly general categorized note',
        '--cat', 'general'])
    queue_id = json.loads(result.output)['queue_id']

    assert _hints(data_dir, queue_id)[0] == 'general'


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


def test_omitted_category_and_importance_stay_unset(mm_runner):
    """Verify an omitted flag still reaches the queue as NULL.

    Mutation: reading every parameter as explicit, which would pin the
        defaults on every write and permanently silence the
        extractor's own category and importance.
    Oracle: both hint columns NULL when neither flag is passed.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, [
        'remember', 'a note with no category or importance flag'])
    queue_id = json.loads(result.output)['queue_id']

    assert _hints(data_dir, queue_id) == (None, None)
