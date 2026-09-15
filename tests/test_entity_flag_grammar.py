"""A caller can name any entity the store can hold.

`--entities` was comma-separated with no escape and no repeatable
form, so a distinguished name -- which always contains a comma --
arrived as three entities and no quoting expressed it. The queue
column is JSON, so a name a comma-free path stored survived the write
whole; the only surface that could not express one was the flag, which
made retyping a shredded name reproduce the shredding.

`--entity` takes one name per occurrence and splits nothing.
"""

import json

from memman.store.db import read_active
from memman.store.factory import open_backend
from tests.conftest import invoke, parse_remember

DN = 'OU=Servers,DC=example,DC=com'


def _queue_entities(data_dir, queue_id):
    """Return the decoded `hint_entities` of one queue row."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        raw = conn.execute(
            'select hint_entities from queue where id = ?',
            (queue_id,)).fetchone()[0]
    return None if raw is None else json.loads(raw)


def test_one_entity_containing_commas_stays_one_entity(mm_runner):
    """Verify a distinguished name enqueues as a single name.

    Mutation: splitting the option value on commas -- the defect
        itself -- which turns one name into three.
    Oracle: the hand-written DN, decoded off the queue row, against a
        one-element list. Asserting only that the DN is present would
        pass on the split too, since `OU=Servers` is a prefix of it;
        the teeth are on the LENGTH.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'a note about a directory container',
        '--entity', DN])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_entities(data_dir, queue_id) == [DN]


def test_the_option_repeats_to_name_several_entities(mm_runner):
    """Verify each occurrence contributes one name, in input order.

    Mutation: declaring the option single-valued, so a second
        occurrence overwrites the first and the earlier names are
        silently dropped.
    Oracle: the three hand-written names in the order typed, decoded
        off the queue row.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'a note naming three things',
        '--entity', 'kombu',
        '--entity', DN,
        '--entity', 'redis'])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_entities(data_dir, queue_id) == ['kombu', DN, 'redis']


def test_a_comma_in_a_replace_entity_survives_too(mm_runner):
    """Verify `replace` expresses the same name, not just `remember`.

    Repairing a row whose entity list is wrong means retyping the name
    on `replace`, so a fix landing on `remember` alone leaves the
    shredding reachable on the one command written to undo it.

    Mutation: adding the repeatable option to `remember` and leaving
        `replace` splitting its value.
    Oracle: the hand-written DN against a one-element list, decoded
        off the `replace` row's queue entry.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note about a container', '--no-reconcile'])
    old = parse_remember(first, mm_runner)

    result = invoke(mm_runner, [
        'replace', old['id'], 'the corrected note', '--entity', DN])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_entities(data_dir, queue_id) == [DN]


def test_replace_without_the_option_still_inherits(mm_runner):
    """Verify an omitted option inherits the target's list.

    A repeatable option's omitted value is an empty tuple, not the
    empty string the single-valued form used, so the omission test has
    to read the parameter source rather than the value.

    Mutation: testing the VALUE for emptiness, which reads an omitted
        option as a deliberate clear and drops the inherited list.
    Oracle: the target row's own stored list, read straight off the
        backend. The stored row carries the enrichment's names beside
        the typed one, so comparing against the typed name alone
        would fail on correct code.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'the broker is kombu', '--no-reconcile',
        '--entity', 'kombu'])
    old = parse_remember(first, mm_runner)

    name = read_active(data_dir) or 'default'
    stored = open_backend(name, data_dir).nodes.get(old['id']).entities
    assert 'kombu' in stored

    result = invoke(mm_runner, [
        'replace', old['id'], 'the broker is redis now'])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_entities(data_dir, queue_id) == stored


def test_an_empty_typed_entity_clears_the_inherited_list(mm_runner):
    """Verify a typed empty value still clears, rather than inheriting.

    Mutation: dropping empty occurrences before the parameter-source
        test, which makes a deliberate clear indistinguishable from an
        omission and re-inherits the list the caller meant to empty.
    Oracle: the target's stored name, which must be ABSENT from the
        replace row's column -- the column's no-entities form is NULL.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'the broker is kombu', '--no-reconcile',
        '--entity', 'kombu'])
    old = parse_remember(first, mm_runner)

    result = invoke(mm_runner, [
        'replace', old['id'], 'the broker is gone', '--entity', ''])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    assert _queue_entities(data_dir, queue_id) is None
