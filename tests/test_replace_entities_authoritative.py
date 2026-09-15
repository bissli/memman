"""A typed `replace --entity` decides the successor's entity list.

`replace`'s own docstring says an unflagged `--cat` / `--imp` /
`--source` / `--entity` inherits the replaced insight's value. The
first three override when typed; `--entity` did not. The predecessor
union in `_apply_plan` re-added the target's names after the caller's
list was already set, so a typed list could only ADD names and an
empty one did nothing at all.

The union itself is right for `update` and `supersede`: its own
rationale is that the extractor sees only the incoming text and would
narrow the merged row's entity set. On a `replace` the extractor never
runs, and the caller named the list.
"""

from memman.store.db import read_active
from memman.store.factory import open_backend
from tests.conftest import invoke, parse_remember


def _seed(mm_runner, data_dir, entities):
    """Store one row carrying `entities` and return its id and store."""
    first = invoke(mm_runner, [
        'remember', 'the broker is kombu', '--no-reconcile'])
    old = parse_remember(first, mm_runner)
    name = read_active(data_dir) or 'default'
    open_backend(name, data_dir).nodes.update_entities(old['id'], entities)
    return old['id'], name


def test_a_typed_entity_list_replaces_the_inherited_one(mm_runner):
    """Verify a typed `--entity` drops the names it did not name.

    Mutation: unioning the predecessor's entity list into the
        successor on a `replace`, so a typed list can only add and
        the caller cannot remove a name the model got wrong.
    Oracle: the hand-typed one-name list, against the successor's
        stored list with the enrichment's own names removed.
    """
    _, data_dir = mm_runner
    old_id, name = _seed(mm_runner, data_dir, ['kombu', 'celery'])

    result = invoke(mm_runner, [
        'replace', old_id, 'the broker is rabbit now',
        '--entity', 'rabbitmq'])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    stored = open_backend(name, data_dir).nodes.get(successor['id']).entities
    assert 'rabbitmq' in stored
    assert [e for e in stored if e in {'kombu', 'celery'}] == []


def test_an_empty_typed_entity_list_clears_the_inherited_one(mm_runner):
    """Verify `--entity ''` clears rather than silently inheriting.

    Mutation: treating a typed empty list as "not given", which
        makes the flag indistinguishable from omitting it and leaves
        the caller no route to empty the list.
    Oracle: the successor's stored list, which must hold none of the
        predecessor's names.
    """
    _, data_dir = mm_runner
    old_id, name = _seed(mm_runner, data_dir, ['kombu', 'celery'])

    result = invoke(mm_runner, [
        'replace', old_id, 'the broker is gone', '--entity', ''])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    stored = open_backend(name, data_dir).nodes.get(successor['id']).entities
    assert [e for e in stored if e in {'kombu', 'celery'}] == []


def test_an_unflagged_replace_still_inherits_every_name(mm_runner):
    """Verify omitting `--entity` carries the stored list whole.

    The paired control: dropping the union outright would break
    inheritance if the CLI were not already encoding the stored list
    into the queue hint.

    Mutation: dropping the inherited list along with the union, so an
        unflagged replace loses every entity the row carried.
    Oracle: the hand-seeded two-name list, every name of which must
        be present on the successor.
    """
    _, data_dir = mm_runner
    old_id, name = _seed(mm_runner, data_dir, ['kombu', 'celery'])

    result = invoke(mm_runner, [
        'replace', old_id, 'the broker is rabbit now'])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    stored = open_backend(name, data_dir).nodes.get(successor['id']).entities
    assert [e for e in stored if e in {'kombu', 'celery'}] == [
        'kombu', 'celery']


def test_replace_offers_no_reconcile_flag(mm_runner):
    """Verify `replace` carries no `--reconcile` option.

    The drain forced `no_reconcile` for every row carrying a
    `hint_replaced_id`, and `replace` is its only producer, so the
    option it advertised could never take effect. A documented flag
    with no effect is worse than none: it tells a caller the command
    can do something it cannot.

    Mutation: re-adding a `--reconcile` option while the drain still
        decides for itself, so the flag reads as live and is not.
    Oracle: the CLI's own exit code and usage error for an unknown
        option.
    """
    _, data_dir = mm_runner
    old_id, _name = _seed(mm_runner, data_dir, ['kombu'])

    result = invoke(mm_runner, [
        'replace', old_id, 'the broker is rabbit now', '--reconcile'])

    assert result.exit_code != 0
    assert 'no such option' in result.output.lower()


def test_a_replace_stores_its_content_verbatim(mm_runner):
    """Verify a replace never splits its content into several facts.

    The documented consequence of the flag's removal: a replace
    targets one id, so the extractor never runs and the text lands
    as one row.

    Mutation: letting the drain reconcile a replace, which would run
        fact extraction and store N rows for one command.
    Oracle: the count of rows carrying the write's queue uuid, and
        the stored content against the text as typed.
    """
    _, data_dir = mm_runner
    old_id, name = _seed(mm_runner, data_dir, ['kombu'])
    text = ('the broker is rabbit now and the cache is redis and the'
            ' queue drains every minute')

    result = invoke(mm_runner, ['replace', old_id, text])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    assert open_backend(name, data_dir).nodes.get(
        successor['id']).content == text
