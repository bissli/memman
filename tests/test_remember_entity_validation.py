"""`remember` must reject an unusable `--entities` list at enqueue.

The caps live in `_validate_caller_entities`. They once sat in
`_parse_entities`, whose only caller was the DRAIN path, so an
oversized list enqueued cleanly, reported `{"action": "queued"}` at
exit 0, and then died in the worker after `MAX_ATTEMPTS` identical
failures with the content never stored. The caps then reached a second
population they do not govern: the entity list `replace` inherits from
the row it replaces, which the enrichment path writes uncapped. These
tests pin the placement, not the values: 50 entities and 200 chars are
unmeasured and deliberately unchanged here.
"""

import json

from memman.store.db import read_active
from memman.store.factory import open_backend
from tests.conftest import invoke, parse_remember


def _queue_rows(data_dir):
    """Return every (id, status, hint_entities) row in the queue."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select id, status, hint_entities from queue').fetchall()


def test_oversized_entity_list_is_rejected_before_enqueue(mm_runner):
    """Verify 51 entities fails the CLI and writes no queue row.

    Mutation: checking the cap only where the worker reads
        `row.hint_entities`, so the CLI reports success and the write
        dies later -- the defect this test was written against.
    Oracle: the CLI exit code plus a direct count of queue rows,
        which must stay at zero.
    """
    _, data_dir = mm_runner
    entities = ','.join(f'e{i}' for i in range(51))

    result = invoke(mm_runner, [
        'remember', 'a note carrying one entity too many',
        '--entities', entities])

    assert result.exit_code != 0
    assert 'too many entities' in result.output
    assert _queue_rows(data_dir) == []


def test_overlong_single_entity_is_rejected_before_enqueue(mm_runner):
    """Verify one 201-char entity fails the CLI and writes no queue row.

    Mutation: validating only the list LENGTH at enqueue and leaving
        the per-entity length check behind in the drain, which loses
        the write the same way for a different reason.
    Oracle: the CLI exit code plus a direct count of queue rows.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'a note carrying one overlong entity',
        '--entities', 'x' * 201])

    assert result.exit_code != 0
    assert 'entity too long' in result.output
    assert _queue_rows(data_dir) == []


def test_entity_list_at_the_cap_still_enqueues(mm_runner):
    """Verify exactly 50 entities is accepted and reaches the queue.

    Mutation: an off-by-one at the boundary (`>= 50` for the count, or
        `> 200` read as `>= 200` for the length), which would start
        rejecting writes that are legal today.
    Oracle: the enqueued row's own `hint_entities`, split and counted
        back to 50, against a hand-built list straddling the cap.
    """
    _, data_dir = mm_runner
    entities = ','.join(f'e{i}' for i in range(50))

    result = invoke(mm_runner, [
        'remember', 'a note carrying exactly the cap',
        '--entities', entities])

    assert result.exit_code == 0
    queue_id = json.loads(result.output)['queue_id']
    rows = {r[0]: r for r in _queue_rows(data_dir)}
    assert len(rows[queue_id][2].split(',')) == 50


def test_replace_rejects_an_oversized_entity_list_too(mm_runner):
    """Verify `replace` shares the enqueue-time check, not just `remember`.

    `replace` enqueues `hint_entities` through the same column and the
    same drain-side re-parse, so a fix applied only to `remember`
    leaves the identical silent loss reachable one command over.

    Mutation: validating in `remember` alone and leaving `replace`
        enqueuing an unchecked list.
    Oracle: the CLI exit code, plus the queue holding only the row
        from the initial `remember` and none from the `replace`.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note that will be replaced', '--no-reconcile'])
    old = parse_remember(first, mm_runner)
    before = len(_queue_rows(data_dir))

    result = invoke(mm_runner, [
        'replace', old['id'], 'the replacement text',
        '--entities', ','.join(f'e{i}' for i in range(51))])

    assert result.exit_code != 0
    assert 'too many entities' in result.output
    assert len(_queue_rows(data_dir)) == before


def test_replace_inherits_an_oversized_stored_entity_list(mm_runner):
    """Verify `replace` without `--entities` carries a 66-entity list through.

    The enrichment path writes `entities` without passing through
    `_parse_entities` and merges as a monotonic union, so a stored
    list grows past the caller cap on its own. Applying the
    caller-input cap to that inherited list makes the row
    permanently unreplaceable, whatever the replacement text says.

    Mutation: validating the INHERITED entity list against the
        caller-input cap -- the defect this test was written
        against -- or truncating it to 50 instead of passing it
        whole.
    Oracle: a hand-built 66-entity list, counted back off the
        enqueued row and off the stored successor.
    """
    _, data_dir = mm_runner
    grown = [f'ent{i}' for i in range(66)]

    first = invoke(mm_runner, [
        'remember', 'a note whose entity list outgrows the cap',
        '--no-reconcile'])
    old = parse_remember(first, mm_runner)

    name = read_active(data_dir) or 'default'
    backend = open_backend(name, data_dir)
    backend.nodes.update_entities(old['id'], grown)
    assert len(backend.nodes.get(old['id']).entities) == 66

    result = invoke(mm_runner, [
        'replace', old['id'], 'x', '--no-reconcile'])

    assert result.exit_code == 0, result.output
    queue_id = json.loads(result.output)['queue_id']
    rows = {r[0]: r for r in _queue_rows(data_dir)}
    assert rows[queue_id][2].split(',') == grown
    successor = parse_remember(result, mm_runner)
    assert open_backend(name, data_dir).nodes.get(
        successor['id']).entities == grown
