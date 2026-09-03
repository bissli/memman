"""`remember` must reject an unusable `--entities` list at enqueue.

The cap itself lives in `_parse_entities`. Its only caller used to be
the DRAIN path, so an oversized list enqueued cleanly, reported
`{"action": "queued"}` at exit 0, and then died in the worker after
`MAX_ATTEMPTS` identical failures with the content never stored. These
tests pin the placement, not the values: 50 entities and 200 chars are
unmeasured and deliberately unchanged here.
"""

import json

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
