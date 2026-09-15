"""A failed queue row can be retired without raw SQL.

`queue retry` re-pends a failed row, which replays it, so a row whose
own stored content is what breaks it could not be cleared through any
command. `purge` reached `done`, `stale` and the skip ledger and never
`failed`.

A failed row stored nothing, so filing it into `skipped_writes` on the
way out satisfies that table's all-or-nothing rule and hands the
operator the documented recovery: read the content back and re-enter
it.
"""

import json
import uuid

from memman.queue import list_skipped, open_queue_db, purge_failed

from tests.conftest import invoke


def _seed_failed(conn, content: str, *, last_error: str | None,
                 processed_at: int | None = 1_700_000_000) -> int:
    """Insert one `failed` queue row. Returns its id."""
    cur = conn.execute(
        'insert into queue'
        ' (store, content, hint_cat, hint_imp, hint_source,'
        '  hint_entities, status, queue_uuid, queued_at, attempts,'
        '  last_error, processed_at, session_id)'
        " values (?, ?, 'fact', 3, 'test', null, 'failed', ?,"
        "         strftime('%s','now'), 3, ?, ?, 'sess-9')",
        ('default', content, str(uuid.uuid4()), last_error, processed_at))
    conn.commit()
    return cur.lastrowid


def test_purge_failed_files_the_content_before_deleting(queue_conn):
    """Verify a purged failed row leaves its content in the ledger.

    Mutation: deleting the queue row without the ledger insert, which
        loses the only copy of the write; or ordering the delete first,
        so the insert reads a row that is gone.
    Oracle: the hand-written content string, which must be readable
        back through `list_skipped` after the queue row is gone.
    """
    row_id = _seed_failed(
        queue_conn, 'a note that broke the drain',
        last_error='entity too long')

    assert purge_failed(queue_conn) == 1

    assert queue_conn.execute(
        'select id from queue where id = ?', (row_id,)).fetchone() is None
    filed = list_skipped(queue_conn)
    assert [r['content'] for r in filed] == ['a note that broke the drain']
    assert filed[0]['queue_id'] == row_id
    assert filed[0]['session_id'] == 'sess-9'
    assert 'entity too long' in filed[0]['skip_reason']


def test_purge_failed_keeps_the_rows_own_timestamp(queue_conn):
    """Verify the ledger entry is dated to the failure, not the purge.

    Mutation: letting `record_skipped_write` stamp its own clock, which
        dates every purged failure to the purge and sorts it above
        genuinely newer skips under the ledger's `processed_at desc`.
    Oracle: the hand-set `processed_at` on the seeded row, and the
        newest-first order the newer pipeline skip must hold.
    """
    _seed_failed(queue_conn, 'the older failure',
                 last_error='boom', processed_at=1_700_000_000)

    from memman.queue import record_skipped_write
    record_skipped_write(
        queue_conn, 999, 'default', 'a newer pipeline skip',
        'already captured', processed_at=1_800_000_000)

    purge_failed(queue_conn)

    filed = list_skipped(queue_conn)
    assert [r['processed_at'] for r in filed] == [1_800_000_000, 1_700_000_000]
    assert [r['content'] for r in filed] == [
        'a newer pipeline skip', 'the older failure']


def test_purge_failed_survives_a_null_last_error(queue_conn):
    """Verify a failed row with no error text still files.

    Mutation: passing `last_error` straight through, which violates
        `skipped_writes.skip_reason not null` and aborts the purge for
        a row restored from a backup without one.
    Oracle: the row is gone and its content readable, with a reason
        naming the failure.
    """
    _seed_failed(queue_conn, 'no error recorded', last_error=None)

    assert purge_failed(queue_conn) == 1

    filed = list_skipped(queue_conn)
    assert [r['content'] for r in filed] == ['no error recorded']
    assert filed[0]['skip_reason']


def test_purge_failed_deletes_only_the_rows_it_filed(queue_conn):
    """Verify a row that fails mid-purge is not deleted unfiled.

    The ledger entry is the only copy of a failed row's content, so a
    delete wider than the set just filed loses a write outright.

    Mutation: deleting with an unqualified `where status = 'failed'`
        rather than by the ids just read, so a row another connection
        fails between the read and the delete goes without an entry.
    Oracle: the second row's content, which must be readable back from
        either the queue or the ledger once the purge returns.
    """
    first = _seed_failed(queue_conn, 'the old failure', last_error='boom')
    later = _seed_failed(queue_conn, 'the late failure', last_error='boom')
    queue_conn.execute(
        "update queue set status = 'pending' where id = ?", (later,))
    queue_conn.commit()

    import memman.queue as queue_mod
    real_record = queue_mod.record_skipped_write
    seen = []

    def _record_then_fail_the_other(conn, queue_id, *args, **kwargs):
        real_record(conn, queue_id, *args, **kwargs)
        seen.append(queue_id)
        if len(seen) == 1:
            conn.execute(
                "update queue set status = 'failed', last_error = 'late'"
                ' where id = ?', (later,))

    queue_mod.record_skipped_write = _record_then_fail_the_other
    try:
        deleted = purge_failed(queue_conn)
    finally:
        queue_mod.record_skipped_write = real_record

    assert deleted == 1
    assert seen == [first]
    surviving = queue_conn.execute(
        'select content from queue where id = ?', (later,)).fetchone()
    filed = [r['content'] for r in list_skipped(queue_conn)]
    assert surviving is not None or 'the late failure' in filed


def test_purge_failed_names_the_write_handle_in_the_reason(queue_conn):
    """Verify the ledger entry carries the row's queue uuid.

    A row can reach `failed` after its insights are already committed:
    the drain's error handling extends past the store commit. The
    ledger entry then reports a write as lost that is in fact stored,
    and the documented recovery -- re-enter the content -- duplicates
    it. The uuid is the handle that settles which happened, through
    `memman insights by-queue`.

    Mutation: filing only the error text, leaving an operator no way
        to tell a lost write from a stored one before re-entering it.
    Oracle: the row's own `queue_uuid`, read off the queue before the
        purge, against the reason string afterward.
    """
    row_id = _seed_failed(queue_conn, 'a write that may be stored',
                          last_error='boom')
    queue_uuid = queue_conn.execute(
        'select queue_uuid from queue where id = ?', (row_id,)).fetchone()[0]

    purge_failed(queue_conn)

    reason = list_skipped(queue_conn)[0]['skip_reason']
    assert queue_uuid in reason


def test_purge_failed_leaves_other_statuses_alone(queue_conn):
    """Verify only `failed` rows are purged.

    Mutation: dropping the status predicate, which would delete every
        queue row including the pending write the worker is about to
        claim.
    Oracle: the three hand-seeded statuses, of which exactly one is
        `failed`.
    """
    _seed_failed(queue_conn, 'the broken one', last_error='boom')
    for status in ('pending', 'done', 'stale'):
        queue_conn.execute(
            'insert into queue'
            ' (store, content, hint_cat, hint_imp, hint_source,'
            '  hint_entities, status, queue_uuid, queued_at)'
            " values ('default', ?, 'fact', 3, 'test', null, ?, ?,"
            "         strftime('%s','now'))",
            (f'{status} row', status, str(uuid.uuid4())))
    queue_conn.commit()

    assert purge_failed(queue_conn) == 1

    survived = queue_conn.execute(
        'select status from queue order by status').fetchall()
    assert [r[0] for r in survived] == ['done', 'pending', 'stale']


def test_queue_purge_failed_from_the_cli(mm_runner):
    """`queue purge --failed` clears a failed row and reports the count.

    Mutation: `--failed` left out of the flag set, or routed to
        `purge_done`/`purge_stale`, so the command reports a deletion
        that never touched the failed row.
    Oracle: the seeded row's id, absent from the queue afterward, with
        the reported count matching.
    """
    _, data_dir = mm_runner
    conn = open_queue_db(data_dir)
    try:
        row_id = _seed_failed(conn, 'cli failed row', last_error='boom')
    finally:
        conn.close()

    result = invoke(mm_runner, ['scheduler', 'queue', 'purge', '--failed'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['deleted'] == 1

    conn = open_queue_db(data_dir)
    try:
        assert conn.execute(
            'select id from queue where id = ?',
            (row_id,)).fetchone() is None
    finally:
        conn.close()


def test_queue_purge_names_failed_among_its_flags(mm_runner):
    """`queue purge` with no flag names every target it accepts.

    Mutation: adding the flag without adding it to the guard's message
        or its mutual-exclusion set, so `--failed` pairs silently with
        another flag and only one branch runs.
    Oracle: the four flag names, in the no-flag error and in the
        conflicting-pair error.
    """
    bare = invoke(mm_runner, ['scheduler', 'queue', 'purge'])
    assert bare.exit_code != 0
    for flag in ('--done', '--stale', '--skipped', '--failed'):
        assert flag in bare.output

    pair = invoke(
        mm_runner, ['scheduler', 'queue', 'purge', '--failed', '--done'])
    assert pair.exit_code != 0
    assert 'exactly one of' in pair.output
