"""Unit tests for the deferred-write queue.
"""

import sqlite3
import time

import pytest
from memman import queue as queue_mod
from memman.queue import MAX_ATTEMPTS, STALE_CLAIM_SECONDS, STATUS_DONE
from memman.queue import STATUS_FAILED, claim, enqueue, get_row, list_rows
from memman.queue import mark_done, mark_failed, open_queue_db, purge_done
from memman.queue import queue_db, retry_row, stats
from memman.store.errors import BackendError


def test_enqueue_returns_row_id_and_the_uuid_it_stored(queue_conn):
    """Enqueue returns `(row_id, queue_uuid)` and both are the real ones.

    Mutation: returning the row id alone, swapping the pair's order,
        or minting a second uuid for the return value instead of
        handing back the one written into the row.
    Oracle: the `queue_uuid` column read back off each queue row,
        plus a second enqueue proving ids rise and uuids differ.
    """
    id1, uuid1 = enqueue(queue_conn, 'main', 'a')
    id2, uuid2 = enqueue(queue_conn, 'main', 'b')
    assert isinstance(id1, int)
    assert isinstance(uuid1, str)
    assert id2 > id1
    assert uuid1 != uuid2
    stored = dict(queue_conn.execute(
        'select id, queue_uuid from queue').fetchall())
    assert stored[id1] == uuid1
    assert stored[id2] == uuid2


def test_claim_fifo_order(queue_conn):
    """Rows drain in queued_at order, whatever their id order.

    Mutation: ordering `claim`'s select by `id` instead of `queued_at`,
        or reversing the sort.
    Oracle: two rows whose `queued_at` order is the reverse of their id
        order; the row with the lower id but the later `queued_at`
        claims second.
    """
    later, _ = enqueue(queue_conn, 'main', 'queued later')
    earlier, _ = enqueue(queue_conn, 'main', 'queued earlier')
    queue_conn.execute(
        'update queue set queued_at = queued_at + 60 where id = ?', (later,))
    r = claim(queue_conn, worker_pid=1)
    assert r.id == earlier


def test_claim_returns_none_when_empty(queue_conn):
    """Verify claim returns None when the queue holds no pending row.

    Mutation: claim building a QueueRow from an empty result and raising
        TypeError.
    Oracle: An empty queue and the None return.
    """
    assert claim(queue_conn, worker_pid=1) is None


def test_claim_bumps_attempts(queue_conn):
    """Verify each claim increments the row attempts counter.

    Mutation: claim leaving `attempts` at 0, so the MAX_ATTEMPTS cap never
        trips.
    Oracle: A first claim of a fresh row reads attempts == 1, hand-counted.
    """
    rid, _ = enqueue(queue_conn, 'main', 'x')
    r = claim(queue_conn, worker_pid=1)
    assert r.attempts == 1
    assert r.id == rid


def test_claim_hides_freshly_claimed_rows(queue_conn):
    """Verify a claimed row is not claimable again inside the stale window.

    Mutation: The claim predicate ignoring `claimed_at`, so two workers take
        the same row.
    Oracle: The second worker gets None for the only row.
    """
    enqueue(queue_conn, 'main', 'a')
    first = claim(queue_conn, worker_pid=1)
    assert first is not None
    second = claim(queue_conn, worker_pid=2)
    assert second is None


def test_stale_claim_reclaimable_after_timeout(queue_conn, monkeypatch):
    """A claimed row becomes reclaimable once the stale window passes.

    Mutation: `claim` skipping every claimed row whatever its age, so a
        row whose worker died mid-drain is stranded forever.
    Oracle: with `STALE_CLAIM_SECONDS` at 0, a second worker claims the
        row the first still holds.
    """
    monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
    enqueue(queue_conn, 'main', 'a')
    claim(queue_conn, worker_pid=1)
    again = claim(queue_conn, worker_pid=2)
    assert again is not None


def test_store_filter(queue_conn):
    """Verify claim honors the `stores` filter.

    Mutation: claim ignoring `stores`, so it takes the older alpha row.
    Oracle: The claimed row is beta, though alpha was queued first.
    """
    enqueue(queue_conn, 'alpha', 'a')
    enqueue(queue_conn, 'beta', 'b')
    r = claim(queue_conn, worker_pid=1, stores=['beta'])
    assert r.store == 'beta'


def test_claim_holds_a_replace_while_its_queued_target_backs_off(
        queue_conn):
    """A replace waits in the queue until the write it targets leaves it.

    Mutation: `claim` ignoring `replaced_id`, so a replace queued behind
        a target in backoff drains first and degrades to an unlinked
        add.
    Oracle: the target backing off after one failed attempt; the
        replace claims only once the target is done.
    """
    _, target_uuid = enqueue(queue_conn, 'main', 'limit is 24 hours')
    replace_id, _ = enqueue(
        queue_conn, 'main', 'limit is 30 days', replaced_id=target_uuid)
    target = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, target.id, 'transient')

    assert claim(queue_conn, worker_pid=1) is None
    mark_done(queue_conn, target.id)
    assert claim(queue_conn, worker_pid=1).id == replace_id


def test_claim_holds_a_second_replace_of_one_id_behind_the_first(
        queue_conn):
    """Two replaces of one id land in the order they were queued.

    Mutation: `claim` holding a replace only behind its queued target,
        so while the first replace of a stored id backs off the second
        drains, and the retried first then retires the newer text.
    Oracle: the first replace backing off; the second claims only
        once the first is done.
    """
    first_id, _ = enqueue(
        queue_conn, 'main', 'limit is 30 days', replaced_id='stored-id')
    second_id, _ = enqueue(
        queue_conn, 'main', 'limit is 7 days', replaced_id='stored-id')
    first = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, first.id, 'transient')

    assert claim(queue_conn, worker_pid=1) is None
    mark_done(queue_conn, first_id)
    assert claim(queue_conn, worker_pid=1).id == second_id


def test_claim_holds_a_replace_behind_an_earlier_replace_in_its_chain(
        queue_conn):
    """A replace waits behind every earlier pending replace in its store.

    Mutation: `claim` comparing only direct targets, so while a replace
        of a queued replacement backs off, a later replace of the
        original drains first, and the retried write then retires the
        newer text.
    Oracle: C1 replaces a stored id, C2 replaces C1, C3 replaces the
        stored id; with C2 backing off, C3 claims only once C2 is done.
    """
    first_id, first_uuid = enqueue(
        queue_conn, 'main', 'limit is 30 days', replaced_id='stored-id')
    middle_id, _ = enqueue(
        queue_conn, 'main', 'limit is 60 days', replaced_id=first_uuid)
    last_id, _ = enqueue(
        queue_conn, 'main', 'limit is 7 days', replaced_id='stored-id')
    mark_done(queue_conn, claim(queue_conn, worker_pid=1).id)
    middle = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, middle.id, 'transient')

    assert claim(queue_conn, worker_pid=1) is None
    mark_done(queue_conn, middle_id)
    assert claim(queue_conn, worker_pid=1).id == last_id


def test_claim_does_not_hold_a_plain_write_behind_a_backed_off_replace(
        queue_conn):
    """A plain write claims while an earlier replace backs off.

    Mutation: the replace-behind-replace hold applied to every
        candidate, so one retrying replace stalls every later write in
        its store.
    Oracle: a replace backing off; the plain write queued after it
        claims.
    """
    enqueue(queue_conn, 'main', 'limit is 30 days', replaced_id='stored-id')
    plain_id, _ = enqueue(queue_conn, 'main', 'kafka retains by age')
    replacement = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, replacement.id, 'transient')

    assert claim(queue_conn, worker_pid=1).id == plain_id


def test_claim_does_not_hold_a_write_behind_an_unrelated_backoff(
        queue_conn):
    """A plain write claims while an earlier, unrelated write backs off.

    Mutation: the hold covering every later write in the store rather
        than a replace of the same id, which stalls the store behind
        any failure.
    Oracle: two plain writes; the second claims while the first
        backs off.
    """
    enqueue(queue_conn, 'main', 'redis evicts on maxmemory')
    later_id, _ = enqueue(queue_conn, 'main', 'kafka retains by age')
    first = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, first.id, 'transient')

    assert claim(queue_conn, worker_pid=1).id == later_id


def test_mark_done_sets_status_and_clears_claim(queue_conn):
    """Verify mark_done sets status done and clears the claim.

    Mutation: mark_done leaving `claimed_at` set, or not stamping
        `processed_at`.
    Oracle: The row read back after the call.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_DONE
    assert row['claimed_at'] is None
    assert row['processed_at'] is not None


def test_mark_failed_below_threshold_backs_off(queue_conn, monkeypatch):
    """Verify a first failure back-dates the claim by the 60 s backoff.

    The row stays pending with `claimed_at` at `now - STALE_CLAIM_SECONDS +
    60`. A default-timeout reclaim is held off, and a zero-timeout reclaim
    succeeds.

    Mutation: mark_failed clearing `claimed_at` (an instant retry) or leaving
        it at the claim time (a full stale window) instead of back-dating it to
        unlock after the backoff.
    Oracle: Hand-computed `claimed_at` within a second of `now -
        STALE_CLAIM_SECONDS + 60`, a default-timeout reclaim held off, and a
        zero-timeout reclaim that succeeds.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    before = int(time.time())
    mark_failed(queue_conn, r.id, 'transient')
    row = get_row(queue_conn, r.id)
    assert row['status'] == 'pending'
    assert row['last_error'] == 'transient'
    expected = before - STALE_CLAIM_SECONDS + 60
    assert row['claimed_at'] is not None
    assert abs(row['claimed_at'] - expected) <= 1

    again = claim(queue_conn, worker_pid=2)
    assert again is None
    monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
    again = claim(queue_conn, worker_pid=2)
    assert again is not None
    assert again.id == r.id


def test_mark_failed_backoff_grows_with_attempts(queue_conn, monkeypatch):
    """Verify backoff runs 60, 120, 240, 480 s, then the row fails.

    STALE_CLAIM_SECONDS is patched to 0 only while claim runs. mark_failed then
    computes each backoff from the real value.

    Mutation: A constant backoff, or an off-by-one exponent in `60 *
        2**(attempts-1)`.
    Oracle: Hand-computed unlock times under a frozen clock, and status failed
        at MAX_ATTEMPTS.
    """
    fixed_now = 1_000_000
    monkeypatch.setattr('memman.queue.time.time', lambda: fixed_now)
    enqueue(queue_conn, 'main', 'a')

    expected_backoffs = [60, 120, 240, 480, STALE_CLAIM_SECONDS]
    for attempt_idx, backoff in enumerate(expected_backoffs, start=1):
        monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
        r = claim(queue_conn, worker_pid=1)
        monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', STALE_CLAIM_SECONDS)
        assert r is not None, f'attempt {attempt_idx}: nothing to claim'
        assert r.attempts == attempt_idx
        if attempt_idx >= MAX_ATTEMPTS:
            mark_failed(queue_conn, r.id, 'final')
            row = get_row(queue_conn, r.id)
            assert row['status'] == STATUS_FAILED
            return
        mark_failed(queue_conn, r.id, f'attempt {attempt_idx}')
        row = get_row(queue_conn, r.id)
        assert row['status'] == 'pending'
        expected = fixed_now - STALE_CLAIM_SECONDS + backoff
        assert row['claimed_at'] == expected, (
            f'attempt {attempt_idx}: expected unlock at {expected},'
            f' got {row["claimed_at"]} (backoff {backoff}s)')


def test_mark_failed_backoff_caps_at_stale_claim_seconds(
        queue_conn, monkeypatch):
    """Verify backoff never exceeds `STALE_CLAIM_SECONDS`.

    STALE_CLAIM_SECONDS is patched to 0, so the cap binds from the first
    attempt. MAX_ATTEMPTS is raised so the loop never reaches the failed state.

    Mutation: Dropping the `min()` cap, so `claimed_at` drifts from `fixed_now`
        by the growing uncapped backoff.
    Oracle: Hand-computed `claimed_at == fixed_now` on every attempt.
    """
    fixed_now = 1_000_000
    monkeypatch.setattr('memman.queue.time.time', lambda: fixed_now)
    monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
    monkeypatch.setattr(queue_mod, 'MAX_ATTEMPTS', 10)
    enqueue(queue_conn, 'main', 'a')

    for attempt_idx in range(1, 7):
        r = claim(queue_conn, worker_pid=1)
        assert r is not None
        mark_failed(queue_conn, r.id, f'attempt {attempt_idx}')
        row = get_row(queue_conn, r.id)
        assert row['claimed_at'] == fixed_now, (
            f'attempt {attempt_idx}: backoff cap at'
            f' STALE_CLAIM_SECONDS expected; got'
            f' claimed_at={row["claimed_at"]}, expected={fixed_now}')


def test_mark_failed_at_threshold_transitions_to_failed(
        queue_conn, monkeypatch):
    """Once attempts reaches MAX_ATTEMPTS, the row moves to failed.

    Mutation: an off-by-one threshold (`>` for `>=`), or a failed row
        left with its claim, so it never leaves the claim path.
    Oracle: after exactly `MAX_ATTEMPTS` failures the row is `failed`,
        carries the last error, and holds no `claimed_at`.
    """
    monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
    enqueue(queue_conn, 'main', 'a')
    for _ in range(MAX_ATTEMPTS):
        r = claim(queue_conn, worker_pid=1)
        assert r is not None
        mark_failed(queue_conn, r.id, 'kept failing')
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_FAILED
    assert row['last_error'] == 'kept failing'
    assert row['claimed_at'] is None


def test_retry_row_resurrects_failed_row(queue_conn, monkeypatch):
    """retry_row clears a failed row and returns it to pending.

    Mutation: `retry_row` flipping the status but keeping `attempts`,
        so the next failure fails the row again at once, or keeping
        `last_error`.
    Oracle: the retried row reads `pending`, zero attempts, no error.
    """
    monkeypatch.setattr(queue_mod, 'STALE_CLAIM_SECONDS', 0)
    enqueue(queue_conn, 'main', 'a')
    for _ in range(MAX_ATTEMPTS):
        r = claim(queue_conn, worker_pid=1)
        mark_failed(queue_conn, r.id, 'still broken')
    assert retry_row(queue_conn, r.id)
    row = get_row(queue_conn, r.id)
    assert row['status'] == 'pending'
    assert row['attempts'] == 0
    assert row['last_error'] is None


def test_retry_row_noop_on_non_failed(queue_conn):
    """Verify retry_row returns False for a row that is not failed.

    Mutation: retry_row resetting a live pending row, or returning True
        unconditionally.
    Oracle: The False return for a fresh pending row.
    """
    rid, _ = enqueue(queue_conn, 'main', 'a')
    assert not retry_row(queue_conn, rid)


def test_stats_reports_counts_and_oldest_age(queue_conn):
    """Verify stats counts rows by status and reports the oldest age.

    Mutation: stats swapping the pending and done counts, or omitting the
        oldest pending age.
    Oracle: Hand-counted 1 pending, 1 done, 0 failed after two enqueues and one
        completion.
    """
    enqueue(queue_conn, 'main', 'a')
    enqueue(queue_conn, 'main', 'b')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    s = stats(queue_conn)
    assert s['pending'] == 1
    assert s['done'] == 1
    assert s['failed'] == 0
    assert s['oldest_pending_age_seconds'] is not None


def test_purge_done_deletes_completed_rows(queue_conn, monkeypatch):
    """purge_done removes status=done rows older than the grace window.

    Mutation: `purge_done` deleting nothing, or one row only.
    Oracle: two done rows under a zero retention; both are deleted and
        `stats` counts no done rows.
    """
    monkeypatch.setattr(queue_mod, 'DONE_RETENTION_SECONDS', 0)
    enqueue(queue_conn, 'main', 'a')
    enqueue(queue_conn, 'main', 'b')
    r1 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r1.id)
    r2 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r2.id)
    deleted = purge_done(queue_conn)
    assert deleted == 2
    assert stats(queue_conn)['done'] == 0


def test_purge_done_keeps_rows_inside_retention(queue_conn):
    """purge_done with the default retention leaves recent rows alone.

    Mutation: `purge_done` ignoring `DONE_RETENTION_SECONDS`, or its
        cutoff comparison flipped, so a row finished this instant goes.
    Oracle: a row marked done just now survives the default retention.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    deleted = purge_done(queue_conn)
    assert deleted == 0
    assert stats(queue_conn)['done'] == 1


def test_list_rows_returns_preview(queue_conn):
    """Verify list_rows cuts the content preview to 80 characters.

    Mutation: The preview carrying the full content, so a long write floods the
        listing.
    Oracle: A 200-character write yields a preview of at most 80.
    """
    enqueue(queue_conn, 'main', 'a' * 200)
    rows = list_rows(queue_conn)
    assert len(rows) == 1
    assert len(rows[0]['content_preview']) <= 80


def test_list_rows_status_filter(queue_conn):
    """Verify list_rows(status=...) returns only rows of that status.

    Mutation: list_rows ignoring the status argument, or filtering on the wrong
        column.
    Oracle: One done and one pending row; each filter returns its own.
    """
    enqueue(queue_conn, 'main', 'x')
    enqueue(queue_conn, 'main', 'y')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    done_rows = list_rows(queue_conn, status='done')
    pend_rows = list_rows(queue_conn, status='pending')
    assert len(done_rows) == 1
    assert len(pend_rows) == 1
    assert done_rows[0]['status'] == 'done'
    assert pend_rows[0]['status'] == 'pending'


def test_concurrent_claim_across_two_connections(tmp_path):
    """Verify two connections never claim the same row.

    Mutation: claim selecting and updating in separate steps, or leaving its
        update uncommitted, so the other connection claims the row again.
    Oracle: Ten rows drained by alternating connections give ten distinct ids.
    """
    with queue_db(str(tmp_path)) as conn_a, \
            queue_db(str(tmp_path)) as conn_b:
        for _ in range(10):
            enqueue(conn_a, 'main', 'row')

        claimed_ids = []
        for _ in range(5):
            r = claim(conn_a, worker_pid=111)
            if r is not None:
                claimed_ids.append(r.id)
            r = claim(conn_b, worker_pid=222)
            if r is not None:
                claimed_ids.append(r.id)
        assert len(set(claimed_ids)) == len(claimed_ids), (
            'same row claimed by both connections')
        assert len(claimed_ids) == 10


def test_queue_db_context_manager_closes_on_exit(tmp_path):
    """Verify `with queue_db(...)` closes the connection on exit.

    Mutation: queue_db yielding without closing the connection on exit.
    Oracle: ProgrammingError from a query on the connection after the block.
    """
    with queue_db(str(tmp_path)) as conn:
        enqueue(conn, 'main', 'hi')
        assert conn.execute(
            'select count(*) from queue').fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('select 1')


def test_open_queue_db_wraps_unreadable_file_as_backend_error(tmp_path):
    """Verify a non-database `queue.db` fails as BackendError.

    Mutation: Dropping the `sqlite3.Error` translation around the pragma and
        migrate block, so the driver error escapes and the message that names
        the queue file is lost.
    Oracle: `sqlite3.connect` is lazy, so garbage bytes surface at the first
        pragma as `sqlite3.DatabaseError`, a type outside BackendError.
    """
    (tmp_path / 'queue.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError) as excinfo:
        open_queue_db(str(tmp_path))
    assert 'queue.db' in str(excinfo.value)


def test_open_queue_db_wraps_unopenable_path_as_backend_error(tmp_path):
    """A directory at the queue path fails as `BackendError`.

    Mutation: dropping the translation around the `sqlite3.connect` call
        specifically. `connect` is lazy, so the corrupt-bytes test above
        never reaches that handler and cannot catch this.
    Oracle: a directory at `<base_dir>/queue.db` makes `connect` itself
        raise `unable to open database file`.
    """
    (tmp_path / 'queue.db').mkdir()
    with pytest.raises(BackendError) as excinfo:
        open_queue_db(str(tmp_path))
    assert 'queue.db' in str(excinfo.value)


def test_open_queue_db_wraps_uncreatable_base_dir_as_backend_error(tmp_path):
    """A plain file where the data dir belongs fails as `BackendError`.

    Mutation: leaving `Path.mkdir` outside the translation, so `OSError`
        escapes untranslated. `OSError` is not a `sqlite3.Error`, so the
        handlers below the mkdir cannot catch it.
    Oracle: `mkdir(exist_ok=True)` still raises `FileExistsError` when
        the path exists and is not a directory.
    """
    occupied = tmp_path / 'mm'
    occupied.write_text('not a directory')
    with pytest.raises(BackendError) as excinfo:
        open_queue_db(str(occupied))
    assert 'mm' in str(excinfo.value)


def test_open_queue_db_closes_the_connection_when_open_fails(
        tmp_path, monkeypatch):
    """A failed queue open leaves no connection behind.

    Mutation: dropping `conn.close()` from the failure handler, so every
        failed open leaks a file handle. The corrupt-bytes test above
        passes with the connection left open.
    Oracle: the captured connection is independently probed after the
        raise. `ProgrammingError` means closed; a leaked handle raises
        `DatabaseError` on these bytes instead, so the probe
        discriminates on the type, not on the query succeeding.
    """
    captured = []
    real_connect = queue_mod.sqlite3.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr(queue_mod.sqlite3, 'connect', spy_connect)
    (tmp_path / 'queue.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError):
        open_queue_db(str(tmp_path))
    assert captured
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute('select 1')


def test_open_queue_db_closes_the_connection_on_a_non_sqlite_error(
        tmp_path, monkeypatch):
    """A failure that is not a `sqlite3.Error` still closes the handle.

    Mutation: keeping only the `sqlite3.Error` arm of the two-arm
        template `open_db` uses. `_migrate` runs inside the guarded
        block, so any error it raises that is not a driver error
        leaks the connection.
    Oracle: `_migrate` is stubbed to raise `RuntimeError`; the
        captured connection is probed after the raise -- a closed
        handle raises `ProgrammingError`, a leaked one answers.
    """
    captured = []
    real_connect = queue_mod.sqlite3.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    def boom(conn):
        raise RuntimeError('migrate exploded')

    monkeypatch.setattr(queue_mod.sqlite3, 'connect', spy_connect)
    monkeypatch.setattr(queue_mod, '_migrate', boom)
    with pytest.raises(RuntimeError):
        open_queue_db(str(tmp_path))
    assert captured
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute('select 1')
