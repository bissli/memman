"""Unit tests for the deferred-write queue."""

import time

import pytest
from memman.queue import MAX_ATTEMPTS, STALE_CLAIM_SECONDS, STATUS_DONE
from memman.queue import STATUS_FAILED, STATUS_PENDING, claim, enqueue
from memman.queue import get_row, list_rows, mark_done, mark_failed
from memman.queue import open_queue_db, purge_done, queue_db, retry_row, stats


def test_enqueue_returns_incrementing_id(queue_conn):
    """Enqueue returns monotonically increasing row ids.
    """
    id1 = enqueue(queue_conn, 'main', 'a')
    id2 = enqueue(queue_conn, 'main', 'b')
    assert id2 > id1


def test_claim_respects_priority(queue_conn):
    """Higher priority rows are claimed before lower priority.
    """
    low = enqueue(queue_conn, 'main', 'low', priority=0)
    high = enqueue(queue_conn, 'main', 'high', priority=5)
    r = claim(queue_conn, worker_pid=1)
    assert r.id == high


def test_claim_fifo_within_same_priority(queue_conn):
    """Same-priority rows drain in queued_at order.
    """
    first = enqueue(queue_conn, 'main', 'first')
    time.sleep(1.05)
    second = enqueue(queue_conn, 'main', 'second')
    r = claim(queue_conn, worker_pid=1)
    assert r.id == first


def test_claim_returns_none_when_empty(queue_conn):
    """Claim returns None when no pending rows exist.
    """
    assert claim(queue_conn, worker_pid=1) is None


def test_claim_bumps_attempts(queue_conn):
    """Each claim increments the row's attempts counter.
    """
    rid = enqueue(queue_conn, 'main', 'x')
    r = claim(queue_conn, worker_pid=1)
    assert r.attempts == 1
    assert r.id == rid


def test_claim_hides_freshly_claimed_rows(queue_conn):
    """A claimed row is not re-claimable before stale timeout.
    """
    enqueue(queue_conn, 'main', 'a')
    first = claim(queue_conn, worker_pid=1)
    assert first is not None
    second = claim(queue_conn, worker_pid=2)
    assert second is None


def test_stale_claim_reclaimable_after_timeout(queue_conn):
    """A claimed row becomes reclaimable once the stale window passes.
    """
    enqueue(queue_conn, 'main', 'a')
    claim(queue_conn, worker_pid=1)
    again = claim(queue_conn, worker_pid=2, stale_after_seconds=0)
    assert again is not None


def test_store_filter(queue_conn):
    """Claim honors the stores filter argument.
    """
    enqueue(queue_conn, 'alpha', 'a')
    enqueue(queue_conn, 'beta', 'b')
    r = claim(queue_conn, worker_pid=1, stores=['beta'])
    assert r.store == 'beta'


def test_mark_done_sets_status_and_clears_claim(queue_conn):
    """mark_done transitions a row to status=done and frees the claim.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_DONE
    assert row['claimed_at'] is None
    assert row['processed_at'] is not None


def test_mark_failed_below_threshold_backs_off(queue_conn):
    """mark_failed reschedules claimed_at into the past so the row
    becomes reclaimable exactly `backoff_seconds` from now.

    On attempt 1 the backoff is 60 s; the row is still PENDING with a
    claim timestamp `STALE_CLAIM_SECONDS - 60` seconds in the past, so
    a stale-claim reclaim with the default timeout is held off until
    that wait elapses but a zero-timeout reclaim succeeds immediately.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    before = int(time.time())
    mark_failed(queue_conn, r.id, 'transient')
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_PENDING
    assert row['last_error'] == 'transient'
    expected = before - STALE_CLAIM_SECONDS + 60
    assert row['claimed_at'] is not None
    assert abs(row['claimed_at'] - expected) <= 1

    again = claim(queue_conn, worker_pid=2)
    assert again is None
    again = claim(queue_conn, worker_pid=2, stale_after_seconds=0)
    assert again is not None
    assert again.id == r.id


def test_mark_failed_backoff_grows_with_attempts(queue_conn, monkeypatch):
    """Attempt 1 unlocks at +60s, 2 at +120s, 3 at +240s, capped at 600s.
    """
    fixed_now = 1_000_000
    monkeypatch.setattr('memman.queue.time.time', lambda: fixed_now)
    enqueue(queue_conn, 'main', 'a')

    expected_backoffs = [60, 120, 240, 480, STALE_CLAIM_SECONDS]
    for attempt_idx, backoff in enumerate(expected_backoffs, start=1):
        r = claim(queue_conn, worker_pid=1, stale_after_seconds=0)
        assert r is not None, f'attempt {attempt_idx}: nothing to claim'
        assert r.attempts == attempt_idx
        if attempt_idx >= MAX_ATTEMPTS:
            mark_failed(queue_conn, r.id, 'final')
            row = get_row(queue_conn, r.id)
            assert row['status'] == STATUS_FAILED
            return
        mark_failed(queue_conn, r.id, f'attempt {attempt_idx}')
        row = get_row(queue_conn, r.id)
        assert row['status'] == STATUS_PENDING
        expected = fixed_now - STALE_CLAIM_SECONDS + backoff
        assert row['claimed_at'] == expected, (
            f'attempt {attempt_idx}: expected unlock at {expected},'
            f' got {row["claimed_at"]} (backoff {backoff}s)')


def test_mark_failed_backoff_caps_at_stale_claim_seconds(
        queue_conn, monkeypatch):
    """The `min(60 * 2**(attempts-1), STALE_CLAIM_SECONDS)` cap fires
    when max_attempts is high enough for the geometric series to
    exceed the cap before hitting the failed-state branch.

    At default MAX_ATTEMPTS=5 the row transitions to failed before
    the cap is reached. Pass max_attempts=10 so attempt 5
    (backoff 960s -> capped to 600s) and beyond are observable.
    """
    fixed_now = 1_000_000
    monkeypatch.setattr('memman.queue.time.time', lambda: fixed_now)
    enqueue(queue_conn, 'main', 'a')

    for attempt_idx in range(1, 7):
        r = claim(queue_conn, worker_pid=1, stale_after_seconds=0)
        assert r is not None
        mark_failed(
            queue_conn, r.id, f'attempt {attempt_idx}',
            max_attempts=10)
        if attempt_idx >= 5:
            row = get_row(queue_conn, r.id)
            expected = fixed_now
            assert row['claimed_at'] == expected, (
                f'attempt {attempt_idx}: backoff cap at'
                f' STALE_CLAIM_SECONDS expected; got'
                f' claimed_at={row["claimed_at"]}, expected={expected}')


def test_mark_failed_at_threshold_transitions_to_failed(queue_conn):
    """Once attempts reaches MAX_ATTEMPTS, the row moves to failed.
    """
    enqueue(queue_conn, 'main', 'a')
    for _ in range(MAX_ATTEMPTS):
        r = claim(queue_conn, worker_pid=1, stale_after_seconds=0)
        assert r is not None
        mark_failed(queue_conn, r.id, 'kept failing')
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_FAILED
    assert row['last_error'] == 'kept failing'
    assert row['claimed_at'] is None


def test_retry_row_resurrects_failed_row(queue_conn):
    """retry_row clears a failed row and returns it to pending.
    """
    enqueue(queue_conn, 'main', 'a')
    for _ in range(MAX_ATTEMPTS):
        r = claim(queue_conn, worker_pid=1, stale_after_seconds=0)
        mark_failed(queue_conn, r.id, 'still broken')
    assert retry_row(queue_conn, r.id)
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_PENDING
    assert row['attempts'] == 0
    assert row['last_error'] is None


def test_retry_row_noop_on_non_failed(queue_conn):
    """retry_row returns False for rows that are not failed.
    """
    rid = enqueue(queue_conn, 'main', 'a')
    assert not retry_row(queue_conn, rid)


def test_stats_reports_counts_and_oldest_age(queue_conn):
    """Stats aggregates by status and reports oldest pending age.
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


def test_purge_done_deletes_completed_rows(queue_conn):
    """purge_done removes status=done rows older than the grace window.
    """
    enqueue(queue_conn, 'main', 'a')
    enqueue(queue_conn, 'main', 'b')
    r1 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r1.id)
    r2 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r2.id)
    deleted = purge_done(queue_conn, keep_seconds=0)
    assert deleted == 2
    assert stats(queue_conn)['done'] == 0


def test_purge_done_respects_keep_seconds(queue_conn):
    """purge_done with default keep_seconds leaves recent rows alone."""
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r.id)
    deleted = purge_done(queue_conn)
    assert deleted == 0
    assert stats(queue_conn)['done'] == 1


def test_list_rows_returns_preview(queue_conn):
    """list_rows returns dicts with truncated content preview.
    """
    enqueue(queue_conn, 'main', 'a' * 200)
    rows = list_rows(queue_conn)
    assert len(rows) == 1
    assert len(rows[0]['content_preview']) <= 80


def test_list_rows_status_filter(queue_conn):
    """list_rows(status=...) filters rows by status.
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
    """Atomic claim: two connections racing never claim the same row twice.
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
    """`with queue_db(...)` closes the connection on scope exit."""
    import sqlite3
    with queue_db(str(tmp_path)) as conn:
        enqueue(conn, 'main', 'hi')
        assert conn.execute(
            'select count(*) from queue').fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('select 1')
