"""Unit tests for the deferred-write queue."""

import time

import pytest
from memman.queue import MAX_ATTEMPTS, STATUS_DONE, STATUS_FAILED
from memman.queue import STATUS_PENDING, claim, enqueue, get_row, list_rows
from memman.queue import mark_done, mark_failed, open_queue_db, purge_done
from memman.queue import retry_row, stats


@pytest.fixture
def queue_conn(tmp_path):
    """Fresh queue DB in a temp dir."""
    conn = open_queue_db(str(tmp_path))
    yield conn
    conn.close()


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


def test_mark_failed_below_threshold_holds_claim(queue_conn):
    """mark_failed below max_attempts keeps claimed_at set to block re-claim.
    """
    enqueue(queue_conn, 'main', 'a')
    r = claim(queue_conn, worker_pid=1)
    mark_failed(queue_conn, r.id, 'transient')
    row = get_row(queue_conn, r.id)
    assert row['status'] == STATUS_PENDING
    assert row['claimed_at'] is not None
    assert row['last_error'] == 'transient'


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
    """purge_done removes all status=done rows and returns count deleted.
    """
    enqueue(queue_conn, 'main', 'a')
    enqueue(queue_conn, 'main', 'b')
    r1 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r1.id)
    r2 = claim(queue_conn, worker_pid=1)
    mark_done(queue_conn, r2.id)
    deleted = purge_done(queue_conn)
    assert deleted == 2
    assert stats(queue_conn)['done'] == 0


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
    conn_a = open_queue_db(str(tmp_path))
    conn_b = open_queue_db(str(tmp_path))
    try:
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
    finally:
        conn_a.close()
        conn_b.close()
