"""The skip ledger's retraction contract.

`clear_skipped_write` exists so that re-entering a discarded write
retracts its ledger row. Its own docstring names the failure it
prevents: without retraction the ledger reports a write as lost
forever and the documented recovery creates a duplicate.
"""
from memman import queue


def _replay_row(conn, queue_id):
    """Return `(store, content)` of a queue row, as the drain reads it."""
    row = conn.execute(
        'select store, content from queue where id = ?',
        (queue_id,)).fetchone()
    return row[0], row[1]


def test_replay_retracts_the_original_ledger_row(queue_conn):
    """Re-entering a discarded write clears the ledger row it replaces.

    Mutation: keying the retraction on the replay's own fresh
        queue_id, so the row filed against the original id survives
        every replay and the ledger reports the write lost forever.
    Oracle: a hand-built ledger holding exactly one row; after the
        replay's drain retracts it, skipped_writes must be empty
        rather than still holding the original.
    """
    queue.record_skipped_write(
        queue_conn, 41, 'main', 'a novel claim', 'already captured')
    replay_id, _ = queue.enqueue(queue_conn, 'main', 'a novel claim')
    assert replay_id != 41

    queue.clear_skipped_write(queue_conn, *_replay_row(queue_conn, replay_id))

    surviving = queue_conn.execute(
        'select queue_id from skipped_writes').fetchall()
    assert surviving == []


def test_retraction_is_selective_not_whole_table(queue_conn):
    """Replaying one discarded write leaves the other ledger rows alone.

    Mutation: offering only `purge_skipped`, which empties the whole
        table, so retracting the already-captured writes would also
        erase every trivial-content row and the audit trail with them.
    Oracle: two ledger rows, one write replayed; exactly the replayed
        row must be gone and the untouched row must survive, which a
        whole-table purge cannot satisfy.
    """
    queue.record_skipped_write(
        queue_conn, 41, 'main', 'a novel claim', 'already captured')
    queue.record_skipped_write(
        queue_conn, 42, 'main', 'a trivial note', 'trivial content')
    replay_id, _ = queue.enqueue(queue_conn, 'main', 'a novel claim')

    queue.clear_skipped_write(queue_conn, *_replay_row(queue_conn, replay_id))

    surviving = [
        row[0] for row in queue_conn.execute(
            'select queue_id from skipped_writes order by queue_id')]
    assert surviving == [42]


def test_retraction_clears_every_row_sharing_the_key(queue_conn):
    """One content discarded twice retracts both its ledger rows.

    Mutation: a LIMIT 1 or a delete of the newest row alone, which
        leaves an older row asserting that content the store now
        holds is still lost.
    Oracle: two ledger rows carrying the same (store, content) and a
        third carrying different content; after one replay only the
        third survives.
    """
    queue.record_skipped_write(
        queue_conn, 41, 'main', 'a novel claim', 'already captured')
    queue.record_skipped_write(
        queue_conn, 42, 'main', 'a novel claim', 'already captured')
    queue.record_skipped_write(
        queue_conn, 43, 'main', 'another claim', 'already captured')
    replay_id, _ = queue.enqueue(queue_conn, 'main', 'a novel claim')

    retracted = queue.clear_skipped_write(
        queue_conn, *_replay_row(queue_conn, replay_id))

    surviving = [
        row[0] for row in queue_conn.execute(
            'select queue_id from skipped_writes order by queue_id')]
    assert retracted == 2
    assert surviving == [43]


def test_retraction_does_not_cross_stores(queue_conn):
    """The same text discarded in another store keeps its ledger row.

    Mutation: keying on content alone, which retracts a sibling
        store's row on a replay that stored nothing there.
    Oracle: identical content filed under two stores; replaying in
        one leaves the other's row standing.
    """
    queue.record_skipped_write(
        queue_conn, 41, 'main', 'a novel claim', 'already captured')
    queue.record_skipped_write(
        queue_conn, 42, 'domain', 'a novel claim', 'already captured')
    replay_id, _ = queue.enqueue(queue_conn, 'main', 'a novel claim')

    queue.clear_skipped_write(queue_conn, *_replay_row(queue_conn, replay_id))

    surviving = [
        row[0] for row in queue_conn.execute(
            'select queue_id from skipped_writes order by queue_id')]
    assert surviving == [42]
