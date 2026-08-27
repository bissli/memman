"""Tests for the skipped-write ledger: a queued write that stores nothing.

`_process_queue_row` can return without inserting an insight -- the
extractor rejects the content as trivial, or every extracted fact
reconciles onto an existing insight. The drain still marks the row
`done` and `purge_done` deletes it a minute later, so the sidecar
`skipped_writes` table is the only surviving record of what was lost.
"""

import json
import sqlite3
import time

import pytest
from memman.pipeline.remember import skip_reason_for_result
from memman.queue import claim, enqueue, list_skipped, mark_done, purge_done
from memman.queue import purge_store, record_skipped_write, stats


def _last_json(output):
    """Return the last JSON document in a multi-document CLI output.

    A drain that runs the real pipeline prints one document per row
    before its own summary, so `json.loads` on the whole output
    raises `Extra data`.
    """
    decoder = json.JSONDecoder()
    idx, last = 0, None
    while idx < len(output):
        while idx < len(output) and output[idx] in ' \t\r\n':
            idx += 1
        if idx >= len(output):
            break
        last, idx = decoder.raw_decode(output, idx)
    return last


def test_skipped_write_round_trips_full_content(queue_conn):
    """A recorded skip returns its whole content, not a preview.

    Mutation: storing `substr(content, 1, 80)` in record_skipped_write
        or selecting a preview in list_skipped, as `list_rows` does.
    Oracle: a 200-char content string compared for equality.
    """
    content = 'x' * 200
    record_skipped_write(
        queue_conn, queue_id=7, store='main', content=content,
        skip_reason='trivial content', session_id='sess-1')
    rows = list_skipped(queue_conn)
    assert len(rows) == 1
    assert rows[0]['content'] == content
    assert rows[0]['queue_id'] == 7
    assert rows[0]['store'] == 'main'
    assert rows[0]['skip_reason'] == 'trivial content'
    assert rows[0]['session_id'] == 'sess-1'
    assert abs(rows[0]['processed_at'] - int(time.time())) < 5


def test_skipped_write_survives_purge_done(queue_conn):
    """The ledger row outlives the queue row that purge_done deletes.

    Mutation: recording the skip on the queue table (a new status, or
        a column on `queue`) instead of the sidecar table.
    Oracle: purge_done reports one deletion and stats['done'] falls to
        zero, while the ledger still returns the content.
    """
    enqueue(queue_conn, 'main', 'a skipped note')
    row = claim(queue_conn, worker_pid=1)
    record_skipped_write(
        queue_conn, queue_id=row.id, store='main',
        content='a skipped note', skip_reason='trivial content')
    mark_done(queue_conn, row.id)
    assert purge_done(queue_conn, keep_seconds=0) == 1
    assert stats(queue_conn)['done'] == 0
    rows = list_skipped(queue_conn)
    assert [r['content'] for r in rows] == ['a skipped note']


def test_stats_counts_skipped_and_stale(queue_conn):
    """stats() reports the skipped ledger and the stale queue status.

    Mutation: dropping `stale` from the result dict (so a stale row
        counts as nothing), or sourcing `skipped` from the queue
        table's status column, where the value can never appear.
    Oracle: one row forced to status=stale and two ledger rows.
    """
    enqueue(queue_conn, 'main', 'a')
    queue_conn.execute("update queue set status = 'stale'")
    record_skipped_write(
        queue_conn, queue_id=1, store='main', content='a',
        skip_reason='trivial content')
    record_skipped_write(
        queue_conn, queue_id=2, store='main', content='b',
        skip_reason='exact duplicate')
    s = stats(queue_conn)
    assert s['stale'] == 1
    assert s['skipped'] == 2
    assert s['done'] == 0


def test_record_skipped_write_is_idempotent_per_queue_row(queue_conn):
    """Re-recording one queue row replaces its ledger entry.

    Mutation: an autoincrement surrogate key in place of queue_id as
        the primary key, which lets a re-drained row accumulate
        duplicate ledger entries.
    Oracle: two records for queue_id 5 leave exactly one row carrying
        the second reason.
    """
    record_skipped_write(
        queue_conn, queue_id=5, store='main', content='first',
        skip_reason='trivial content')
    record_skipped_write(
        queue_conn, queue_id=5, store='main', content='second',
        skip_reason='exact duplicate')
    rows = list_skipped(queue_conn)
    assert len(rows) == 1
    assert rows[0]['skip_reason'] == 'exact duplicate'


def test_purge_store_drops_its_skipped_writes(queue_conn):
    """Removing a store deletes the content of its skipped writes.

    Mutation: purge_store deleting from `queue` only, leaving the
        removed store's raw content readable in the ledger.
    Oracle: two stores recorded, one purged, one survivor named.
    """
    record_skipped_write(
        queue_conn, queue_id=1, store='gone', content='secret note',
        skip_reason='trivial content')
    record_skipped_write(
        queue_conn, queue_id=2, store='kept', content='kept note',
        skip_reason='trivial content')
    purge_store(queue_conn, 'gone')
    rows = list_skipped(queue_conn)
    assert [r['store'] for r in rows] == ['kept']


def test_skip_reason_reads_the_top_level_extractor_skip():
    """A result-level skip yields its skip_reason verbatim.

    Mutation: reading the `reason` key (the fact-level spelling) at
        the top level, where the key is `skip_reason`.
    Oracle: the literal reason string run_remember returns when
        extract_facts comes back empty.
    """
    result = {
        'id': 'abc', 'content': 'hi', 'action': 'skipped',
        'skip_reason': 'trivial content', 'llm_calls': 1,
        }
    assert skip_reason_for_result(result) == 'trivial content'


def test_skip_reason_reads_an_all_facts_skipped_result():
    """Every fact reconciling away is a skip, and names its reasons.

    Mutation: checking only the top-level `action` key, which a
        fact-bearing result never carries -- the reconcile skip then
        stays silent, the exact hole the ledger exists to close.
    Oracle: two facts, both skipped, with distinct reasons.
    """
    result = {'facts': [
        {'id': 'a', 'action': 'skipped', 'reason': 'exact duplicate'},
        {'id': 'b', 'action': 'skipped', 'reason': 'already captured'},
        ]}
    assert (skip_reason_for_result(result)
            == 'already captured; exact duplicate')


def test_skip_reason_empty_when_one_fact_was_stored():
    """A partial skip stored something, so it is not a lost write.

    Mutation: `any` in place of `all` over the fact actions, which
        files every reconcile that dedupes one fact of several as a
        lost write.
    Oracle: a two-fact result whose first fact was added.
    """
    result = {'facts': [
        {'id': 'a', 'action': 'add'},
        {'id': 'b', 'action': 'skipped', 'reason': 'exact duplicate'},
        ]}
    assert skip_reason_for_result(result) == ''


def test_skip_reason_empty_for_a_plain_add():
    """A stored write records nothing in the ledger.

    Mutation: returning a non-empty reason on the default path, which
        files every successful write as lost.
    Oracle: a single-fact add result.
    """
    assert skip_reason_for_result({'facts': [
        {'id': 'a', 'action': 'add'}]}) == ''
    assert skip_reason_for_result({'facts': []}) == ''
    assert skip_reason_for_result({'action': 'already_committed'}) == ''


def test_skip_reason_tolerates_a_non_dict_result():
    """A malformed result reads as stored, never as an exception.

    Mutation: dropping the isinstance guard, so a `None` from a
        future early return raises inside the drain's try block and
        sends a row that actually succeeded to mark_failed.
    Oracle: None and a list, both of which must come back empty.
    """
    assert skip_reason_for_result(None) == ''
    assert skip_reason_for_result([]) == ''


@pytest.mark.no_auto_drain
def test_drain_records_a_skipped_row_and_still_marks_it_done(
        mm_runner, monkeypatch):
    """The drain files the skip and completes the row.

    Mutation: the drain discarding `_process_queue_row`'s return
        value, or the top-level skip return spelling its reason under
        a key the drain never reads.
    Oracle: the REAL pipeline, with only `extract_facts` stubbed
        empty, must drive one processed row whose content and reason
        come back from `queue skipped`, with the row left `done`.
    """
    from memman.cli import cli
    from memman.llm import extract as llm_extract

    monkeypatch.setattr(llm_extract, 'extract_facts', lambda *a, **kw: [])
    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'a note the extractor drops'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output
    drain = _last_json(res.output)
    assert drain['processed'] == 1
    assert drain['skipped_writes'] == 1

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data['stats']['skipped'] == 1
    assert data['stats']['done'] == 1
    assert data['stats']['pending'] == 0
    assert len(data['rows']) == 1
    assert data['rows'][0]['content'] == 'a note the extractor drops'
    assert data['rows'][0]['skip_reason'] == 'trivial content'


@pytest.mark.no_auto_drain
def test_drain_records_nothing_for_a_stored_write(mm_runner, monkeypatch):
    """A write that lands stays out of the skipped ledger.

    Mutation: recording a ledger row unconditionally in the drain
        loop, which turns the ledger into a duplicate of the queue.
    Oracle: a stub processor returns an `add` fact result; the ledger
        must come back empty.
    """
    from memman.cli import cli

    def _stub_add(row, ctx, executor):
        return {'facts': [{'id': 'a', 'action': 'add'}], 'llm_calls': 1}

    monkeypatch.setattr('memman.cli._process_queue_row', _stub_add)
    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'a note that lands'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output
    assert _last_json(res.output)['skipped_writes'] == 0

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data['rows'] == []
    assert data['stats']['skipped'] == 0


def test_skip_reason_files_a_delete_only_row():
    """A reconcile that only deleted stored nothing of the new text.

    Mutation: treating `deleted` as a stored action, which is what
        `any(f.get('action') != 'skipped')` does -- the write that
        contradicted an existing insight vanishes with only a
        200-char oplog trace.
    Oracle: the fact shape `_apply_plan` returns from its delete
        branch, which returns before `nodes.insert`.
    """
    result = {'facts': [{
        'id': 'a', 'action': 'deleted', 'target_id': 't1',
        'reason': 'contradicted an existing insight',
        }]}
    assert (skip_reason_for_result(result)
            == 'contradicted an existing insight')


def test_skip_reason_empty_when_a_delete_accompanies_an_add():
    """A row that deleted one insight and stored another is not lost.

    Mutation: folding `deleted` into the skipped set without keeping
        the all-or-nothing rule, which files every contradicting
        write that also stored a fact.
    Oracle: a two-fact result, one add and one delete.
    """
    result = {'facts': [
        {'id': 'a', 'action': 'add'},
        {'id': 'b', 'action': 'deleted', 'target_id': 't1'},
        ]}
    assert skip_reason_for_result(result) == ''


def test_delete_branch_names_its_reason(tmp_db, tmp_backend):
    """Both delete outcomes carry a reason the ledger can show.

    Mutation: dropping the `reason` key from either delete return, so
        `skip_reason_for_result` falls back to the bare literal
        'skipped' and the ledger answers nothing.
    Oracle: `_apply_plan` driven through both `soft_delete` outcomes
        -- a live target, then an id that was never stored.
    """
    from memman.pipeline.remember import FactPlan, _apply_plan
    from memman.store.node import insert_insight
    from tests.conftest import make_insight

    def _delete_plan(new_id, target_id):
        return FactPlan(
            action='delete',
            fact_text='postgres was abandoned',
            fact_insight=make_insight(
                id=new_id, content='postgres was abandoned'),
            target_id=target_id,
            embed_vec=None,
            enrichment={},
            causal_edges=[])

    insert_insight(tmp_db, make_insight(
        id='old-1', content='we run on postgres'))
    hit = _apply_plan(tmp_backend, _delete_plan('n-1', 'old-1'),
                      embed_cache={})
    assert hit['action'] == 'deleted'
    assert hit['reason'] == 'contradicted an existing insight'
    assert skip_reason_for_result({'facts': [hit]}) == hit['reason']
    assert tmp_backend.nodes.get('n-1') is None

    miss = _apply_plan(tmp_backend, _delete_plan('n-2', 'never-stored'),
                       embed_cache={})
    assert miss['action'] == 'skipped'
    assert miss['reason'] == 'delete target already gone'
    assert skip_reason_for_result({'facts': [miss]}) == miss['reason']


@pytest.mark.no_auto_drain
def test_ledger_write_failure_does_not_fail_the_row(
        mm_runner, monkeypatch):
    """A ledger write that raises still leaves the row done.

    Mutation: leaving `record_skipped_write` bare inside the drain's
        per-row try, so a locked database sends a row whose pipeline
        already ran to mark_failed and re-runs extraction up to five
        times.
    Oracle: a raising recorder; the queue must show the row `done`
        with nothing failed or pending.
    """
    from memman import queue as queue_mod
    from memman.cli import cli

    def _stub_skip(row, ctx, executor):
        return {
            'id': 'x', 'content': row.content, 'action': 'skipped',
            'skip_reason': 'trivial content', 'llm_calls': 1,
            }

    def _boom(*a, **kw):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr('memman.cli._process_queue_row', _stub_skip)
    monkeypatch.setattr(queue_mod, 'record_skipped_write', _boom)
    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'a note the ledger drops'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data['stats']['done'] == 1
    assert data['stats']['failed'] == 0
    assert data['stats']['pending'] == 0
    assert data['rows'] == []


def test_drain_clears_a_stale_ledger_row_when_the_retry_stores(
        queue_conn):
    """A retry that stores must retract the earlier ledger entry.

    Mutation: only ever inserting in the drain, so a row that skipped
        on attempt 1 and stored on attempt 2 is reported lost
        forever, and the documented recovery re-enters a duplicate.
    Oracle: a ledger row recorded for queue id 1, then cleared.
    """
    from memman.queue import clear_skipped_write
    record_skipped_write(
        queue_conn, queue_id=1, store='main', content='a',
        skip_reason='trivial content')
    assert len(list_skipped(queue_conn)) == 1
    clear_skipped_write(queue_conn, 1)
    assert list_skipped(queue_conn) == []


def test_purge_skipped_empties_the_ledger(queue_conn):
    """The ledger has a purge verb of its own.

    Mutation: no purge path at all, which is the state before this
        change -- the table grows without bound because every
        exact-duplicate restatement files a full content copy and
        only `store remove` ever clears one.
    Oracle: two recorded rows, both deleted, count returned.
    """
    from memman.queue import purge_skipped
    record_skipped_write(
        queue_conn, queue_id=1, store='main', content='a',
        skip_reason='trivial content')
    record_skipped_write(
        queue_conn, queue_id=2, store='main', content='b',
        skip_reason='exact duplicate')
    assert purge_skipped(queue_conn) == 2
    assert list_skipped(queue_conn) == []
    assert stats(queue_conn)['skipped'] == 0


def test_list_skipped_is_newest_first_and_honors_limit(queue_conn):
    """The ledger lists newest first and stops at the limit.

    Mutation: `asc` in place of `desc` in the order-by, which buries
        the recent skip the operator is hunting under old ones, or
        ignoring `limit` and returning everything.
    Oracle: three rows stamped with distinct explicit `processed_at`
        values; the two newest, in order.
    """
    for qid, stamp in ((1, 100), (2, 300), (3, 200)):
        record_skipped_write(
            queue_conn, queue_id=qid, store='main', content=f'c{qid}',
            skip_reason='trivial content')
        queue_conn.execute(
            'update skipped_writes set processed_at = ? where queue_id = ?',
            (stamp, qid))
    rows = list_skipped(queue_conn, limit=2)
    assert [r['queue_id'] for r in rows] == [2, 3]


@pytest.mark.no_auto_drain
def test_a_reconciled_duplicate_is_filed_by_the_real_pipeline(mm_runner):
    """The real pipeline's reconcile skip reaches the ledger.

    Mutation: renaming or dropping the per-fact `reason` key in
        `_apply_plan`'s skip return, which degrades every reconcile
        skip to the bare literal 'skipped' -- the operator can no
        longer tell a fold from an extractor drop, which is the
        ledger's whole purpose.
    Oracle: the same note remembered twice. The stub embedder hashes
        content, so the second write hits cosine 1.0, fires the
        exact-match rung with no LLM call, and must land in the
        ledger naming that reason.
    """
    from memman.cli import cli

    r, data_dir = mm_runner
    note = 'Redis caches session tokens for the poller'
    for _ in range(2):
        res = r.invoke(cli, ['--data-dir', data_dir, 'remember', note])
        assert res.exit_code == 0, res.output
        res = r.invoke(cli, [
            '--data-dir', data_dir, 'scheduler', 'drain',
            '--limit', '5', '--timeout', '10'])
        assert res.exit_code == 0, res.output

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert len(data['rows']) == 1
    assert data['rows'][0]['content'] == note
    assert data['rows'][0]['skip_reason'] == 'exact duplicate'


@pytest.mark.no_auto_drain
def test_no_reconcile_write_is_never_filed(mm_runner):
    """The documented escape hatch leaves no ledger row.

    Mutation: `--no-reconcile` failing to bypass either drop path, so
        the flag every skill recommends as the fix for silent loss
        loses writes itself.
    Oracle: the same note stored twice verbatim; neither is filed and
        both reach the store.
    """
    from memman.cli import cli

    r, data_dir = mm_runner
    note = 'Postgres DSN lives in the env file, never in git'
    for _ in range(2):
        res = r.invoke(cli, [
            '--data-dir', data_dir, 'remember', note, '--no-reconcile'])
        assert res.exit_code == 0, res.output
        res = r.invoke(cli, [
            '--data-dir', data_dir, 'scheduler', 'drain',
            '--limit', '5', '--timeout', '10'])
        assert res.exit_code == 0, res.output

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)['rows'] == []


@pytest.mark.no_auto_drain
def test_a_crash_recovery_redrain_files_nothing(mm_runner):
    """Re-draining a committed row must not report it lost.

    Mutation: the `already_committed` early return handing back a
        skipped shape (or None without the guard), which tells the
        operator a write vanished while it sits in the store -- and
        the documented recovery then enters a duplicate. Also the
        drain never calling `clear_skipped_write`, which leaves a
        ledger entry from an earlier attempt standing after a later
        one stored.
    Oracle: a stored row reset to pending, given a stale ledger entry
        by hand, and drained again.
    """
    from memman.cli import cli
    from memman.queue import queue_db

    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'the poller retries on 401'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output

    with queue_db(data_dir) as conn:
        conn.execute(
            "update queue set status = 'pending', claimed_at = null,"
            ' processed_at = null')
        row_id = conn.execute('select id from queue').fetchone()[0]
        record_skipped_write(
            conn, queue_id=row_id, store='main',
            content='stale entry from an earlier attempt',
            skip_reason='trivial content')
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)['rows'] == []


def test_queue_purge_skipped_from_the_cli(mm_runner):
    """`queue purge --skipped` empties the ledger.

    Mutation: `--skipped` routed to `purge_done`/`purge_stale`, or
        rejected by the mutual-exclusion guard, leaving the operator
        no way to reclaim a ledger that keeps full content forever.
    Oracle: one recorded row, purged through the CLI, count reported.
    """
    from memman.cli import cli
    from memman.queue import queue_db

    r, data_dir = mm_runner
    with queue_db(data_dir) as conn:
        record_skipped_write(
            conn, queue_id=1, store='main', content='a note',
            skip_reason='trivial content')

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'purge',
        '--skipped'])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)['deleted'] == 1

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'skipped'])
    assert json.loads(res.output)['rows'] == []

    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'queue', 'purge',
        '--done', '--skipped'])
    assert res.exit_code != 0
