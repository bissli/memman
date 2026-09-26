"""D1: one field per job - source, queue_uuid.

`source` is provenance stored verbatim; idempotency rides on a uuid4
minted at enqueue. These tests pin the decomposition end to end
through the real queue drain.
"""

import json
import sqlite3

from memman.store.db import store_dir
from tests.conftest import force_drain, invoke, parse_remember


def _queue_row(data_dir, queue_id):
    """Return the queue row's `queue_uuid`."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()[0]


def _stored(data_dir, store, where, params):
    """Rows of (id, source, queue_uuid) from the store."""
    db_path = f'{store_dir(data_dir, store)}/memman.db'
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            'select id, source, queue_uuid from insights'
            f' where {where} and deleted_at is null', params).fetchall()


def _requeue(data_dir, queue_id):
    """Flip a drained queue row back to pending (simulated replay)."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        conn.execute(
            "update queue set status = 'pending', attempts = 0,"
            ' claimed_at = null, worker_pid = null,'
            ' processed_at = null where id = ?', (queue_id,))
        conn.commit()


def test_source_round_trips_verbatim(mm_runner):
    """The default `user` source is stored as `'user'`, not `queue:N`.

    Mutation: restoring the `!= 'user'` mapping at the CLI enqueue.
    Oracle: default write stores `'user'`; `--source agent` stores
        `'agent'`.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'a default sourced note'])
    raw1 = json.loads(r1.output)
    r2 = invoke(mm_runner, [
        'remember', 'an agent sourced note',
        '--source', 'agent'])
    raw2 = json.loads(r2.output)
    u1 = _queue_row(data_dir, raw1['queue_id'])
    u2 = _queue_row(data_dir, raw2['queue_id'])
    assert _stored(
        data_dir, raw1['store'], 'queue_uuid = ?', (u1,))[0][1] == 'user'
    assert _stored(
        data_dir, raw2['store'], 'queue_uuid = ?', (u2,))[0][1] == 'agent'


def test_source_defaults_to_user_for_programmatic_enqueue(mm_runner):
    """A bare `enqueue()` with no hint yields `source = 'user'`.

    Mutation: dropping the `or 'user'` at the drain - a programmatic
        enqueue (hint_source None) would write NULL into a column the
        recall filter compares with `=`.
    Oracle: direct enqueue, drained, stores `'user'`.
    """
    from memman.queue import enqueue, queue_db
    _, data_dir = mm_runner
    with queue_db(data_dir) as conn:
        row_id, _ = enqueue(conn, store='default',
                            content='programmatic enqueue note')
    force_drain(data_dir)
    queue_uuid = _queue_row(data_dir, row_id)
    rows = _stored(data_dir, 'default', 'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][1] == 'user'


def test_replace_inherits_source(mm_runner):
    """`replace` without `--source` keeps the old insight's source.

    Mutation: dropping the replace-side fix - its own
        `source_explicit` guard (independent of the remember-side
        mapping) discarded the inherited source as a None hint, and
        the drain then fell back to the default.
    Oracle: the replacement row carries `'agent'` from the original.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'original agent note',
        '--source', 'agent'])
    old = parse_remember(r1, mm_runner)
    r2 = invoke(mm_runner, [
        'replace', old['id'], 'updated agent note'])
    assert r2.exit_code == 0, r2.output
    raw2 = json.loads(r2.output)
    queue_uuid = _queue_row(data_dir, raw2['queue_id'])
    rows = _stored(data_dir, raw2['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][1] == 'agent'


def test_idempotency_keyed_on_queue_uuid(mm_runner):
    """A replay skips; a second write does not.

    Mutation: keying the drain replay check on `source` (the first
        `user` write would suppress every later default write) or on
        the integer row id.
    Oracle: two separate writes both store; re-queueing the first
        row and re-draining adds nothing.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, ['remember', 'first note'])
    raw1 = json.loads(r1.output)
    r2 = invoke(mm_runner, ['remember', 'second note'])
    raw2 = json.loads(r2.output)
    assert raw1['store'] == raw2['store']
    u1 = _queue_row(data_dir, raw1['queue_id'])
    u2 = _queue_row(data_dir, raw2['queue_id'])
    rows = _stored(data_dir, raw1['store'],
                   'queue_uuid in (?, ?)', (u1, u2))
    assert len(rows) == 2

    _requeue(data_dir, raw1['queue_id'])
    force_drain(data_dir)
    rows_after = _stored(data_dir, raw1['store'],
                         'queue_uuid in (?, ?)', (u1, u2))
    assert len(rows_after) == 2


def test_idempotency_check_runs_for_explicit_source(mm_runner):
    """The replay check fires even when a source hint is present.

    Mutation: restoring the old `hint_source is None` precondition -
        a replayed row with an explicit source would store twice.
    Oracle: re-queueing an `--source agent` row and re-draining
        leaves exactly one stored insight for its uuid.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'explicit source replay note',
        '--source', 'agent'])
    raw = json.loads(r1.output)
    queue_uuid = _queue_row(data_dir, raw['queue_id'])
    _requeue(data_dir, raw['queue_id'])
    force_drain(data_dir)
    rows = _stored(data_dir, raw['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1


def test_queue_uuid_survives_counter_rewind(mm_runner):
    """A rebuilt queue.db that reuses row id 1 must not skip the write.

    `backup.restore` replaces queue.db wholesale and rewinds the
    AUTOINCREMENT counter; with the integer id as the key, a fresh
    enqueue drawing a used id is silently dropped.

    Mutation: keying idempotency on the queue row id.
    Oracle: after deleting queue.db, a second write that draws the
        same row id still stores (two insights total).
    """
    import os
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'note before rewind'])
    raw1 = json.loads(r1.output)
    for suffix in ('', '-wal', '-shm'):
        try:
            os.remove(f'{data_dir}/queue.db{suffix}')
        except FileNotFoundError:
            pass
    r2 = invoke(mm_runner, [
        'remember', 'note after rewind'])
    raw2 = json.loads(r2.output)
    assert raw2['queue_id'] == raw1['queue_id'], (
        'fixture failed to rewind the AUTOINCREMENT counter')
    rows = _stored(data_dir, raw1['store'], '1 = 1', ())
    assert len(rows) == 2


def test_insight_column_lists_are_identical_across_backends():
    """`_INSIGHT_COLUMNS` and `_INSIGHT_COLS` are byte-identical.

    A transposition of any two columns between backends is invisible
    to the type checker and to every single-backend test - each
    backend would round-trip its own transposed order happily.

    Mutation: transposing any two adjacent columns in one constant.
    Oracle: pure string compare, no server needed; the Postgres
        constant is read from source text since psycopg may be
        absent.
    """
    import ast
    import inspect
    from pathlib import Path

    from memman.store import node as node_mod
    from memman.store.node import _INSIGHT_COLUMNS
    pg_path = (
        Path(inspect.getsourcefile(node_mod)).parent / 'postgres.py')
    tree = ast.parse(pg_path.read_text())
    pg_value = None
    for stmt in ast.walk(tree):
        if (isinstance(stmt, ast.Assign)
                and any(getattr(t, 'id', '') == '_INSIGHT_COLS'
                        for t in stmt.targets)):
            pg_value = ast.literal_eval(stmt.value)
    assert pg_value == _INSIGHT_COLUMNS


def test_expected_insight_columns_covers_new_fields(backend):
    """`doctor.EXPECTED_INSIGHT_COLUMNS` matches the live schema.

    Mutation: adding a column to the schema but not to doctor -
        `check_schema_columns` would then pass on a store doctor
        cannot actually vouch for.
    Oracle: every expected column exists on a freshly created store.
    """
    from memman.doctor import EXPECTED_INSIGHT_COLUMNS
    present = backend.introspect_columns('insights')
    assert EXPECTED_INSIGHT_COLUMNS <= present
    assert {'queue_uuid'} <= present
