"""D1: idempotency keyed on queue_uuid, and column-list parity.

Idempotency rides on a uuid4 minted at enqueue, not the queue row's
integer id. These tests pin that decomposition end to end through the
real queue drain.
"""

import json
import sqlite3

from memman.store.db import store_dir
from tests.conftest import force_drain, invoke


def _queue_row(data_dir, queue_id):
    """Return the queue row's `queue_uuid`."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()[0]


def _stored(data_dir, store, where, params):
    """Rows of (id, queue_uuid) from the store."""
    db_path = f'{store_dir(data_dir, store)}/memman.db'
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            'select id, queue_uuid from insights'
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


def test_idempotency_keyed_on_queue_uuid(mm_runner):
    """A replay skips; a second write does not.

    Mutation: keying the drain replay check on the integer row id.
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
