"""The shipped SQLite schema has to order `insights list` from an index.

`query_insights` (`store/node.py`) filters on `deleted_at is null` and
sorts `importance desc, created_at desc` under a limit. Without an
index carrying that order SQLite reads every active row into a temp
b-tree before honoring the limit, so the cost grows with the store
while the indexed form stops at the limit.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memman.store.db import open_db
from memman.store.sqlite import SqliteBackend
from tests.conftest import make_insight, set_created_at

INDEX = 'idx_insights_deleted_importance_created'


def _seed(backend):
    """Insert rows with a distinct (importance, created_at) each.

    The sort has to be total, or the two plans are free to break ties
    differently and the id comparison stops meaning anything.
    """
    for n in range(12):
        backend.nodes.insert(make_insight(
            id=f'row-{n:02d}',
            content=f'listing fixture row {n}',
            importance=(n % 5) + 1))
        set_created_at(backend, f'row-{n:02d}',
                       datetime(2026, 1, 1, tzinfo=timezone.utc)
                       + timedelta(days=n))


def _plan_of(db, sql):
    """The `explain query plan` steps for one statement, joined."""
    rows = db._conn.execute('explain query plan ' + sql).fetchall()
    return ' | '.join(r[3] for r in rows)


def test_the_insights_list_query_takes_its_order_from_an_index(tmp_path):
    """Verify the shipped schema sorts `insights list` in the index.

    Mutation: dropping the `(deleted_at, importance, created_at)`
        declaration from `_BASELINE_SCHEMA`, narrowing it to
        `(deleted_at, importance)`, or reversing it to
        `(importance, deleted_at)`. Each one puts a temp b-tree back
        in the plan - measured 1.289 -> 4.520 ms on a 1,312-row
        store, growing with the store because the sort reads every
        active row before the limit applies.
    Oracle: sqlite's own `explain query plan`, asserted in both
        directions - the shipped store reports no sort step, and the
        same store with the index dropped reports one. The returned
        ids are compared across both states, so a plan that sorts
        correctly and an index that reorders rows cannot both pass.

    Notes
    -----
    - The statement under test is captured from `nodes.query` rather
      than written out here, so a change to the verb's own `order by`
      is measured instead of being shadowed by a stale copy.
    """
    store = tmp_path / 'listing'
    db = open_db(str(store))
    backend = SqliteBackend(db)
    _seed(backend)

    captured: list[str] = []
    db._conn.set_trace_callback(captured.append)
    indexed_rows = backend.nodes.query(limit=5)
    db._conn.set_trace_callback(None)
    sql = next(s for s in captured
               if 'order by importance desc' in ' '.join(s.split()))

    indexed_plan = _plan_of(db, sql)
    assert INDEX in indexed_plan
    assert 'ORDER BY' not in indexed_plan
    db.close()

    # A fresh handle, because a connection that dropped the index
    # keeps re-planning the statement it already prepared.
    raw = sqlite3.connect(Path(store) / 'memman.db')
    raw.execute(f'drop index {INDEX}')
    unindexed_plan = ' | '.join(
        r[3] for r in raw.execute('explain query plan ' + sql).fetchall())
    unindexed_ids = [r[0] for r in raw.execute(sql).fetchall()]
    raw.close()

    assert 'ORDER BY' in unindexed_plan
    assert INDEX not in unindexed_plan
    assert [i.id for i in indexed_rows] == unindexed_ids
    assert len(indexed_rows) == 5
