"""The shipped SQLite schema has to order the `--basic` listing from an index.

`query_insights` (`store/node.py`) filters on `deleted_at is null and
superseded_by is null`, matches content alone, and sorts `created_at
desc` under a limit. Its one caller is `recall --basic` (`cli.py`), not
the default scored path. Without an index carrying that order SQLite
reads every active row into a temp b-tree before honoring the limit,
so the cost grows with the store while the indexed form stops at the
limit.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memman.store.db import open_db
from memman.store.sqlite import SqliteBackend
from tests.conftest import make_insight, set_created_at

INDEX = 'idx_insights_current_listing'


def _seed(backend):
    """Insert rows with a distinct `created_at` each.

    The sort has to be total, or the two plans are free to break ties
    differently and the id comparison stops meaning anything.
    """
    for n in range(12):
        backend.nodes.insert(make_insight(
            id=f'row-{n:02d}',
            content=f'listing fixture row {n}'))
        set_created_at(backend, f'row-{n:02d}',
                       datetime(2026, 1, 1, tzinfo=timezone.utc)
                       + timedelta(days=n))


def _plan_of(db, sql):
    """The `explain query plan` steps for one statement, joined."""
    rows = db._conn.execute('explain query plan ' + sql).fetchall()
    return ' | '.join(r[3] for r in rows)


def test_the_basic_listing_takes_its_order_from_an_index(tmp_path):
    """Verify the shipped schema sorts the `--basic` listing in the index.

    Mutation: dropping the `(deleted_at, superseded_by, created_at)`
        declaration from `_BASELINE_SCHEMA`, reordering it so
        `created_at` leads, or keeping an `importance` term in the
        verb's `order by`, which the index cannot serve. Each one puts
        a temp b-tree back in the plan, and its cost grows with the
        store because the sort reads every active row before the limit
        applies.
    Oracle: sqlite's own `explain query plan`, asserted in both
        directions - the shipped store reports no sort step, and the
        same store with the index dropped reports one. The returned
        ids are compared across both states and against the seeded
        order, newest first.

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
               if 'order by created_at desc' in ' '.join(s.split()))

    indexed_plan = _plan_of(db, sql)
    assert 'importance' not in sql
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
    assert unindexed_ids == [f'row-{n:02d}' for n in (11, 10, 9, 8, 7)]


def test_the_basic_listing_matches_content_alone(tmp_path):
    """Verify `--basic` matches a query word in the content and nowhere else.

    Mutation: keeping the `entities` or `keywords` LIKE arms in
        `query_insights`, which return a row whose content lacks the
        word.
    Oracle: the captured statement text, and the one matching row.
    """
    db = open_db(str(tmp_path / 'match'))
    backend = SqliteBackend(db)
    backend.nodes.insert(make_insight(id='hit', content='kombu broker'))
    backend.nodes.insert(make_insight(id='miss', content='celery worker'))

    captured: list[str] = []
    db._conn.set_trace_callback(captured.append)
    rows = backend.nodes.query(keyword='kombu', limit=5)
    db._conn.set_trace_callback(None)
    db.close()
    sql = ' '.join(next(s for s in captured if 'like' in s).split())

    assert [r.id for r in rows] == ['hit']
    assert 'entities' not in sql
    assert 'keywords' not in sql
