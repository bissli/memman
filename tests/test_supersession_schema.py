"""Schema canaries for the `superseded_by` column (0.33.0).

`superseded_by text` is nullable, carries no foreign key, and is
appended LAST at every touch point. These tests pin the paths a
silent omission would degrade: doctor's schema audit, the migration
payload round-trip on both migrator halves, and the baseline index
that makes a pre-0.33.0 store fail at open.
"""

import inspect
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.store import node as node_mod
from memman.store.db import open_db, store_dir
from memman.store.errors import BackendError
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteBackend, SqliteMigrator
from tests.conftest import make_insight


def _drop_column(conn, column):
    """Drop `column` and, first, every index whose DDL names it."""
    indexes = conn.execute(
        "select name from sqlite_master where type = 'index'"
        " and tbl_name = 'insights' and sql like ?",
        (f'%{column}%',)).fetchall()
    for (name,) in indexes:
        conn.execute(f'drop index {name}')
    conn.execute(f'alter table insights drop column {column}')
    conn.commit()


def _seed_store(data_dir, store):
    """Create a store whose row `p-1` is superseded by `p-2`.

    Every column adjacent to `superseded_by` in the migrators' column
    lists carries a distinct value, so a positional shift lands the
    pointer somewhere audible instead of in a same-typed neighbor.
    """
    sdir = store_dir(data_dir, store)
    db = open_db(sdir)
    write_fingerprint(SqliteBackend(db), seed_default_fingerprint())
    insert_insight(db, make_insight(
        id='p-1', content='first statement', session_id='sess-p1',
        queue_uuid='queue-p1'))
    insert_insight(db, make_insight(
        id='p-2', content='second statement', session_id='sess-p2',
        queue_uuid='queue-p2'))
    db._exec(
        'update insights set superseded_by = ?, corroboration_count = 5'
        ' where id = ?', ('p-2', 'p-1'))
    db.close()
    return Path(sdir) / 'memman.db'


def test_expected_insight_columns_covers_superseded_by(backend):
    """`doctor.EXPECTED_INSIGHT_COLUMNS` names the new column.

    Mutation: adding the column to the baseline schemas but not to
        doctor -- `check_schema_columns` passes on a store doctor
        cannot vouch for.
    Oracle: the constant names the column AND a freshly created
        store carries it, on both backends.
    """
    from memman.doctor import EXPECTED_INSIGHT_COLUMNS
    present = backend.introspect_columns('insights')
    assert 'superseded_by' in EXPECTED_INSIGHT_COLUMNS
    assert 'superseded_by' in present


def test_superseded_by_round_trips_through_migration(tmp_path):
    """Gather -> apply -> gather preserves the pointer and its neighbors.

    Mutation: omitting `superseded_by` from gather's optional list,
        from its index map, or from apply's insert list -- a rebuild
        silently drops every pointer, and the fleet's 9,2xx
        supersessions with it.
    Oracle: the payload row carries `p-2` in between, the applied
        store returns it, and the adjacent `corroboration_count`,
        `queue_uuid` and `session_id` keep their distinct values.
    """
    data_dir = str(tmp_path)
    _seed_store(data_dir, 'src')
    m = SqliteMigrator(data_dir)
    payload = m.gather('src')
    by_id = {i.id: i for i in payload.insights}
    assert by_id['p-1'].superseded_by == 'p-2'
    assert by_id['p-2'].superseded_by is None
    m.apply('dst', payload)
    again = {i.id: i for i in m.gather('dst').insights}
    assert again['p-1'].superseded_by == 'p-2'
    assert again['p-1'].corroboration_count == 5
    assert again['p-1'].queue_uuid == 'queue-p1'
    assert again['p-1'].session_id == 'sess-p1'
    assert again['p-2'].superseded_by is None


def test_postgres_migrator_names_superseded_by_on_both_halves():
    """The Postgres gather select and apply insert both carry the column.

    Mutation: adding `superseded_by` to gather's select but not to
        apply's insert list, or the reverse -- every Postgres
        migration then drops or nulls the pointer.
    Oracle: source text of `PostgresMigrator` (read from source since
        psycopg may be absent) names the column in the gather select
        and in the apply insert list.
    """
    src = (Path(inspect.getsourcefile(node_mod)).parent
           / 'postgres.py').read_text()
    _, _, migrator = src.partition('class PostgresMigrator')
    _, _, gather_body = migrator.partition('def gather')
    _, _, select_list = gather_body.partition('select id, content')
    assert 'superseded_by' in select_list[:600]
    _, _, apply_body = migrator.partition('def apply')
    _, _, insert_list = apply_body.partition('insert into {schema}.insights')
    assert 'superseded_by' in insert_list[:800]


def test_open_db_refuses_a_store_missing_superseded_by(tmp_path):
    """A 0.32.x-shape store fails at open with the schema diagnostic.

    `create table if not exists` no-ops on an existing table, so the
    ONLY statement that raises for a store already carrying
    `corroboration_count` is the baseline index on the newest column.

    Mutation: dropping `idx_insights_current_listing` from
        `_BASELINE_SCHEMA`, or declaring it without `superseded_by`
        among its columns -- the 0.32.x store opens silently and
        fails later with a raw OperationalError deep in a read path.
    Oracle: `open_db` raises BackendError naming the schema and the
        missing column.
    """
    data_dir = str(tmp_path)
    db_path = _seed_store(data_dir, 'v032')
    with closing(sqlite3.connect(str(db_path))) as conn:
        _drop_column(conn, 'superseded_by')
    with pytest.raises(BackendError, match='superseded_by'):
        open_db(store_dir(data_dir, 'v032'))
