"""Schema canaries for the corroboration_count column (0.19.0).

`corroboration_count integer not null default 0` is appended after
`queue_uuid` at every touch point. These tests pin the paths a silent
omission would degrade: doctor's schema audit, the migration payload
round-trip, and the gather probe that must substitute 0 when reading
a pre-0.19.0 store.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.store.db import open_db, store_dir
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteBackend, SqliteMigrator
from tests.conftest import make_insight


def _seed_store(data_dir, store):
    """Create a store with one insight whose corroboration_count is 5."""
    sdir = store_dir(data_dir, store)
    db = open_db(sdir)
    write_fingerprint(SqliteBackend(db), seed_default_fingerprint())
    insert_insight(db, make_insight(id='c-1', content='corrob row'))
    db._exec(
        'update insights set corroboration_count = 5 where id = ?',
        ('c-1',))
    db.close()
    return Path(sdir) / 'memman.db'


def test_expected_insight_columns_covers_corroboration_count(backend):
    """`doctor.EXPECTED_INSIGHT_COLUMNS` names the new column.

    Mutation: adding the column to the baseline schemas but not to
        doctor — `check_schema_columns` would then pass on a store
        doctor cannot actually vouch for.
    Oracle: the constant names the column AND a freshly created
        store carries it, on both backends.
    """
    from memman.doctor import EXPECTED_INSIGHT_COLUMNS
    present = backend.introspect_columns('insights')
    assert 'corroboration_count' in EXPECTED_INSIGHT_COLUMNS
    assert 'corroboration_count' in present


def test_migration_payload_round_trips_corroboration_count(tmp_path):
    """gather -> apply preserves a non-zero corroboration_count.

    Mutation: dropping `corroboration_count` from gather's select or
        from apply's unconditional insert list — a rebuild silently
        zeroes live counts.
    Oracle: a count of 5 written before gather reappears in the
        applied store, and the payload row carries it in between.
    """
    data_dir = str(tmp_path)
    _seed_store(data_dir, 'src')
    m = SqliteMigrator(data_dir)
    payload = m.gather('src')
    by_id = {i.id: i for i in payload.insights}
    assert by_id['c-1'].corroboration_count == 5
    m.apply('dst', payload)
    dst_db = Path(store_dir(data_dir, 'dst')) / 'memman.db'
    with closing(sqlite3.connect(
            f'file:{dst_db}?mode=ro', uri=True)) as conn:
        row = conn.execute(
            'select corroboration_count from insights where id = ?',
            ('c-1',)).fetchone()
    assert row[0] == 5


def test_gather_from_pre_phase2_schema_defaults_zero(tmp_path):
    """gather substitutes 0 when the source lacks the column.

    The rebuild gathers from stores on the 0.18.x schema, where
    `corroboration_count` does not exist; an unconditional select
    raises on the first store and nothing is rebuilt.

    Mutation: adding the column unconditionally to gather's select.
    Oracle: gather succeeds against a column-less store and every
        payload row defaults to 0.
    """
    data_dir = str(tmp_path)
    db_path = _seed_store(data_dir, 'old')
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute('drop index idx_insights_corroboration')
        conn.execute(
            'alter table insights drop column corroboration_count')
        conn.commit()
    payload = SqliteMigrator(data_dir).gather('old')
    assert payload.insights
    assert all(
        i.corroboration_count == 0 for i in payload.insights)


def test_postgres_gather_probe_includes_corroboration_count():
    """The Postgres gather probe's IN-list names the new column.

    `has_corrob` derives from a merged `information_schema.columns`
    query with a hardcoded IN-list; extending the ordered optional
    list without extending the IN-list leaves `has_corrob`
    permanently False, so the gather silently substitutes 0 on
    every Postgres store.

    Mutation: adding `corroboration_count` to the ordered optional
        list but not to the IN-list filter.
    Oracle: source text of the gather probe (read from source since
        psycopg may be absent) names the column inside the IN-list
        AND reads it back through the derived index map.
    """
    import inspect

    from memman.store import node as node_mod
    src = (Path(inspect.getsourcefile(node_mod)).parent
           / 'postgres.py').read_text()
    _, _, gather_body = src.partition('def gather')
    _, _, in_list = gather_body.partition('column_name in')
    assert "'corroboration_count'" in in_list[:300]
    assert "idx['corroboration_count']" in gather_body
