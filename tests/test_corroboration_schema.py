"""Schema canaries for the corroboration_count column (0.19.0).

`corroboration_count integer not null default 0` is appended after
`queue_uuid` at every touch point. These tests pin the paths a silent
omission would degrade: doctor's schema audit, the migration payload
round-trip, and the baseline index that makes a pre-0.19.0 store fail
at open.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.store.db import open_db, store_dir
from memman.store.errors import BackendError
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
        doctor -- `check_schema_columns` would then pass on a store
        doctor cannot actually vouch for.
    Oracle: the constant names the column AND a freshly created
        store carries it, on both backends.
    """
    from memman.doctor import EXPECTED_INSIGHT_COLUMNS
    present = backend.introspect_columns('insights')
    assert 'corroboration_count' in EXPECTED_INSIGHT_COLUMNS
    assert 'corroboration_count' in present


def test_migration_payload_round_trips_corroboration_count(tmp_path):
    """Gather -> apply preserves a non-zero corroboration_count.

    Mutation: dropping `corroboration_count` from gather's select or
        from apply's unconditional insert list -- a rebuild silently
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


def test_open_db_refuses_store_missing_corroboration_column(tmp_path):
    """A 0.18.x-shape store fails at open with the schema diagnostic.

    `create table if not exists` no-ops on an existing table, so the
    ONLY statement that raises for a store already carrying
    session_id/queue_uuid is the baseline index on the newest column.

    Mutation: dropping `idx_insights_corroboration` from
        `_BASELINE_SCHEMA` -- the 0.18.x store opens silently and
        fails later with a raw OperationalError deep in a read path.
    Oracle: `open_db` raises BackendError naming the schema and the
        missing column.
    """
    import pytest

    data_dir = str(tmp_path)
    db_path = _seed_store(data_dir, 'v018')
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute('drop index idx_insights_corroboration')
        conn.execute(
            'alter table insights drop column corroboration_count')
        conn.commit()
    with pytest.raises(BackendError, match='predates the current schema'):
        open_db(store_dir(data_dir, 'v018'))


def test_increment_corroboration_contract_both_backends(backend):
    """Liveness guard and adopt-only-when-null hold on BOTH backends.

    Every pipeline-level corroboration test drives SQLite; this pins
    the changed UPDATE statement itself on the parametrized backend
    pair.

    Mutation: dropping `deleted_at is null` from one backend's
        UPDATE, or reverting its coalesce argument order.
    Oracle: rowcount-derived returns plus direct column reads --
        live bump True with the key preserved, null key adopted,
        dead target False with the counter unchanged.
    """
    backend.nodes.insert(make_insight(
        id='cc-1', content='keyed row', queue_uuid='q-created'))
    assert backend.nodes.increment_corroboration(
        'cc-1', queue_uuid='q-restate') is True
    row = backend.nodes.get('cc-1')
    assert row.corroboration_count == 1
    assert row.queue_uuid == 'q-created'

    backend.nodes.insert(make_insight(id='cc-2', content='keyless row'))
    assert backend.nodes.increment_corroboration(
        'cc-2', queue_uuid='q-adopt') is True
    assert backend.nodes.get('cc-2').queue_uuid == 'q-adopt'

    backend.nodes.soft_delete('cc-1')
    assert backend.nodes.increment_corroboration(
        'cc-1', queue_uuid='q-late') is False
    dead = backend.nodes.get_include_deleted('cc-1')
    assert dead.corroboration_count == 1
