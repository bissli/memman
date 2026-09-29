"""Tests for memman.store.oplog -- operation logging, stats, and trim.
"""

from datetime import datetime, timedelta, timezone

from memman.store.db import open_db
from memman.store.model import format_timestamp
from memman.store.node import insert_insight, soft_delete_insight
from memman.store.oplog import MAX_OPLOG_ENTRIES, OPLOG_RETENTION_DAYS
from memman.store.oplog import get_oplog_stats, trim_oplog_by_age
from tests.conftest import make_insight


class TestOplogStats:
    """get_oplog_stats queries against real schema.
    """

    def test_stats_no_crash(self, tmp_db):
        """Verify get_oplog_stats returns a valid dict on an empty DB.

        Mutation: get_oplog_stats raising, or omitting operation_counts, when
            no rows exist.
        Oracle: total_active of 0 and a dict-typed operation_counts.
        """
        stats = get_oplog_stats(tmp_db)
        assert stats['total_active'] == 0
        assert isinstance(stats['operation_counts'], dict)

    def test_stats_counts_active(self, tmp_db):
        """Verify total_active excludes soft-deleted insights.

        Mutation: counting soft-deleted rows in total_active.
        Oracle: two inserts and one soft delete, expecting 1.
        """
        insert_insight(tmp_db, make_insight(id='a'))
        insert_insight(tmp_db, make_insight(id='b'))
        soft_delete_insight(tmp_db, 'b')
        stats = get_oplog_stats(tmp_db)
        assert stats['total_active'] == 1


def _insert_at(db, created_at_dt):
    """Insert a raw oplog row at an explicit created_at.
    """
    db._exec(
        'insert into oplog (operation, insight_id, detail, created_at)'
        ' values (?, ?, ?, ?)',
        ('test_op', 'some-id', 'some-detail',
         format_timestamp(created_at_dt)))


class TestOplogTrim:
    """trim_oplog_by_age: age-based retention.
    """

    def test_trim_deletes_rows_older_than_retention(self, tmp_path):
        """Verify rows older than OPLOG_RETENTION_DAYS are removed.

        Mutation: comparing created_at with the wrong cutoff, or trimming
            nothing.
        Oracle: rows placed 1 and 90 days past retention and 10 days old; 2
            deleted, 1 remains.
        """
        db = open_db(str(tmp_path))
        try:
            now = datetime.now(timezone.utc)
            _insert_at(db, now - timedelta(days=OPLOG_RETENTION_DAYS + 1))
            _insert_at(db, now - timedelta(days=OPLOG_RETENTION_DAYS + 90))
            _insert_at(db, now - timedelta(days=10))
            deleted = trim_oplog_by_age(db)
            assert deleted == 2
            (remaining,) = db._query(
                'select count(*) from oplog').fetchone()
            assert remaining == 1
        finally:
            db.close()

    def test_trim_preserves_recent_rows(self, tmp_path):
        """Verify rows inside the retention window are untouched.

        Mutation: a cutoff too short (such as 30 days), which deletes recent
            rows.
        Oracle: five rows aged 0 to 179 days against a 180-day retention.
        """
        db = open_db(str(tmp_path))
        try:
            now = datetime.now(timezone.utc)
            for days_back in (0, 1, 30, 90, 179):
                _insert_at(db, now - timedelta(days=days_back))
            deleted = trim_oplog_by_age(db)
            assert deleted == 0
            (remaining,) = db._query(
                'select count(*) from oplog').fetchone()
            assert remaining == 5
        finally:
            db.close()

    def test_trim_noop_on_empty_table(self, tmp_path):
        """Verify trim_oplog_by_age returns 0 on an empty table.

        Mutation: returning a nonzero count or raising with no rows.
        Oracle: the returned count of 0.
        """
        db = open_db(str(tmp_path))
        try:
            assert trim_oplog_by_age(db) == 0
        finally:
            db.close()


class TestOplogTrimInMaintenance:
    """oplog.log is insert-only; trim runs in maintenance_step.
    """

    def test_log_does_not_trim(self, tmp_db, tmp_backend):
        """Verify log() leaves an oplog past the cap untrimmed.

        Mutation: putting the cap delete back on the write path, which makes
            every write a delete and breaks the single-statement Postgres log.
        Oracle: a row count equal to the number of log calls.
        """
        over_cap = MAX_OPLOG_ENTRIES + 50
        for i in range(over_cap):
            tmp_backend.oplog.log(
                operation='probe', insight_id=str(i), detail='')
        row = tmp_db._query('select count(*) from oplog').fetchone()
        assert row[0] == over_cap

    def test_maintenance_step_trims(self, tmp_db, tmp_backend):
        """Verify maintenance_step caps the oplog at MAX_OPLOG_ENTRIES.

        Mutation: maintenance_step skipping the cap delete or using an
            off-by-one bound.
        Oracle: MAX_OPLOG_ENTRIES + 50 rows in, exactly MAX_OPLOG_ENTRIES left.
        """
        over_cap = MAX_OPLOG_ENTRIES + 50
        for i in range(over_cap):
            tmp_backend.oplog.log(
                operation='probe', insight_id=str(i), detail='')

        tmp_backend.oplog.maintenance_step()

        row = tmp_db._query('select count(*) from oplog').fetchone()
        assert row[0] == MAX_OPLOG_ENTRIES

    def test_maintenance_step_reclaims_more_than_one_page(
            self, tmp_db, tmp_backend):
        """Verify maintenance_step's vacuum returns many free pages at once.

        Mutation: running `pragma incremental_vacuum(200)` through
            `conn.execute`, which steps the pragma once and frees one
            page per call.
        Oracle: sqlite's own `pragma freelist_count`, before and after,
            on a store with well over 200 free pages.
        """
        for i in range(600):
            tmp_backend.oplog.log(
                operation='probe', insight_id=str(i), detail='x' * 2000)
        tmp_db._exec('delete from oplog')
        freed_before = tmp_db._query('pragma freelist_count').fetchone()[0]

        tmp_backend.oplog.maintenance_step()

        freed_after = tmp_db._query('pragma freelist_count').fetchone()[0]
        assert freed_before > 200
        assert freed_before - freed_after == 200
