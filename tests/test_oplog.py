"""Tests for memman.store.oplog -- operation logging, stats, and trim."""

from datetime import datetime, timedelta, timezone

from memman.store.db import open_db
from memman.store.model import format_timestamp
from memman.store.node import insert_insight, soft_delete_insight
from memman.store.oplog import MAX_OPLOG_ENTRIES, OPLOG_RETENTION_DAYS
from memman.store.oplog import get_oplog_stats, log_op, trim_oplog_by_age
from tests.conftest import make_insight


class TestOplogStats:
    """get_oplog_stats queries against real schema."""

    def test_stats_no_crash(self, tmp_db):
        """Returns valid dict on empty DB."""
        stats = get_oplog_stats(tmp_db)
        assert stats['total_active'] == 0
        assert stats['never_accessed'] == 0
        assert isinstance(stats['operation_counts'], dict)

    def test_stats_counts_active(self, tmp_db):
        """Counts exclude soft-deleted insights."""
        insert_insight(tmp_db, make_insight(id='a'))
        insert_insight(tmp_db, make_insight(id='b'))
        soft_delete_insight(tmp_db, 'b')
        stats = get_oplog_stats(tmp_db)
        assert stats['total_active'] == 1

    def test_stats_never_accessed(self, tmp_db):
        """Counts insights with access_count = 0."""
        insert_insight(tmp_db, make_insight(id='a'))
        tmp_db._exec(
            "UPDATE insights SET access_count = 1 WHERE id = 'a'")
        insert_insight(tmp_db, make_insight(id='b'))
        stats = get_oplog_stats(tmp_db)
        assert stats['never_accessed'] == 1


def _insert_at(db, created_at_dt):
    """Insert a raw oplog row at an explicit created_at for test setup."""
    db._exec(
        'INSERT INTO oplog (operation, insight_id, detail, created_at)'
        ' VALUES (?, ?, ?, ?)',
        ('test_op', 'some-id', 'some-detail',
         format_timestamp(created_at_dt)))


class TestOplogTrim:
    """trim_oplog_by_age: age-based retention (B13)."""

    def test_trim_deletes_rows_older_than_retention(self, tmp_path):
        """Rows older than retention_days are removed."""
        db = open_db(str(tmp_path))
        try:
            now = datetime.now(timezone.utc)
            _insert_at(db, now - timedelta(days=OPLOG_RETENTION_DAYS + 1))
            _insert_at(db, now - timedelta(days=OPLOG_RETENTION_DAYS + 90))
            _insert_at(db, now - timedelta(days=10))
            deleted = trim_oplog_by_age(db)
            assert deleted == 2
            (remaining,) = db._query(
                'SELECT COUNT(*) FROM oplog').fetchone()
            assert remaining == 1
        finally:
            db.close()

    def test_trim_preserves_recent_rows(self, tmp_path):
        """Rows inside retention window are untouched."""
        db = open_db(str(tmp_path))
        try:
            now = datetime.now(timezone.utc)
            for days_back in (0, 1, 30, 90, 179):
                _insert_at(db, now - timedelta(days=days_back))
            deleted = trim_oplog_by_age(db)
            assert deleted == 0
            (remaining,) = db._query(
                'SELECT COUNT(*) FROM oplog').fetchone()
            assert remaining == 5
        finally:
            db.close()

    def test_trim_noop_on_empty_table(self, tmp_path):
        """trim_oplog_by_age returns 0 when there is nothing to delete."""
        db = open_db(str(tmp_path))
        try:
            assert trim_oplog_by_age(db) == 0
        finally:
            db.close()

    def test_trim_respects_custom_retention_days(self, tmp_path):
        """Caller can override retention_days for testing or aggressive trim.
        """
        db = open_db(str(tmp_path))
        try:
            now = datetime.now(timezone.utc)
            _insert_at(db, now - timedelta(days=5))
            _insert_at(db, now - timedelta(days=15))
            deleted = trim_oplog_by_age(db, retention_days=10)
            assert deleted == 1
        finally:
            db.close()

    def test_log_op_still_honors_max_entries_cap(self, tmp_path):
        """log_op's id-based cap remains -- age-trim is additive, not a replacement.
        """
        db = open_db(str(tmp_path))
        try:
            for i in range(5):
                log_op(db, f'op-{i}', f'id-{i}', 'detail')
            (count,) = db._query(
                'SELECT COUNT(*) FROM oplog').fetchone()
            assert count == 5
        finally:
            db.close()


class TestOplogTrimInMaintenance:
    """oplog.log is INSERT-only; trim moves to maintenance_step."""

    def test_log_does_not_trim(self, tmp_db, tmp_backend):
        """Stuffing the oplog past the cap does NOT trim during log()."""
        over_cap = MAX_OPLOG_ENTRIES + 50
        for i in range(over_cap):
            tmp_backend.oplog.log(
                operation='probe', insight_id=str(i), detail='')
        row = tmp_db._query('SELECT COUNT(*) FROM oplog').fetchone()
        assert row[0] == over_cap

    def test_maintenance_step_trims(self, tmp_db, tmp_backend):
        """maintenance_step() caps the oplog at MAX_OPLOG_ENTRIES."""
        over_cap = MAX_OPLOG_ENTRIES + 50
        for i in range(over_cap):
            tmp_backend.oplog.log(
                operation='probe', insight_id=str(i), detail='')

        tmp_backend.oplog.maintenance_step()

        row = tmp_db._query('SELECT COUNT(*) FROM oplog').fetchone()
        assert row[0] == MAX_OPLOG_ENTRIES
