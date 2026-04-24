"""Tests for oplog age-based retention (B13).

Covers trim_oplog_by_age: rows older than retention_days are removed,
recent rows survive, and the function is safe on an empty table.
"""

from datetime import datetime, timedelta, timezone

from memman.model import format_timestamp
from memman.store.db import open_db
from memman.store.oplog import OPLOG_RETENTION_DAYS, log_op, trim_oplog_by_age


def _insert_at(db, created_at_dt):
    """Insert a raw oplog row at an explicit created_at for test setup."""
    db._exec(
        'INSERT INTO oplog (operation, insight_id, detail, created_at)'
        ' VALUES (?, ?, ?, ?)',
        ('test_op', 'some-id', 'some-detail',
         format_timestamp(created_at_dt)))


def test_trim_deletes_rows_older_than_retention(tmp_path):
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


def test_trim_preserves_recent_rows(tmp_path):
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


def test_trim_noop_on_empty_table(tmp_path):
    """trim_oplog_by_age returns 0 when there is nothing to delete."""
    db = open_db(str(tmp_path))
    try:
        assert trim_oplog_by_age(db) == 0
    finally:
        db.close()


def test_trim_respects_custom_retention_days(tmp_path):
    """Caller can override retention_days for e.g. testing or aggressive trim.
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


def test_log_op_still_honors_max_entries_cap(tmp_path):
    """log_op's id-based cap remains — age-trim is additive, not a replacement.
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
