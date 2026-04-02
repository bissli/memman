"""Tests for open_db_lightweight — background thread DB connection."""

from memman.store.db import open_db, open_db_lightweight


class TestOpenDbLightweight:
    """open_db_lightweight connects with correct PRAGMAs."""

    def test_connects_and_returns_db(self, tmp_path):
        """Returns a DB wrapper with valid connection."""
        open_db(str(tmp_path))
        db = open_db_lightweight(str(tmp_path))
        try:
            result = db._conn.execute('SELECT 1').fetchone()
            assert result == (1,)
        finally:
            db.close()

    def test_wal_mode_set(self, tmp_path):
        """WAL journal mode is active."""
        open_db(str(tmp_path))
        db = open_db_lightweight(str(tmp_path))
        try:
            mode = db._conn.execute(
                'PRAGMA journal_mode').fetchone()[0]
            assert mode == 'wal'
        finally:
            db.close()

    def test_foreign_keys_enabled(self, tmp_path):
        """Foreign keys pragma is ON."""
        open_db(str(tmp_path))
        db = open_db_lightweight(str(tmp_path))
        try:
            fk = db._conn.execute(
                'PRAGMA foreign_keys').fetchone()[0]
            assert fk == 1
        finally:
            db.close()

    def test_busy_timeout_set(self, tmp_path):
        """Busy timeout is 5000ms."""
        open_db(str(tmp_path))
        db = open_db_lightweight(str(tmp_path))
        try:
            timeout = db._conn.execute(
                'PRAGMA busy_timeout').fetchone()[0]
            assert timeout == 5000
        finally:
            db.close()

    def test_skips_migration(self, tmp_path):
        """Does not create tables when DB file does not exist."""
        db = open_db_lightweight(str(tmp_path))
        try:
            tables = db._conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table'").fetchall()
            assert len(tables) == 0
        finally:
            db.close()

    def test_reads_existing_schema(self, tmp_path):
        """Can query existing tables created by open_db."""
        full_db = open_db(str(tmp_path))
        full_db.close()

        db = open_db_lightweight(str(tmp_path))
        try:
            count = db._conn.execute(
                'SELECT COUNT(*) FROM insights').fetchone()[0]
            assert count == 0
        finally:
            db.close()
