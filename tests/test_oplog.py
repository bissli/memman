"""Tests for memman.store.oplog — operation logging and stats."""

from memman.store.node import insert_insight, soft_delete_insight
from memman.store.oplog import get_oplog_stats
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
