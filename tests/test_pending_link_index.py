"""B10a: partial index on linked_at for the pending-link scheduler scan.

Verifies the canonical schema declares idx_insights_pending_link and
that the pending-link query planner selects it.
"""

from memman.store.db import open_db


def test_index_is_present_in_canonical_schema(tmp_path):
    """A fresh DB has idx_insights_pending_link."""
    db = open_db(str(tmp_path))
    try:
        rows = db._conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='index' AND name='idx_insights_pending_link'"
            ).fetchall()
        assert rows, 'idx_insights_pending_link missing from baseline schema'
    finally:
        db.close()


def test_pending_link_query_uses_index(tmp_path):
    """EXPLAIN QUERY PLAN picks the partial index.
    """
    db = open_db(str(tmp_path))
    try:
        plan = db._conn.execute(
            'EXPLAIN QUERY PLAN SELECT id FROM insights'
            ' WHERE linked_at IS NULL AND deleted_at IS NULL'
            ' ORDER BY created_at ASC LIMIT 10'
            ).fetchall()
    finally:
        db.close()
    assert any(
        'idx_insights_pending_link' in (row[3] if len(row) > 3 else '')
        for row in plan), plan
