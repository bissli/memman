"""auto_prune ordering should prefer older inactive records over fresh ones.

Existing `test_prefers_stale_over_fresh` differentiates by access_count;
this test differentiates by created_at on records with identical
access_count and importance, so the recency tiebreak (`COALESCE
last_accessed_at, created_at) ASC`) is exercised independently.
"""

from datetime import datetime, timedelta, timezone

from memman.model import format_timestamp
from memman.store.node import auto_prune, get_insight_by_id, insert_insight
from tests.conftest import make_insight


def _backdate(db, insight_id, days):
    """Push an insight's created_at into the past."""
    when = format_timestamp(
        datetime.now(timezone.utc) - timedelta(days=days))
    db._exec(
        'UPDATE insights SET created_at = ?, updated_at = ? WHERE id = ?',
        (when, when, insight_id))


def test_old_unaccessed_pruned_before_fresh_unaccessed(tmp_db):
    """Two unaccessed prunable records: the older one goes first."""
    insert_insight(tmp_db, make_insight(
        id='old-row', content='old content', importance=2))
    insert_insight(tmp_db, make_insight(
        id='fresh-row', content='fresh content', importance=2))

    _backdate(tmp_db, 'old-row', days=30)

    pruned = auto_prune(tmp_db, 1)
    assert pruned == 1

    assert get_insight_by_id(tmp_db, 'fresh-row') is not None
    assert get_insight_by_id(tmp_db, 'old-row') is None
