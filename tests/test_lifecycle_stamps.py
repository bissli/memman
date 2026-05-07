"""Tests for lifecycle stamp and pending-link store functions."""

from datetime import datetime, timezone

from memman.store.model import format_timestamp
from memman.store.node import count_pending_links, get_active_insight_ids
from memman.store.node import get_insight_by_id, get_pending_link_ids
from memman.store.node import insert_insight, reset_for_rebuild
from memman.store.node import stamp_enriched, stamp_linked
from tests.conftest import make_insight


class TestStampEnriched:
    """stamp_enriched sets enriched_at timestamp.

    Only the enriched_at variant stays as a column-correctness check;
    stamp_linked is covered transitively by `TestGetPendingLinkIds`
    (a stamp_linked column-name typo would surface there as a row
    failing to disappear from the pending list).
    """

    def test_sets_timestamp(self, tmp_db):
        """Insight's enriched_at is set to the given timestamp."""
        insert_insight(tmp_db, make_insight(id='se-1', content='a'))
        ts = format_timestamp(datetime(2025, 6, 1, tzinfo=timezone.utc))
        stamp_enriched(tmp_db, 'se-1', ts)
        row = tmp_db._query(
            'SELECT enriched_at FROM insights WHERE id = ?',
            ('se-1',)).fetchone()
        assert row[0] == ts


class TestGetPendingLinkIds:
    """get_pending_link_ids returns unlinked, non-deleted insights."""

    def test_returns_unlinked(self, tmp_db):
        """Insights with NULL linked_at are returned."""
        insert_insight(tmp_db, make_insight(id='pl-1', content='a'))
        insert_insight(tmp_db, make_insight(id='pl-2', content='b'))
        ids = get_pending_link_ids(tmp_db, 10)
        assert set(ids) == {'pl-1', 'pl-2'}

    def test_excludes_linked(self, tmp_db):
        """Insights with non-NULL linked_at are excluded."""
        insert_insight(tmp_db, make_insight(id='pl-3', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_linked(tmp_db, 'pl-3', ts)
        ids = get_pending_link_ids(tmp_db, 10)
        assert 'pl-3' not in ids

    def test_excludes_deleted(self, tmp_db):
        """Soft-deleted insights are excluded."""
        from memman.store.node import soft_delete_insight
        insert_insight(tmp_db, make_insight(id='pl-4', content='a'))
        soft_delete_insight(tmp_db, 'pl-4')
        ids = get_pending_link_ids(tmp_db, 10)
        assert 'pl-4' not in ids

    def test_respects_limit(self, tmp_db):
        """Limit caps the number of IDs returned."""
        for i in range(5):
            insert_insight(tmp_db, make_insight(
                id=f'pl-l{i}', content=f'c{i}'))
        ids = get_pending_link_ids(tmp_db, 3)
        assert len(ids) == 3


class TestGetActiveInsightIds:
    """get_active_insight_ids returns non-deleted IDs in creation order."""

    def test_returns_active_ordered(self, tmp_db):
        """Active insight IDs returned in created_at ASC order."""
        insert_insight(tmp_db, make_insight(id='ai-1', content='first'))
        insert_insight(tmp_db, make_insight(id='ai-2', content='second'))
        ids = get_active_insight_ids(tmp_db)
        assert ids == ['ai-1', 'ai-2']

    def test_excludes_deleted(self, tmp_db):
        """Soft-deleted insights are excluded."""
        from memman.store.node import soft_delete_insight
        insert_insight(tmp_db, make_insight(id='ai-3', content='a'))
        soft_delete_insight(tmp_db, 'ai-3')
        ids = get_active_insight_ids(tmp_db)
        assert 'ai-3' not in ids


class TestCountPendingLinks:
    """count_pending_links counts unlinked, non-deleted insights."""

    def test_counts_pending(self, tmp_db):
        """Returns count of insights with NULL linked_at."""
        insert_insight(tmp_db, make_insight(id='cp-1', content='a'))
        insert_insight(tmp_db, make_insight(id='cp-2', content='b'))
        assert count_pending_links(tmp_db) == 2

    def test_excludes_linked(self, tmp_db):
        """Linked insights are not counted."""
        insert_insight(tmp_db, make_insight(id='cp-3', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_linked(tmp_db, 'cp-3', ts)
        assert count_pending_links(tmp_db) == 0


class TestResetForRebuild:
    """reset_for_rebuild clears linked_at and enriched_at for given IDs."""

    def test_clears_both_timestamps(self, tmp_db):
        """Both linked_at and enriched_at are set to NULL."""
        insert_insight(tmp_db, make_insight(id='rb-1', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_linked(tmp_db, 'rb-1', ts)
        stamp_enriched(tmp_db, 'rb-1', ts)
        reset_for_rebuild(tmp_db, ['rb-1'])
        row = tmp_db._query(
            'SELECT linked_at, enriched_at FROM insights WHERE id = ?',
            ('rb-1',)).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_empty_list_is_noop(self, tmp_db):
        """Passing empty list does nothing."""
        reset_for_rebuild(tmp_db, [])


class TestInsightDataclassExposesStamps:
    """The Insight dataclass returned by `get` reflects stamped timestamps.
    """

    def test_linked_at_round_trips(self, tmp_db):
        """`get_insight_by_id(...).linked_at` is non-None after stamp_linked.
        """
        insert_insight(tmp_db, make_insight(id='dx-1', content='a'))
        ts = format_timestamp(datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        stamp_linked(tmp_db, 'dx-1', ts)
        ins = get_insight_by_id(tmp_db, 'dx-1')
        assert ins is not None
        assert ins.linked_at is not None
        assert ins.linked_at.tzinfo is not None
        assert ins.enriched_at is None

    def test_enriched_at_round_trips(self, tmp_db):
        """`get_insight_by_id(...).enriched_at` is non-None after stamp_enriched.
        """
        insert_insight(tmp_db, make_insight(id='dx-2', content='b'))
        ts = format_timestamp(datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        stamp_enriched(tmp_db, 'dx-2', ts)
        ins = get_insight_by_id(tmp_db, 'dx-2')
        assert ins is not None
        assert ins.enriched_at is not None
        assert ins.linked_at is None

    def test_unstamped_insight_has_none_stamps(self, tmp_db):
        """Fresh insight returns linked_at and enriched_at as None."""
        insert_insight(tmp_db, make_insight(id='dx-3', content='c'))
        ins = get_insight_by_id(tmp_db, 'dx-3')
        assert ins is not None
        assert ins.linked_at is None
        assert ins.enriched_at is None
