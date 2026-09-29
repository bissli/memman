"""Tests for lifecycle stamp and pending-enrich store functions.
"""

from datetime import datetime, timezone

from memman.store.model import format_timestamp
from memman.store.node import count_pending_enrich, get_active_insight_ids
from memman.store.node import get_insight_by_id, get_pending_enrich_ids
from memman.store.node import insert_insight, reset_for_rebuild
from memman.store.node import soft_delete_insight, stamp_enrich_attempted
from memman.store.node import stamp_enriched
from tests.conftest import make_insight


class TestStampEnriched:
    """stamp_enriched sets enriched_at timestamp.

    stamp_enrich_attempted is covered by `TestGetPendingEnrichIds`, where
    a column-name typo leaves a row in the pending list.
    """

    def test_sets_timestamp(self, tmp_db):
        """Verify stamp_enriched writes the given timestamp to enriched_at.

        Mutation: stamp_enriched writing the wrong column or the current time.
        Oracle: the exact formatted timestamp passed in.
        """
        insert_insight(tmp_db, make_insight(id='se-1', content='a'))
        ts = format_timestamp(datetime(2025, 6, 1, tzinfo=timezone.utc))
        stamp_enriched(tmp_db, 'se-1', ts)
        row = tmp_db._query(
            'select enriched_at from insights where id = ?',
            ('se-1',)).fetchone()
        assert row[0] == ts


class TestGetPendingEnrichIds:
    """get_pending_enrich_ids returns unattempted, non-deleted insights.
    """

    def test_returns_unattempted(self, tmp_db):
        """Verify unattempted insights are returned.

        Mutation: get_pending_enrich_ids dropping the `deleted_at is
            null` or `replaced_by is null` clause, or its select
            list dropping a freshly inserted id.
        Oracle: the returned id set equals both freshly inserted ids.
        """
        insert_insight(tmp_db, make_insight(id='pl-1', content='a'))
        insert_insight(tmp_db, make_insight(id='pl-2', content='b'))
        ids = get_pending_enrich_ids(tmp_db, 10)
        assert set(ids) == {'pl-1', 'pl-2'}

    def test_excludes_attempted(self, tmp_db):
        """Verify attempted insights are excluded.

        Mutation: get_pending_enrich_ids dropping its
            `enrich_attempted_at is null` clause, so a stamped row
            never leaves the pending list.
        Oracle: the returned id list excludes the stamped id.
        """
        insert_insight(tmp_db, make_insight(id='pl-3', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_enrich_attempted(tmp_db, 'pl-3', ts)
        ids = get_pending_enrich_ids(tmp_db, 10)
        assert 'pl-3' not in ids

    def test_excludes_deleted(self, tmp_db):
        """Verify soft-deleted insights are excluded.

        Mutation: get_pending_enrich_ids dropping its `deleted_at is
            null` clause, so a soft-deleted row still comes back
            pending.
        Oracle: the returned id list excludes the deleted id.
        """
        insert_insight(tmp_db, make_insight(id='pl-4', content='a'))
        soft_delete_insight(tmp_db, 'pl-4')
        ids = get_pending_enrich_ids(tmp_db, 10)
        assert 'pl-4' not in ids

    def test_respects_limit(self, tmp_db):
        """Verify the limit caps the number of IDs returned.

        Mutation: get_pending_enrich_ids ignoring its `limit`
            parameter and returning every pending row.
        Oracle: exact returned length of 3 against 5 pending rows.
        """
        for i in range(5):
            insert_insight(tmp_db, make_insight(
                id=f'pl-l{i}', content=f'c{i}'))
        ids = get_pending_enrich_ids(tmp_db, 3)
        assert len(ids) == 3


class TestGetActiveInsightIds:
    """get_active_insight_ids returns non-deleted IDs in creation order.
    """

    def test_returns_active_ordered(self, tmp_db):
        """Verify active insight IDs come back in created_at ascending order.

        Mutation: ordering descending, or unordered.
        Oracle: the hand-listed insertion order ['ai-1', 'ai-2'].
        """
        insert_insight(tmp_db, make_insight(id='ai-1', content='first'))
        insert_insight(tmp_db, make_insight(id='ai-2', content='second'))
        ids = get_active_insight_ids(tmp_db)
        assert ids == ['ai-1', 'ai-2']

    def test_excludes_deleted(self, tmp_db):
        """Verify soft-deleted insights are absent from the active IDs.

        Mutation: dropping the `deleted_at is null` filter.
        Oracle: absence of the soft-deleted id.
        """
        insert_insight(tmp_db, make_insight(id='ai-3', content='a'))
        soft_delete_insight(tmp_db, 'ai-3')
        ids = get_active_insight_ids(tmp_db)
        assert 'ai-3' not in ids


class TestCountPendingEnrich:
    """count_pending_enrich counts unattempted, non-deleted insights.
    """

    def test_counts_pending(self, tmp_db):
        """Verify the count covers insights with NULL enrich_attempted_at.

        Mutation: count_pending_enrich dropping its
            `enrich_attempted_at is null` clause and counting every
            active row.
        Oracle: exact count of 2 against two freshly inserted rows.
        """
        insert_insight(tmp_db, make_insight(id='cp-1', content='a'))
        insert_insight(tmp_db, make_insight(id='cp-2', content='b'))
        assert count_pending_enrich(tmp_db) == 2

    def test_excludes_attempted(self, tmp_db):
        """Verify attempted insights are not counted.

        Mutation: count_pending_enrich dropping its
            `enrich_attempted_at is null` clause, counting a stamped
            row as still pending.
        Oracle: exact count of 0 after stamping the one row.
        """
        insert_insight(tmp_db, make_insight(id='cp-3', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_enrich_attempted(tmp_db, 'cp-3', ts)
        assert count_pending_enrich(tmp_db) == 0


class TestResetForRebuild:
    """reset_for_rebuild clears both enrichment stamps for given IDs.
    """

    def test_clears_both_timestamps(self, tmp_db):
        """Verify reset sets both enrichment stamps to NULL.

        Mutation: reset_for_rebuild clearing only one of the two
            columns, leaving the other stamp behind.
        Oracle: both columns read back None for the given id after
            stamping both, then resetting.
        """
        insert_insight(tmp_db, make_insight(id='rb-1', content='a'))
        ts = format_timestamp(datetime.now(timezone.utc))
        stamp_enrich_attempted(tmp_db, 'rb-1', ts)
        stamp_enriched(tmp_db, 'rb-1', ts)
        reset_for_rebuild(tmp_db, ['rb-1'])
        row = tmp_db._query(
            'select enrich_attempted_at, enriched_at from insights where id = ?',
            ('rb-1',)).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_empty_list_is_noop(self, tmp_db):
        """Verify reset_for_rebuild with an empty list does nothing and does not raise.

        Mutation: building `in ()` SQL from an empty id list, which errors.
        Oracle: the call returns without an exception.
        """
        reset_for_rebuild(tmp_db, [])


class TestInsightDataclassExposesStamps:
    """The Insight dataclass returned by `get` reflects stamped timestamps.
    """

    def test_enrich_attempted_at_round_trips(self, tmp_db):
        """Verify enrich_attempted_at reads back after stamp_enrich_attempted.

        Mutation: `_scan_insight` reading the enriched_at column
            index into `enrich_attempted_at`, or `parse_timestamp`
            dropping tzinfo.
        Oracle: enrich_attempted_at not-None with tzinfo set, while
            enriched_at stays None.
        """
        insert_insight(tmp_db, make_insight(id='dx-1', content='a'))
        ts = format_timestamp(datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        stamp_enrich_attempted(tmp_db, 'dx-1', ts)
        ins = get_insight_by_id(tmp_db, 'dx-1')
        assert ins is not None
        assert ins.enrich_attempted_at is not None
        assert ins.enrich_attempted_at.tzinfo is not None
        assert ins.enriched_at is None

    def test_enriched_at_round_trips(self, tmp_db):
        """Verify enriched_at reads back after stamp_enriched.

        Mutation: `_scan_insight` reading the enrich_attempted_at
            column index into `enriched_at`, mixing up the two
            stamps.
        Oracle: enriched_at not-None while enrich_attempted_at stays
            None.
        """
        insert_insight(tmp_db, make_insight(id='dx-2', content='b'))
        ts = format_timestamp(datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        stamp_enriched(tmp_db, 'dx-2', ts)
        ins = get_insight_by_id(tmp_db, 'dx-2')
        assert ins is not None
        assert ins.enriched_at is not None
        assert ins.enrich_attempted_at is None

    def test_unstamped_insight_has_none_stamps(self, tmp_db):
        """Verify a fresh insight has both stamps None.

        Mutation: `_scan_insight` setting either stamp attribute
            whenever its column reads a falsy value rather than
            skipping the assignment.
        Oracle: both attributes read back None on an unstamped row.
        """
        insert_insight(tmp_db, make_insight(id='dx-3', content='c'))
        ins = get_insight_by_id(tmp_db, 'dx-3')
        assert ins is not None
        assert ins.enrich_attempted_at is None
        assert ins.enriched_at is None
