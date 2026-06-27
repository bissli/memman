"""Unit tests for memman.backup.cron (parsing + native translation)."""

from datetime import datetime

import pytest
from memman.backup.cron import cron_matches, cron_to_launchd
from memman.backup.cron import cron_to_oncalendar


class TestCronMatches:
    """Local-time matching with the Vixie dom/dow OR-rule."""

    def test_daily_3am_matches_only_at_3am(self):
        """`0 3 * * *` matches 03:00 and rejects other times."""
        assert cron_matches('0 3 * * *', datetime(2026, 6, 27, 3, 0))
        assert not cron_matches('0 3 * * *', datetime(2026, 6, 27, 4, 0))
        assert not cron_matches('0 3 * * *', datetime(2026, 6, 27, 3, 1))

    def test_step_minutes_match_quarter_hours(self):
        """`*/15 * * * *` matches 0/15/30/45 and rejects 7."""
        for minute in (0, 15, 30, 45):
            assert cron_matches(
                '*/15 * * * *', datetime(2026, 6, 27, 1, minute))
        assert not cron_matches('*/15 * * * *', datetime(2026, 6, 27, 1, 7))

    def test_sunday_zero_and_seven_equivalent(self):
        """dow `0` and `7` both match Sunday; Monday does not."""
        sunday = datetime(2026, 6, 28, 0, 0)
        assert cron_matches('0 0 * * 0', sunday)
        assert cron_matches('0 0 * * 7', sunday)
        assert not cron_matches('0 0 * * 0', datetime(2026, 6, 29, 0, 0))

    def test_vixie_or_rule_dom_or_dow(self):
        """`0 0 1 * 5` matches the 1st OR any Friday, but not neither."""
        assert cron_matches('0 0 1 * 5', datetime(2026, 6, 1, 0, 0))
        assert cron_matches('0 0 1 * 5', datetime(2026, 6, 5, 0, 0))
        assert not cron_matches('0 0 1 * 5', datetime(2026, 6, 3, 0, 0))

    def test_invalid_exprs_raise(self):
        """Bad field count, out-of-range value, and zero step raise."""
        with pytest.raises(ValueError):
            cron_matches('0 3 * *', datetime(2026, 6, 27, 3, 0))
        with pytest.raises(ValueError):
            cron_matches('99 3 * * *', datetime(2026, 6, 27, 3, 0))
        with pytest.raises(ValueError):
            cron_matches('*/0 * * * *', datetime(2026, 6, 27, 3, 0))


class TestCronToOnCalendar:
    """systemd OnCalendar rendering (local time, no UTC suffix)."""

    def test_table_examples(self):
        """The five locked cron -> OnCalendar translations."""
        assert cron_to_oncalendar('0 3 * * *') == '*-*-* 03:00:00'
        assert cron_to_oncalendar('*/15 * * * *') == '*-*-* *:0,15,30,45:00'
        assert cron_to_oncalendar('30 2 1 * *') == '*-*-01 02:30:00'
        assert cron_to_oncalendar('0 0 * * 0') == 'Sun *-*-* 00:00:00'
        assert (cron_to_oncalendar('0 9 * * 1-5')
                == 'Mon,Tue,Wed,Thu,Fri *-*-* 09:00:00')

    def test_no_utc_suffix(self):
        """OnCalendar is local time -- never carries a UTC token."""
        assert 'UTC' not in cron_to_oncalendar('0 3 * * *')


class TestCronToLaunchd:
    """launchd StartCalendarInterval rendering (Weekday 0=Sunday)."""

    def test_single_value_returns_one_dict(self):
        """All-single fields collapse to one dict; `*` fields omitted."""
        assert cron_to_launchd('0 3 * * *') == {'Minute': 0, 'Hour': 3}
        assert cron_to_launchd('30 2 1 * *') == {
            'Minute': 30, 'Hour': 2, 'Day': 1}
        assert cron_to_launchd('0 0 * * 0') == {
            'Minute': 0, 'Hour': 0, 'Weekday': 0}

    def test_multi_value_returns_array(self):
        """A multi-value field expands to a cartesian array of dicts."""
        assert cron_to_launchd('*/15 * * * *') == [
            {'Minute': 0}, {'Minute': 15}, {'Minute': 30}, {'Minute': 45}]
        weekdays = cron_to_launchd('0 9 * * 1-5')
        assert weekdays == [
            {'Minute': 0, 'Hour': 9, 'Weekday': n} for n in (1, 2, 3, 4, 5)]

    def test_both_dom_and_dow_concatenate_or_groups(self):
        """Restricting dom AND dow yields a Day group plus a Weekday group."""
        result = cron_to_launchd('0 0 1 * 5')
        assert {'Minute': 0, 'Hour': 0, 'Day': 1} in result
        assert {'Minute': 0, 'Hour': 0, 'Weekday': 5} in result
        assert len(result) == 2
