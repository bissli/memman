"""Unit tests for memman.backup.cron (parsing + native translation).
"""

from datetime import datetime

import pytest
from memman.backup.cron import cron_matches, cron_to_launchd
from memman.backup.cron import cron_to_oncalendar


class TestCronMatches:
    """Local-time matching with the Vixie dom/dow OR-rule.
    """

    def test_daily_3am_matches_only_at_3am(self):
        """Verify `0 3 * * *` matches 03:00 and rejects other minutes and hours.

        Mutation: ignoring the hour or minute field.
        Oracle: hand-picked datetimes at 03:00, 04:00, and 03:01.
        """
        assert cron_matches('0 3 * * *', datetime(2026, 6, 27, 3, 0))
        assert not cron_matches('0 3 * * *', datetime(2026, 6, 27, 4, 0))
        assert not cron_matches('0 3 * * *', datetime(2026, 6, 27, 3, 1))

    def test_step_minutes_match_quarter_hours(self):
        """Verify `*/15 * * * *` matches 0/15/30/45 and rejects 7.

        Mutation: a step parsed as an offset or as a range end.
        Oracle: the hand-listed quarter-hour minutes.
        """
        for minute in (0, 15, 30, 45):
            assert cron_matches(
                '*/15 * * * *', datetime(2026, 6, 27, 1, minute))
        assert not cron_matches('*/15 * * * *', datetime(2026, 6, 27, 1, 7))

    def test_sunday_zero_and_seven_equivalent(self):
        """Verify dow 0 and 7 both match Sunday and Monday does not.

        Mutation: rejecting 7 as out of range, or mapping it to another day.
        Oracle: 2026-06-28 (a Sunday) and 2026-06-29 (a Monday) by calendar.
        """
        sunday = datetime(2026, 6, 28, 0, 0)
        assert cron_matches('0 0 * * 0', sunday)
        assert cron_matches('0 0 * * 7', sunday)
        assert not cron_matches('0 0 * * 0', datetime(2026, 6, 29, 0, 0))

    def test_vixie_or_rule_dom_or_dow(self):
        """Verify `0 0 1 * 5` matches the 1st or any Friday, not neither.

        Mutation: AND-ing the dom and dow fields instead of OR-ing them.
        Oracle: 2026-06-01 (Monday, 1st), 06-05 (Friday), 06-03 (neither).
        """
        assert cron_matches('0 0 1 * 5', datetime(2026, 6, 1, 0, 0))
        assert cron_matches('0 0 1 * 5', datetime(2026, 6, 5, 0, 0))
        assert not cron_matches('0 0 1 * 5', datetime(2026, 6, 3, 0, 0))

    def test_month_field_restricts_to_that_month(self):
        """Verify `0 0 1 6 *` matches June 1 and rejects May 1.

        Mutation: ignoring the month field.
        Oracle: hand-picked June and May dates.
        """
        assert cron_matches('0 0 1 6 *', datetime(2026, 6, 1, 0, 0))
        assert not cron_matches('0 0 1 6 *', datetime(2026, 5, 1, 0, 0))

    def test_invalid_exprs_raise(self):
        """Verify a bad field count, an out-of-range value, and a zero step raise.

        Mutation: accepting a malformed expression, or a zero step that would
            divide by zero at match time.
        Oracle: pytest.raises(ValueError) on hand-built bad expressions.
        """
        with pytest.raises(ValueError):
            cron_matches('0 3 * *', datetime(2026, 6, 27, 3, 0))
        with pytest.raises(ValueError):
            cron_matches('99 3 * * *', datetime(2026, 6, 27, 3, 0))
        with pytest.raises(ValueError):
            cron_matches('*/0 * * * *', datetime(2026, 6, 27, 3, 0))


class TestCronToOnCalendar:
    """systemd OnCalendar rendering (local time, no UTC suffix).
    """

    def test_table_examples(self):
        """Verify five cron expressions translate to the expected OnCalendar text.

        Mutation: wrong zero padding, weekday names, or minute list expansion.
        Oracle: hand-written systemd OnCalendar strings.
        """
        assert cron_to_oncalendar('0 3 * * *') == '*-*-* 03:00:00'
        assert cron_to_oncalendar('*/15 * * * *') == '*-*-* *:0,15,30,45:00'
        assert cron_to_oncalendar('30 2 1 * *') == '*-*-01 02:30:00'
        assert cron_to_oncalendar('0 0 * * 0') == 'Sun *-*-* 00:00:00'
        assert (cron_to_oncalendar('0 9 * * 1-5')
                == 'Mon,Tue,Wed,Thu,Fri *-*-* 09:00:00')

    def test_month_field_renders(self):
        """Verify a restricted month renders as a zero-padded date component.

        Mutation: rendering the month unpadded or dropping it.
        Oracle: the literal '*-06-01 00:00:00'.
        """
        assert cron_to_oncalendar('0 0 1 6 *') == '*-06-01 00:00:00'

    def test_no_utc_suffix(self):
        """Verify the OnCalendar text carries no UTC token.

        Mutation: appending ' UTC', which would shift the schedule off
            local time.
        Oracle: substring absence in the rendered text.
        """
        assert 'UTC' not in cron_to_oncalendar('0 3 * * *')


class TestCronToLaunchd:
    """launchd StartCalendarInterval rendering (Weekday 0=Sunday).
    """

    def test_single_value_returns_one_dict(self):
        """Verify single-valued fields give one dict and `*` fields are omitted.

        Mutation: emitting wildcard keys, or a list for a single-value
            expression.
        Oracle: hand-written launchd dicts.
        """
        assert cron_to_launchd('0 3 * * *') == {'Minute': 0, 'Hour': 3}
        assert cron_to_launchd('30 2 1 * *') == {
            'Minute': 30, 'Hour': 2, 'Day': 1}
        assert cron_to_launchd('0 0 * * 0') == {
            'Minute': 0, 'Hour': 0, 'Weekday': 0}
        assert cron_to_launchd('0 0 1 6 *') == {
            'Minute': 0, 'Hour': 0, 'Month': 6, 'Day': 1}

    def test_multi_value_returns_array(self):
        """Verify a multi-value field expands to one dict per value.

        Mutation: collapsing the range or step to one dict, or off-by-one
            values.
        Oracle: hand-listed minute and weekday dicts.
        """
        assert cron_to_launchd('*/15 * * * *') == [
            {'Minute': 0}, {'Minute': 15}, {'Minute': 30}, {'Minute': 45}]
        weekdays = cron_to_launchd('0 9 * * 1-5')
        assert weekdays == [
            {'Minute': 0, 'Hour': 9, 'Weekday': n} for n in (1, 2, 3, 4, 5)]

    def test_both_dom_and_dow_concatenate_or_groups(self):
        """Verify restricting dom and dow yields a Day group plus a Weekday group.

        Mutation: cross-multiplying dom and dow into one AND group.
        Oracle: the two hand-written dicts and a length of 2.
        """
        result = cron_to_launchd('0 0 1 * 5')
        assert {'Minute': 0, 'Hour': 0, 'Day': 1} in result
        assert {'Minute': 0, 'Hour': 0, 'Weekday': 5} in result
        assert len(result) == 2
