"""Tests for memman.model -- Insight dataclass and helpers.
"""

from datetime import datetime, timezone

from memman.store.model import format_timestamp, parse_timestamp


def test_format_timestamp():
    """Verify format_timestamp renders a UTC datetime with a Z suffix.

    Mutation: rendering the +00:00 offset or dropping the seconds.
    Oracle: the hand-written string '2024-01-15T14:30:45Z'.
    """
    dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
    assert format_timestamp(dt) == '2024-01-15T14:30:45Z'


def test_parse_timestamp_z():
    """Verify parse_timestamp reads a Z-suffix timestamp.

    Mutation: failing on the Z suffix or shifting the hour.
    Oracle: the year and hour of the hand-written input.
    """
    dt = parse_timestamp('2024-01-15T14:30:45Z')
    assert dt.year == 2024
    assert dt.hour == 14


def test_parse_timestamp_offset():
    """Verify parse_timestamp reads a +00:00 suffix timestamp.

    Mutation: rejecting the numeric-offset form.
    Oracle: the year of the hand-written input.
    """
    dt = parse_timestamp('2024-01-15T14:30:45+00:00')
    assert dt.year == 2024
