"""Tests for memman.model -- Insight dataclass and helpers."""

from datetime import datetime, timezone

from memman.store.model import VALID_CATEGORIES, Insight, format_timestamp
from memman.store.model import parse_timestamp


def test_valid_categories():
    """The five categories are accepted; `general` and unknowns are not.

    Mutation: the `general` literal returning to `VALID_CATEGORIES`.
    Oracle: the five-member set, with `general` and `bogus` outside it.
    """
    assert VALID_CATEGORIES == {'preference', 'decision', 'fact',
                                'insight', 'context'}
    assert 'general' not in VALID_CATEGORIES
    assert 'bogus' not in VALID_CATEGORIES


def test_semantic_default_values():
    """Pin the semantically-meaningful `category` dataclass default.

    Mutation: the `general` literal returning as the `Insight` default.
    Oracle: the dataclass default.
    """
    ins = Insight()
    assert ins.category == 'fact'


def test_format_timestamp():
    """Verify Z-suffix timestamp format."""
    dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
    assert format_timestamp(dt) == '2024-01-15T14:30:45Z'


def test_parse_timestamp_z():
    """Parse Z-suffix timestamp."""
    dt = parse_timestamp('2024-01-15T14:30:45Z')
    assert dt.year == 2024
    assert dt.hour == 14


def test_parse_timestamp_offset():
    """Parse +00:00 suffix timestamp."""
    dt = parse_timestamp('2024-01-15T14:30:45+00:00')
    assert dt.year == 2024
