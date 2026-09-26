"""Tests for memman.model -- Insight dataclass and helpers."""

from datetime import datetime, timezone

from memman.store.model import VALID_CATEGORIES, Insight, format_timestamp
from memman.store.model import parse_timestamp


def test_parse_entities_null():
    """JSON 'null' produces empty list, not None.

    Real branch: `parse_entities` has an explicit `if entities is None`
    fallback after json.loads. Removing that line leaks a `None`-typed
    `.entities` to downstream `.append`/iteration which crashes.
    """
    ins = Insight()
    ins.parse_entities('null')
    assert ins.entities == []


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
    """Pin semantically-meaningful dataclass defaults.

    These two are real downstream-consumer contracts: changing either
    of them silently shifts LLM-output fallbacks.

    Mutation: the `general` literal returning as the `Insight` default.
    Oracle: the dataclass defaults.
    """
    ins = Insight()
    assert ins.category == 'fact'
    assert ins.importance == 3


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
