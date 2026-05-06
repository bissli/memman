"""Tests for memman.model -- Insight/Edge dataclasses and helpers."""

from datetime import datetime, timezone

from memman.store.model import VALID_CATEGORIES, VALID_EDGE_TYPES, Edge
from memman.store.model import Insight, base_weight, format_float
from memman.store.model import format_timestamp, is_immune, parse_timestamp


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
    """All 6 categories accepted, invalid rejected."""
    for cat in ('preference', 'decision', 'fact',
                'insight', 'context', 'general'):
        assert cat in VALID_CATEGORIES
    assert 'bogus' not in VALID_CATEGORIES


def test_parse_metadata_null():
    """JSON 'null' produces empty dict, not None.

    Real branch: parse_metadata has an explicit None fallback after
    json.loads.
    """
    e = Edge()
    e.parse_metadata('null')
    assert e.metadata == {}


def test_parse_metadata_invalid_json():
    """Invalid JSON produces empty dict via try/except fallback.

    Real branch: removing the try/except would propagate JSONDecodeError
    through every Edge read with corrupted JSON.
    """
    e = Edge()
    e.parse_metadata('not json')
    assert e.metadata == {}


def test_valid_edge_types():
    """All 4 edge types accepted, invalid rejected."""
    for et in ('temporal', 'semantic', 'causal', 'entity'):
        assert et in VALID_EDGE_TYPES
    assert 'narrative' not in VALID_EDGE_TYPES


def test_semantic_default_values():
    """Pin semantically-meaningful dataclass defaults.

    These four are real downstream-consumer contracts: changing any
    of them silently shifts graph behavior or LLM-output fallbacks.
    """
    ins = Insight()
    assert ins.category == 'general'
    assert ins.importance == 3
    e = Edge()
    assert e.edge_type == 'semantic'
    assert e.weight == 0.5


def test_base_weight_values():
    """Verify base_weight maps importance correctly."""
    assert base_weight(5) == 1.0
    assert base_weight(4) == 0.8
    assert base_weight(3) == 0.5
    assert base_weight(2) == 0.3
    assert base_weight(1) == 0.15


def test_is_immune():
    """Verify immunity rules."""
    assert is_immune(4, 0) is True
    assert is_immune(5, 0) is True
    assert is_immune(1, 3) is True
    assert is_immune(3, 2) is False
    assert is_immune(1, 0) is False


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


def test_format_float():
    """Verify 4 decimal place formatting."""
    assert format_float(0.85) == '0.8500'
    assert format_float(1.0) == '1.0000'
