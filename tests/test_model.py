"""Tests for memman.model -- Insight/Edge dataclasses and helpers."""

from datetime import datetime, timezone

from memman.store.model import VALID_CATEGORIES, VALID_EDGE_TYPES, Edge
from memman.store.model import Insight, format_float, format_timestamp
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
    """The three edge types the store writes are accepted, others not.

    Mutation: readmitting a type nothing mints, which `graph link`
        would then accept into a store whose check constraint
        rejects it.
    Oracle: the literal member set, named one by one.
    """
    for et in ('temporal', 'semantic', 'entity'):
        assert et in VALID_EDGE_TYPES
    assert VALID_EDGE_TYPES == {'temporal', 'semantic', 'entity'}
    assert 'narrative' not in VALID_EDGE_TYPES


def test_schema_check_constraints_match_valid_edge_types():
    """Both backends' DDL admits exactly the types the code validates.

    The edge-type set is written three times: `VALID_EDGE_TYPES`, the
    SQLite baseline DDL, and the Postgres baseline DDL. The schema
    strings stay literal because a store's schema is a fixed baseline,
    so nothing but this test couples them.

    Mutation: adding or removing a member of `VALID_EDGE_TYPES`
        without editing both DDL strings -- the application would
        then accept an edge type the store's check constraint
        rejects, or reject one it allows.
    Oracle: the member list parsed back out of each DDL string,
        compared against the constant.
    """
    import re

    from memman.store.db import _BASELINE_SCHEMA
    from memman.store.postgres import PG_BASELINE_SCHEMA

    pattern = re.compile(r'edge_type in \(([^)]*)\)')
    for label, ddl in (('sqlite', _BASELINE_SCHEMA),
                       ('postgres', PG_BASELINE_SCHEMA)):
        found = pattern.search(ddl)
        assert found is not None, f'no edge_type check in {label} DDL'
        members = {m.strip().strip("'") for m in found.group(1).split(',')}
        assert members == VALID_EDGE_TYPES, label


def test_semantic_default_values():
    """Pin semantically-meaningful dataclass defaults.

    These four are real downstream-consumer contracts: changing any
    of them silently shifts graph behavior or LLM-output fallbacks.

    Mutation: the `general` literal returning as the `Insight` default.
    Oracle: the dataclass defaults.
    """
    ins = Insight()
    assert ins.category == 'fact'
    assert ins.importance == 3
    e = Edge()
    assert e.edge_type == 'semantic'
    assert e.weight == 0.5


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
