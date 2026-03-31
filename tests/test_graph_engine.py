"""Tests for mnemon.graph.engine — constants hash and edge orchestration."""

from mnemon.graph.engine import compute_constants_hash


def test_constants_hash_deterministic():
    """Calling twice returns the same value."""
    assert compute_constants_hash() == compute_constants_hash()


def test_constants_hash_changes_on_entity_limit(monkeypatch):
    """Changing MAX_ENTITY_LINKS produces a different hash."""
    original = compute_constants_hash()
    import mnemon.graph.engine as engine
    monkeypatch.setattr(engine, 'MAX_ENTITY_LINKS', 999)
    changed = compute_constants_hash()
    assert changed != original


def test_constants_hash_changes_on_proximity_limit(monkeypatch):
    """Changing MAX_PROXIMITY_EDGES produces a different hash."""
    original = compute_constants_hash()
    import mnemon.graph.engine as engine
    monkeypatch.setattr(engine, 'MAX_PROXIMITY_EDGES', 999)
    changed = compute_constants_hash()
    assert changed != original
