"""Tests for mnemon.graph.engine — constants hash and edge orchestration."""

from mnemon.graph.engine import compute_constants_hash, consolidate_pending
from mnemon.graph.engine import rebuild_auto_edges
from mnemon.store.edge import get_all_edges, insert_edge
from mnemon.store.node import insert_insight
from tests.conftest import make_edge, make_insight


def test_constants_hash_deterministic():
    """Calling twice returns the same value."""
    assert compute_constants_hash() == compute_constants_hash()


def test_constants_hash_changes_on_entity_limit(monkeypatch):
    """Changing MAX_ENTITY_LINKS produces a different hash."""
    original = compute_constants_hash()
    from mnemon.graph import engine
    monkeypatch.setattr(engine, 'MAX_ENTITY_LINKS', 999)
    changed = compute_constants_hash()
    assert changed != original


def test_constants_hash_changes_on_proximity_limit(monkeypatch):
    """Changing MAX_PROXIMITY_EDGES produces a different hash."""
    original = compute_constants_hash()
    from mnemon.graph import engine
    monkeypatch.setattr(engine, 'MAX_PROXIMITY_EDGES', 999)
    changed = compute_constants_hash()
    assert changed != original


def test_rebuild_does_not_block_consolidation(tmp_db):
    """After rebuild, consolidate_pending can still process insights."""
    insert_insight(tmp_db, make_insight(
        id='rbc-1', content='rebuild then consolidate test'))
    insert_insight(tmp_db, make_insight(
        id='rbc-2', content='second insight for consolidation'))

    rebuild_auto_edges(tmp_db)
    processed = consolidate_pending(tmp_db)
    assert processed > 0


def test_rebuild_preserves_manual_edge_metadata(tmp_db):
    """Manual claude entity edge metadata survives rebuild with auto edges."""
    ins1 = make_insight(
        id='hw-1', content='Python web framework',
        entities=['Python'])
    ins2 = make_insight(
        id='hw-2', content='Python data analysis',
        entities=['Python'])
    insert_insight(tmp_db, ins1)
    insert_insight(tmp_db, ins2)

    manual_edge = make_edge(
        source_id='hw-1', target_id='hw-2',
        edge_type='entity', weight=0.5,
        metadata={'entity': 'Python', 'created_by': 'claude'})
    insert_edge(tmp_db, manual_edge)

    rebuild_auto_edges(tmp_db)

    edges = get_all_edges(tmp_db)
    match = [e for e in edges
             if e.source_id == 'hw-1' and e.target_id == 'hw-2'
             and e.edge_type == 'entity']
    assert len(match) == 1
    assert match[0].metadata.get('created_by') == 'claude'
    assert match[0].weight >= 0.5
