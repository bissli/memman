"""Tests for edge deletion store functions.

Covers delete_auto_edges_for_node, delete_auto_edges_by_type,
count_auto_edges_by_type, and temporal proximity helpers.
"""

import pytest
from memman.store.edge import count_auto_edges_by_type
from memman.store.edge import count_low_weight_temporal_proximity
from memman.store.edge import delete_auto_edges_by_type
from memman.store.edge import delete_auto_edges_for_node
from memman.store.edge import delete_low_weight_temporal_proximity
from memman.store.edge import get_all_edges, insert_edge
from memman.store.node import insert_insight
from tests.conftest import make_edge, make_insight


_EDGE_KW = {
    'entity': {'weight': 0.5, 'extra': {'entity': 'Go'}},
    'semantic': {'weight': 0.7, 'extra': {'cosine': '0.70'}},
    'causal': {'weight': 0.9, 'extra': {}},
}


def _setup_node_pair(db, id_a='n-1', id_b='n-2'):
    """Insert two insights for edge tests."""
    insert_insight(db, make_insight(id=id_a, content=f'node {id_a}'))
    insert_insight(db, make_insight(id=id_b, content=f'node {id_b}'))


@pytest.mark.parametrize('edge_type,created_by,expected_remaining', [
    ('entity', 'claude', 1),
    ('entity', 'manual', 1),
    ('entity', 'auto', 0),
    ('entity', None, 0),
    ('semantic', 'claude', 1),
    ('semantic', 'auto', 0),
    ('semantic', None, 0),
    ('causal', 'claude', 1),
    ('causal', 'llm', 0),
])
def test_delete_auto_edges_for_node_by_type(
        tmp_db, edge_type, created_by, expected_remaining):
    """`delete_auto_edges_for_node` preserves manual edges, drops auto/llm.

    Per-edge-type contract: only edges WITHOUT a recognized human
    `created_by` (or with the type-specific auto markers) are deleted.
    """
    _setup_node_pair(tmp_db)
    spec = _EDGE_KW[edge_type]
    metadata = dict(spec['extra'])
    if created_by is not None:
        metadata['created_by'] = created_by
    insert_edge(tmp_db, make_edge(
        source_id='n-1', target_id='n-2',
        edge_type=edge_type, weight=spec['weight'],
        metadata=metadata))
    delete_auto_edges_for_node(tmp_db, 'n-1', edge_type)
    assert len(get_all_edges(tmp_db)) == expected_remaining


class TestDeleteAutoEdgesForNodeScoping:
    """Per-node deletion only affects the target node."""

    def test_scopes_to_node(self, tmp_db):
        """Edges on other nodes are untouched."""
        insert_insight(tmp_db, make_insight(
            id='s-1', content='a'))
        insert_insight(tmp_db, make_insight(
            id='s-2', content='b'))
        insert_insight(tmp_db, make_insight(
            id='s-3', content='c'))
        insert_edge(tmp_db, make_edge(
            source_id='s-1', target_id='s-2',
            edge_type='entity', weight=0.5,
            metadata={'entity': 'Go'}))
        insert_edge(tmp_db, make_edge(
            source_id='s-2', target_id='s-3',
            edge_type='entity', weight=0.5,
            metadata={'entity': 'Python'}))
        delete_auto_edges_for_node(tmp_db, 's-1', 'entity')
        edges = get_all_edges(tmp_db)
        assert len(edges) == 1
        assert edges[0].source_id == 's-2'

    def test_deletes_as_target(self, tmp_db):
        """Edges where node is target are also deleted."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-2', target_id='n-1',
            edge_type='entity', weight=0.5,
            metadata={'entity': 'Go'}))
        delete_auto_edges_for_node(tmp_db, 'n-1', 'entity')
        assert len(get_all_edges(tmp_db)) == 0


class TestDeleteAutoEdgesByType:
    """Global deletion for reindex."""

    def test_semantic_deletes_only_auto(self, tmp_db):
        """Global semantic delete removes auto but preserves null creator."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='semantic', weight=0.7,
            metadata={'created_by': 'auto', 'cosine': '0.70'}))
        insert_insight(tmp_db, make_insight(
            id='n-3', content='c'))
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-3',
            edge_type='semantic', weight=0.6,
            metadata={'cosine': '0.60'}))
        delete_auto_edges_by_type(tmp_db, 'semantic')
        edges = get_all_edges(tmp_db)
        assert len(edges) == 1
        assert edges[0].metadata.get('cosine') == '0.60'

    def test_entity_preserves_claude_and_manual(self, tmp_db):
        """Global entity delete preserves claude and manual."""
        _setup_node_pair(tmp_db)
        insert_insight(tmp_db, make_insight(
            id='n-3', content='c'))
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='entity', weight=0.5,
            metadata={'created_by': 'claude'}))
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-3',
            edge_type='entity', weight=0.5,
            metadata={'created_by': 'auto'}))
        delete_auto_edges_by_type(tmp_db, 'entity')
        edges = get_all_edges(tmp_db)
        assert len(edges) == 1
        assert edges[0].metadata['created_by'] == 'claude'

    def test_causal_preserves_llm_claude_manual(self, tmp_db):
        """Global causal delete preserves llm, claude, manual."""
        insert_insight(tmp_db, make_insight(
            id='c-1', content='a'))
        insert_insight(tmp_db, make_insight(
            id='c-2', content='b'))
        insert_insight(tmp_db, make_insight(
            id='c-3', content='c'))
        insert_insight(tmp_db, make_insight(
            id='c-4', content='d'))
        insert_edge(tmp_db, make_edge(
            source_id='c-1', target_id='c-2',
            edge_type='causal', weight=0.9,
            metadata={'created_by': 'llm'}))
        insert_edge(tmp_db, make_edge(
            source_id='c-1', target_id='c-3',
            edge_type='causal', weight=0.8,
            metadata={'created_by': 'claude'}))
        insert_edge(tmp_db, make_edge(
            source_id='c-1', target_id='c-4',
            edge_type='causal', weight=0.7,
            metadata={'created_by': 'manual'}))
        delete_auto_edges_by_type(tmp_db, 'causal')
        edges = get_all_edges(tmp_db)
        assert len(edges) == 3


class TestCountAutoEdges:
    """Count functions match delete filters."""

    def test_count_matches_delete(self, tmp_db):
        """Count returns same number as rows that would be deleted."""
        _setup_node_pair(tmp_db)
        insert_insight(tmp_db, make_insight(
            id='n-3', content='c'))
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='entity', weight=0.5,
            metadata={'created_by': 'auto'}))
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-3',
            edge_type='entity', weight=0.5,
            metadata={'created_by': 'claude'}))
        insert_edge(tmp_db, make_edge(
            source_id='n-2', target_id='n-3',
            edge_type='entity', weight=0.5,
            metadata={}))
        count = count_auto_edges_by_type(tmp_db, 'entity')
        assert count == 2
        delete_auto_edges_by_type(tmp_db, 'entity')
        assert len(get_all_edges(tmp_db)) == 1


class TestTemporalProximity:
    """Temporal proximity weight-based deletion."""

    def test_deletes_below_threshold(self, tmp_db):
        """Proximity edges below min_weight are deleted."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='temporal', weight=0.05,
            metadata={'sub_type': 'proximity'}))
        delete_low_weight_temporal_proximity(tmp_db, 0.1)
        assert len(get_all_edges(tmp_db)) == 0

    def test_preserves_above_threshold(self, tmp_db):
        """Proximity edges above min_weight survive."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='temporal', weight=0.5,
            metadata={'sub_type': 'proximity'}))
        delete_low_weight_temporal_proximity(tmp_db, 0.1)
        assert len(get_all_edges(tmp_db)) == 1

    def test_preserves_backbone(self, tmp_db):
        """Backbone temporal edges are never deleted."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='temporal', weight=0.01,
            metadata={'sub_type': 'backbone'}))
        delete_low_weight_temporal_proximity(tmp_db, 0.1)
        assert len(get_all_edges(tmp_db)) == 1

    def test_count_matches(self, tmp_db):
        """Count returns same as rows that would be deleted."""
        _setup_node_pair(tmp_db)
        insert_edge(tmp_db, make_edge(
            source_id='n-1', target_id='n-2',
            edge_type='temporal', weight=0.05,
            metadata={'sub_type': 'proximity'}))
        count = count_low_weight_temporal_proximity(tmp_db, 0.1)
        assert count == 1
