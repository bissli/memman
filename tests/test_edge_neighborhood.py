"""EdgeStore.get_neighborhood Protocol verb tests.

Verifies the bounded BFS verb returns the expected (id, hop, etype)
triples on a small fixture graph, and that depth + edge_filter both
clamp the traversal as documented.
"""

import inspect

from memman.store.backend import EdgeStore
from memman.store.model import Edge, Insight


def _seed(backend) -> None:
    """Build a 5-node graph: a -> b -> c (semantic), a -> d (causal),
    plus an isolated e.
    """
    for i, content in (('a', 'A'), ('b', 'B'), ('c', 'C'),
                       ('d', 'D'), ('e', 'E')):
        backend.nodes.insert(Insight(id=i, content=content))
    backend.edges.upsert(Edge(
        source_id='a', target_id='b', edge_type='semantic',
        weight=1.0))
    backend.edges.upsert(Edge(
        source_id='b', target_id='c', edge_type='semantic',
        weight=1.0))
    backend.edges.upsert(Edge(
        source_id='a', target_id='d', edge_type='causal',
        weight=1.0))


def test_protocol_verb_signature():
    """EdgeStore.get_neighborhood is a Protocol verb."""
    assert hasattr(EdgeStore, 'get_neighborhood')
    sig = inspect.signature(EdgeStore.get_neighborhood)
    assert 'seed_id' in sig.parameters
    assert 'depth' in sig.parameters
    assert 'edge_filter' in sig.parameters


def test_depth_one(backend):
    """depth=1 reaches the immediate neighbors of `a`."""
    _seed(backend)
    triples = backend.edges.get_neighborhood('a', depth=1)
    nbrs = {nid for nid, _hop, _etype in triples}
    assert nbrs == {'b', 'd'}


def test_depth_two(backend):
    """depth=2 reaches `c` via `b`."""
    _seed(backend)
    triples = backend.edges.get_neighborhood('a', depth=2)
    nbrs = {nid for nid, _hop, _etype in triples}
    assert nbrs == {'b', 'c', 'd'}


def test_edge_filter_causal(backend):
    """edge_filter='causal' walks only causal edges."""
    _seed(backend)
    triples = backend.edges.get_neighborhood(
        'a', depth=2, edge_filter='causal')
    nbrs = {nid for nid, _hop, _etype in triples}
    assert nbrs == {'d'}


def test_isolated_seed_returns_empty(backend):
    """A node with no edges produces no triples."""
    _seed(backend)
    triples = backend.edges.get_neighborhood('e', depth=2)
    assert triples == []


def test_hop_values_are_correct(backend):
    """Each triple's hop value matches its distance from the seed."""
    _seed(backend)
    triples = backend.edges.get_neighborhood('a', depth=2)
    by_id = {nid: hop for nid, hop, _etype in triples}
    assert by_id['b'] == 1
    assert by_id['d'] == 1
    assert by_id['c'] == 2


def test_get_many_hydrates_in_order(backend):
    """NodeStore.get_many returns insights in input order, drops misses.
    """
    backend.nodes.insert(Insight(id='gm-1', content='one'))
    backend.nodes.insert(Insight(id='gm-2', content='two'))
    insights = backend.nodes.get_many(['gm-2', 'absent', 'gm-1'])
    assert [i.id for i in insights] == ['gm-2', 'gm-1']
