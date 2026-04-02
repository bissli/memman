"""Graph integration tests ported from Go integration_test.go.

These tests exercise temporal, entity, causal, semantic, BFS, and engine
modules against a real SQLite database.
"""

from datetime import datetime, timedelta, timezone

from memman.embed.vector import serialize_vector
from memman.graph.bfs import BFSOptions, bfs
from memman.graph.engine import fast_edges
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import build_embed_cache, create_semantic_edges
from memman.graph.temporal import create_temporal_edge
from memman.store.edge import get_edges_by_node_and_type, insert_edge
from memman.store.node import insert_insight, soft_delete_insight
from memman.store.node import update_embedding
from tests.conftest import make_edge, make_insight

# --- Temporal ---


class TestTemporalBackboneChain:
    """Two insights with same source produce bidirectional backbone edges."""

    def test_temporal_backbone_chain(self, tmp_db):
        """Insert two insights with same source; verify 2 backbone temporal edges."""
        now = datetime.now(timezone.utc)
        ins1 = make_insight(
            id='t-1', content='first insight', source='proj-a',
            created_at=now - timedelta(hours=1))
        ins2 = make_insight(
            id='t-2', content='second insight', source='proj-a',
            created_at=now)
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        count = create_temporal_edge(tmp_db, ins2)
        assert count >= 2

        edges = get_edges_by_node_and_type(tmp_db, 't-2', 'temporal')
        assert len(edges) >= 2

        directions = set()
        for e in edges:
            if e.metadata.get('sub_type') == 'backbone':
                directions.add(e.metadata.get('direction'))
        assert 'precedes' in directions
        assert 'succeeds' in directions


class TestTemporalProximityDecay:
    """Closer insights get higher temporal proximity weight."""

    def test_temporal_proximity_decay(self, tmp_db):
        """A close insight has higher weight than a far one."""
        now = datetime.now(timezone.utc)
        close = make_insight(
            id='tp-1', content='close insight', source='other',
            created_at=now - timedelta(minutes=30))
        far = make_insight(
            id='tp-2', content='far insight', source='other',
            created_at=now - timedelta(hours=12))
        current = make_insight(
            id='tp-3', content='current insight', source='proj-b',
            created_at=now)
        insert_insight(tmp_db, close)
        insert_insight(tmp_db, far)
        insert_insight(tmp_db, current)

        create_temporal_edge(tmp_db, current)

        edges = get_edges_by_node_and_type(tmp_db, 'tp-3', 'temporal')
        proximity_edges = [
            e for e in edges
            if e.metadata.get('sub_type') == 'proximity'
            ]
        assert len(proximity_edges) >= 2

        weight_by_target = {}
        for e in proximity_edges:
            target = e.target_id if e.source_id == 'tp-3' else e.source_id
            weight_by_target[target] = e.weight

        assert weight_by_target.get('tp-1', 0) > weight_by_target.get('tp-2', 0)


class TestTemporalNoSource:
    """Single insight with no prior creates 0 backbone edges."""

    def test_temporal_no_source(self, tmp_db):
        """A lone insight in an empty DB produces 0 temporal edges."""
        ins = make_insight(id='ts-1', content='solo insight', source='solo')
        insert_insight(tmp_db, ins)

        count = create_temporal_edge(tmp_db, ins)
        assert count == 0


# --- Entity ---


class TestEntityCoOccurrence:
    """Shared entity creates bidirectional entity edges."""

    def test_entity_co_occurrence(self, tmp_db):
        """Two insights sharing 'Go' produce entity edges."""
        ins1 = make_insight(
            id='e-1', content='Go is fast', entities=['Go'])
        ins2 = make_insight(
            id='e-2', content='Go concurrency', entities=['Go'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        count = create_entity_edges(tmp_db, ins2)
        assert count >= 2

        edges = get_edges_by_node_and_type(tmp_db, 'e-2', 'entity')
        assert len(edges) >= 1


class TestEntityNoShared:
    """No shared entities means 0 entity edges."""

    def test_entity_no_shared(self, tmp_db):
        """Two insights with disjoint entities produce 0 edges."""
        ins1 = make_insight(
            id='en-1', content='Go is fast', entities=['Go'])
        ins2 = make_insight(
            id='en-2', content='Python web', entities=['Python'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        count = create_entity_edges(tmp_db, ins2)
        assert count == 0


class TestEntityEmpty:
    """Insight with no entities produces 0 edges."""

    def test_entity_empty(self, tmp_db):
        """No entities on insight means no entity edges."""
        ins = make_insight(id='ee-1', content='no entities', entities=[])
        insert_insight(tmp_db, ins)

        count = create_entity_edges(tmp_db, ins)
        assert count == 0


class TestEntityHyphenatedUserProvided:
    """User-provided entities with non-standard patterns create edges."""

    def test_user_provided_hyphenated_entity_creates_edges(self, tmp_db):
        """User-provided entity like 'caching-layer' creates entity edges."""
        ins1 = make_insight(
            id='hyp-1', content='caching layer design',
            entities=['caching-layer'])
        ins2 = make_insight(
            id='hyp-2', content='caching layer implementation',
            entities=['caching-layer'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)
        count = create_entity_edges(tmp_db, ins2)
        assert count >= 2


class TestEntityIdfWeightedEdges:
    """Entity edges carry IDF-based weights, not fixed 1.0."""

    def test_rare_entity_high_weight(self, tmp_db):
        """Rare entity shared by 2 of many insights gets high weight."""
        for i in range(10):
            insert_insight(tmp_db, make_insight(
                id=f'idf-{i}', content=f'content {i}',
                entities=['Python']))
        rare = make_insight(
            id='idf-rare', content='rare thing',
            entities=['UniqueEntity'])
        insert_insight(tmp_db, rare)
        target = make_insight(
            id='idf-rare2', content='also rare',
            entities=['UniqueEntity', 'Python'])
        insert_insight(tmp_db, target)

        create_entity_edges(tmp_db, target)

        edges = get_edges_by_node_and_type(tmp_db, 'idf-rare2', 'entity')
        rare_edges = [
            e for e in edges
            if e.metadata.get('entity') == 'UniqueEntity'
            ]
        common_edges = [
            e for e in edges
            if e.metadata.get('entity') == 'Python'
            ]
        assert len(rare_edges) > 0
        assert len(common_edges) > 0
        assert rare_edges[0].weight > common_edges[0].weight


# --- BFS ---


class TestBFSBasic:
    """A→B→C graph: B at hop 1, C at hop 2, D disconnected."""

    def test_bfs_basic(self, tmp_db):
        """BFS from A reaches B (hop 1) and C (hop 2) but not D."""
        insert_insight(tmp_db, make_insight(id='a', content='node A'))
        insert_insight(tmp_db, make_insight(id='b', content='node B'))
        insert_insight(tmp_db, make_insight(id='c', content='node C'))
        insert_insight(tmp_db, make_insight(id='d', content='node D'))

        insert_edge(tmp_db, make_edge(
            source_id='a', target_id='b',
            edge_type='semantic', weight=1.0))
        insert_edge(tmp_db, make_edge(
            source_id='b', target_id='c',
            edge_type='semantic', weight=1.0))

        result = bfs(tmp_db, 'a', BFSOptions(max_depth=3, max_nodes=0))
        ids = {r['insight'].id for r in result}
        assert 'b' in ids
        assert 'c' in ids
        assert 'd' not in ids

        hops = {r['insight'].id: r['hop'] for r in result}
        assert hops['b'] == 1
        assert hops['c'] == 2


class TestBFSMaxHops:
    """maxHops=1 stops at direct neighbors."""

    def test_bfs_max_hops(self, tmp_db):
        """BFS with max_depth=1 finds B but not C (2 hops away)."""
        insert_insight(tmp_db, make_insight(id='h-a', content='node A'))
        insert_insight(tmp_db, make_insight(id='h-b', content='node B'))
        insert_insight(tmp_db, make_insight(id='h-c', content='node C'))

        insert_edge(tmp_db, make_edge(
            source_id='h-a', target_id='h-b',
            edge_type='semantic', weight=1.0))
        insert_edge(tmp_db, make_edge(
            source_id='h-b', target_id='h-c',
            edge_type='semantic', weight=1.0))

        result = bfs(tmp_db, 'h-a', BFSOptions(max_depth=1, max_nodes=0))
        ids = {r['insight'].id for r in result}
        assert 'h-b' in ids
        assert 'h-c' not in ids


class TestBFSMaxNodes:
    """max_nodes caps the result count."""

    def test_bfs_max_nodes(self, tmp_db):
        """BFS with max_nodes=1 returns at most 1 node."""
        insert_insight(tmp_db, make_insight(id='m-a', content='node A'))
        insert_insight(tmp_db, make_insight(id='m-b', content='node B'))
        insert_insight(tmp_db, make_insight(id='m-c', content='node C'))

        insert_edge(tmp_db, make_edge(
            source_id='m-a', target_id='m-b',
            edge_type='semantic', weight=1.0))
        insert_edge(tmp_db, make_edge(
            source_id='m-a', target_id='m-c',
            edge_type='semantic', weight=1.0))

        result = bfs(tmp_db, 'm-a', BFSOptions(max_depth=3, max_nodes=1))
        assert len(result) == 1


class TestBFSSkipsDeleted:
    """Soft-deleted nodes excluded from BFS results."""

    def test_bfs_skips_deleted(self, tmp_db):
        """BFS does not return soft-deleted neighbor."""
        insert_insight(tmp_db, make_insight(id='sd-a', content='node A'))
        insert_insight(tmp_db, make_insight(id='sd-b', content='node B'))
        insert_insight(tmp_db, make_insight(id='sd-c', content='node C'))

        insert_edge(tmp_db, make_edge(
            source_id='sd-a', target_id='sd-b',
            edge_type='semantic', weight=1.0))
        insert_edge(tmp_db, make_edge(
            source_id='sd-a', target_id='sd-c',
            edge_type='semantic', weight=1.0))

        soft_delete_insight(tmp_db, 'sd-b')

        result = bfs(tmp_db, 'sd-a', BFSOptions(max_depth=3, max_nodes=0))
        ids = {r['insight'].id for r in result}
        assert 'sd-b' not in ids
        assert 'sd-c' in ids


# --- Semantic ---


class TestSemanticEdgesHighCosine:
    """Similar embeddings create semantic edges."""

    def test_semantic_edges_high_cosine(self, tmp_db):
        """Nearly identical vectors exceed AUTO threshold and create edges."""
        ins1 = make_insight(id='sv-1', content='vector one')
        ins2 = make_insight(id='sv-2', content='vector two')
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.99, 0.01, 0.0, 0.0]
        update_embedding(tmp_db, 'sv-1', serialize_vector(vec1))
        update_embedding(tmp_db, 'sv-2', serialize_vector(vec2))

        cache = {'sv-1': vec1, 'sv-2': vec2}
        count = create_semantic_edges(tmp_db, ins1, embed_cache=cache)
        assert count >= 2

        edges = get_edges_by_node_and_type(tmp_db, 'sv-1', 'semantic')
        assert len(edges) >= 1


class TestSemanticEdgesLowCosine:
    """Orthogonal embeddings produce 0 semantic edges."""

    def test_semantic_edges_low_cosine(self, tmp_db):
        """Orthogonal vectors yield cosine ~0, no edges created."""
        ins1 = make_insight(id='sl-1', content='vector one')
        ins2 = make_insight(id='sl-2', content='vector two')
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        update_embedding(tmp_db, 'sl-1', serialize_vector(vec1))
        update_embedding(tmp_db, 'sl-2', serialize_vector(vec2))

        cache = {'sl-1': vec1, 'sl-2': vec2}
        count = create_semantic_edges(tmp_db, ins1, embed_cache=cache)
        assert count == 0


class TestSemanticEdgesNoEmbedding:
    """No embeddings in cache means 0 semantic edges."""

    def test_semantic_edges_no_embedding(self, tmp_db):
        """Insight without embedding in cache produces 0 edges."""
        ins = make_insight(id='sne-1', content='no embedding')
        insert_insight(tmp_db, ins)

        count = create_semantic_edges(tmp_db, ins, embed_cache=None)
        assert count == 0


# --- Engine ---


class TestFastEdgesEngine:
    """fast_edges generates temporal + entity + causal edges (no semantic)."""

    def test_fast_edges_creates_edges(self, tmp_db):
        """Two insights with shared entity and same source create edges."""
        now = datetime.now(timezone.utc)
        ins1 = make_insight(
            id='eng-1', content='Go uses SQLite for storage',
            source='proj', entities=['Go', 'SQLite'],
            created_at=now - timedelta(hours=1))
        ins2 = make_insight(
            id='eng-2', content='Go concurrency patterns',
            source='proj', entities=['Go'],
            created_at=now)
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        result = fast_edges(tmp_db, ins2)
        assert set(result.keys()) == {'temporal', 'entity'}
        assert result['temporal'] >= 2
        assert result['entity'] >= 1


# --- Build Embed Cache ---


class TestBuildEmbedCache:
    """Loads embedding vectors from DB into dict."""

    def test_build_embed_cache(self, tmp_db):
        """Stored embeddings are deserialized into the cache dict."""
        ins1 = make_insight(id='bc-1', content='first')
        ins2 = make_insight(id='bc-2', content='second')
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]
        update_embedding(tmp_db, 'bc-1', serialize_vector(vec1))
        update_embedding(tmp_db, 'bc-2', serialize_vector(vec2))

        cache = build_embed_cache(tmp_db)
        assert cache is not None
        assert 'bc-1' in cache
        assert 'bc-2' in cache
        assert len(cache['bc-1']) == 3
        assert abs(cache['bc-1'][0] - 1.0) < 0.001


class TestBuildEmbedCacheEmpty:
    """No embeddings returns None."""

    def test_build_embed_cache_empty(self, tmp_db):
        """Empty DB (no embeddings) returns None from build_embed_cache."""
        cache = build_embed_cache(tmp_db)
        assert cache is None


# --- Edge-worthy filtering ---


class TestEdgeWorthyFiltering:
    """Entity edges created for legitimate shared entities."""

    def test_tech_dict_allowed(self, tmp_db):
        """Two insights sharing 'Redis' produce entity edges."""
        ins1 = make_insight(
            id='ew-3', content='Redis cache',
            entities=['Redis'])
        ins2 = make_insight(
            id='ew-4', content='Redis session store',
            entities=['Redis'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)
        count = create_entity_edges(tmp_db, ins2)
        assert count >= 2

    def test_camelcase_allowed(self, tmp_db):
        """Two insights sharing 'DataProcessor' produce entity edges."""
        ins1 = make_insight(
            id='ew-5', content='DataProcessor init',
            entities=['DataProcessor'])
        ins2 = make_insight(
            id='ew-6', content='DataProcessor run',
            entities=['DataProcessor'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)
        count = create_entity_edges(tmp_db, ins2)
        assert count >= 2

    def test_url_allowed(self, tmp_db):
        """Two insights sharing a URL produce entity edges."""
        url = 'https://example.com/api'
        ins1 = make_insight(
            id='ew-7', content='see docs', entities=[url])
        ins2 = make_insight(
            id='ew-8', content='api ref', entities=[url])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)
        count = create_entity_edges(tmp_db, ins2)
        assert count >= 2


# --- Temporal window ---


class TestTemporalOutsideWindow:
    """Insight outside the 4h window gets no proximity edges."""

    def test_outside_window_no_proximity(self, tmp_db):
        """5-hour-old insight produces 0 proximity edges."""
        now = datetime.now(timezone.utc)
        old = make_insight(
            id='tw-1', content='old', source='p',
            created_at=now - timedelta(hours=5))
        new = make_insight(
            id='tw-2', content='new', source='q',
            created_at=now)
        insert_insight(tmp_db, old)
        insert_insight(tmp_db, new)
        create_temporal_edge(tmp_db, new)
        proximity_edges = [
            e for e in get_edges_by_node_and_type(
                tmp_db, 'tw-2', 'temporal')
            if e.metadata.get('sub_type') == 'proximity']
        assert len(proximity_edges) == 0
