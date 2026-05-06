"""Tests for database quality fixes: entity normalization, constants."""

from datetime import datetime, timezone

from memman.graph.enrichment import enrich_with_llm
from memman.graph.entity import create_entity_edges, normalize_entity
from memman.graph.temporal import create_temporal_edge
from memman.store.edge import count_insights_with_entity
from memman.store.edge import find_insights_with_entity
from memman.store.edge import get_edges_by_node_and_type
from memman.store.node import insert_insight, update_entities
from tests.conftest import make_insight

# --- Fix 1: ANCHOR_TOP_K ---


# --- Fix 2: Entity Normalization ---


class TestNormalizeEntity:
    """normalize_entity lowercases and strips."""

    def test_lowercase(self):
        """PascalCase becomes lowercase."""
        assert normalize_entity('Thesis') == 'thesis'

    def test_strip(self):
        """Whitespace is trimmed."""
        assert normalize_entity('  SOFR  ') == 'sofr'

    def test_mixed(self):
        """Combined case and whitespace."""
        assert normalize_entity(' Credit P&L ') == 'credit p&l'

    def test_empty(self):
        """Empty string stays empty."""
        assert normalize_entity('') == ''

    def test_already_normalized(self):
        """Already-lowercase string unchanged."""
        assert normalize_entity('pnl_opp') == 'pnl_opp'


class TestEntitySqlCaseInsensitive:
    """Entity SQL queries match case-insensitively."""

    def test_find_entity_case_insensitive(self, tmp_db):
        """find_insights_with_entity matches across case variants."""
        insert_insight(tmp_db, make_insight(
            id='ci-1', content='uses Thesis',
            entities=['Thesis']))
        insert_insight(tmp_db, make_insight(
            id='ci-2', content='about thesis',
            entities=['thesis']))

        ids = find_insights_with_entity(tmp_db, 'THESIS', 'ci-1', 10)
        assert 'ci-2' in ids

    def test_count_entity_case_insensitive(self, tmp_db):
        """count_insights_with_entity counts across case variants."""
        insert_insight(tmp_db, make_insight(
            id='cc-1', content='a', entities=['SOFR']))
        insert_insight(tmp_db, make_insight(
            id='cc-2', content='b', entities=['sofr']))
        insert_insight(tmp_db, make_insight(
            id='cc-3', content='c', entities=['Sofr']))

        cnt = count_insights_with_entity(tmp_db, 'sofr', 'cc-1')
        assert cnt == 2


class TestEntityEdgeCaseInsensitive:
    """Entity edges connect insights with case-variant entities."""

    def test_entity_edge_across_cases(self, tmp_db, tmp_backend):
        """Insights with 'Thesis' and 'thesis' get connected."""
        insert_insight(tmp_db, make_insight(
            id='ee-1', content='about Thesis',
            entities=['Thesis']))
        ins2 = make_insight(
            id='ee-2', content='thesis stuff',
            entities=['thesis'])
        insert_insight(tmp_db, ins2)

        count = create_entity_edges(tmp_backend, ins2)
        assert count >= 2

        edges = get_edges_by_node_and_type(tmp_db, 'ee-2', 'entity')
        targets = {
            e.target_id if e.source_id == 'ee-2' else e.source_id
            for e in edges}
        assert 'ee-1' in targets


class TestUpdateEntitiesDedup:
    """update_entities deduplicates case variants."""

    def test_dedup_case_variants(self, tmp_db):
        """Storing ['Thesis', 'thesis'] keeps only the first."""
        insert_insight(tmp_db, make_insight(
            id='ud-1', content='test'))
        update_entities(tmp_db, 'ud-1', ['Thesis', 'thesis', 'THESIS'])

        from memman.store.node import get_insight_by_id
        ins = get_insight_by_id(tmp_db, 'ud-1')
        assert len(ins.entities) == 1
        assert ins.entities[0] == 'Thesis'

    def test_dedup_preserves_distinct(self, tmp_db):
        """Distinct entities are all kept."""
        insert_insight(tmp_db, make_insight(
            id='ud-2', content='test'))
        update_entities(tmp_db, 'ud-2', ['Python', 'Go', 'Rust'])

        from memman.store.node import get_insight_by_id
        ins = get_insight_by_id(tmp_db, 'ud-2')
        assert len(ins.entities) == 3


class TestEnrichmentMergeCaseInsensitive:
    """Enrichment merge prevents case-variant duplicates."""

    def test_merge_skips_case_variant(self):
        """LLM entity 'thesis' is not added when 'Thesis' exists."""
        import json
        from unittest.mock import MagicMock

        insight = make_insight(
            id='em-ci-1', content='Thesis analysis',
            entities=['Thesis'],
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))

        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            'entities': ['thesis', 'analysis'],
            'keywords': ['thesis'],
            'summary': 'test',
            'semantic_facts': ['test'],
            })

        result = enrich_with_llm(insight, mock_client)
        lower_entities = [e.lower() for e in result['entities']]
        assert lower_entities.count('thesis') == 1
        assert 'analysis' in lower_entities


# --- Fix 5: Temporal Constants ---


class TestTemporalConstants:
    """MAX_PROXIMITY_EDGES cap is enforced."""

    def test_proximity_capped_at_5(self, tmp_db, tmp_backend):
        """Exactly 5 proximity neighbors connected when 8 candidates qualify.

        The cap is meaningful only if more than 5 eligible neighbors
        exist; the assertion has to verify the cap *fires*, not just
        that it isn't exceeded. With 8 same-window candidates and
        MAX_PROXIMITY_EDGES=5, the function picks 5.
        """
        for i in range(8):
            insert_insight(tmp_db, make_insight(
                id=f'pc-{i}', content=f'insight {i}', source='other'))

        current = make_insight(
            id='pc-current', content='current', source='proj-x')
        insert_insight(tmp_db, current)

        create_temporal_edge(tmp_backend, current)

        edges = get_edges_by_node_and_type(
            tmp_db, 'pc-current', 'temporal')
        proximity = [
            e for e in edges
            if e.metadata.get('sub_type') == 'proximity'
            ]
        unique_neighbors = {
            e.target_id if e.source_id == 'pc-current' else e.source_id
            for e in proximity}
        assert len(unique_neighbors) == 5, (
            f'expected exactly 5 capped neighbors out of 8 candidates;'
            f' got {len(unique_neighbors)}: {unique_neighbors}')
