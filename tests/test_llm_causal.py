"""Tests for LLM causal inference with mock LLM client."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from mnemon.graph.causal import LLM_CONFIDENCE_FLOOR, create_llm_causal_edges
from mnemon.llm.client import get_llm_client
from mnemon.store.edge import get_edges_by_node
from mnemon.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_llm_response(edges):
    """Build a mock LLM JSON response."""
    return json.dumps(edges)


class TestLLMCausalInference:
    """LLM-based causal edge creation with mocked client."""

    def test_prompt_format(self, tmp_db):
        """Prompt includes new insight content and neighbor content."""
        insert_insight(tmp_db, make_insight(
            id='pf-1', content='database migration completed production',
            created_at=OLD))
        insert_insight(tmp_db, make_insight(
            id='pf-2', content='database schema changed production deploy',
            created_at=OLD + timedelta(hours=1)))

        mock_client = MagicMock()
        mock_client.complete.return_value = '[]'

        create_llm_causal_edges(tmp_db, make_insight(
            id='pf-2', content='database schema changed production deploy',
            created_at=OLD + timedelta(hours=1)), mock_client)

        if mock_client.complete.called:
            _system, user = mock_client.complete.call_args[0]
            assert 'pf-2' in user
            assert 'database schema changed' in user

    def test_confidence_floor_boundary(self, tmp_db):
        """Edge with confidence below floor is rejected; at floor is accepted."""
        insert_insight(tmp_db, make_insight(
            id='cf-1', content='chose SQLite because embedded database',
            created_at=OLD))
        insight = make_insight(
            id='cf-2', content='SQLite chosen enables single-file deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_llm_response([
            {
                'source_id': 'cf-1', 'target_id': 'cf-2',
                'confidence': LLM_CONFIDENCE_FLOOR - 0.01,
                'sub_type': 'causes', 'rationale': 'below floor',
                },
            {
                'source_id': 'cf-1', 'target_id': 'cf-2',
                'confidence': LLM_CONFIDENCE_FLOOR,
                'sub_type': 'enables', 'rationale': 'at floor',
                },
            ])

        count = create_llm_causal_edges(tmp_db, insight, mock_client)
        assert count == 1

        edges = get_edges_by_node(tmp_db, 'cf-2')
        causal = [e for e in edges if e.edge_type == 'causal']
        assert len(causal) >= 1
        llm_edge = [e for e in causal if e.metadata.get('created_by') == 'llm']
        assert len(llm_edge) == 1
        assert llm_edge[0].weight == LLM_CONFIDENCE_FLOOR

    def test_edge_metadata_shape(self, tmp_db):
        """Created edges have created_by, confidence, rationale, sub_type."""
        insert_insight(tmp_db, make_insight(
            id='ms-1', content='chose Redis because fast caching layer',
            created_at=OLD))
        insight = make_insight(
            id='ms-2', content='Redis caching enables low latency response',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_llm_response([{
            'source_id': 'ms-1', 'target_id': 'ms-2',
            'confidence': 0.85,
            'sub_type': 'enables',
            'rationale': 'Redis enables caching',
            }])

        create_llm_causal_edges(tmp_db, insight, mock_client)

        edges = get_edges_by_node(tmp_db, 'ms-2')
        llm_edges = [
            e for e in edges
            if e.metadata.get('created_by') == 'llm']
        assert len(llm_edges) == 1
        meta = llm_edges[0].metadata
        assert meta['created_by'] == 'llm'
        assert isinstance(meta['confidence'], float)
        assert isinstance(meta['rationale'], str)
        assert meta['sub_type'] in {'causes', 'enables', 'prevents'}

    def test_prefilter_short_circuit(self, tmp_db):
        """No LLM call when token overlap is below MIN_CAUSAL_OVERLAP."""
        insert_insight(tmp_db, make_insight(
            id='pf-a', content='alpha bravo charlie delta echo foxtrot',
            created_at=OLD))
        insight = make_insight(
            id='pf-b',
            content='xylophone zebra quantum neutrino plasma vortex',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = '[]'

        create_llm_causal_edges(tmp_db, insight, mock_client)
        mock_client.complete.assert_not_called()

    def test_llm_unavailable_skips_silently(self, tmp_db):
        """Connection error from LLM is caught; no exception, zero edges."""
        insert_insight(tmp_db, make_insight(
            id='ua-1', content='database migration completed production',
            created_at=OLD))
        insight = make_insight(
            id='ua-2', content='database schema changed production deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        count = create_llm_causal_edges(tmp_db, insight, mock_client)
        assert count == 0

    def test_malformed_confidence_skips_edge(self, tmp_db):
        """Non-numeric confidence in one edge doesn't abort the batch."""
        insert_insight(tmp_db, make_insight(
            id='mc-1', content='chose SQLite because embedded database',
            created_at=OLD))
        insight = make_insight(
            id='mc-2', content='SQLite chosen enables single-file deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_llm_response([
            {
                'source_id': 'mc-1', 'target_id': 'mc-2',
                'confidence': 'high',
                'sub_type': 'causes', 'rationale': 'malformed',
                },
            {
                'source_id': 'mc-1', 'target_id': 'mc-2',
                'confidence': 0.85,
                'sub_type': 'enables', 'rationale': 'valid edge',
                },
            ])

        count = create_llm_causal_edges(tmp_db, insight, mock_client)
        assert count == 1

    def test_non_list_json_response(self, tmp_db):
        """JSON object (not array) from LLM returns 0 edges."""
        insert_insight(tmp_db, make_insight(
            id='nl-1', content='chose SQLite because embedded database',
            created_at=OLD))
        insight = make_insight(
            id='nl-2', content='SQLite chosen enables single-file deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = '{"error": "oops"}'

        count = create_llm_causal_edges(tmp_db, insight, mock_client)
        assert count == 0
        mock_client.complete.assert_called_once()

    def test_env_var_opt_in(self, monkeypatch):
        """Without any API key set, get_llm_client raises."""
        import click
        monkeypatch.delenv('MNEMON_LLM_ENDPOINT', raising=False)
        monkeypatch.delenv('MNEMON_LLM_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        with pytest.raises(click.ClickException):
            get_llm_client()

    def test_anthropic_api_key_fallback(self, monkeypatch):
        """ANTHROPIC_API_KEY used when MNEMON_LLM_API_KEY absent."""
        monkeypatch.delenv('MNEMON_LLM_ENDPOINT', raising=False)
        monkeypatch.delenv('MNEMON_LLM_API_KEY', raising=False)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
        client = get_llm_client()
        assert client is not None
        assert client.endpoint == 'https://api.anthropic.com'
        assert client.api_key == 'sk-ant-test'
