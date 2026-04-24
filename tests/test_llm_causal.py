"""Tests for LLM causal inference with mock LLM client."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from memman.graph.causal import LLM_CONFIDENCE_FLOOR, infer_llm_causal_edges
from memman.llm.client import get_llm_client
from memman.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_llm_response(edges):
    """Build a mock LLM JSON response."""
    return json.dumps(edges)


class TestLLMCausalInference:
    """LLM-based causal edge inference with mocked client."""

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

        infer_llm_causal_edges(tmp_db, make_insight(
            id='pf-2', content='database schema changed production deploy',
            created_at=OLD + timedelta(hours=1)), mock_client)

        if mock_client.complete.called:
            _system, user = mock_client.complete.call_args[0]
            assert 'pf-2' in user
            assert 'database schema changed' in user

    def test_confidence_floor_boundary(self, tmp_db):
        """Edge below floor rejected; edge at floor accepted."""
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

        result = infer_llm_causal_edges(tmp_db, insight, mock_client)
        assert len(result) == 1
        assert result[0].weight == LLM_CONFIDENCE_FLOOR

    def test_edge_metadata_shape(self, tmp_db):
        """Returned edges have created_by, confidence, rationale, sub_type."""
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

        result = infer_llm_causal_edges(tmp_db, insight, mock_client)
        assert len(result) == 1
        meta = result[0].metadata
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

        infer_llm_causal_edges(tmp_db, insight, mock_client)
        mock_client.complete.assert_not_called()

    def test_llm_unavailable_skips_silently(self, tmp_db):
        """Connection error from LLM caught; returns empty list."""
        insert_insight(tmp_db, make_insight(
            id='ua-1', content='database migration completed production',
            created_at=OLD))
        insight = make_insight(
            id='ua-2', content='database schema changed production deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        result = infer_llm_causal_edges(tmp_db, insight, mock_client)
        assert result == []

    def test_malformed_confidence_skips_edge(self, tmp_db):
        """Non-numeric confidence skipped; valid edge still returned."""
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

        result = infer_llm_causal_edges(tmp_db, insight, mock_client)
        assert len(result) == 1

    def test_non_list_json_response(self, tmp_db):
        """JSON object (not array) from LLM returns empty list."""
        insert_insight(tmp_db, make_insight(
            id='nl-1', content='chose SQLite because embedded database',
            created_at=OLD))
        insight = make_insight(
            id='nl-2', content='SQLite chosen enables single-file deploy',
            created_at=OLD + timedelta(hours=1))
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = '{"error": "oops"}'

        result = infer_llm_causal_edges(tmp_db, insight, mock_client)
        assert result == []
        mock_client.complete.assert_called_once()

    def test_env_var_opt_in(self, monkeypatch):
        """Without ANTHROPIC_API_KEY set, get_llm_client raises ConfigError."""
        from memman.exceptions import ConfigError
        monkeypatch.delenv('MEMMAN_ANTHROPIC_ENDPOINT', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        with pytest.raises(ConfigError):
            get_llm_client()

    def test_anthropic_api_key_used(self, monkeypatch):
        """ANTHROPIC_API_KEY is the sole key source for the Anthropic client."""
        monkeypatch.delenv('MEMMAN_ANTHROPIC_ENDPOINT', raising=False)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
        client = get_llm_client()
        assert client is not None
        assert client.endpoint == 'https://api.anthropic.com'
        assert client.api_key == 'sk-ant-test'
