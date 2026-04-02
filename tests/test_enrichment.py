"""Tests for LLM-based insight enrichment."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.graph.engine import link_pending
from memman.graph.enrichment import build_enriched_text, enrich_with_llm
from memman.graph.semantic import build_embed_cache
from memman.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_enrichment_response(
        entities=None, keywords=None,
        summary='test summary', facts=None) -> str:
    """Build a mock LLM enrichment JSON response."""
    return json.dumps({
        'entities': entities or ['Python', 'FastAPI'],
        'keywords': keywords or ['web', 'framework'],
        'summary': summary,
        'semantic_facts': facts or ['Python is used'],
        })


def _read_enrichment_columns(db, insight_id: str) -> dict:
    """Read enrichment columns directly from DB."""
    row = db._conn.execute(
        'SELECT keywords, summary, semantic_facts, entities'
        ' FROM insights WHERE id = ?',
        (insight_id,)).fetchone()
    if row is None:
        return {}
    return {
        'keywords': json.loads(row[0]) if row[0] else None,
        'summary': row[1],
        'semantic_facts': json.loads(row[2]) if row[2] else None,
        'entities': json.loads(row[3]) if row[3] else None,
        }


class TestEnrichWithLLM:
    """LLM enrichment extraction with mocked client."""

    def test_happy_path(self):
        """Valid LLM response returns all enrichment fields."""
        insight = make_insight(
            id='hp-1', content='Python web framework comparison',
            entities=['Python'], created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            entities=['Python', 'FastAPI', 'Django'],
            keywords=['web', 'framework', 'comparison'],
            summary='Comparing Python web frameworks',
            facts=['FastAPI is faster', 'Django is mature'])

        result = enrich_with_llm(insight, mock_client)

        assert result['keywords'] == ['web', 'framework', 'comparison']
        assert result['summary'] == 'Comparing Python web frameworks'
        assert result['semantic_facts'] == [
            'FastAPI is faster', 'Django is mature']
        assert 'Python' in result['entities']
        assert 'FastAPI' in result['entities']
        assert 'Django' in result['entities']

    def test_llm_unavailable_returns_empty(self):
        """ConnectionError from LLM returns empty dict, no crash."""
        insight = make_insight(
            id='ua-1', content='test content', created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        result = enrich_with_llm(insight, mock_client)
        assert result == {}

    def test_malformed_json_returns_empty(self):
        """Unparseable LLM output returns empty dict."""
        insight = make_insight(
            id='mj-1', content='test content', created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.return_value = 'not json at all'

        result = enrich_with_llm(insight, mock_client)
        assert result == {}

    def test_llm_client_none_skips(self, tmp_db):
        """link_pending with llm_client=None skips enrichment."""
        insight = make_insight(
            id='nc-1', content='test content', created_at=OLD)
        insert_insight(tmp_db, insight)

        count = link_pending(tmp_db, llm_client=None, max_batch=1)
        assert count == 1

        cols = _read_enrichment_columns(tmp_db, 'nc-1')
        assert cols['keywords'] is None
        assert cols['summary'] is None

    def test_entity_merge_deduplicates(self):
        """LLM entities merge with existing without duplicates."""
        insight = make_insight(
            id='em-1', content='Python FastAPI web',
            entities=['Python'], created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            entities=['Python', 'FastAPI'])

        result = enrich_with_llm(insight, mock_client)

        assert 'Python' in result['entities']
        assert 'FastAPI' in result['entities']
        assert result['entities'].count('Python') == 1

    def test_entity_merge_empty_llm(self):
        """Empty LLM entities preserve existing regex entities."""
        insight = make_insight(
            id='ee-1', content='test content',
            entities=['Go'], created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            entities=[])

        result = enrich_with_llm(insight, mock_client)
        assert 'Go' in result['entities']

    def test_entity_merge_malformed(self):
        """Malformed entities field does not clobber existing."""
        insight = make_insight(
            id='ef-1', content='test content',
            entities=['Docker'], created_at=OLD)

        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            'entities': 'not a list',
            'keywords': ['test'],
            'summary': 'test',
            'semantic_facts': ['test'],
            })

        result = enrich_with_llm(insight, mock_client)
        assert 'Docker' in result['entities']


class TestReEmbed:
    """Re-embedding with keyword-enriched text."""

    def test_reembed_uses_keywords(self, tmp_db):
        """embed_client.embed() called with keyword-appended text."""
        insight = make_insight(
            id='re-1', content='Python web framework',
            created_at=OLD)
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response(
            keywords=['web', 'framework'])

        mock_embed = MagicMock()
        mock_embed.available.return_value = True
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]

        embed_cache = build_embed_cache(tmp_db) or {}
        link_pending(
            tmp_db, embed_cache=embed_cache,
            llm_client=mock_llm, embed_client=mock_embed,
            max_batch=1)

        mock_embed.embed.assert_called_once()
        call_text = mock_embed.embed.call_args[0][0]
        assert '[KEYWORDS: web framework]' in call_text
        assert 'Python web framework' in call_text

    def test_reembed_skipped_when_no_embed_client(self, tmp_db):
        """No embed_client means no re-embed attempt."""
        insight = make_insight(
            id='rs-1', content='test content', created_at=OLD)
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response()

        link_pending(
            tmp_db, llm_client=mock_llm, embed_client=None,
            max_batch=1)

        cols = _read_enrichment_columns(tmp_db, 'rs-1')
        assert cols['keywords'] is not None

    def test_embed_failure_still_stamps_linked_at(self, tmp_db):
        """Embed crash doesn't prevent linked_at stamp."""
        insight = make_insight(
            id='ef-1', content='test content', created_at=OLD)
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response()

        mock_embed = MagicMock()
        mock_embed.available.return_value = True
        mock_embed.embed.side_effect = RuntimeError('embed crashed')

        link_pending(
            tmp_db, llm_client=mock_llm, embed_client=mock_embed,
            max_batch=1)

        row = tmp_db._conn.execute(
            'SELECT linked_at FROM insights WHERE id = ?',
            ('ef-1',)).fetchone()
        assert row[0] is not None

        cols = _read_enrichment_columns(tmp_db, 'ef-1')
        assert cols['keywords'] is not None


class TestEnrichmentPurity:
    """enrich_with_llm should be pure (no DB writes)."""

    def test_enrichment_does_not_write_db_directly(self, tmp_db):
        """enrich_with_llm returns data without writing to DB."""
        insight = make_insight(
            id='pw-1', content='purity test content',
            entities=['Python'], created_at=OLD)
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=['test'], summary='test summary',
            facts=['fact one'])

        result = enrich_with_llm(insight, mock_client)

        assert result['keywords'] == ['test']
        assert result['summary'] == 'test summary'

        row = tmp_db._conn.execute(
            'SELECT keywords, summary, semantic_facts'
            ' FROM insights WHERE id = ?',
            ('pw-1',)).fetchone()
        assert row[0] is None, (
            'enrich_with_llm should not write keywords to DB')
        assert row[1] is None, (
            'enrich_with_llm should not write summary to DB')
        assert row[2] is None, (
            'enrich_with_llm should not write semantic_facts to DB')

    def test_markdown_fence_json_parsed(self):
        """LLM response wrapped in ```json fence is parsed correctly."""
        insight = make_insight(
            id='mf-1', content='fence test content', created_at=OLD)

        fenced_json = '```json\n' + _make_enrichment_response(
            keywords=['fenced'], summary='fenced summary') + '\n```'

        mock_client = MagicMock()
        mock_client.complete.return_value = fenced_json

        result = enrich_with_llm(insight, mock_client)
        assert result['keywords'] == ['fenced']
        assert result['summary'] == 'fenced summary'


class TestBuildEnrichedText:
    """build_enriched_text utility."""

    def test_appends_keywords(self):
        """Keywords are appended in bracket format."""
        result = build_enriched_text('hello world', ['foo', 'bar'])
        assert result == 'hello world [KEYWORDS: foo bar]'

    def test_empty_keywords_returns_content(self):
        """No keywords means original content returned."""
        result = build_enriched_text('hello world', [])
        assert result == 'hello world'
