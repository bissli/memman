"""Tests for LLM-based insight enrichment."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.graph.engine import link_pending
from memman.graph.enrichment import build_enriched_text, enrich_with_llm
from memman.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_enrichment_response(keywords=None, summary='test summary') -> str:
    """Build a mock LLM enrichment JSON response."""
    return json.dumps({
        'keywords': keywords or ['web', 'framework'],
        'summary': summary,
        })


def _content_containing(*names) -> str:
    """Content that names each of `names`."""
    return 'cap body ' + ' '.join(names)


def _read_enrichment_columns(db, insight_id: str) -> dict:
    """Read enrichment columns directly from DB."""
    row = db._conn.execute(
        'SELECT keywords, summary, entities'
        ' FROM insights WHERE id = ?',
        (insight_id,)).fetchone()
    if row is None:
        return {}
    return {
        'keywords': json.loads(row[0]) if row[0] else None,
        'summary': row[1],
        'entities': json.loads(row[2]) if row[2] else None,
        }


class TestEnrichWithLLM:
    """LLM enrichment extraction with mocked client."""

    def test_happy_path(self):
        """Valid LLM response returns all enrichment fields."""
        insight = make_insight(
            id='hp-1',
            content=(
                'Python has several mature web frameworks: FastAPI for '
                'async-first APIs, Django for full-stack with batteries '
                'included, and Flask for minimal microframeworks. Each '
                'targets different sweet spots in the deployment surface.'),
            entities=['Python'])

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=['web', 'framework', 'comparison'],
            summary='Comparing Python web frameworks')

        result = enrich_with_llm(insight, mock_client)

        assert result['keywords'] == ['web', 'framework', 'comparison']
        assert result['summary'] == 'Comparing Python web frameworks'

    def test_enrichment_returns_only_keywords_and_summary(self):
        """Verify the result carries no `entities` or `semantic_facts` key.

        Mutation: `enrich_with_llm` copying the model's `entities`
            or `semantic_facts` into its result dict, so a caller
            reads fields outside the enrichment contract.
        Oracle: the LLM body below carries its own `entities` and
            `semantic_facts`, so a surviving key proves the drop
            never happened, not that the mock omitted them.
        """
        insight = make_insight(
            id='ks-1', content='Redis backs the session cache',
            entities=['caller-tag'])

        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            'entities': ['Redis'],
            'keywords': ['redis', 'cache'],
            'summary': 'Redis backs the session cache',
            'semantic_facts': ['Redis backs the session cache'],
            })

        result = enrich_with_llm(insight, mock_client)

        assert set(result) == {'keywords', 'summary'}

    def test_llm_unavailable_returns_empty(self):
        """ConnectionError from LLM returns empty dict, no crash."""
        insight = make_insight(
            id='ua-1', content='test content')

        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        result = enrich_with_llm(insight, mock_client)
        assert result == {}

    def test_undecodable_body_returns_empty_fields(self):
        """Verify a body that decodes on neither draw yields empty fields.

        Mutation: returning `{}` here as for a failed call, which
            leaves the row unstamped, so the stranded-row sweep
            re-enriches it on every drain and bills the call each time.
        Oracle: the literal empty-valued dict, against the `{}` a
            failed call returns in `test_llm_unavailable_returns_empty`.
        """
        insight = make_insight(
            id='mj-1', content='test content')

        mock_client = MagicMock()
        mock_client.complete.return_value = 'not json at all'

        result = enrich_with_llm(insight, mock_client)
        assert result == {'keywords': [], 'summary': ''}

    def test_llm_failure_logged_at_warning(self, caplog):
        """An LLM exception during enrichment is logged at WARNING."""
        import logging
        insight = make_insight(id='warn-1', content='test content')
        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        with caplog.at_level(logging.WARNING, logger='memman'):
            result = enrich_with_llm(insight, mock_client)

        assert result == {}
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_parse_error_logged_at_warning(self, caplog):
        """Verify an undecodable (e.g. truncated) body logs at WARNING.

        Mutation: logging the decode failure at debug, which hides a
            row stamped enriched with no keywords from the default log.
        Oracle: the captured record levels.
        """
        import logging
        insight = make_insight(id='warn-2', content='test content')
        mock_client = MagicMock()
        mock_client.complete.return_value = 'not json at all'

        with caplog.at_level(logging.WARNING, logger='memman'):
            enrich_with_llm(insight, mock_client)

        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_unresolvable_metadata_role_still_links(
            self, tmp_db, tmp_backend, monkeypatch):
        """An unavailable metadata client links the row unenriched.

        Mutation: resolving `slow` before the loop instead of
            at the row that needs it, which turns a missing credential
            into a raise and leaves a store with nothing to enrich
            unable to relink at all.
        Oracle: the enrichment columns, which stay null, beside the
            processed count, which does not.
        """
        from memman.graph import engine as engine_mod

        def _unavailable(*args, **kwargs):
            raise RuntimeError('no LLM credential')

        monkeypatch.setattr(engine_mod, 'get_llm_client', _unavailable)
        insert_insight(tmp_db, make_insight(
            id='nc-1', content='test content'))

        count = link_pending(tmp_backend, max_batch=1)
        assert count == 1

        cols = _read_enrichment_columns(tmp_db, 'nc-1')
        assert cols['keywords'] is None
        assert cols['summary'] is None

    def test_keywords_capped(self):
        """An over-long LLM keyword list is capped to the salience limit.

        Mutation: dropping the `[:MAX_ENRICH_KEYWORDS]` slice, so an
            over-eager model inflates the keyword-enriched embed.
        Oracle: 40 proposed keywords capped at `MAX_ENRICH_KEYWORDS`.
        """
        from memman.graph.enrichment import MAX_ENRICH_KEYWORDS
        insight = make_insight(id='cap-1', content='cap body')
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            'keywords': [f'k{i}' for i in range(40)],
            'summary': 'summary',
            })

        result = enrich_with_llm(insight, mock_client)

        assert len(result['keywords']) == MAX_ENRICH_KEYWORDS


class TestReEmbed:
    """Re-embedding with keyword-enriched text."""

    def test_reembed_uses_keywords(self, tmp_db, tmp_backend):
        """embed_client.embed() called with keyword-appended text."""
        insight = make_insight(
            id='re-1', content='Python web framework')
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response(
            keywords=['web', 'framework'])

        mock_embed = MagicMock()
        mock_embed.available.return_value = True
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        mock_embed.model = 'voyage-3-lite'

        link_pending(
            tmp_backend,
            metadata_llm_client=mock_llm, embed_client=mock_embed,
            max_batch=1)

        mock_embed.embed.assert_called_once()
        call_text = mock_embed.embed.call_args[0][0]
        assert '[KEYWORDS: web framework]' in call_text
        assert 'Python web framework' in call_text

    def test_reembed_skipped_when_no_embed_client(self, tmp_db, tmp_backend):
        """No embed_client means no re-embed attempt."""
        insight = make_insight(
            id='rs-1', content='test content')
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response()

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm, embed_client=None,
            max_batch=1)

        cols = _read_enrichment_columns(tmp_db, 'rs-1')
        assert cols['keywords'] is not None

    def test_embed_failure_still_stamps_linked_at(self, tmp_db, tmp_backend):
        """Embed crash doesn't prevent linked_at stamp."""
        insight = make_insight(
            id='ef-1', content='test content')
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response()

        mock_embed = MagicMock()
        mock_embed.available.return_value = True
        mock_embed.embed.side_effect = RuntimeError('embed crashed')

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm, embed_client=mock_embed,
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
            entities=['Python'])
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=['test'], summary='test summary')

        result = enrich_with_llm(insight, mock_client)

        assert result['keywords'] == ['test']
        assert result['summary'] == 'test summary'

        row = tmp_db._conn.execute(
            'SELECT keywords, summary'
            ' FROM insights WHERE id = ?',
            ('pw-1',)).fetchone()
        assert row[0] is None, (
            'enrich_with_llm should not write keywords to DB')
        assert row[1] is None, (
            'enrich_with_llm should not write summary to DB')

    def test_markdown_fence_json_parsed(self):
        """LLM response wrapped in ```json fence is parsed correctly."""
        insight = make_insight(
            id='mf-1', content='fence test content')

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


class TestLengthCaps:
    """Per-string length guardrails on LLM keywords (F5)."""

    def test_overlong_keyword_dropped_not_truncated(self):
        """An over-long keyword is dropped, never truncated.

        A truncated keyword still lands in the enriched-text embed,
        preserving the pathology under a new name.

        Mutation: truncating to `MAX_ENRICH_STRING_CHARS` instead of
            dropping.
        Oracle: neither the over-long value nor any prefix of it
            appears in the result; the valid sibling survives.
        """
        insight = make_insight(id='cap-1', content='cap body')
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=['cache', 'k' * 250])
        result = enrich_with_llm(insight, mock_client)
        assert 'cache' in result['keywords']
        assert all(not k.startswith('kkk') for k in result['keywords'])

    def test_length_cap_boundary_sits_at_the_measured_200(self):
        """A 200-char keyword survives; its 201-char sibling drops.

        The literals pin the fleet-measured constant itself, not
        just the comparison: a drift to 2000 (or a `>=` flip) is a
        silent policy change every mid-range input misses.

        Mutation: `>` flipped to `>=`, or `MAX_ENRICH_STRING_CHARS`
            drifting from the measured 200.
        Oracle: hand-built strings straddling the real threshold --
            exactly 200 chars kept, 201 dropped.
        """
        at_cap = 'a' * 200
        over_cap = 'b' * 201
        insight = make_insight(id='cap-4', content='cap body')
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=[at_cap, over_cap, 'cache'])
        result = enrich_with_llm(insight, mock_client)
        assert at_cap in result['keywords']
        assert over_cap not in result['keywords']
        assert 'cache' in result['keywords']

    def test_length_cap_applies_before_count_cap(self):
        """12 valid + 3 over-long keywords yield 12, not 9.

        The 3 over-long inputs are listed FIRST, so a count-cap-first
        ordering provably drops 3 valid keywords from the tail.

        Mutation: applying the count cap before the length cap.
        Oracle: exactly `MAX_ENRICH_KEYWORDS` valid keywords survive.
        """
        from memman.graph.enrichment import MAX_ENRICH_KEYWORDS
        overlong = [('x' * 250) + str(i) for i in range(3)]
        valid = [f'keyword-{i}' for i in range(MAX_ENRICH_KEYWORDS)]
        insight = make_insight(id='cap-2', content='cap body')
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            keywords=overlong + valid)
        result = enrich_with_llm(insight, mock_client)
        assert result['keywords'] == valid


class _SequenceClient:
    """Answer a scripted list of responses; raise once it is spent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, *, stage, max_tokens=None):
        self.calls.append({'stage': stage, 'max_tokens': max_tokens})
        if not self.responses:
            raise AssertionError(
                f'complete called {len(self.calls)} times;'
                ' the script is spent')
        return self.responses.pop(0)


# A body cut mid keyword while the provider reported a finish_reason
# of `stop`, so nothing but the parse failure names it as unusable.
_CUT_MID_KEYWORD_BODY = (
    '{\n  "keywords": [\n    "handoff",\n'
    '    "environment-deployed-resources.')


def test_enrichment_rerolls_a_body_that_does_not_parse():
    """Verify an unparsable enrichment body is re-rolled, not dropped.

    Mutation: dropping the re-roll, so a cut body leaves the row
        unenriched while the drain stamps a prompt_version on it, and
        no later stage retries.
    Oracle: the keywords of the second response, and two calls.
    """
    insight = make_insight(content=_content_containing('Python'))
    client = _SequenceClient([
        _CUT_MID_KEYWORD_BODY,
        _make_enrichment_response(keywords=['python'])])
    result = enrich_with_llm(insight, client)
    assert result['keywords'] == ['python']
    assert len(client.calls) == 2


def test_enrichment_does_not_reroll_a_body_that_parses():
    """Verify the ordinary enrichment is billed once.

    Mutation: re-rolling unconditionally, which doubles the enrichment
        bill on every drained row.
    Oracle: one call; a second exhausts the scripted client and raises.
    """
    insight = make_insight(
        content=_content_containing('Python', 'FastAPI'))
    client = _SequenceClient([_make_enrichment_response()])
    result = enrich_with_llm(insight, client)
    assert result['keywords'] == ['web', 'framework']
    assert len(client.calls) == 1


def test_enrichment_rerolls_at_most_once():
    """Verify an undecodable model is billed twice, never more.

    Mutation: an unbounded retry loop on the drain's hottest stage.
    Oracle: two calls, and the empty fields the caller stamps as a
        terminal outcome.
    """
    insight = make_insight()
    client = _SequenceClient(['not json', 'still not json', 'nor this'])
    assert enrich_with_llm(insight, client) == {'keywords': [], 'summary': ''}
    assert len(client.calls) == 2
