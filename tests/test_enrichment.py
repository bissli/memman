"""Tests for LLM-based insight enrichment."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.graph.engine import link_pending
from memman.graph.enrichment import enrich_with_llm
from memman.store.node import insert_insight
from tests.conftest import make_insight

OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_enrichment_response(summary='test summary') -> str:
    """Build a mock LLM enrichment JSON response."""
    return json.dumps({'summary': summary})


def _content_containing(*names) -> str:
    """Content that names each of `names`."""
    return 'cap body ' + ' '.join(names)


def _read_enrichment_columns(db, insight_id: str) -> dict:
    """Read enrichment columns directly from DB."""
    row = db._conn.execute(
        'SELECT summary FROM insights WHERE id = ?',
        (insight_id,)).fetchone()
    if row is None:
        return {}
    return {'summary': row[0]}


class TestEnrichWithLLM:
    """LLM enrichment extraction with mocked client."""

    def test_happy_path(self):
        """Verify a valid LLM response returns its summary.

        Mutation: reading the summary from the wrong key, or blanking
            one well under the near-copy length guard.
        Oracle: the summary the mocked body carries.
        """
        insight = make_insight(
            id='hp-1',
            content=(
                'Python has several mature web frameworks: FastAPI for '
                'async-first APIs, Django for full-stack with batteries '
                'included, and Flask for minimal microframeworks. Each '
                'targets different sweet spots in the deployment surface.'))

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            summary='Comparing Python web frameworks')

        result = enrich_with_llm(insight, mock_client)

        assert result == {'summary': 'Comparing Python web frameworks'}

    def test_enrichment_returns_only_the_summary(self):
        """Verify the result carries no key beside `summary`.

        Mutation: `enrich_with_llm` copying the model's `keywords`,
            `entities` or `semantic_facts` into its result dict, so a
            caller reads fields outside the enrichment contract.
        Oracle: the LLM body below carries all three, so a surviving
            key proves the drop never happened, not that the mock
            omitted them.
        """
        insight = make_insight(
            id='ks-1', content='Redis backs the session cache')

        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            'entities': ['Redis'],
            'keywords': ['redis', 'cache'],
            'summary': 'Redis backs the session cache',
            'semantic_facts': ['Redis backs the session cache'],
            })

        result = enrich_with_llm(insight, mock_client)

        assert set(result) == {'summary'}

    def test_llm_unavailable_returns_empty(self):
        """ConnectionError from LLM returns empty dict, no crash."""
        insight = make_insight(
            id='ua-1', content='test content')

        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError('unreachable')

        result = enrich_with_llm(insight, mock_client)
        assert result == {}

    def test_undecodable_body_returns_an_empty_summary(self):
        """Verify a body that decodes on neither draw yields an empty summary.

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
        assert result == {'summary': ''}

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
            row stamped enriched with no summary from the default log.
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
        assert cols['summary'] is None


class TestReEmbed:
    """Re-embedding a pending row's raw content."""

    def test_reembed_embeds_the_raw_content(self, tmp_db, tmp_backend):
        """Verify link_pending embeds the row's content and nothing else.

        Mutation: appending enrichment output to the embedded text, so
            a rebuilt row's vector drifts from a fresh write's.
        Oracle: the one text the embed mock received, against the
            stored content.
        """
        insight = make_insight(
            id='re-1', content='Python web framework')
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            'keywords': ['web', 'framework'],
            'summary': 'A framework.'})

        mock_embed = MagicMock()
        mock_embed.available.return_value = True
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        mock_embed.model = 'voyage-3-lite'

        link_pending(
            tmp_backend,
            metadata_llm_client=mock_llm, embed_client=mock_embed,
            max_batch=1)

        mock_embed.embed.assert_called_once_with('Python web framework')

    def test_reembed_skipped_when_no_embed_client(self, tmp_db, tmp_backend):
        """Verify a pass with no embed client still stores the summary.

        Mutation: skipping the enrichment write whenever no vector
            comes back, which bills the LLM call and keeps nothing.
        Oracle: the stored summary, against the mocked body's.
        """
        insight = make_insight(
            id='rs-1', content='test content for the pass with no embedder')
        insert_insight(tmp_db, insight)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_enrichment_response()

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm, embed_client=None,
            max_batch=1)

        cols = _read_enrichment_columns(tmp_db, 'rs-1')
        assert cols['summary'] == 'test summary'

    def test_embed_failure_still_stamps_linked_at(self, tmp_db, tmp_backend):
        """Verify an embed crash still stamps `linked_at` and keeps the summary.

        Mutation: letting the embed exception abort the transaction,
            which leaves the row pending and re-bills it every pass.
        Oracle: the stored `linked_at` and summary.
        """
        insight = make_insight(
            id='ef-1', content='test content for the pass whose embed fails')
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
        assert cols['summary'] == 'test summary'


class TestEnrichmentPurity:
    """enrich_with_llm should be pure (no DB writes)."""

    def test_enrichment_does_not_write_db_directly(self, tmp_db):
        """Verify enrich_with_llm returns the summary without writing it.

        Mutation: an `update_enrichment` call inside `enrich_with_llm`,
            which writes outside the caller's transaction.
        Oracle: the returned summary, beside the still-null column.
        """
        insight = make_insight(
            id='pw-1', content='purity test content for the summary')
        insert_insight(tmp_db, insight)

        mock_client = MagicMock()
        mock_client.complete.return_value = _make_enrichment_response(
            summary='test summary')

        result = enrich_with_llm(insight, mock_client)

        assert result == {'summary': 'test summary'}
        row = tmp_db._conn.execute(
            'SELECT summary FROM insights WHERE id = ?',
            ('pw-1',)).fetchone()
        assert row[0] is None, (
            'enrich_with_llm should not write summary to DB')

    def test_markdown_fence_json_parsed(self):
        """Verify a body wrapped in a json code fence still parses.

        Mutation: parsing the raw body without stripping the fence,
            which returns an empty summary for a well-formed reply.
        Oracle: the summary inside the fence.
        """
        insight = make_insight(
            id='mf-1', content='fence test content for the parser')

        fenced_json = '```json\n' + _make_enrichment_response(
            summary='fenced summary') + '\n```'

        mock_client = MagicMock()
        mock_client.complete.return_value = fenced_json

        result = enrich_with_llm(insight, mock_client)
        assert result == {'summary': 'fenced summary'}


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


# A body cut mid summary while the provider reported a finish_reason
# of `stop`, so nothing but the parse failure names it as unusable.
_CUT_MID_SUMMARY_BODY = (
    '{\n  "summary": "The handoff names the'
    ' environment-deployed-resources.')


def test_enrichment_rerolls_a_body_that_does_not_parse():
    """Verify an unparsable enrichment body is re-rolled, not dropped.

    Mutation: dropping the re-roll, so a cut body leaves the row
        unenriched while the drain stamps a prompt_version on it, and
        no later stage retries.
    Oracle: the summary of the second response, and two calls.
    """
    insight = make_insight(content=_content_containing('Python'))
    client = _SequenceClient([
        _CUT_MID_SUMMARY_BODY,
        _make_enrichment_response(summary='Python.')])
    result = enrich_with_llm(insight, client)
    assert result == {'summary': 'Python.'}
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
    assert result == {'summary': 'test summary'}
    assert len(client.calls) == 1


def test_enrichment_rerolls_at_most_once():
    """Verify an undecodable model is billed twice, never more.

    Mutation: an unbounded retry loop on the drain's hottest stage.
    Oracle: two calls, and the empty summary the caller stamps as a
        terminal outcome.
    """
    insight = make_insight()
    client = _SequenceClient(['not json', 'still not json', 'nor this'])
    assert enrich_with_llm(insight, client) == {'summary': ''}
    assert len(client.calls) == 2
