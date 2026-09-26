"""Tests for enriched_at column lifecycle in link_pending."""

from unittest.mock import MagicMock

from memman.graph.engine import link_pending
from memman.store.node import insert_insight
from tests.conftest import insert_pending as _insert_pending
from tests.conftest import make_insight


class TestEnrichedAtColumn:
    """enriched_at column exists and is backfilled from migration."""

    def test_column_exists(self, tmp_db):
        """Fresh DB has enriched_at column."""
        cols = tmp_db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        col_names = {row[1] for row in cols}
        assert 'enriched_at' in col_names

    def test_backfill_from_linked_at(self, tmp_db):
        """Insights with linked_at get enriched_at backfilled."""
        insert_insight(tmp_db, make_insight(
            id='bf-1', content='backfill test'))
        row = tmp_db._conn.execute(
            'SELECT enriched_at, linked_at FROM insights'
            " WHERE id = 'bf-1'").fetchone()
        assert row[0] == row[1]


class TestEnrichedAtOnLinkPending:
    """link_pending sets enriched_at only when LLM enrichment succeeds."""

    def test_no_llm_sets_linked_at_only(
            self, tmp_db, tmp_backend, monkeypatch):
        """An unreachable LLM sets linked_at but leaves enriched_at NULL."""
        from memman.graph import engine as engine_mod

        def _unavailable(role, *args, **kwargs):
            raise RuntimeError(f'no credential for {role}')

        monkeypatch.setattr(engine_mod, 'get_llm_client', _unavailable)
        _insert_pending(tmp_db, 'nl-1', 'test without llm')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'nl-1'")

        link_pending(tmp_backend)

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'nl-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None

    def test_llm_success_sets_enriched_at(self, tmp_db, tmp_backend):
        """Verify an enrichment plus a vector stamps enriched_at.

        Mutation: inverting the vector check to `new_vec is None`, which
            stamps only the rows whose embed failed and leaves every
            enriched row to the stranded-row sweep.
        Oracle: the enriched_at column after one pass with a working
            LLM and embedder.
        """
        from memman.embed.fingerprint import bound_embedder

        _insert_pending(tmp_db, 'ls-1', 'test with llm enrichment')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'ls-1'")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"keywords": ["test"], "summary": "test"}')

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm,
            embed_client=bound_embedder(tmp_backend))

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'ls-1'").fetchone()
        assert row[0] is not None
        assert row[1] is not None

    def test_reembed_failure_skips_stamp_enriched(
            self, tmp_db, tmp_backend, monkeypatch, caplog):
        """A re-embed failure mid-link must NOT mark the insight enriched.

        Pre-F.4 the failure was silent at debug level and the insight
        flipped to `enriched_at != NULL` despite carrying a stale
        embedding. Now the failure logs at warn and `stamp_enriched`
        is skipped so the row is retried on the next pass.
        """
        import logging

        _insert_pending(tmp_db, 'rf-1', 'reembed-fail content')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'rf-1'")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"keywords": ["alpha"], "summary": "s"}')

        class _FailingClient:
            available = staticmethod(lambda: True)
            model = 'mock'

            def embed(self, text):
                raise RuntimeError('forced reembed failure')

        with caplog.at_level(logging.WARNING, logger='memman'):
            link_pending(
                tmp_backend, metadata_llm_client=mock_llm,
                embed_client=_FailingClient())

        warned = [r for r in caplog.records
                  if 'Re-embed failed' in r.getMessage()]
        assert warned

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'rf-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None, (
            'enriched_at must stay NULL when the re-embed failed')

    def test_skipped_embed_leaves_a_vectorless_row_unstamped(
            self, tmp_db, tmp_backend):
        """Verify an embed that cannot run leaves a vectorless row unstamped.

        Mutation: stamping whenever the enrichment returned, so an
            embedder whose probe failed mid-outage stamps the row with
            no vector, and the stranded-row sweep never revisits it.
        Oracle: the row's enriched_at, beside its linked_at, which the
            pass does set.
        """
        _insert_pending(tmp_db, 'sk-1', 'skipped embed content')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'sk-1'")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"keywords": ["alpha"], "summary": "s"}')
        unavailable = MagicMock()
        unavailable.available.return_value = False

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm,
            embed_client=unavailable)

        row = tmp_db._conn.execute(
            'SELECT linked_at, enriched_at FROM insights'
            " WHERE id = 'sk-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None

    def test_vectorless_row_gets_a_vector_without_keywords(
            self, tmp_db, tmp_backend):
        """Verify link_pending embeds a vectorless row with no keywords.

        Mutation: embedding only when the enrichment carries keywords,
            so a row the write stored without a vector, whose retry
            enrichment finds no keywords, is stamped enriched while
            still vectorless and is never revisited.
        Oracle: the store's embedding set, which lacks the row before
            the pass.
        """
        from memman.embed.fingerprint import bound_embedder

        _insert_pending(tmp_db, 'nv-1', 'vectorless content')
        tmp_db._conn.execute(
            'UPDATE insights SET enriched_at = NULL'
            " WHERE id = 'nv-1'")
        before = tmp_db._conn.execute(
            "SELECT embedding FROM insights WHERE id = 'nv-1'").fetchone()
        assert before[0] is None

        mock_llm = MagicMock()
        mock_llm.complete.return_value = '{"keywords": [], "summary": "s"}'

        link_pending(
            tmp_backend, metadata_llm_client=mock_llm,
            embed_client=bound_embedder(tmp_backend))

        after = tmp_db._conn.execute(
            "SELECT embedding FROM insights WHERE id = 'nv-1'").fetchone()
        assert after[0] is not None
