"""Tests for enriched_at column lifecycle in enrich_pending.
"""

import logging
from unittest.mock import MagicMock

from memman.embed.fingerprint import bound_embedder
from memman.pipeline import enrich as enrich_mod
from memman.pipeline.enrich import enrich_pending
from memman.store.node import insert_insight
from tests.conftest import insert_pending as _insert_pending
from tests.conftest import make_insight


class TestEnrichedAtColumn:
    """enriched_at column exists and starts equal to enrich_attempted_at.
    """

    def test_column_exists(self, tmp_db):
        """Verify a fresh DB has the enriched_at column.

        Mutation: dropping enriched_at from the baseline schema.
        Oracle: the column names reported by pragma table_info.
        """
        cols = tmp_db._conn.execute(
            'pragma table_info(insights)').fetchall()
        col_names = {row[1] for row in cols}
        assert 'enriched_at' in col_names

    def test_new_insight_has_no_enrichment_stamps(self, tmp_db):
        """Verify a new insight has NULL enriched_at and enrich_attempted_at.

        Mutation: insert_insight stamping enrich_attempted_at at
            insert time, silently opting a new row out of enrichment
            while enriched_at stays NULL.
        Oracle: enriched_at and enrich_attempted_at both read back
            NULL right after insert.
        """
        insert_insight(tmp_db, make_insight(
            id='bf-1', content='backfill test'))
        row = tmp_db._conn.execute(
            'select enriched_at, enrich_attempted_at from insights'
            " where id = 'bf-1'").fetchone()
        assert tuple(row) == (None, None)


class TestEnrichedAtOnEnrichPending:
    """enrich_pending sets enriched_at only when LLM enrichment succeeds.
    """

    def test_no_llm_sets_enrich_attempted_at_only(
            self, tmp_db, tmp_backend, monkeypatch):
        """Verify an unreachable LLM stamps only enrich_attempted_at.

        Mutation: dropping the `if enrichment and new_vec is not
            None` guard before stamp_enriched, so a row with no
            enrichment still flips to enriched.
        Oracle: enrich_attempted_at not-None and enriched_at None
            read back after get_llm_client raises.
        """
        def _unavailable(*args, **kwargs):
            raise RuntimeError('no LLM credential')

        monkeypatch.setattr(enrich_mod, 'get_llm_client', _unavailable)
        _insert_pending(tmp_db, 'nl-1', 'test without llm')
        tmp_db._conn.execute(
            'update insights set enriched_at = NULL'
            " where id = 'nl-1'")

        enrich_pending(tmp_backend)

        row = tmp_db._conn.execute(
            'select enrich_attempted_at, enriched_at from insights'
            " where id = 'nl-1'").fetchone()
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
        _insert_pending(tmp_db, 'ls-1', 'test with llm enrichment')
        tmp_db._conn.execute(
            'update insights set enriched_at = NULL'
            " where id = 'ls-1'")

        mock_llm = MagicMock()
        mock_llm.model = 'test-llm'
        mock_llm.complete.return_value = '{"summary": "test"}'

        enrich_pending(
            tmp_backend, llm_client=mock_llm,
            embed_client=bound_embedder(tmp_backend))

        row = tmp_db._conn.execute(
            'select enrich_attempted_at, enriched_at from insights'
            " where id = 'ls-1'").fetchone()
        assert row[0] is not None
        assert row[1] is not None

    def test_reembed_failure_skips_stamp_enriched(
            self, tmp_db, tmp_backend, monkeypatch, caplog):
        """Verify a re-embed failure leaves enriched_at NULL and warns.

        Mutation: dropping the `new_vec is not None` guard so
            stamp_enriched runs despite the failed embed, or logging
            the failure at debug instead of warning so it never
            surfaces operationally.
        Oracle: a WARNING-level "Re-embed failed" log record, and
            enriched_at read back NULL while enrich_attempted_at is
            not.
        """
        _insert_pending(tmp_db, 'rf-1', 'reembed-fail content')
        tmp_db._conn.execute(
            'update insights set enriched_at = NULL'
            " where id = 'rf-1'")

        mock_llm = MagicMock()
        mock_llm.model = 'test-llm'
        mock_llm.complete.return_value = '{"summary": "s"}'

        class _FailingClient:
            available = staticmethod(lambda: True)
            model = 'mock'

            def embed(self, text):
                raise RuntimeError('forced reembed failure')

        with caplog.at_level(logging.WARNING, logger='memman'):
            enrich_pending(
                tmp_backend, llm_client=mock_llm,
                embed_client=_FailingClient())

        warned = [r for r in caplog.records
                  if 'Re-embed failed' in r.getMessage()]
        assert warned

        row = tmp_db._conn.execute(
            'select enrich_attempted_at, enriched_at from insights'
            " where id = 'rf-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None, (
            'enriched_at must stay NULL when the re-embed failed')

    def test_skipped_embed_leaves_a_vectorless_row_unstamped(
            self, tmp_db, tmp_backend):
        """Verify an embed that cannot run leaves a vectorless row unstamped.

        Mutation: stamping whenever the enrichment returned, so an
            embedder whose probe failed mid-outage stamps the row with
            no vector, and the stranded-row sweep never revisits it.
        Oracle: the row's enriched_at, beside its
            enrich_attempted_at, which the pass does set.
        """
        _insert_pending(tmp_db, 'sk-1', 'skipped embed content')
        tmp_db._conn.execute(
            'update insights set enriched_at = NULL'
            " where id = 'sk-1'")

        mock_llm = MagicMock()
        mock_llm.model = 'test-llm'
        mock_llm.complete.return_value = '{"summary": "s"}'
        unavailable = MagicMock()
        unavailable.available.return_value = False

        enrich_pending(
            tmp_backend, llm_client=mock_llm,
            embed_client=unavailable)

        row = tmp_db._conn.execute(
            'select enrich_attempted_at, enriched_at from insights'
            " where id = 'sk-1'").fetchone()
        assert row[0] is not None
        assert row[1] is None

    def test_vectorless_row_gets_a_vector_on_retry(
            self, tmp_db, tmp_backend):
        """Verify enrich_pending embeds a vectorless row on its retry pass.

        Mutation: embedding only on the row's first enrichment pass, so
            a row the write stored without a vector, retried on a later
            pass, is stamped enriched while still vectorless and is
            never revisited.
        Oracle: the store's embedding set, which lacks the row before
            the pass.
        """
        _insert_pending(tmp_db, 'nv-1', 'vectorless content')
        tmp_db._conn.execute(
            'update insights set enriched_at = NULL'
            " where id = 'nv-1'")
        before = tmp_db._conn.execute(
            "select embedding from insights where id = 'nv-1'").fetchone()
        assert before[0] is None

        mock_llm = MagicMock()
        mock_llm.model = 'test-llm'
        mock_llm.complete.return_value = '{"summary": "s"}'

        enrich_pending(
            tmp_backend, llm_client=mock_llm,
            embed_client=bound_embedder(tmp_backend))

        after = tmp_db._conn.execute(
            "select embedding from insights where id = 'nv-1'").fetchone()
        assert after[0] is not None
