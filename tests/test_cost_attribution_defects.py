"""Spend memman cannot attribute.

The scheduler's drain and relink paths both hand enrichment a client
built for `slow_canonical`, so the metadata model
`compute_prompt_version` stamps on the row is not the model that
enriched it, and the cost tool prices those calls at the wrong role.
"""
from memman import config
from memman.graph.engine import link_pending
from memman.store.node import insert_insight
from tests.conftest import make_insight

SENTINEL_METADATA_MODEL = 'sentinel/metadata-only-model'


def test_link_pending_enriches_on_the_metadata_role(
        tmp_db, tmp_backend, monkeypatch):
    """A caller passing no client still enriches on slow_metadata.

    Mutation: falling back to the caller's canonical client - or to
        its `None` - for the enrichment call, which is what the
        scheduler drain and the maintenance relink path both trigger
        today, so the row is enriched by one model and stamped with
        another.
    Oracle: a model id given ONLY to `MEMMAN_LLM_MODEL_SLOW_METADATA`,
        against the `.model` of the client that actually reaches
        `enrich_with_llm`. The shipped env gives both slow roles the
        same model, so the roles have to be split here or no
        assertion can tell them apart.
    """
    from memman.graph import enrichment as enrichment_mod
    from memman.llm import client as client_mod

    config._load_file_cache()
    monkeypatch.setitem(
        config._FILE_CACHE, config.LLM_MODEL_SLOW_METADATA,
        SENTINEL_METADATA_MODEL)
    client_mod._ROLE_CACHE.clear()
    monkeypatch.setattr(
        client_mod, '_ROLE_CACHE', {}, raising=False)

    seen: list = []

    def _spy(insight, llm_client, **kwargs):
        seen.append(llm_client)
        return {}

    monkeypatch.setattr(enrichment_mod, 'enrich_with_llm', _spy)
    insert_insight(tmp_db, make_insight(
        id='ca-1', content='Postgres carries the Warrant ledger.'))

    processed = link_pending(tmp_backend, max_batch=1, store_name='test')

    assert processed == 1
    assert len(seen) == 1
    assert getattr(seen[0], 'model', None) == SENTINEL_METADATA_MODEL
    assert seen[0].model != config.require(config.LLM_MODEL_SLOW_CANONICAL)
