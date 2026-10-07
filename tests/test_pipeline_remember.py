"""Tests for `pipeline.remember`'s embed count and summary model stamp.

`test_a_write_embeds_once_after_enrichment` pins the write's embed
contract: one call, on the raw content. The `summary_model` tests pin
which model id a write records beside its summary.
"""

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight


def test_a_write_embeds_once_after_enrichment(tmp_backend, monkeypatch):
    """Verify a write embeds its raw content exactly once.

    Mutation: the write embeds a second, billed text whose vector it
        discards, or embeds the content with enrichment output
        appended.
    Oracle: the spy's call list, against the content the write stored.
    """
    ec = bound_embedder(tmp_backend)
    calls: list[str] = []
    original_embed = ec.embed

    def spy_embed(text):
        calls.append(text)
        return original_embed(text)

    monkeypatch.setattr(ec, 'embed', spy_embed)

    parent = make_insight(
        id='embed-once-1', content='Redis backs the session cache')
    run_remember(tmp_backend, parent, ec=ec)

    assert calls == ['Redis backs the session cache']


def test_a_write_whose_embed_fails_stays_unenriched(
        tmp_backend, monkeypatch):
    """Verify a write stored without a vector is left for the sweep.

    Mutation: `_apply_plan` stamping `enriched_at` whenever the
        enrichment returned, so the vectorless row falls outside the
        stranded-row sweep, which selects `enriched_at is null`, and
        never gets a vector.
    Oracle: the stored row's `enriched_at`, beside its summary, which
        proves the enrichment itself landed.
    """
    ec = bound_embedder(tmp_backend)

    def failing_embed(text):
        raise RuntimeError('forced embed failure')

    monkeypatch.setattr(ec, 'embed', failing_embed)

    content = (
        'Redis backs the session cache for the web tier, evicts keys'
        ' under an LRU policy, and replicates to a standby node in a'
        ' second zone')
    parent = make_insight(id='embed-fail-1', content=content)
    res = run_remember(tmp_backend, parent, ec=ec)

    stored = tmp_backend.nodes.get(res['id'])
    assert stored.summary
    assert stored.enriched_at is None


def test_a_write_whose_enrichment_never_decodes_is_stamped(
        tmp_backend, monkeypatch):
    """Verify a write whose enrichment body never decodes is stamped.

    Mutation: treating a body that decodes on neither draw as a
        failed call, which leaves `enriched_at` null, so the
        stranded-row sweep re-enriches the row on every drain of its
        store and bills the call each time.
    Oracle: the stored row's `enriched_at`.
    """
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        lambda self, system, user, **kwargs: 'not json at all')
    ec = bound_embedder(tmp_backend)

    parent = make_insight(
        id='undecodable-1', content='Redis backs the session cache')
    res = run_remember(tmp_backend, parent, ec=ec)

    stored = tmp_backend.nodes.get(res['id'])
    assert stored.enriched_at is not None


def test_a_write_whose_enrichment_call_fails_stays_unenriched(
        tmp_backend, monkeypatch):
    """Verify a write whose enrichment call raises is left for the sweep.

    Mutation: `_apply_plan` stamping whenever the embed succeeded, so a
        row with no enrichment reads as enriched and the stranded-row
        sweep never enriches it.
    Oracle: the stored row's `enriched_at`.
    """
    def failing_complete(self, system, user, **kwargs):
        raise ConnectionError('forced enrichment failure')

    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete', failing_complete)
    ec = bound_embedder(tmp_backend)

    parent = make_insight(
        id='enrich-fail-1', content='Redis backs the session cache')
    res = run_remember(tmp_backend, parent, ec=ec)

    stored = tmp_backend.nodes.get(res['id'])
    assert stored.enriched_at is None


class _StubLLM:
    """An LLM client that answers one fixed body under a fixed model id.
    """

    def __init__(self, model, body):
        self.model = model
        self.body = body

    def complete(self, system, user, **kwargs):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def test_a_write_stamps_the_clients_model_on_its_summary(
        backend, monkeypatch):
    """Verify a write stores the model id of the client that summarized it.

    Mutation: `run_remember` passing the configured model or a
        constant to `_apply_plan` instead of `llm_client.model`, or
        `_apply_plan` dropping the argument so the column stays null.
    Oracle: the model id of the stub client, a string no config holds.
    """
    monkeypatch.setattr(
        'memman.pipeline.remember.get_llm_client',
        lambda: _StubLLM('stub-vendor/stub-model', '{"summary": "Cache."}'))
    ec = bound_embedder(backend)

    res = run_remember(
        backend, make_insight(
            id='sm-write-1',
            content='Redis backs the session cache for the web tier'),
        ec=ec)

    row = backend.nodes.get_raw(res['id'])
    assert row.summary == 'Cache.'
    assert row.summary_model == 'stub-vendor/stub-model'


def test_a_write_whose_enrichment_fails_leaves_summary_model_unset(
        backend, monkeypatch):
    """Verify a write whose enrichment call raised records no model.

    Mutation: `_apply_plan` writing `summary_model` outside the
        `if enrichment` guard, so a row that got no model outcome
        names a model anyway.
    Oracle: the stored row's `summary_model`, null beside its null
        summary.
    """
    stub = _StubLLM(
        'stub-vendor/stub-model', ConnectionError('forced failure'))
    monkeypatch.setattr(
        'memman.pipeline.remember.get_llm_client', lambda: stub)
    ec = bound_embedder(backend)

    res = run_remember(
        backend, make_insight(
            id='sm-write-2', content='Redis backs the session cache'),
        ec=ec)

    row = backend.nodes.get_raw(res['id'])
    assert row.summary is None
    assert row.summary_model is None
