"""Tests for `pipeline.remember`'s prompt-version pin and embed count.

`compute_prompt_version` pins the replay hash the reconcile-free
pipeline still depends on. `test_a_write_embeds_once_after_enrichment`
pins the write's embed contract: one call, on the enriched text.
"""

from memman.embed.fingerprint import bound_embedder
from memman.graph.enrichment import build_enriched_text
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight


def test_a_write_embeds_once_after_enrichment(tmp_backend, monkeypatch):
    """Verify a write embeds the enriched text exactly once.

    Mutation: the write embeds the raw text before enrichment as
        well as the enriched text after it, a second billed call whose
        vector the write discards.
    Oracle: the spy's call list, against the one text
        `build_enriched_text` produces for this content and the
        conftest mock's keywords.
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
    res = run_remember(
        tmp_backend, parent, 'Redis backs the session cache',
        ec=ec)

    keywords = res['facts'][0]['enrichment']['keywords']
    expected = build_enriched_text(
        'Redis backs the session cache', keywords)
    assert calls == [expected]


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
    res = run_remember(
        tmp_backend, parent, content, ec=ec)

    stored = tmp_backend.nodes.get(res['facts'][0]['id'])
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
    res = run_remember(
        tmp_backend, parent, 'Redis backs the session cache',
        ec=ec)

    stored = tmp_backend.nodes.get(res['facts'][0]['id'])
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
    res = run_remember(
        tmp_backend, parent, 'Redis backs the session cache',
        ec=ec)

    stored = tmp_backend.nodes.get(res['facts'][0]['id'])
    assert stored.enriched_at is None


def test_prompt_version_unchanged_by_length_caps():
    """The length caps live post-parse; the prompt hash is pinned.

    The pin is a tripwire, not a constant: any deliberate change to a
    hashed input moves it, and re-pinning is the right answer once the
    author has weighed the cost. That cost is what the tripwire
    surfaces -- every stored row in every store goes stale at once,
    and only a `graph rebuild --stale` clears it.

    Two inputs move this value and neither is a length cap: the
    enrichment prompt, and the configured
    `MEMMAN_LLM_MODEL`, which the key folds in and which
    the suite seeds from `INSTALL_DEFAULTS` -- changing that default
    re-pins this test, deliberately.

    Mutation: "fixing" the length caps inside a system prompt, or any
        other incidental edit to a hashed input -- the hash moves and
        every stored row goes stale for a change nobody intended.
    Oracle: the hash of the replayable prompt plus the seeded
        metadata model, pinned.
    """
    from memman.pipeline.remember import compute_prompt_version
    assert compute_prompt_version() == '397b65af1c473aee'
