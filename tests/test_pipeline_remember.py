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
        ec=ec, store_name='test')

    keywords = res['facts'][0]['enrichment']['keywords']
    expected = build_enriched_text(
        'Redis backs the session cache', keywords)
    assert calls == [expected]


def test_prompt_version_unchanged_by_length_caps():
    """The length caps live post-parse; the prompt hash is pinned.

    The pin is a tripwire, not a constant: any deliberate change to a
    hashed input moves it, and re-pinning is the right answer once the
    author has weighed the cost. That cost is what the tripwire
    surfaces -- every stored row in every store goes stale at once,
    and only a `graph rebuild --stale` clears it.

    Two inputs move this value and neither is a length cap: the
    enrichment prompt, and the configured
    `MEMMAN_LLM_MODEL_SLOW`, which the key folds in and which
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
