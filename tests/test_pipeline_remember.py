"""Tests for `pipeline.remember`'s prompt-version pin.

`compute_prompt_version` is the one invariant this module still
covers: the reconcile candidate shortlist and its ranking are gone
along with the LLM-judged reconcile path, so the only cross-cutting
contract left to pin is the replay hash.
"""


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
    assert compute_prompt_version() == '4512702d00ebce0e'
