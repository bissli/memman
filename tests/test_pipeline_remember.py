"""Tests for `pipeline.remember`'s planning and prompt pinning.

Covers the two invariants the drain's write path cannot express in
its own output: the reconcile candidate list must carry the
strongest near-duplicate rather than the first ones the cache
happened to yield, and `compute_prompt_version` must not move
except deliberately.
"""


def test_reconcile_candidates_ranked_by_similarity(monkeypatch):
    """The strongest near-duplicate must reach the reconcile candidate list.

    Regression: the cosine candidates were appended in embed_cache order
    and capped at MAX_SIMILAR_FOR_RECONCILE, so a high-cosine insight that
    sorts last could be crowded out by weaker earlier ones.
    """
    import math
    from unittest.mock import MagicMock

    from memman.llm import extract as llm_extract
    from memman.pipeline import remember as rem
    from tests.conftest import make_insight

    captured = {}

    def _fake_reconcile(client, facts, similar):
        captured['similar'] = list(similar)
        return [{'fact': facts[0]['text'], 'action': 'NONE',
                 'target_id': None, 'merged_text': None}]

    monkeypatch.setattr(llm_extract, 'reconcile_memories', _fake_reconcile)

    fact_vec = [1.0, 0.0]
    med = [0.6, math.sqrt(1 - 0.6 * 0.6)]
    top = [0.95, math.sqrt(1 - 0.95 * 0.95)]

    insights_by_id = {}
    embed_cache = {}
    for i in range(10):
        ins = make_insight(id=f'dec{i}', content=f'decoy body number {i}')
        insights_by_id[ins.id] = ins
        embed_cache[ins.id] = list(med)
    topins = make_insight(id='TOP', content='topmost candidate body')
    insights_by_id[topins.id] = topins
    embed_cache[topins.id] = list(top)

    fact = {'text': 'zzqq alpha brandnew', 'category': 'fact',
            'importance': 3, 'entities': []}
    parent = make_insight(id='parent', content='zzqq alpha brandnew')
    ec = MagicMock()
    ec.embed.return_value = fact_vec

    rem._plan_fact(
        fact, parent, '', False, False, False,
        insights_by_id, embed_cache, set(),
        MagicMock(), MagicMock(), ec, MagicMock(), MagicMock())

    ids = [cid for cid, _content in captured.get('similar', [])]
    assert 'TOP' in ids, f'top-cosine insight crowded out; candidates={ids}'


def test_prompt_version_unchanged_by_length_caps():
    """The length caps live post-parse; the prompt hash is pinned.

    The pin is a tripwire, not a constant: any deliberate change to a
    hashed input moves it, and re-pinning is the right answer once the
    author has weighed the cost. That cost is what the tripwire
    surfaces -- every stored row in every store goes stale at once,
    and only a `graph rebuild --stale` clears it.

    Two inputs now move this value and neither is a length cap: the
    enrichment prompt and the causal prompt. So does the configured
    `MEMMAN_LLM_MODEL_SLOW_METADATA`, which the key folds in and which
    the suite seeds from `INSTALL_DEFAULTS` -- changing that default
    re-pins this test, deliberately.

    Mutation: "fixing" the length caps inside a system prompt, or any
        other incidental edit to a hashed input -- the hash moves and
        every stored row goes stale for a change nobody intended.
    Oracle: the hash of the two replayable prompts plus the seeded
        metadata model, pinned.
    """
    from memman.pipeline.remember import compute_prompt_version
    assert compute_prompt_version() == '6a60ef0080b1ab9f'
