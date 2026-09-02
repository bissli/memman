"""Tests for hot-path observability in `pipeline.remember`.

The `apply_all` block in `remember.run_remember` calls
`refresh_effective_importance` and falls back to a quiet default
(0.0) when it raises. F.4 turns the silent exception into a
`logger.warning` line so operators can see real failures (e.g., DB
pool exhaustion, schema drift) without losing the transactional
fallback.
"""

import logging


def _new_insight(content: str):
    """Build a fresh Insight for the pipeline test."""
    import uuid
    from datetime import datetime, timezone

    from memman.store.model import Insight
    now = datetime.now(timezone.utc)
    return Insight(
        id=str(uuid.uuid4()),
        content=content,
        category='fact',
        importance=3,
        entities=[],
        source='test',
        access_count=0,
        created_at=now,
        updated_at=now,
        deleted_at=None,
        last_accessed_at=None,
        effective_importance=0.0)


def test_refresh_effective_importance_failure_is_logged(
        tmp_backend, monkeypatch, caplog):
    """A raising `refresh_effective_importance` produces a warn line.

    The fallback (`ei = 0.0`) still applies so the transaction
    completes, but the failure is visible in operator logs.
    """
    from memman.embed.fingerprint import bound_embedder
    from memman.pipeline.remember import run_remember

    def _boom(*args, **kwargs):
        raise RuntimeError('forced importance failure')

    monkeypatch.setattr(
        tmp_backend.nodes, 'refresh_effective_importance', _boom)
    insight = _new_insight('Go uses sqlite for persistent storage')
    with caplog.at_level(logging.WARNING, logger='memman'):
        run_remember(
            tmp_backend, insight,
            content=insight.content,
            ec=bound_embedder(tmp_backend), store_name='test',
            no_reconcile=True)
    matches = [
        r for r in caplog.records
        if 'refresh_effective_importance failed' in r.getMessage()]
    assert matches


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

    The pin is a tripwire, not a constant: any deliberate write-path
    prompt change moves it, and re-pinning is the right answer once
    the author has weighed the cost. That cost is what the tripwire
    exists to surface -- every stored row in every store goes stale
    at once, and `graph rebuild --stale` re-runs enrichment and
    linking only, so it cannot repair an extraction or reconciliation
    change. The value below moved when the reconcile prompt began
    demanding the id of the memory a NONE verdict names.

    Mutation: "fixing" the length caps inside a system prompt, or any
        other incidental prompt edit -- the hash moves, every stored
        row's prompt_version goes stale, and doctor reports a
        fleet-wide rebuild that would not address the edit.
    Oracle: the hash of the four shipped write-path prompts, pinned.
    """
    from memman.pipeline.remember import compute_prompt_version
    assert compute_prompt_version() == '65b17a58c930e540'
