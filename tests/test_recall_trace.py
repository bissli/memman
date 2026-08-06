"""Phase-level recall trace events (F2).

`intent_aware_recall` hoists one `trace.is_enabled()` read and emits
per-phase events (anchors, traversal, rerank) behind it. These tests
pin the hoist (the read can fall through to a file read on the
synchronous hot path) and the id-based rerank movement metric.
"""

from memman import trace
from memman.embed.fingerprint import stored_fingerprint
from memman.search.recall import intent_aware_recall
from tests.conftest import make_insight


def _seed(backend, count=8, category='fact'):
    for i in range(count):
        backend.nodes.insert(make_insight(
            id=f'tr-{category}-{i}', category=category,
            content=f'alpha shared topic row {i}'))


def _vec512(second):
    """Unit vector [1, second, 0, ...]/norm at the mock dim (512)."""
    import math
    n = math.sqrt(1.0 + second * second)
    v = [0.0] * 512
    v[0] = 1.0 / n
    v[1] = second / n
    return v


def test_is_enabled_read_once_per_recall(tmp_backend, monkeypatch):
    """The debug gate is read once per recall, not once per event site.

    `is_enabled` falls through to a file read when `MEMMAN_DEBUG` is
    unset, so a per-site call is a real hot-path regression.

    Mutation: removing the `enabled` hoist (re-reading
        `trace.is_enabled()` at each event site).
    Oracle: a counting spy sees exactly one call across a full
        disabled-mode recall.
    """
    _seed(tmp_backend)
    calls = []
    monkeypatch.setattr(
        trace, 'is_enabled', lambda: calls.append(1) or False)
    resp = intent_aware_recall(
        tmp_backend, 'alpha shared topic', None, [], 5,
        fingerprint=stored_fingerprint(tmp_backend),
        intent_override='GENERAL')
    assert resp['results']
    assert len(calls) == 1


def test_rerank_event_reports_moved_by_id_not_score(
        tmp_backend, monkeypatch):
    """`recall_rerank.moved` diffs shortlist ids, not scores.

    The reranker replaces every score, so a score-based diff always
    reports "all moved" and the event could never justify or kill the
    cross-encoder.

    Mutation: computing `moved` from score changes instead of id
        positions.
    Oracle: a spy reranker that preserves order while rewriting every
        score must yield moved == 0.
    """
    _seed(tmp_backend)
    events = []
    monkeypatch.setattr(trace, 'is_enabled', lambda: True)
    monkeypatch.setattr(
        trace, 'event',
        lambda name, **fields: events.append((name, fields)))

    class _IdentityRerank:
        def rerank(self, query, docs, top_k=None):
            return [(i, 0.9 - 0.001 * i) for i in range(len(docs))]

    monkeypatch.setattr('memman.rerank.get_client', _IdentityRerank)
    resp = intent_aware_recall(
        tmp_backend, 'alpha shared topic', None, [], 5,
        fingerprint=stored_fingerprint(tmp_backend),
        intent_override='GENERAL', rerank=True)
    assert resp['meta']['reranked'] is True
    rr = [f for n, f in events if n == 'recall_rerank']
    assert rr
    assert rr[0]['moved'] == 0


def test_anchor_event_reports_vector_hits_against_anchor_k(
        tmp_backend, monkeypatch):
    """`recall_anchors` reports the raw vector hit count, not proxies.

    This event is the measurement Phase 1 deferred: whether a
    selective filter makes the vector scan return fewer than `k`
    anchors. Reporting the fused pool or `anchor_k` in its place
    would answer a different question.

    Mutation: reporting the fused pool size (10 here) or `anchor_k`
        (35) as `vector_hits`.
    Oracle: 10 matching rows of which only 6 are embedded — the
        event must carry vector_hits == 6 with anchor_k == 35.
    """
    _seed(tmp_backend, count=10, category='preference')
    for i in range(6):
        tmp_backend.nodes.update_embedding(
            f'tr-preference-{i}', _vec512(0.3 + 0.01 * i),
            'voyage-3-lite')
    events = []
    monkeypatch.setattr(trace, 'is_enabled', lambda: True)
    monkeypatch.setattr(
        trace, 'event',
        lambda name, **fields: events.append((name, fields)))
    qv = [0.0] * 512
    qv[0] = 1.0
    intent_aware_recall(
        tmp_backend, 'zzz unmatched query', qv, [], 35,
        fingerprint=stored_fingerprint(tmp_backend),
        intent_override='GENERAL', category='preference')
    ev = [f for n, f in events if n == 'recall_anchors']
    assert ev
    assert ev[0]['anchor_k'] == 35
    assert ev[0]['vector_hits'] == 6
