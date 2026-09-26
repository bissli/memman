"""Recall channel degradation, and the unfiltered anchor count.

A failing vector or keyword channel leaves recall to answer from the
surviving channels, and an unfiltered recall keeps `ANCHOR_TOP_K`
recency anchors at any limit.

Vector-path tests seed deliberately correlated embeddings via
`nodes.update_embedding` rather than going through the autouse mock
embedder: the mock builds SHA-256 unit vectors whose pairwise cosine
is ~1/sqrt(dim) of arbitrary sign, so which rows anchor is noise and a
vector-path assertion would be testing the hash.
"""

import math
from datetime import datetime, timedelta, timezone

from memman.search.recall import ANCHOR_TOP_K, intent_aware_recall
from tests.conftest import make_insight, set_created_at

NOW = datetime.now(timezone.utc)


def _seed(backend, count, category, content_fmt):
    ids = []
    for i in range(count):
        iid = f'{category}-{i}'
        backend.nodes.insert(make_insight(
            id=iid, category=category, content=content_fmt.format(i=i)))
        set_created_at(backend, iid, NOW - timedelta(minutes=i))
        ids.append(iid)
    return ids


def test_unfiltered_recall_anchor_k_unchanged(backend):
    """Unfiltered recall keeps `ANCHOR_TOP_K` anchors at any limit.

    A bare `max(ANCHOR_TOP_K, limit)` would silently override the
    ablation harness's `anchor_top_k` sweep on every unfiltered
    config; unfiltered recall must stay byte-identical to 0.17.3.

    Mutation: applying the `max()` anchor bump unconditionally -
        `limit=50` would then produce 50 time anchors.
    Oracle: `meta['anchor_count'] == ANCHOR_TOP_K` with 60 rows, a
        keyword-dark query, and `limit` above `ANCHOR_TOP_K`.
    """
    _seed(backend, 60, 'fact', 'filler row body {i}')
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', None, 50)
    assert resp['meta']['anchor_count'] == ANCHOR_TOP_K


def _vec512(second):
    """Unit vector [1, second, 0, ...]/norm at the snapshot dim (512)."""
    n = math.sqrt(1.0 + second * second)
    v = [0.0] * 512
    v[0] = 1.0 / n
    v[1] = second / n
    return v


def test_recall_survives_a_raising_session_verb(backend, monkeypatch):
    """Verify a failing vector channel degrades instead of returning nothing.

    A dimension mismatch, a missing pgvector extension, or a statement
    timeout makes `similarities` / `vector_anchors` raise. Recall must
    keep the keyword and time channels and still answer. No other path
    computes vector scores, so the degrade path is the only thing
    standing between an operator error and an empty recall.

    Mutation: letting either exception escape `intent_aware_recall`,
        or returning an empty result set instead of falling through to
        the surviving channels.
    Oracle: the same query run against a healthy session, whose row
        count the degraded run must match (both channels reach every
        seeded row here), with the similarity signal at 0.0 throughout
        the degraded run.
    """
    _seed(backend, 12, 'fact', 'kombu serialization body {i}')
    query_vec = _vec512(0.2)

    healthy = intent_aware_recall(
        backend, 'kombu serialization body', query_vec, 10)

    def _raise(self, *args, **kwargs):
        raise RuntimeError('forced session failure')

    with backend.recall_session() as probe:
        session_cls = type(probe)
    monkeypatch.setattr(session_cls, 'similarities', _raise)
    monkeypatch.setattr(session_cls, 'vector_anchors', _raise)

    degraded = intent_aware_recall(
        backend, 'kombu serialization body', query_vec, 10)

    assert len(degraded['results']) == len(healthy['results']) > 0
    assert all(r['signals']['similarity'] == 0.0
               for r in degraded['results'])
    assert all(r['signals']['keyword'] > 0.0
               for r in degraded['results'])


def test_recall_survives_a_failed_keyword_channel(backend, monkeypatch):
    """Verify a dead keyword channel still answers instead of raising.

    Mutation: letting the exception escape, or substituting a
        non-zero keyword score for the channel that just failed.
    Oracle: the SAME query run healthy, which scores at least one row
        above zero on the keyword signal; the degraded run must score
        every row at exactly 0.0 and still return rows.

    Notes
    -----
    - This is the channel with no fallback: a store with no
      embeddings has only keyword and time, so its degrade path
      decides whether an operator error returns a wrong answer or
      raises.
    - The response deliberately carries no degradation flag. A caller
      reads the per-row `signals`, where an all-zero keyword column
      is what a dead channel looks like.
    """
    _seed(backend, 12, 'fact', 'kombu serialization body {i}')

    healthy = intent_aware_recall(
        backend, 'kombu serialization body', None, 10)
    assert any(r['signals']['keyword'] > 0.0
               for r in healthy['results'])

    def _raise(self, *args, **kwargs):
        raise RuntimeError('forced index failure')

    with backend.recall_session() as probe:
        session_cls = type(probe)
    monkeypatch.setattr(session_cls, 'keyword_counts', _raise)

    degraded = intent_aware_recall(
        backend, 'kombu serialization body', None, 10)

    assert degraded['results'], 'time anchors should still answer'
    assert all(r['signals']['keyword'] == 0.0
               for r in degraded['results'])
