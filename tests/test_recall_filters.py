"""Filtered recall (`--cat`/`--source`) fills to the limit.

D2: the old CLI over-fetched `limit * 3`, post-filtered in Python and
truncated, silently under-returning whenever matching rows ranked
below the unfiltered top `3 * limit`. The fix pushes the predicate
into the anchor scans and filters after the weighted-sum sort, before
rerank.

Vector-path tests seed deliberately correlated embeddings via
`nodes.update_embedding` rather than going through the autouse mock
embedder: the mock builds SHA-256 unit vectors whose pairwise cosine
(~1/sqrt(dim)) sits below `VECTOR_SEARCH_MIN_SIM = 0.10`, so
mock-embedded fixtures return zero vector anchors and any vector-path
assertion would be vacuously green.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest
from memman.embed.fingerprint import stored_fingerprint
from memman.search.recall import ANCHOR_TOP_K, intent_aware_recall
from tests.conftest import make_insight, set_created_at

NOW = datetime.now(timezone.utc)


def _seed(backend, count, category, content_fmt, *, days_old=0,
          prefix=''):
    ids = []
    for i in range(count):
        iid = f'{prefix}{category}-{i}'
        backend.nodes.insert(make_insight(
            id=iid, category=category, content=content_fmt.format(i=i)))
        set_created_at(
            backend, iid,
            NOW - timedelta(days=days_old, minutes=i))
        ids.append(iid)
    return ids


def test_filtered_recall_fills_to_limit(backend):
    """A category filter returns `limit` rows when enough rows match.

    Mutation: reverting to post-filtering (fetch an unfiltered
        `limit * 3`, filter, truncate) — the 15 matching rows are old
        and keyword-dark, so no unfiltered anchor scan surfaces them
        and the post-filter returns zero.
    Oracle: exactly `limit` results, every one in the filtered
        category.
    """
    _seed(backend, 45, 'fact', 'alpha topic note {i}')
    _seed(backend, 15, 'preference', 'quiet other subject {i}',
          days_old=10)
    resp = intent_aware_recall(
        backend, 'alpha topic note', None, [], 10,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', category='preference')
    assert len(resp['results']) == 10
    assert all(r['insight'].category == 'preference'
               for r in resp['results'])


def test_unfiltered_recall_anchor_k_unchanged(backend):
    """Unfiltered recall keeps `ANCHOR_TOP_K` anchors at any limit.

    A bare `max(ANCHOR_TOP_K, limit)` would silently override the
    ablation harness's `anchor_top_k` sweep on every unfiltered
    config; unfiltered recall must stay byte-identical to 0.17.3.

    Mutation: applying the `max()` anchor bump unconditionally —
        `limit=50` would then produce 50 time anchors.
    Oracle: `meta['anchor_count'] == ANCHOR_TOP_K` with 60 rows, a
        keyword-dark query, and `limit` above `ANCHOR_TOP_K`.
    """
    _seed(backend, 60, 'fact', 'filler row body {i}')
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', None, [], 50,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL')
    assert resp['meta']['anchor_count'] == ANCHOR_TOP_K


def test_filtered_recall_above_anchor_top_k(backend):
    """`--limit 50` with 60 matching rows returns 50, not ANCHOR_TOP_K.

    Mutation: leaving `anchor_k` at `ANCHOR_TOP_K` under a filter —
        time anchors then cap the candidate pool at 30 and only 30
        rows return.
    Oracle: exactly 50 results from 60 keyword-dark matching rows.
    """
    _seed(backend, 60, 'preference', 'quiet other subject {i}')
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', None, [], 50,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', category='preference')
    assert len(resp['results']) == 50
    assert all(r['insight'].category == 'preference'
               for r in resp['results'])


def test_filter_does_not_block_graph_traversal(backend):
    """A matching row reachable only through a non-matching hop returns.

    Traversal is deliberately unfiltered: a hop through a
    non-matching neighbour is correct, and only the final result set
    is filtered.

    Mutation: applying the category filter inside the beam-search
        neighbour loop (or to the traversal lookups), blocking the
        hop through the non-matching bridge row.
    Oracle: `p-far` is connected only via the `fact` bridge and is
        too old to be a time anchor, so its presence proves the
        traversal crossed the non-matching hop.
    """
    from tests.conftest import make_edge
    _seed(backend, 30, 'preference', 'quiet other subject {i}',
          prefix='fill-')
    backend.nodes.insert(make_insight(
        id='p-near', category='preference', content='p near body'))
    set_created_at(backend, 'p-near', NOW + timedelta(minutes=5))
    backend.nodes.insert(make_insight(
        id='g-bridge', category='fact', content='g bridge body'))
    backend.nodes.insert(make_insight(
        id='p-far', category='preference', content='p far body'))
    set_created_at(backend, 'p-far', NOW - timedelta(days=30))
    for a, b in [('p-near', 'g-bridge'), ('g-bridge', 'p-far')]:
        backend.edges.upsert(make_edge(
            source_id=a, target_id=b, edge_type='semantic', weight=1.0))
        backend.edges.upsert(make_edge(
            source_id=b, target_id=a, edge_type='semantic', weight=1.0))
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', None, [], 0,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', category='preference')
    ids = {r['insight'].id for r in resp['results']}
    assert 'p-far' in ids
    assert 'g-bridge' not in ids


def _vec512(second):
    """Unit vector [1, second, 0, ...]/norm at the snapshot dim (512)."""
    n = math.sqrt(1.0 + second * second)
    v = [0.0] * 512
    v[0] = 1.0 / n
    v[1] = second / n
    return v


@pytest.mark.parametrize('with_snapshot', [False, True])
def test_session_vector_anchors_filter_before_topk(
        backend, backend_kind, with_snapshot):
    """`SqliteRecallSession.vector_anchors` itself filters before top-k.

    This is the PRIMARY vector path on an all-SQLite host — the
    fallback test below deliberately breaks the session verb, so it
    proves nothing about the verb's own eligibility filter (a
    mutation stripping `category`/`source` from `vector_anchors`
    left the whole suite green before this test existed). Covers
    both branches: `_meta_cache` (snapshot miss) and
    `snapshot.insights` (snapshot present).

    Mutation: dropping the eligibility filter from either branch of
        `SqliteRecallSession.vector_anchors` — the top-35 cut over
        all 70 vectors then keeps the 30 higher-similarity
        non-matching rows and only 5 matching vector hits survive.
    Oracle: all 35 results carry via='hybrid' (time + vector agree
        on the 35 newest matching rows, whose similarity rank
        matches their recency rank by construction).
    """
    from pathlib import Path

    from memman.store.factory import BACKENDS
    from memman.store.snapshot import write_snapshot
    if with_snapshot and not (
            BACKENDS[backend_kind].migrator_cls
            .snapshot_features.supports_recall_snapshot):
        pytest.skip(
            f'{backend_kind} declares supports_recall_snapshot=False:'
            f' the snapshot-present branch is unreachable, not untested')
    pref_ids = _seed(backend, 40, 'preference', 'quiet other subject {i}')
    fact_ids = _seed(backend, 30, 'fact', 'plain filler body {i}')
    for i, iid in enumerate(pref_ids):
        backend.nodes.update_embedding(
            iid, _vec512(0.3 + 0.002 * i), 'voyage-3-lite')
    for iid in fact_ids:
        backend.nodes.update_embedding(
            iid, _vec512(0.1), 'voyage-3-lite')
    if with_snapshot:
        sdir = str(Path(backend._db.path).parent)
        assert write_snapshot(
            backend._db, sdir, stored_fingerprint(backend)) is True
    qv = [0.0] * 512
    qv[0] = 1.0
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', qv, [], 35,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', category='preference')
    assert len(resp['results']) == 35
    assert all(r['insight'].category == 'preference'
               for r in resp['results'])
    assert all(r['via'] == 'hybrid' for r in resp['results'])


def test_vector_cache_fallback_fills_to_anchor_k(backend, monkeypatch):
    """The in-memory vector fallback searches a filtered input dict.

    Forces the fallback by making the session verb raise, then checks
    the fallback returned `anchor_k` MATCHING vector hits — not
    merely that the surviving hits match. Non-matching rows carry
    strictly higher similarity, so an unfiltered top-k is dominated
    by rows the filter then discards.

    Mutation: post-filtering `vector_search_from_cache`'s hits
        instead of filtering its input dict — the top-35 cut over all
        70 vectors keeps 30 non-matching rows and only 5 matching
        vector hits survive.
    Oracle: all 35 results carry via='hybrid' (time + vector agree on
        the 35 newest matching rows, whose similarity rank matches
        their recency rank by construction).
    """
    qv = [0.0] * 512
    qv[0] = 1.0
    pref_ids = _seed(backend, 40, 'preference', 'quiet other subject {i}')
    fact_ids = _seed(backend, 30, 'fact', 'plain filler body {i}')
    for i, iid in enumerate(pref_ids):
        backend.nodes.update_embedding(
            iid, _vec512(0.3 + 0.002 * i), 'test-model')
    for iid in fact_ids:
        backend.nodes.update_embedding(iid, _vec512(0.1), 'test-model')

    def _raise(self, query_vec, **kwargs):
        raise RuntimeError('forced session miss')

    # Patch the session class this backend actually yields: naming
    # SqliteRecallSession outright leaves the Postgres slot unpatched,
    # so its session verb never raises and the fallback under test is
    # never reached.
    with backend.recall_session(stored_fingerprint(backend)) as sess:
        session_cls = type(sess)
    monkeypatch.setattr(session_cls, 'vector_anchors', _raise)
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', qv, [], 35,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', category='preference')
    assert len(resp['results']) == 35
    assert all(r['insight'].category == 'preference'
               for r in resp['results'])
    assert all(r['via'] == 'hybrid' for r in resp['results'])


def test_filter_precedes_rerank(backend, monkeypatch):
    """The cross-encoder shortlist contains only filter-matching rows.

    Filtering after rerank spends the 100-slot cross-encoder window
    on rows about to be discarded.

    Mutation: moving the result filter below the rerank block.
    Oracle: a spy rerank client records the shortlist documents; every
        one must belong to the filtered category and none to the
        marker category.
    """
    seen_docs = []

    class _SpyRerank:
        def rerank(self, query, docs, top_k=None):
            seen_docs.extend(docs)
            return [(i, 1.0 - 0.01 * i) for i in range(len(docs))]

    monkeypatch.setattr(
        'memman.rerank.get_client', _SpyRerank)
    _seed(backend, 10, 'preference', 'alpha shared topic pref {i}')
    _seed(backend, 10, 'fact', 'alpha shared topic gen {i}')
    resp = intent_aware_recall(
        backend, 'alpha shared topic', None, [], 10,
        fingerprint=stored_fingerprint(backend),
        intent_override='GENERAL', rerank=True, category='preference')
    assert resp['meta']['reranked'] is True
    assert seen_docs, 'rerank spy never called'
    assert all('pref' in d for d in seen_docs)
    assert not any('gen' in d for d in seen_docs)
