"""The vector anchor channel carries no absolute cosine floor.

`VECTOR_SEARCH_MIN_SIM = 0.10` used to gate this channel. A fixed
cosine means different things under different embedding models, so a
store whose cosines center low - a sparse store, or a weaker provider
- silently lost anchors to it. The floor is gone; only the sign
boundary remains, which is model-invariant because an orthogonal row
is orthogonal under every model.

Both tests here drive `intent_aware_recall` rather than
`vector_anchors` directly: the deleted constant lived at the recall
call site, so the verb alone cannot show the behavior either way.
"""

import math
from datetime import datetime, timedelta, timezone

from memman.search.recall import ANCHOR_TOP_K, intent_aware_recall
from tests.conftest import EMBEDDING_DIM, make_insight, set_created_at

QUERY = 'orthogonal probe request'


def _vec_at_cosine(cos: float, *, axis: int = 1) -> list[float]:
    """Unit vector whose cosine against `QUERY_VEC` is exactly `cos`.

    The first component carries the cosine and `axis` carries the
    remainder, so two vectors at the same cosine can still be made
    distinct from each other by choosing different axes.
    """
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = cos
    vec[axis] = math.sqrt(1.0 - cos * cos)
    return vec


QUERY_VEC = _vec_at_cosine(1.0, axis=1)

# Notes:
# - 0.05 straddles the deleted 0.10 floor from below while staying
#   positive, which is the whole point: it is the band the floor used
#   to swallow.
# - Postgres pins the embedding column at the fingerprint's width, so
#   these must be EMBEDDING_DIM wide, not a readable 4.
FAINT_VEC = _vec_at_cosine(0.05)
OPPOSED_VEC = _vec_at_cosine(-0.5)
CROWD_VEC = _vec_at_cosine(-0.5, axis=2)


def _seed_crowded_store(backend):
    """Fill the time channel with `ANCHOR_TOP_K` recent decoys.

    The two rows under test are dated a hundred days back so the
    recency channel cannot seed them, share no token with `QUERY` so
    the keyword channel cannot either, and carry no edges so
    traversal cannot reach them. The vector channel is their only
    route into the anchor set.
    """
    now = datetime.now(timezone.utc)
    for i in range(ANCHOR_TOP_K + 4):
        iid = f'crowd-{i}'
        backend.nodes.insert(
            make_insight(id=iid, content=f'zzz crowd body {i}'))
        backend.nodes.update_embedding(iid, CROWD_VEC, 'test-model')
        set_created_at(backend, iid, now - timedelta(minutes=i + 1))

    for iid, vec in (('faint', FAINT_VEC), ('opposed', OPPOSED_VEC)):
        backend.nodes.insert(
            make_insight(id=iid, content=f'zzz {iid} body'))
        backend.nodes.update_embedding(iid, vec, 'test-model')
        set_created_at(backend, iid, now - timedelta(days=100))


def _returned_ids(backend):
    """Ids `intent_aware_recall` returns for `QUERY`, unranked."""
    resp = intent_aware_recall(
        backend, QUERY, QUERY_VEC, ANCHOR_TOP_K + 10)
    return {r['insight'].id for r in resp['results']}


def test_faint_positive_cosine_still_anchors(backend):
    """Verify a 0.05-cosine row reaches the results with no other route.

    Mutation: reinstating any positive cosine floor on the vector
        anchor channel - the deleted 0.10, or a smaller one - which
        drops `faint` from every channel at once and so from the
        response.
    Oracle: a hand-built store where `faint`'s cosine is exactly its
        vector's first component, 0.05, straddling the old 0.10 floor
        from below; `crowd-*` occupy all ANCHOR_TOP_K recency slots.
    """
    _seed_crowded_store(backend)

    assert 'faint' in _returned_ids(backend)


def test_negative_cosine_does_not_anchor(backend):
    """Verify the sign boundary survives the floor's deletion.

    Mutation: dropping the positives-only filter along with
        `min_sim`, which would admit a row pointing away from the
        query as an anchor whenever fewer than k rows point toward
        it.
    Oracle: `opposed` sits at cosine -0.50 and, like `faint`, has no
        keyword, recency or edge route - so its presence would have
        to come from the vector channel.
    """
    _seed_crowded_store(backend)

    assert 'opposed' not in _returned_ids(backend)
