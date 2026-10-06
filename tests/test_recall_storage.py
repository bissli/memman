"""Storage-layer contracts the live recall path depends on.

`run_recall` reads the store on every request: the candidate
universe is `nodes.get_all_active()`.
"""

import numpy as np
import pytest
from memman.search.recall import run_recall
from tests.conftest import _vec, make_insight


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Per-pair float64 cosine similarity, the oracle for the matmul path.
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.dot(av, bv)) / float(
        np.linalg.norm(av) * np.linalg.norm(bv))


def test_similarities_omits_nonpositive_and_unembedded(backend):
    """Verify `similarities` returns positives only, keyed by id.

    `sim_cache` is read with `.get(id, 0.0)`, so a row that is absent
    and a row scoring 0.0 must be indistinguishable to the caller.

    Mutation: returning every row including non-positive cosines,
        which would let an anti-correlated row contribute a negative
        semantic term to the traversal score.
    Oracle: a hand-built store where one row's vector is the query,
        one is its exact negation, and one carries no embedding.
    """

    dim = 512
    same = [0.0] * dim
    same[0] = 1.0
    opposite = [0.0] * dim
    opposite[0] = -1.0

    for iid, vec in (('sim-same', same), ('sim-opposite', opposite),
                     ('sim-none', None)):
        backend.nodes.insert(make_insight(id=iid, content=f'body {iid}'))
        if vec is not None:
            backend.nodes.update_embedding(iid, vec, 'test-model')

    with backend.recall_session() as session:
        sims = session.similarities(same)

    assert sims['sim-same'] == pytest.approx(1.0)
    assert 'sim-opposite' not in sims
    assert 'sim-none' not in sims


def test_ragged_embedding_widths_do_not_break_recall(tmp_backend):
    """Verify a half-swapped store still recalls, off-width rows at 0.0.

    A partial `embed swap` leaves two embedding widths in one store.
    The test is SQLite-only: pgvector's `vector(N)` column is fixed
    width, so a Postgres store cannot hold two widths.

    Mutation: building one matrix over all widths, which raises on a
        ragged `np.array` and takes down every recall until an
        operator repairs the store.
    Oracle: three 512-wide rows against one 8-wide row, queried at
        512; the three must score and the outlier must be absent
        rather than fatal.
    """

    query = [0.0] * 512
    query[0] = 1.0
    for i in range(3):
        iid = f'wide-{i}'
        tmp_backend.nodes.insert(
            make_insight(id=iid, content=f'modal width body {i}'))
        tmp_backend.nodes.update_embedding(iid, query, 'test-model')
    tmp_backend.nodes.insert(
        make_insight(id='narrow-0', content='off width body'))
    tmp_backend.nodes.update_embedding(
        'narrow-0', [0.5] * 8, 'test-model')

    with tmp_backend.recall_session() as session:
        sims = session.similarities(query)
        anchors = session.vector_anchors(query, k=10)

    assert set(sims) == {'wide-0', 'wide-1', 'wide-2'}
    assert {a for a, _s in anchors} == {'wide-0', 'wide-1', 'wide-2'}


def test_replaced_row_is_not_returned_by_recall(backend):
    """Verify a replaced row leaves the candidate universe entirely.

    Mutation: omitting `replaced_by is null` from `get_all_active`,
        so the predecessor re-enters the pool and ranks beside its
        successor as an equal.
    Oracle: the returned id set against the three current rows, on
        both backends.
    """
    for n in range(4):
        backend.nodes.insert(
            make_insight(id=f'sup-{n}',
                         content=f'replaced probe body {n} kombu'))
    assert backend.nodes.mark_replaced('sup-2', 'sup-3') is True

    resp = run_recall(
        backend, 'replaced probe kombu', None, 10)

    returned = {r['insight'].id for r in resp['results']}
    assert returned == {'sup-0', 'sup-1', 'sup-3'}


def test_vector_anchors_accepts_a_k_past_the_search_width_cap(backend):
    """Verify a large k returns anchors in place of raising.

    Mutation: Postgres ef_search set to 4 * k with no clamp, which
        pgvector refuses above 1000, so a branch over-ask drops the
        vector channel.
    Oracle: one embedded row, which a working search returns. A first
        small search loads pgvector, which only then checks the width.
    """
    backend.nodes.insert(make_insight(id='row-a', content='grackle'))
    backend.nodes.update_embedding('row-a', _vec(1.0), 'model-x')

    with backend.recall_session() as session:
        session.vector_anchors(_vec(1.0), k=1)
        anchors = session.vector_anchors(_vec(1.0), k=300)

    assert [row_id for row_id, _ in anchors] == ['row-a']


def test_minority_width_query_still_scores_its_own_rows(tmp_backend):
    """Verify a query at the LESS common width still scores its rows.

    A store part-way through `embed reembed` to a different-dimension
    model holds two widths while `bound_embedder` still produces query
    vectors at one of them. Scoring only the majority width blanks the
    entire vector channel for such a query, including the rows it can
    score.

    Mutation: reducing the stored embeddings to a single modal width
        and comparing every query against that one matrix.
    Oracle: five rows at width A against two at width B, queried at
        B; the two B rows must score and the five A rows must not.
    """
    query = [1.0] + [0.0] * 511
    for i in range(5):
        iid = f'majority-{i}'
        tmp_backend.nodes.insert(
            make_insight(id=iid, content=f'majority width body {i}'))
        tmp_backend.nodes.update_embedding(iid, [0.5] * 8, 'test-model')
    for i in range(2):
        iid = f'minority-{i}'
        tmp_backend.nodes.insert(
            make_insight(id=iid, content=f'minority width body {i}'))
        tmp_backend.nodes.update_embedding(iid, query, 'test-model')

    with tmp_backend.recall_session() as session:
        sims = session.similarities(query)
        anchors = session.vector_anchors(query, k=10)

    assert set(sims) == {'minority-0', 'minority-1'}
    assert {a for a, _s in anchors} == {'minority-0', 'minority-1'}


def test_malformed_embedding_blob_does_not_break_recall(tmp_backend):
    """Verify a truncated float64 blob is skipped without failing recall.

    `np.frombuffer(blob, dtype='<f8')` raises on a length that is not
    a multiple of 8. Raising inside the session build would take down
    the whole vector channel rather than one row.

    Mutation: dropping the `len(blob) % 8` guard in `_load`.
    Oracle: two well-formed rows alongside one truncated blob written
        directly to the column; the two must still score.
    """
    query = [1.0] + [0.0] * 511
    for i in range(2):
        iid = f'sound-{i}'
        tmp_backend.nodes.insert(
            make_insight(id=iid, content=f'sound body {i}'))
        tmp_backend.nodes.update_embedding(iid, query, 'test-model')
    tmp_backend.nodes.insert(
        make_insight(id='truncated-0', content='truncated body'))
    tmp_backend._db._exec(
        'update insights set embedding = ? where id = ?',
        (b'\x00' * 13, 'truncated-0'))

    with tmp_backend.recall_session() as session:
        sims = session.similarities(query)

    assert set(sims) == {'sound-0', 'sound-1'}


def test_similarities_matches_per_pair_cosine(backend, backend_kind):
    """Verify the matmul agrees with a per-pair cosine to storage precision.

    Mutation: dropping the query-norm divisor, dividing by the wrong
        axis's norms, or letting `_row_ids` drift out of step with the
        matrix rows. Each lands far outside the tolerance.
    Oracle: `_cosine_similarity` computed per row over the same
        vectors.
    """
    # Notes:
    # - BLAS sums a matrix-vector product in a different order than a
    #   per-pair dot, so the two agree to a float ulp, never exactly.
    # - SQLite keeps float64 blobs and is held to a float ulp.
    #   pgvector's `vector` stores float4, so single-precision epsilon
    #   is its floor.
    # - `anchor_score` is min-max normalized over the query's own
    #   candidate pool, so a last-bit change in one similarity
    #   rescales every row. Ordering churn far larger than this
    #   tolerance follows from any numeric change on this path.
    tolerance = 1e-6 if backend_kind == 'postgres' else 1e-12
    dim = 512
    query = [0.03 * ((i % 7) - 3) for i in range(dim)]
    vectors = {}
    for n in range(12):
        iid = f'parity-{n}'
        vec = [0.01 * (((i * (n + 2)) % 11) - 5) for i in range(dim)]
        vectors[iid] = vec
        backend.nodes.insert(
            make_insight(id=iid, content=f'parity body {n}'))
        backend.nodes.update_embedding(iid, vec, 'test-model')

    with backend.recall_session() as session:
        sims = session.similarities(query)

    checked = 0
    for iid, vec in vectors.items():
        expected = _cosine_similarity(query, vec)
        if expected > 0:
            assert iid in sims, f'{iid} scored {expected} but is absent'
            assert sims[iid] == pytest.approx(expected, abs=tolerance)
            checked += 1
        else:
            assert iid not in sims
    assert checked >= 4, f'only {checked} rows exercised the positive path'
