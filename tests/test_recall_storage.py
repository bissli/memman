"""Storage-layer contracts the live recall path depends on.

`intent_aware_recall` reads the store on every request: the candidate
universe is `nodes.get_all_active()`.
"""

import pytest
from memman.search.recall import intent_aware_recall
from tests.conftest import make_insight


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
    Building a matrix over them would raise on a ragged `np.array`,
    taking down every recall until an operator repaired the store.

    Mutation: building one matrix over all widths, which raises on a
        ragged `np.array` and takes down every recall until an
        operator repairs the store.
    Oracle: three 512-wide rows against one 8-wide row, queried at
        512; the three must score and the outlier must be absent
        rather than fatal.

    Notes
    -----
    - Sqlite-only: pgvector's `vector(N)` column is fixed-width, so a
      Postgres store cannot hold two widths for this to exercise.
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


def test_superseded_row_is_not_returned_by_recall(backend):
    """Verify a superseded row leaves the candidate universe entirely.

    Mutation: omitting `superseded_by is null` from `get_all_active`,
        so the predecessor re-enters the pool and ranks beside its
        successor as an equal.
    Oracle: the returned id set against the three current rows, on
        both backends.
    """
    for n in range(4):
        backend.nodes.insert(
            make_insight(id=f'sup-{n}',
                         content=f'superseded probe body {n} kombu'))
    assert backend.nodes.supersede('sup-2', 'sup-3') is True

    resp = intent_aware_recall(
        backend, 'superseded probe kombu', None, 10)

    returned = {r['insight'].id for r in resp['results']}
    assert returned == {'sup-0', 'sup-1', 'sup-3'}


def test_minority_width_query_still_scores_its_own_rows(tmp_backend):
    """Verify a query at the LESS common width still scores its rows.

    A store part-way through `embed reembed` to a different-dimension
    model holds two widths while `bound_embedder` still produces query
    vectors at one of them. Scoring only the majority width blanks the
    entire vector channel for such a query, including the rows it can
    score.

    Mutation: reducing the stored embeddings to a single modal width
        and comparing every query against that one matrix - the exact
        shape this replaced.
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
    """Verify a blob that is not whole float64 values is skipped, not fatal.

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
    """Verify the matmul agrees with `cosine_similarity` to 1e-12.

    The session scores with one matrix-vector product where the old
    path called `cosine_similarity` per row. Both are float64, but
    BLAS sums a matrix-vector product in a different order than a
    per-pair dot, so the two agree to a float ulp rather than
    exactly. A real defect here -- a missing norm, a transposed
    matmul, rows misaligned with their ids -- lands far outside 1e-12.

    Mutation: dropping the query-norm divisor, dividing by the wrong
        axis's norms, or letting `_row_ids` drift out of step with the
        matrix rows.
    Oracle: `embed.vector.cosine_similarity` computed per row over
        the same vectors.

    Notes
    -----
    - The tolerance follows the backend's declared storage precision.
      SQLite keeps float64 blobs, so it is held to a float ulp;
      pgvector's `vector` is float4, declared as
      `embedding_dtype='float32'` in the Postgres migrator features,
      so single-precision epsilon is the floor there and demanding
      1e-12 of it would assert something the storage cannot
      represent.
    - `anchor_score` is min-max normalized over the query's own
      candidate pool, so a last-bit change in one similarity rescales
      every row. Ordering churn far larger than this tolerance is
      expected from any numeric change on this path, and is
      amplification rather than a logic difference.
    """
    from memman.embed.vector import cosine_similarity

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
        expected = cosine_similarity(query, vec)
        if expected > 0:
            assert iid in sims, f'{iid} scored {expected} but is absent'
            assert sims[iid] == pytest.approx(expected, abs=tolerance)
            checked += 1
        else:
            assert iid not in sims
    assert checked >= 4, f'only {checked} rows exercised the positive path'
