"""Storage-layer contracts the live recall path depends on.

`intent_aware_recall` reads the store on every request: the candidate
universe is `nodes.get_all_active()` and the graph is one
`edges.adjacency()` read. These pin the two verbs that replaced the
removed recall snapshot.
"""

import pytest
from memman.search.recall import _bidirectional_adjacency, intent_aware_recall
from tests.conftest import make_insight


def _seed_graph(backend):
    """Insert four insights and five edges of mixed type and direction."""
    from memman.store.model import Edge
    ids = [f'adj-{c}' for c in 'abcd']
    for i, iid in enumerate(ids):
        backend.nodes.insert(
            make_insight(id=iid, content=f'adjacency body {i}'))
    specs = [
        (ids[0], ids[1], 'entity', 0.8),
        (ids[1], ids[2], 'semantic', 0.6),
        (ids[2], ids[3], 'causal', 1.0),
        (ids[0], ids[3], 'temporal', 0.3),
        (ids[3], ids[0], 'entity', 0.5),
        ]
    for source_id, target_id, edge_type, weight in specs:
        e = Edge()
        e.source_id = source_id
        e.target_id = target_id
        e.edge_type = edge_type
        e.weight = weight
        backend.edges.upsert(e)
    return ids


def test_adjacency_matches_edges_all(backend):
    """Verify the projection read reproduces `all()`'s graph exactly.

    Mutation: losing `edge_type` or `weight` in the projection, or
        keying the map on target instead of source - either silently
        changes which nodes traversal can reach.
    Oracle: the same adjacency rebuilt independently from
        `edges.all()`, whose full `Edge` dataclasses carry the columns
        the projection omits.

    Notes
    -----
    - The mirror is asserted separately, in
      `test_bidirectional_mirror_adds_the_reverse_hop`. Passing both
      sides of the comparison here through `_bidirectional_adjacency`
      would make any mutation of the mirror cancel out.
    """
    ids = _seed_graph(backend)

    from_projection = backend.edges.adjacency()
    from_dataclass: dict[str, list[tuple[str, str, float]]] = {}
    for e in backend.edges.all():
        from_dataclass.setdefault(e.source_id, []).append(
            (e.target_id, e.edge_type, e.weight))

    def normalize(adjacency):
        return {k: sorted(v) for k, v in adjacency.items()}

    assert normalize(from_projection) == normalize(from_dataclass)
    # Anti-vacuity: an empty projection would satisfy the equality.
    assert set(from_projection) == {ids[0], ids[1], ids[2], ids[3]}


def test_bidirectional_mirror_adds_the_reverse_hop():
    """Verify the mirror makes a one-way edge reachable from its target.

    Beam search walks edges as undirected, but `causal` edges and
    manual `graph link` rows are stored one-way only, so without the
    mirror the traversal can never reach a cause from its effect.

    Mutation: returning the directed map unchanged, or mirroring only
        some edge types.
    Oracle: a hand-built one-way map; the reverse entry is asserted
        by value, not against another call of the same helper.
    """
    directed = {'cause': [('effect', 'causal', 0.9)]}

    mirrored = _bidirectional_adjacency(directed)

    assert mirrored['cause'] == [('effect', 'causal', 0.9)]
    assert mirrored['effect'] == [('cause', 'causal', 0.9)]


def test_adjacency_does_not_mutate_its_input(backend):
    """Verify the bidirectional mirror leaves the directed map intact.

    Recall keeps the directed map for the source-keyed causal lookup
    after mirroring it for traversal, so an in-place mirror would make
    every causal edge look bidirectional and corrupt WHY ordering.

    Mutation: mirroring in place (`directed.setdefault(...).append`)
        instead of building a new map.
    Oracle: a deep copy of the directed map taken before mirroring.
    """
    _seed_graph(backend)
    directed = backend.edges.adjacency()
    before = {k: list(v) for k, v in directed.items()}
    _bidirectional_adjacency(directed)
    assert directed == before


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
        anchors = session.vector_anchors(query, k=10, min_sim=0.1)

    assert set(sims) == {'wide-0', 'wide-1', 'wide-2'}
    assert {a for a, _s in anchors} == {'wide-0', 'wide-1', 'wide-2'}


def test_vectors_for_ids_round_trips_on_both_backends(backend):
    """Verify `vectors_for_ids` returns the stored vector, by id.

    This is the verb the MMR block consumes. MMR ships disabled
    (`MMR_LAMBDA = 1.0`), so without a direct test the Postgres
    implementation - including its `pgvector_to_list` conversion -
    never executes at all.

    Mutation: returning the wrong row for an id, dropping the
        pgvector-to-list conversion, or returning every id rather
        than only the ones asked for.
    Oracle: distinct hand-built vectors, each identified by its own
        leading element.
    """
    dim = 512
    wanted = {}
    for n in range(3):
        iid = f'vfi-{n}'
        vec = [float(n + 1)] + [0.0] * (dim - 1)
        wanted[iid] = vec
        backend.nodes.insert(
            make_insight(id=iid, content=f'vectors-for-ids body {n}'))
        backend.nodes.update_embedding(iid, vec, 'test-model')
    backend.nodes.insert(
        make_insight(id='vfi-absent', content='never asked for'))

    with backend.recall_session() as session:
        got = session.vectors_for_ids(['vfi-0', 'vfi-2', 'vfi-absent'])

    assert set(got) == {'vfi-0', 'vfi-2'}
    assert got['vfi-0'][0] == pytest.approx(1.0)
    assert got['vfi-2'][0] == pytest.approx(3.0)
    assert len(got['vfi-0']) == dim


def test_dangling_edge_does_not_enter_the_candidate_pool(tmp_backend):
    """Verify an edge to a soft-deleted row scores nothing and costs nothing.

    A soft-delete leaves the row's edges in place, so the graph can
    point at an id `get_all_active()` does not return. Such a
    neighbour must not take a scored slot, a visit-budget slot, or a
    beam push, and must not inflate `meta.traversed`.

    Mutation: scoring an unresolvable neighbour anyway - the
        pre-change shape, where it entered `score_map`, consumed a
        `max_visited` slot and was pushed onto the beam.
    Oracle: `meta.traversed` and the returned id set, against a store
        whose only live rows are the two that were never deleted.
    """
    from memman.store.model import Edge

    for n in range(3):
        tmp_backend.nodes.insert(
            make_insight(id=f'dang-{n}',
                         content=f'dangling probe body {n} kombu'))
    for target in ('dang-1', 'dang-2'):
        e = Edge()
        e.source_id, e.target_id = 'dang-0', target
        e.edge_type, e.weight = 'entity', 0.9
        tmp_backend.edges.upsert(e)

    # Soft-delete WITHOUT touching edges, exactly as an out-of-band
    # write or a future soft-delete path would leave the graph.
    tmp_backend._db._exec(
        "update insights set deleted_at = '2026-01-01T00:00:00+00:00'"
        ' where id = ?', ('dang-2',))

    resp = intent_aware_recall(
        tmp_backend, 'dangling probe kombu', None, 10,
        intent_override='GENERAL')

    returned = {r['insight'].id for r in resp['results']}
    assert returned == {'dang-0', 'dang-1'}
    assert resp['meta']['traversed'] == 2


def test_minority_width_query_still_scores_its_own_rows(tmp_backend):
    """Verify a query at the LESS common width still scores its rows.

    A store part-way through `embed reembed` to a different-dimension
    model holds two widths while `bound_embedder` still produces query
    vectors at one of them. Scoring only the majority width blanks the
    entire vector channel for such a query, including the rows it can
    score, and `--min-score` then starts dropping rows too.

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
        anchors = session.vector_anchors(query, k=10, min_sim=0.1)
        vectors = session.vectors_for_ids(
            ['majority-0', 'minority-0'])

    assert set(sims) == {'minority-0', 'minority-1'}
    assert {a for a, _s in anchors} == {'minority-0', 'minority-1'}
    # vectors_for_ids reads each id at whatever width it was stored at
    assert len(vectors['majority-0']) == 8
    assert len(vectors['minority-0']) == 512


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
    - `graph_score` is min-max normalized over the query's own
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
