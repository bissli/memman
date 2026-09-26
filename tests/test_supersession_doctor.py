"""Doctor checks for supersession: pointer integrity.

The pointer has no foreign key, so `check_supersession_integrity` is
the only enforcement of its validity.
"""

from memman.doctor import check_supersession_integrity
from memman.store.sqlite import SqliteBackend
from tests.conftest import make_insight


def _sql(backend, sqlite_sql, postgres_sql, params=()):
    """Run one raw statement against whichever backend is under test."""
    if isinstance(backend, SqliteBackend):
        backend._db._exec(sqlite_sql, params)
    else:
        with backend._conn.cursor() as cur:
            cur.execute(postgres_sql.format(s=backend._schema), params)
        backend._conn.commit()


def _point(backend, row_id, target):
    """Set `superseded_by` by raw SQL, touching nothing else."""
    _sql(backend,
         'update insights set superseded_by = ? where id = ?',
         'update {s}.insights set superseded_by = %s where id = %s',
         (target, row_id))


def test_integrity_passes_on_a_clean_chain_with_a_forgotten_target(backend):
    """Verify a well-formed chain passes even when the successor is forgotten.

    Mutation: treating a forgotten successor as dangling, which would
        fail every chain whose head was forgotten and make `forget` on
        a head a doctor failure.
    Oracle: two chains, one whose successor is soft-deleted, both
        built through the store verbs; every population empty.
    """
    for rid in ('c-1', 'c-2', 'c-3', 'c-4'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    assert backend.nodes.supersede('c-1', 'c-2') is True
    assert backend.nodes.supersede('c-3', 'c-4') is True
    assert backend.nodes.soft_delete('c-4') is True

    result = check_supersession_integrity(backend)
    assert result['name'] == 'supersession_integrity'
    assert result['status'] == 'pass'
    assert result['detail']['counts'] == {
        'dangling': 0, 'self_pointer': 0, 'unterminated': 0}


def test_integrity_fails_on_a_dangling_pointer(backend):
    """Verify a pointer at an id absent from the table fails the check.

    Mutation: counting pointers instead of resolving them against the
        table, or resolving them against the ACTIVE set (a forgotten
        target then reads as dangling).
    Oracle: one pointer at a never-stored id -> fail naming the row;
        the sibling pointing at a forgotten row stays out of the list.
    """
    for rid in ('d-1', 'd-2', 'd-3'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    _point(backend, 'd-1', 'ghost')
    assert backend.nodes.supersede('d-2', 'd-3') is True
    assert backend.nodes.soft_delete('d-3') is True

    result = check_supersession_integrity(backend)
    assert result['status'] == 'fail'
    assert result['detail']['dangling'] == ['d-1']
    assert result['detail']['counts']['dangling'] == 1


def test_integrity_fails_on_a_self_pointer_and_passes_a_join(backend):
    """Verify a self-pointer fails while two predecessors on one successor pass.

    Mutation: treating a join (a replace's predecessor plus a curated
        sibling converging on one successor) as a failure, which the
        live fleet's own supersession history trips on; or dropping
        the self-pointer population.
    Oracle: `m-1` and `m-2` both superseded by `m-3` pass every
        population; `s-1` pointing at itself, set by raw SQL, fails
        naming the row.
    """
    for rid in ('m-1', 'm-2', 'm-3', 's-1'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    assert backend.nodes.supersede('m-1', 'm-3') is True
    assert backend.nodes.supersede('m-2', 'm-3') is True
    joined = check_supersession_integrity(backend)
    assert joined['status'] == 'pass'
    assert 'multi_predecessor' not in joined['detail']

    _point(backend, 's-1', 's-1')
    result = check_supersession_integrity(backend)
    assert result['status'] == 'fail'
    assert result['detail']['self_pointer'] == ['s-1']


def test_integrity_fails_on_a_pointer_cycle(backend):
    """Verify a chain that never reaches a row without a pointer fails.

    A two-row cycle trips none of the other populations: both rows
    leave every active read and the doctor would pass.

    Mutation: dropping the `unterminated` population, or computing it
        as "pointer at a superseded row", which also flags every
        middle row of a legitimate chain.
    Oracle: `x-1 -> x-2 -> x-1` set by raw SQL fails naming both rows,
        while the legitimate chain `c-1 -> c-2 -> c-3` passes.
    """
    for rid in ('c-1', 'c-2', 'c-3', 'x-1', 'x-2'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    assert backend.nodes.supersede('c-1', 'c-2') is True
    assert backend.nodes.supersede('c-2', 'c-3') is True
    clean = check_supersession_integrity(backend)
    assert clean['status'] == 'pass'

    _point(backend, 'x-1', 'x-2')
    _point(backend, 'x-2', 'x-1')
    result = check_supersession_integrity(backend)
    assert result['status'] == 'fail'
    assert result['detail']['unterminated'] == ['x-1', 'x-2']
