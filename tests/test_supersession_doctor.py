"""Doctor checks for supersession: pointer integrity and index predicates.

The pointer has no foreign key, so `check_supersession_integrity` is
the only enforcement of its validity. `create index if not exists`
matches by name, so a partial index redeclared in code keeps its old
predicate on a live store until dropped; `check_partial_index_predicates`
is what finds that.
"""

from memman.doctor import check_partial_index_predicates
from memman.doctor import check_supersession_integrity
from memman.store.model import Edge
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
        'dangling': 0, 'superseded_with_edges': 0,
        'multi_predecessor': 0, 'self_pointer': 0, 'unterminated': 0}


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


def test_integrity_fails_on_a_superseded_row_with_edges(backend):
    """Verify a superseded row that kept an edge fails the check.

    Mutation: dropping the edge population, so an entity-edge
        regression that links into history passes the doctor.
    Oracle: the pointer set by raw SQL after the edges exist, so the
        edges survive -> fail naming the row.
    """
    for rid in ('e-1', 'e-2'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    backend.edges.upsert(Edge(
        source_id='e-1', target_id='e-2', edge_type='semantic', weight=0.5))
    _point(backend, 'e-1', 'e-2')

    result = check_supersession_integrity(backend)
    assert result['status'] == 'fail'
    assert result['detail']['superseded_with_edges'] == ['e-1']


def test_integrity_fails_on_a_fork_and_a_self_pointer(backend):
    """Verify two predecessors on one successor, and a self-pointer, fail.

    Mutation: dropping either population; a hand edit that forks a
        chain or points a row at itself then passes.
    Oracle: `m-1` and `m-2` both pointing at `m-3`, and `s-1` pointing
        at itself, set by raw SQL -> the successor and the row named.
    """
    for rid in ('m-1', 'm-2', 'm-3', 's-1'):
        backend.nodes.insert(make_insight(id=rid, content=f'row {rid}'))
    _point(backend, 'm-1', 'm-3')
    _point(backend, 'm-2', 'm-3')
    _point(backend, 's-1', 's-1')

    result = check_supersession_integrity(backend)
    assert result['status'] == 'fail'
    assert result['detail']['multi_predecessor'] == ['m-3']
    assert result['detail']['self_pointer'] == ['s-1']


def test_partial_index_predicates_pass_on_a_fresh_store(backend):
    """Verify the shipped baseline declares every partial index correctly.

    Mutation: a baseline partial index whose WHERE names
        `deleted_at is null` without `superseded_by is null`.
    Oracle: the index definitions read back from the catalog of a
        store the baseline just created, on both backends.
    """
    backend.nodes.insert(make_insight(id='p-1', content='row'))
    result = check_partial_index_predicates(backend)
    assert result['name'] == 'partial_index_predicates'
    assert result['status'] == 'pass'
    assert result['detail']['stale'] == []
    # SQLite declares one partial index on insights (pending-link);
    # Postgres adds the GIN and HNSW ones.
    expected = 1 if isinstance(backend, SqliteBackend) else 3
    assert result['detail']['checked'] == expected


def test_partial_index_predicates_fail_on_a_stale_definition(backend):
    """Verify an index kept from the previous schema is reported by name.

    `create index if not exists` matches by NAME, so a live store
    migrated by hand keeps the old predicate until the index is
    dropped.

    Mutation: checking column presence instead of the index DDL, or
        matching `superseded_by` anywhere in the definition rather
        than inside the predicate.
    Oracle: the pending-link index recreated with the 0.32.x predicate
        -> fail naming that index and a remedy that says to drop it.
    """
    backend.nodes.insert(make_insight(id='p-1', content='row'))
    if isinstance(backend, SqliteBackend):
        name = 'idx_insights_pending_link'
        backend._db._exec(f'drop index {name}', ())
        backend._db._exec(
            f'create index {name} on insights(linked_at, created_at)'
            ' where linked_at is null and deleted_at is null', ())
    else:
        # Postgres truncates identifiers to 63 bytes, and the catalog
        # reports the truncated name.
        name = f'idx_insights_pending_link_{backend._schema}'[:63]
        with backend._conn.cursor() as cur:
            cur.execute(f'drop index {backend._schema}.{name}')
            cur.execute(
                f'create index {name} on {backend._schema}.insights'
                '(linked_at, created_at)'
                ' where linked_at is null and deleted_at is null')
        backend._conn.commit()

    result = check_partial_index_predicates(backend)
    assert result['status'] == 'fail'
    assert result['detail']['stale'] == [name]
    assert 'drop' in result['detail']['remedy']


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
    assert result['detail']['multi_predecessor'] == []


def test_partial_index_predicates_fail_on_a_retired_index(tmp_backend):
    """Verify the retired listing index is reported when a store still has it.

    `alter table` alone leaves `idx_insights_deleted_importance_created`
    in place, and it has no predicate for the stale check to read.

    Mutation: checking predicates only, so the retired index survives
        every doctor run as write amplification.
    Oracle: the old index recreated by name on a fresh store -> fail
        naming it under `retired`, with a remedy that says to drop it.
    """
    tmp_backend.nodes.insert(make_insight(id='p-1', content='row'))
    tmp_backend._db._exec(
        'create index idx_insights_deleted_importance_created'
        ' on insights(deleted_at, importance, created_at)', ())

    result = check_partial_index_predicates(tmp_backend)
    assert result['status'] == 'fail'
    assert result['detail']['retired'] == [
        'idx_insights_deleted_importance_created']
    assert result['detail']['stale'] == []
    assert 'drop' in result['detail']['remedy']
