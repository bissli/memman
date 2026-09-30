"""Store-layer contracts for replacement: the verb and the active predicate.

`nodes.mark_replaced(pred, succ)` sets `replaced_by` on a CURRENT row;
every active read then excludes the row exactly as it excludes a
soft-deleted one. These pin the verb's guard and the predicate at the
reads the pipeline and the doctor depend on.
"""

from tests.conftest import make_insight


def _seed_pair(backend):
    """Insert `p-1` and its successor `p-2`, both current.
    """
    backend.nodes.insert(make_insight(id='p-1', content='first statement'))
    backend.nodes.insert(make_insight(id='p-2', content='second statement'))


def test_mark_replaced_refuses_a_target_that_is_not_current(backend):
    """Verify `mark_replaced` links a current row once and never a gone one.

    Mutation: dropping `replaced_by is null` from the guard, so a
        second call re-points the chain (a fork); or dropping
        `deleted_at is null`, so a forgotten row acquires a successor.
    Oracle: the bool return and the pointer read back through
        `get_include_deleted` after each call.
    """
    _seed_pair(backend)
    backend.nodes.insert(make_insight(id='p-3', content='third statement'))
    backend.nodes.insert(make_insight(id='f-1', content='forgotten'))
    backend.nodes.soft_delete('f-1')

    assert backend.nodes.mark_replaced('p-1', 'p-2') is True
    assert backend.nodes.mark_replaced('p-1', 'p-3') is False
    assert backend.nodes.get_include_deleted('p-1').replaced_by == 'p-2'
    assert backend.nodes.mark_replaced('f-1', 'p-2') is False
    assert backend.nodes.get_include_deleted('f-1').replaced_by is None
    assert backend.nodes.mark_replaced('missing', 'p-2') is False


def test_replaced_row_is_not_returned_by_the_basic_listing(backend):
    """Verify every by-id and listing read excludes a replaced row.

    Mutation: leaving `replaced_by is null` off `query`, `get`,
        `get_all_active` or `count_active`.
    Oracle: the id sets each verb returns after one replacement,
        against `get_include_deleted`, which must still see the row.
    """
    _seed_pair(backend)
    assert backend.nodes.mark_replaced('p-1', 'p-2') is True

    assert [i.id for i in backend.nodes.query(limit=10)] == ['p-2']
    assert backend.nodes.get('p-1') is None
    assert backend.nodes.get_include_deleted('p-1').id == 'p-1'
    assert {i.id for i in backend.nodes.get_all_active()} == {'p-2'}
    assert backend.nodes.get_active_ids() == ['p-2']
    assert backend.nodes.count_active() == 1
    assert backend.nodes.count_total() == 2


def test_stats_reports_current_replaced_and_deleted_separately(backend):
    """Verify the three stats buckets partition every row exactly once.

    Mutation: counting replaced rows in `total_insights`, leaving
        them out of every bucket, or counting a row that is both
        replaced and forgotten twice.
    Oracle: 3 current, 1 replaced, 2 forgotten (one of them also
        replaced) -> (3, 1, 2), summing to `count_total`.
    """
    for n in range(3):
        backend.nodes.insert(make_insight(id=f'c-{n}', content=f'current {n}'))
    backend.nodes.insert(make_insight(id='s-1', content='replaced'))
    backend.nodes.insert(make_insight(id='s-2', content='replaced then gone'))
    backend.nodes.insert(make_insight(id='f-1', content='forgotten'))
    assert backend.nodes.mark_replaced('s-1', 'c-0') is True
    assert backend.nodes.mark_replaced('s-2', 'c-1') is True
    assert backend.nodes.soft_delete('s-2') is True
    assert backend.nodes.soft_delete('f-1') is True

    stats = backend.nodes.stats()
    assert (stats.total_insights, stats.replaced_insights,
            stats.deleted_insights) == (3, 1, 2)
    assert (stats.total_insights + stats.replaced_insights
            + stats.deleted_insights) == backend.nodes.count_total()


def test_pending_enrich_count_matches_its_id_list_after_replacement(
        backend):
    """Verify the count/iter maintenance pairs move together.

    Mutation: adding the predicate to `get_pending_enrich_ids` but not
        `count_pending_enrich` (or the reverse), so the re-enrich
        gate never reaches zero.
    Oracle: the count equals the length of the id list on both sides
        of the replacement.
    """
    _seed_pair(backend)
    assert backend.nodes.count_pending_enrich() == 2
    assert backend.nodes.mark_replaced('p-1', 'p-2') is True

    ids = backend.nodes.get_pending_enrich_ids(limit=100)
    assert ids == ['p-2']
    assert backend.nodes.count_pending_enrich() == len(ids)


def test_predecessors_read_back_on_both_backends(backend):
    """Verify the history walk's backward step on each backend.

    Mutation: a transposed column in `predecessors`' select, or a
        missing `replaced_by` predicate that returns every row.
    Oracle: `p-1` read back whole through `predecessors('p-2')`, and
        nothing through `predecessors('p-1')`.
    """
    _seed_pair(backend)
    assert backend.nodes.mark_replaced('p-1', 'p-2') is True

    preds = backend.nodes.predecessors('p-2')
    assert [(i.id, i.content, i.replaced_by) for i in preds] == [
        ('p-1', 'first statement', 'p-2')]
    assert backend.nodes.predecessors('p-1') == []
