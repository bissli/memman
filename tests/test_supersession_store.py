"""Store-layer contracts for supersession: the verb and the active predicate.

`nodes.supersede(pred, succ)` sets `superseded_by` on a CURRENT row and
removes its edges; every active read then excludes the row exactly as
it excludes a soft-deleted one. These pin the verb's guard and the
predicate at the reads the pipeline and the doctor depend on.
"""

from memman.store.model import Edge
from tests.conftest import make_insight


def _edge(backend, source_id, target_id, edge_type='semantic', weight=0.7):
    """Upsert one directed edge."""
    backend.edges.upsert(Edge(
        source_id=source_id, target_id=target_id,
        edge_type=edge_type, weight=weight))


def _seed_pair(backend):
    """Insert `p-1` and its successor `p-2`, both current."""
    backend.nodes.insert(make_insight(id='p-1', content='first statement'))
    backend.nodes.insert(make_insight(id='p-2', content='second statement'))


def test_supersede_refuses_a_target_that_is_not_current(backend):
    """Verify `supersede` links a current row once and never a gone one.

    Mutation: dropping `superseded_by is null` from the guard, so a
        second call re-points the chain (a fork); or dropping
        `deleted_at is null`, so a forgotten row acquires a successor.
    Oracle: the bool return and the pointer read back through
        `get_include_deleted` after each call.
    """
    _seed_pair(backend)
    backend.nodes.insert(make_insight(id='p-3', content='third statement'))
    backend.nodes.insert(make_insight(id='f-1', content='forgotten'))
    backend.nodes.soft_delete('f-1')

    assert backend.nodes.supersede('p-1', 'p-2') is True
    assert backend.nodes.supersede('p-1', 'p-3') is False
    assert backend.nodes.get_include_deleted('p-1').superseded_by == 'p-2'
    assert backend.nodes.supersede('f-1', 'p-2') is False
    assert backend.nodes.get_include_deleted('f-1').superseded_by is None
    assert backend.nodes.supersede('missing', 'p-2') is False


def test_supersede_removes_the_predecessors_edges(backend):
    """Verify the verb leaves the predecessor edgeless in both directions.

    Mutation: a verb that sets the pointer but leaves the edges, so
        the walker keeps hopping through a row `get_all_active` no
        longer returns.
    Oracle: `edges.by_node` on the predecessor after the call, and
        the untouched edge between the two survivors.
    """
    _seed_pair(backend)
    backend.nodes.insert(make_insight(id='q-1', content='neighbor'))
    _edge(backend, 'p-1', 'q-1', 'entity', 0.8)
    _edge(backend, 'q-1', 'p-1', 'entity', 0.8)
    _edge(backend, 'p-2', 'p-1', 'semantic', 0.6)
    _edge(backend, 'q-1', 'p-2', 'semantic', 0.5)

    assert backend.nodes.supersede('p-1', 'p-2') is True

    assert backend.edges.by_node('p-1') == []
    remaining = {(e.source_id, e.target_id) for e in backend.edges.by_node('q-1')}
    assert remaining == {('q-1', 'p-2')}


def test_superseded_row_is_not_returned_by_the_basic_listing(backend):
    """Verify every by-id and listing read excludes a superseded row.

    Mutation: leaving `superseded_by is null` off `query`, `get`,
        `get_many`, `get_all_active` or `count_active`.
    Oracle: the id sets each verb returns after one supersession,
        against `get_include_deleted`, which must still see the row.
    """
    _seed_pair(backend)
    assert backend.nodes.supersede('p-1', 'p-2') is True

    assert [i.id for i in backend.nodes.query(limit=10)] == ['p-2']
    assert backend.nodes.get('p-1') is None
    assert backend.nodes.get_include_deleted('p-1').id == 'p-1'
    assert [i.id for i in backend.nodes.get_many(['p-1', 'p-2'])] == ['p-2']
    assert {i.id for i in backend.nodes.get_all_active()} == {'p-2'}
    assert backend.nodes.get_active_ids() == ['p-2']
    assert backend.nodes.count_active() == 1
    assert backend.nodes.count_total() == 2


def test_stats_reports_current_superseded_and_deleted_separately(backend):
    """Verify the three stats buckets partition every row exactly once.

    Mutation: counting superseded rows in `total_insights`, leaving
        them out of every bucket, or counting a row that is both
        superseded and forgotten twice.
    Oracle: 3 current, 1 superseded, 2 forgotten (one of them also
        superseded) -> (3, 1, 2), summing to `count_total`.
    """
    for n in range(3):
        backend.nodes.insert(make_insight(id=f'c-{n}', content=f'current {n}'))
    backend.nodes.insert(make_insight(id='s-1', content='superseded'))
    backend.nodes.insert(make_insight(id='s-2', content='superseded then gone'))
    backend.nodes.insert(make_insight(id='f-1', content='forgotten'))
    assert backend.nodes.supersede('s-1', 'c-0') is True
    assert backend.nodes.supersede('s-2', 'c-1') is True
    assert backend.nodes.soft_delete('s-2') is True
    assert backend.nodes.soft_delete('f-1') is True

    stats = backend.nodes.stats()
    assert (stats.total_insights, stats.superseded_insights,
            stats.deleted_insights) == (3, 1, 2)
    assert (stats.total_insights + stats.superseded_insights
            + stats.deleted_insights) == backend.nodes.count_total()
    assert sum(stats.by_category.values()) == 3


def test_increment_counters_ignore_a_superseded_row(backend):
    """Verify neither counter moves on a superseded row.

    Mutation: no `superseded_by is null` guard on
        `increment_access_count` or `increment_corroboration`.
    Oracle: both counters read back unchanged on the predecessor,
        `increment_corroboration` returns False, and the same calls
        still move the successor's counters.
    """
    _seed_pair(backend)
    assert backend.nodes.supersede('p-1', 'p-2') is True

    backend.nodes.increment_access_count('p-1')
    assert backend.nodes.increment_corroboration('p-1', queue_uuid='q-1') is False
    old = backend.nodes.get_include_deleted('p-1')
    assert (old.access_count, old.corroboration_count) == (0, 0)

    backend.nodes.increment_access_count('p-2')
    assert backend.nodes.increment_corroboration('p-2', queue_uuid='q-2') is True
    new = backend.nodes.get('p-2')
    assert (new.access_count, new.corroboration_count) == (1, 1)


def test_pending_link_count_matches_its_id_list_after_supersession(backend):
    """Verify the count/iter maintenance pairs move together.

    Mutation: adding the predicate to `get_pending_link_ids` but not
        `count_pending_links` (or the reverse), so the relink gate
        never reaches zero; or leaving `count_orphans` on
        `deleted_at` alone, so an edgeless superseded row reads as an
        orphan forever.
    Oracle: the count equals the length of the id list on both sides
        of the supersession, and the orphan pair counts only the
        current, edgeless successor.
    """
    _seed_pair(backend)
    assert backend.nodes.count_pending_links() == 2
    assert backend.nodes.supersede('p-1', 'p-2') is True

    ids = backend.nodes.get_pending_link_ids(limit=100)
    assert ids == ['p-2']
    assert backend.nodes.count_pending_links() == len(ids)
    assert backend.nodes.count_orphans() == (1, 1)


def test_predecessors_and_unsupersede_on_both_backends(backend):
    """Verify the backward walk and the compare-and-swap clear on each backend.

    Mutation: a transposed column in `predecessors`' select, or an
        `unsupersede` that ignores the expected successor and clears
        any pointer.
    Oracle: `p-1` read back whole through `predecessors('p-2')`; the
        clear refused for a stale successor and accepted for the right
        one, with the row current afterwards.
    """
    _seed_pair(backend)
    backend.nodes.insert(make_insight(id='p-3', content='third statement'))
    assert backend.nodes.supersede('p-1', 'p-2') is True

    preds = backend.nodes.predecessors('p-2')
    assert [(i.id, i.content, i.superseded_by) for i in preds] == [
        ('p-1', 'first statement', 'p-2')]
    assert backend.nodes.predecessors('p-1') == []

    assert backend.nodes.unsupersede('p-1', 'p-3') is False
    assert backend.nodes.get_include_deleted('p-1').superseded_by == 'p-2'
    assert backend.nodes.unsupersede('p-1', 'p-2') is True
    assert backend.nodes.get('p-1').superseded_by is None
    assert backend.nodes.unsupersede('p-1', 'p-2') is False
