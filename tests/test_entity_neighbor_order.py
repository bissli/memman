"""The entity neighbor set is the same five rows on either backend.

`create_entity_edges` asks for at most `MAX_ENTITY_LINKS` carriers of
each entity and writes an edge to every one it gets, in both
directions. The set the query returns is therefore baked into the
stored graph at write time, so two stores holding identical content
must answer with identical neighbors or their entity graphs diverge
permanently.

Postgres used to order the candidates by `id`, a random uuid4, while
SQLite ordered by `created_at desc`. Both sides now order by
`created_at desc, id`, which makes each order total.

One key is not enough on its own. Postgres took `created_at` from the
transaction clock at microsecond resolution while SQLite stamped it
from a per-row clock read cut to the second, so rows written in
different transactions inside one second sorted one way on SQLite and
another on Postgres. Both backends now stamp the column from the same
Python clock at the same resolution, which is what makes the shared
ordering key mean the same thing on either side.
"""

from datetime import datetime, timedelta, timezone

from tests.conftest import make_insight, set_created_at

ENTITY = 'kombu'
BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def _seed_carriers(backend):
    """Store eight carriers of one entity, the newest three tied.

    Ids are chosen so that lexicographic order disagrees with age: the
    two oldest rows carry the lowest ids, which is what an `order by
    id` implementation returns and a `created_at desc` one never does.
    """
    ages = [
        ('aa-oldest', 7),
        ('ab-older', 6),
        ('zc-mid', 5),
        ('zd-mid', 4),
        ('ze-mid', 3),
        ('zf-tied', 0),
        ('zg-tied', 0),
        ('zh-tied', 0),
        ]
    for insight_id, days_old in ages:
        backend.nodes.insert(make_insight(
            id=insight_id, content=f'the broker is {insight_id}',
            entities=[ENTITY]))
        set_created_at(backend, insight_id, BASE - timedelta(days=days_old))
    return [insight_id for insight_id, _days in ages]


def test_entity_neighbors_are_the_newest_five_on_either_backend(backend):
    """Verify the neighbor set is newest-five, tiebroken by ascending id.

    Mutation: ordering by `id` (the shipped Postgres form), dropping
        the `, i.id` tiebreak so the three tied rows come back in
        query-plan order, or reversing the sort to oldest-first.
    Oracle: the hand-assigned ages. The three newest rows tie exactly,
        so their ascending ids decide the order, and the two oldest
        rows hold the lowest ids and must not appear at all.
    """
    _seed_carriers(backend)

    ids = backend.edges.find_with_entity(
        ENTITY, exclude_id='no-such-row', limit=5)

    assert ids == ['zf-tied', 'zg-tied', 'zh-tied', 'ze-mid', 'zd-mid']


def test_entity_neighbors_exclude_the_row_asking(backend):
    """Verify `exclude_id` drops the asking row without shortening the set.

    Mutation: dropping the `i.id <> ?` predicate, which would let a
        row link to itself and push the fifth-newest carrier out.
    Oracle: the hand-assigned ages with the newest carrier removed,
        so the set slides one row older.
    """
    _seed_carriers(backend)

    ids = backend.edges.find_with_entity(ENTITY, exclude_id='zf-tied', limit=5)

    assert ids == ['zg-tied', 'zh-tied', 'ze-mid', 'zd-mid', 'zc-mid']


def test_the_stamp_resolution_is_the_same_on_either_backend(backend):
    """Verify server-stamped `created_at` values agree across backends.

    The ordering key is only shared if it means the same thing on
    both sides. This test stamps through the production insert path
    rather than `set_created_at`, which writes one value to both
    backends and so cannot see a resolution mismatch.

    Mutation: letting Postgres fall back to its `default now()`, the
        transaction clock at microsecond resolution, so rows written
        inside one second order by sub-second time on Postgres and by
        id on SQLite -- a different neighbor SET for identical
        content.
    Oracle: the stamp's own text. A second-granular stamp has no
        sub-second field, so `created_at` formatted back must end in
        whole seconds and two rows inserted inside one second must
        compare equal.
    """
    for insight_id in ('res-1', 'res-2'):
        backend.nodes.insert(make_insight(
            id=insight_id, content=f'a row named {insight_id}',
            entities=[ENTITY]))

    first = backend.nodes.get('res-1').created_at
    second = backend.nodes.get('res-2').created_at

    assert first.microsecond == 0
    assert second.microsecond == 0
    assert first == second


def test_a_repeated_entity_name_yields_its_row_once(backend):
    """Verify a name stored twice on one row returns that row once.

    Mutation: matching entities with a lateral unnest instead of an
        `exists` subquery, which yields one result row per matching
        array element and spends two of the five link slots on one
        neighbor.
    Oracle: the two hand-seeded carriers, one of which holds the name
        twice in case variants that fold together under `lower(trim())`.
    """
    backend.nodes.insert(make_insight(
        id='dup-1', content='the broker is kombu',
        entities=[ENTITY, ENTITY.upper()]))
    set_created_at(backend, 'dup-1', BASE)
    backend.nodes.insert(make_insight(
        id='dup-2', content='kombu retries are exponential',
        entities=[ENTITY]))
    set_created_at(backend, 'dup-2', BASE - timedelta(days=1))

    ids = backend.edges.find_with_entity(
        ENTITY, exclude_id='no-such-row', limit=5)

    assert ids == ['dup-1', 'dup-2']
