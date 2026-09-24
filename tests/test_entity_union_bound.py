"""The predecessor entity union is bounded.

A successor unions each retired predecessor's entity list into its
own, so that the merged row still names what its predecessors' clauses
are about. The union was monotonic and unbounded: a row superseded
repeatedly accumulated every name any predecessor ever carried.

An inherited name whose carrier is now superseded matches no active
row, so it spends one `find_with_entity` query and no edge budget.
The edge loop therefore never breaks and runs the whole list, making
the query count per link pass grow without limit. The names also feed
the keyword channel, which has no length normalization, so a long
list can only raise a row's keyword score.

`MAX_ROW_ENTITIES` bounds the result. The successor's own names lead
the list, so the cut falls on the oldest inherited names -- the ones
that already produce no edge.
"""

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from memman.store.model import MAX_ROW_ENTITIES
from tests.conftest import make_insight
from tests.test_supersession_pipeline import _parent

OWN = ['redis', 'kombu']


def _stub_supersede(monkeypatch):
    """Arrange a write whose fact supersedes the seeded row."""
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: (
            ('CONTRADICTS', []) if memory[0] == 'old-1'
            else ('UNRELATED', [])))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)
    monkeypatch.setattr(
        'memman.graph.enrichment.enrich_with_llm',
        lambda insight, client: {})


def test_the_union_is_capped_at_max_row_entities(tmp_backend, monkeypatch):
    """Verify a successor's entity list stops at the cap.

    Mutation: unioning without the slice -- the defect itself -- so
        the successor carries every inherited name however many there
        are.
    Oracle: the cap constant against the stored list's length, with
        the predecessor seeded well past it so an off-by-one at the
        boundary cannot hide.
    """
    inherited = [f'inh{i}' for i in range(MAX_ROW_ENTITIES + 20)]
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu', entities=inherited))
    _stub_supersede(monkeypatch)

    parent = _parent('the broker is redis now')
    parent.entities = list(OWN)
    res = run_remember(
        tmp_backend, parent, 'the broker is redis now',
        ec=bound_embedder(tmp_backend), store_name='test')

    stored = tmp_backend.nodes.get(res['facts'][0]['id']).entities
    assert len(stored) == MAX_ROW_ENTITIES


def test_the_cut_falls_on_the_oldest_inherited_names(
        tmp_backend, monkeypatch):
    """Verify the successor's own names survive the cut, in order.

    The edge budget is spent in list order, so a cap that dropped the
    head would leave a row whose edges describe only its ancestors.

    Mutation: slicing the tail of the union, or unioning the
        predecessor's names ahead of the successor's, either of which
        drops the names taken from the text actually stored.
    Oracle: the two hand-named own entities, which must be the first
        two stored, against the last inherited name, which must be
        absent.
    """
    inherited = [f'inh{i}' for i in range(MAX_ROW_ENTITIES + 20)]
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu', entities=inherited))
    _stub_supersede(monkeypatch)

    parent = _parent('the broker is redis now')
    parent.entities = list(OWN)
    res = run_remember(
        tmp_backend, parent, 'the broker is redis now',
        ec=bound_embedder(tmp_backend), store_name='test')

    stored = tmp_backend.nodes.get(res['facts'][0]['id']).entities
    assert stored[:2] == OWN
    assert inherited[0] in stored
    assert inherited[-1] not in stored


def test_a_short_union_is_left_whole(tmp_backend, monkeypatch):
    """Verify a list under the cap keeps every inherited name.

    The paired control: a cap that truncated unconditionally, or an
    off-by-one reading the bound as exclusive, would pass the two
    tests above and silently shed names from an ordinary row.

    Mutation: slicing to a fixed length rather than to the cap, or
        applying the cap where the union is short.
    Oracle: the two hand-named own entities and the three inherited
        ones, every one of which must be present. The stored row also
        carries whatever the enrichment named, so this is a subset
        check rather than an equality.
    """
    inherited = ['kombu', 'celery', 'amqp']
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu', entities=inherited))
    _stub_supersede(monkeypatch)

    parent = _parent('the broker is redis now')
    parent.entities = list(OWN)
    res = run_remember(
        tmp_backend, parent, 'the broker is redis now',
        ec=bound_embedder(tmp_backend), store_name='test')

    stored = tmp_backend.nodes.get(res['facts'][0]['id']).entities
    assert set(OWN + inherited) <= set(stored)
