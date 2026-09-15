"""One entity name, one edge, one stored form.

Three defects shared the list `create_entity_edges` walks.

- The edge primary key is `(source_id, target_id, edge_type)` and
  ignores the `entity` label, so a second entity naming a neighbor an
  earlier entity already linked spent two budget slots, created no
  row, and overwrote the first entity's label. The returned count went
  to the caller as `edges_created.entity`, so the reported number
  exceeded the edges that exist.
- `store/node.py::update_entities` folds case and whitespace variants
  on the way into the column, but the in-memory list does not, so the
  edge builder, the result JSON and the oplog delta all saw a list the
  store does not hold.
- A whitespace-only name from the model passed the `if e` filter,
  because a blank string is truthy, and merged under the key `''`.
"""

from memman.embed.fingerprint import bound_embedder
from memman.graph.entity import MAX_TOTAL_ENTITY_EDGES, create_entity_edges
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight
from tests.test_supersession_pipeline import _parent


def _stored_entity_edges(backend, source_id):
    """Return `(target_id, entity_label)` per entity edge on a row."""
    rows = []
    for edge in backend.edges.by_source_and_type(source_id, 'entity'):
        rows.append((edge.target_id, (edge.metadata or {}).get('entity')))
    return sorted(rows)


def test_the_edge_budget_is_spent_on_distinct_neighbors(tmp_backend):
    """Verify two entities sharing a neighbor cost one edge, not two.

    Mutation: counting every upsert attempt against
        `MAX_TOTAL_ENTITY_EDGES` rather than each distinct neighbor,
        so a repeated neighbor burns budget, writes no new row and
        relabels the edge an earlier entity created.
    Oracle: the entity edges actually present on the row, counted
        and read back off the store, against the count
        `create_entity_edges` returns.
    """
    for i in range(3):
        tmp_backend.nodes.insert(make_insight(
            id=f'peer-{i}', content=f'peer row {i}',
            entities=['shared-alpha', 'shared-beta']))
    row = make_insight(
        id='new-1', content='the new row',
        entities=['shared-alpha', 'shared-beta'])
    tmp_backend.nodes.insert(row)

    reported = create_entity_edges(tmp_backend, row)

    edges = _stored_entity_edges(tmp_backend, 'new-1')
    assert reported == len(edges) * 2
    assert sorted(t for t, _label in edges) == ['peer-0', 'peer-1', 'peer-2']
    assert {label for _t, label in edges} == {'shared-alpha'}


def test_a_rare_entity_is_not_starved_by_a_repeated_neighbor(tmp_backend):
    """Verify budget spent on duplicate neighbors does not strand a name.

    Mutation: letting repeated neighbors exhaust
        `MAX_TOTAL_ENTITY_EDGES` so a later entity in the list gets
        no edge at all.
    Oracle: the rare entity's own neighbor appears as an edge
        target, read off the store.
    """
    common = [f'common-{i:02d}' for i in range(30)]
    for i in range(6):
        tmp_backend.nodes.insert(make_insight(
            id=f'peer-{i}', content=f'peer row {i}', entities=list(common)))
    tmp_backend.nodes.insert(make_insight(
        id='rare-holder', content='the rare row', entities=['rare-name']))
    row = make_insight(
        id='new-1', content='the new row', entities=common + ['rare-name'])
    tmp_backend.nodes.insert(row)

    reported = create_entity_edges(tmp_backend, row)

    targets = [t for t, _label in _stored_entity_edges(tmp_backend, 'new-1')]
    assert reported <= MAX_TOTAL_ENTITY_EDGES
    assert 'rare-holder' in targets


def test_the_stored_and_reported_entity_lists_agree(tmp_backend,
                                                    monkeypatch):
    """Verify one name in two cases yields one entity everywhere.

    Mutation: deduping only inside `update_entities`, so the column
        folds the variants while the in-memory list the edge builder,
        the result JSON and the oplog delta read keeps both.
    Oracle: the stored column read back off the store, against the
        entity list the result dict reports.
    """
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': 'the broker is kombu', 'category': 'fact',
             'importance': 3, 'entities': ['kombu']}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('UNRELATED', []))

    parent = _parent('the broker')
    parent.entities = ['Kombu']
    res = run_remember(
        tmp_backend, parent, 'the broker',
        ec=bound_embedder(tmp_backend), store_name='test')

    stored = tmp_backend.nodes.get(res['facts'][0]['id']).entities
    assert res['facts'][0]['entities'] == stored
    assert len([e for e in stored if e.lower() == 'kombu']) == 1


def test_a_whitespace_only_entity_from_the_model_is_dropped(tmp_backend):
    """Verify a blank name never reaches the stored entity list.

    A blank string is truthy, so an `if e` filter keeps it and its
    merge key is the empty string.

    Mutation: filtering on the raw value rather than the stripped
        one, in either the enrichment merge or the fact extractor.
    Oracle: the stored entity list, which must hold no name that is
        empty after stripping.
    """
    from unittest.mock import MagicMock

    from memman.graph.enrichment import enrich_with_llm
    import json as _json

    insight = make_insight(id='ws-1', content='body', entities=[])
    client = MagicMock()
    client.complete.return_value = _json.dumps({
        'entities': ['   ', '\t', 'Redis'],
        'keywords': ['k'],
        'summary': 's',
        'semantic_facts': ['f'],
        })

    result = enrich_with_llm(insight, client)

    assert [e for e in result['entities'] if not e.strip()] == []
    assert 'Redis' in result['entities']
