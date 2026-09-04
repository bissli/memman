"""Supersession reads identically to a soft delete, for the same history.

The active predicate gains a second clause at roughly ninety sites.
Enumerating them in tests would pin the list, not the property. This
builds the same store twice, supersedes the predecessor in one and
soft-deletes it in the other, and asserts every read, count and
edge-build agrees. One predicate site left on `deleted_at is null`
alone makes the two stores diverge somewhere below.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from memman.graph.entity import create_entity_edges
from memman.search.recall import intent_aware_recall
from memman.store.model import Edge
from tests.conftest import _mock_embed, make_insight, set_created_at

_ROWS = [
    ('p-1', 'alpha service moved to the kombu broker', ['kombu', 'alpha']),
    ('p-2', 'alpha service now uses the redis broker', ['redis', 'alpha']),
    ('q-1', 'beta dashboard reads alpha metrics', ['alpha', 'beta']),
    ('r-1', 'gamma pipeline runs nightly', ['gamma']),
    ('t-1', 'alpha deploys ride the tuesday train', ['alpha', 'train']),
    ('t-2', 'delta cache keys rotate hourly', ['delta']),
    ('t-3', 'epsilon exports land in the alpha bucket', ['alpha', 'epsilon']),
    ]
_EDGES = [
    ('p-1', 'q-1', 'entity', 0.8), ('q-1', 'p-1', 'entity', 0.8),
    ('p-1', 'p-2', 'semantic', 0.6), ('p-2', 'p-1', 'semantic', 0.6),
    ('q-1', 'r-1', 'semantic', 0.4), ('p-2', 'q-1', 'entity', 0.5),
    ('t-1', 'q-1', 'entity', 0.7), ('t-3', 't-1', 'semantic', 0.3),
    ]


@pytest.fixture
def twin_backends(request, backend_kind, tmp_path):
    """Two isolated backends of one kind, torn down together."""
    from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
    opened = []
    if backend_kind == 'sqlite':
        from memman.store.sqlite import drop_sqlite_store, open_sqlite_backend
        data_dir = str(tmp_path / 'memman')
        opened.extend(
            (name, open_sqlite_backend(name, data_dir))
            for name in ('twin_a', 'twin_b'))
        cleanup = [lambda n=n: drop_sqlite_store(n, data_dir) for n, _ in opened]
    else:
        from memman.store.postgres import drop_postgres_store
        from memman.store.postgres import open_postgres_backend
        from tests.conftest import _safe_store_name
        pg_dsn = request.getfixturevalue('pg_dsn')
        base = _safe_store_name(request.node.name)
        for suffix in ('a', 'b'):
            name = f'{base}_{suffix}'
            try:
                drop_postgres_store(name, pg_dsn)
            except Exception:
                pass
            opened.append((name, open_postgres_backend(name, pg_dsn)))
        cleanup = [lambda n=n: drop_postgres_store(n, pg_dsn) for n, _ in opened]
    for _, b in opened:
        b.meta.set(META_KEY, seed_default_fingerprint().to_json())
    try:
        yield opened[0][1], opened[1][1]
    finally:
        for _, b in opened:
            try:
                b.close()
            except Exception:
                pass
        for fn in cleanup:
            try:
                fn()
            except Exception:
                pass


def _build(backend, *, supersede):
    """Seed rows, vectors, enrichment and edges, then retire `p-1`."""
    embedder = SimpleNamespace(dim=512)
    # Distinct, fixed timestamps: the anchor pool orders on
    # created_at, and SQLite stamps whole seconds, so two stores
    # built seconds apart would tie-break differently.
    for n, (rid, content, entities) in enumerate(_ROWS):
        backend.nodes.insert(make_insight(
            id=rid, content=content, entities=entities,
            prompt_version='pv-1', model_id='m-1'))
        set_created_at(backend, rid,
                       datetime(2026, 3, 1, tzinfo=timezone.utc)
                       + timedelta(hours=n))
        backend.nodes.update_embedding(
            rid, _mock_embed(embedder, content), 'test-model')
    backend.nodes.update_enrichment(
        'p-1', keywords=['kombu'], summary='old broker', semantic_facts=['f'])
    backend.nodes.update_enrichment(
        'q-1', keywords=['beta'], summary='dashboard', semantic_facts=['g'])
    for source_id, target_id, edge_type, weight in _EDGES:
        backend.edges.upsert(Edge(
            source_id=source_id, target_id=target_id,
            edge_type=edge_type, weight=weight))
    if supersede:
        assert backend.nodes.supersede('p-1', 'p-2') is True
    else:
        assert backend.nodes.soft_delete('p-1') is True


def _recall_view(backend, query):
    """Ids, scores and traversal count of one recall, rounded for compare."""
    resp = intent_aware_recall(
        backend, query, None, 10, intent_override='GENERAL')
    rows = [(r['insight'].id, round(r['score'], 9),
             {k: round(v, 9) for k, v in r['signals'].items()})
            for r in resp['results']]
    return rows, resp['meta']['traversed']


def test_supersession_reads_identically_to_a_soft_delete(twin_backends):
    """Verify every read agrees between a superseded and a deleted predecessor.

    Mutation: any one of the active-predicate sites left on
        `deleted_at is null` alone -- `get_all_active` returns the
        row into the pool, `count_with_entity` inflates `doc_freq` so
        the fresh entity edge weight drops, `count_orphans` counts a
        phantom, `keyword_counts` scores it, `provenance_distribution`
        reports it stale.
    Oracle: store B, where the predecessor is soft-deleted, which is
        the shipped behavior every read already agrees on.
    """
    superseded, deleted = twin_backends
    _build(superseded, supersede=True)
    _build(deleted, supersede=False)

    for query in ('alpha broker kombu', 'beta dashboard metrics', 'gamma'):
        assert _recall_view(superseded, query) == _recall_view(deleted, query)
    assert {r['insight'].id for r in intent_aware_recall(
        superseded, 'alpha broker kombu', None, 10,
        intent_override='GENERAL')['results']}.isdisjoint({'p-1'})

    assert superseded.nodes.count_active() == deleted.nodes.count_active()
    assert superseded.nodes.count_orphans() == deleted.nodes.count_orphans()
    assert superseded.edges.degree_distribution() == deleted.edges.degree_distribution()
    assert superseded.nodes.enrichment_coverage() == deleted.nodes.enrichment_coverage()
    assert superseded.nodes.embedding_stats() == deleted.nodes.embedding_stats()
    assert (superseded.nodes.embedding_size_distribution()
            == deleted.nodes.embedding_size_distribution())
    assert (superseded.nodes.provenance_distribution()
            == deleted.nodes.provenance_distribution())
    assert (superseded.nodes.count_pending_links()
            == deleted.nodes.count_pending_links())
    assert (superseded.nodes.count_stale_insights('pv-2')
            == deleted.nodes.count_stale_insights('pv-2'))
    assert (superseded.nodes.stats().total_insights
            == deleted.nodes.stats().total_insights)
    assert (superseded.edges.count_dangling_by_type()
            == deleted.edges.count_dangling_by_type())
    assert (superseded.oplog.stats().total_active
            == deleted.oplog.stats().total_active)
    with superseded.recall_session() as sa, deleted.recall_session() as sb:
        assert (sa.keyword_counts({'alpha', 'kombu'})
                == sb.keyword_counts({'alpha', 'kombu'}))
        assert 'p-1' not in sa.keyword_counts({'kombu'})

    for backend in (superseded, deleted):
        fresh = make_insight(
            id='s-1', content='sigma note about alpha', entities=['alpha'])
        backend.nodes.insert(fresh)
        create_entity_edges(backend, fresh)
    weights = [
        sorted((e.target_id, e.edge_type, round(e.weight, 9))
               for e in b.edges.by_node('s-1') if e.source_id == 's-1')
        for b in (superseded, deleted)]
    assert weights[0] == weights[1]
    assert weights[0]
    assert all(target != 'p-1' for target, _, _ in weights[0])
