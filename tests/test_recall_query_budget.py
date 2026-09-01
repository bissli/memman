"""Recall must issue a constant number of database queries.

Postgres is network-bound, so a per-frontier-node read is the
dominant cost there, and it grows with the traversal rather than with
anything the caller asked for. Nothing else in the suite can see that
shape: the results are identical either way and only the query count
moves.
"""

import pytest


@pytest.mark.postgres
def test_scored_recall_issues_constant_query_count(tmp_path, pg_dsn):
    """Verify one recall costs a small constant number of queries.

    Mutation: reinstating a per-frontier-node lookup - `edges.by_node`
        or `nodes.get` inside the traversal - which no other test can
        see, because the results are identical and only the query
        count explodes.
    Oracle: a counting wrapper around `psycopg.Cursor.execute`, whose
        total must stay under a ceiling far below the node count.

    Notes
    -----
    - Asserts a ceiling, not an exact count: connection setup runs
      pgvector type introspection whose call count is not this
      module's contract. The ceiling is what separates "constant" from
      "one per node", and 60 nodes with five edges each would put the
      old shape three orders of magnitude above it.
    """
    import psycopg
    from memman.search.recall import intent_aware_recall
    from memman.store.model import Edge
    from memman.store.postgres import drop_postgres_store
    from memman.store.postgres import open_postgres_backend
    from tests.conftest import make_insight

    store = 'query_budget'
    try:
        drop_postgres_store(store, pg_dsn)
    except Exception:
        pass
    backend = open_postgres_backend(store, pg_dsn)
    try:
        from memman.embed.fingerprint import seed_default_fingerprint
        backend.meta.set(
            'embed_fingerprint', seed_default_fingerprint().to_json())

        dim = 512
        node_count = 60
        ids = []
        for n in range(node_count):
            iid = f'budget-{n:03d}'
            ids.append(iid)
            backend.nodes.insert(make_insight(
                id=iid,
                content=f'query budget probe body {n} alpha beta gamma'))
            backend.nodes.update_embedding(
                iid,
                [0.01 * (((i * (n + 1)) % 13) - 6) for i in range(dim)],
                'test-model')
        for a in range(node_count):
            for b in range(a + 1, min(a + 6, node_count)):
                e = Edge()
                e.source_id, e.target_id = ids[a], ids[b]
                e.edge_type, e.weight = 'entity', 0.5
                backend.edges.upsert(e)

        query_vec = [0.01 * ((i % 13) - 6) for i in range(dim)]
        original = psycopg.Cursor.execute
        count = 0

        def counting(self, query, *args, **kwargs):
            nonlocal count
            count += 1
            return original(self, query, *args, **kwargs)

        psycopg.Cursor.execute = counting
        try:
            resp = intent_aware_recall(
                backend, 'query budget probe alpha beta gamma',
                query_vec, 10)
        finally:
            psycopg.Cursor.execute = original

        assert resp['meta']['traversed'] == node_count
        assert count <= 25, (
            f'{count} queries for a {node_count}-node recall: the'
            f' count must not scale with the node count')
    finally:
        backend.close()
        try:
            drop_postgres_store(store, pg_dsn)
        except Exception:
            pass
