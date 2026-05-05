"""Backend Protocol-shape regression guard.

Asserts the four distributed-shaping decisions baked into the
`Backend` Protocol surface:

1. Timestamp ownership at the boundary -- `NodeStore.insert`,
   `EdgeStore.upsert`, `Oplog.log`, `NodeStore.stamp_linked`,
   `NodeStore.stamp_enriched` accept no timestamp argument.
2. `Backend.write_lock` is a Protocol verb.
3. `Backend.transaction()` and `Backend.readonly_context()` are
   Protocol verbs returning context managers.
4. `Insight.created_at`, `Insight.updated_at`, `Edge.created_at`
   carry no `default_factory` -- backends stamp these server-side.

Guards against a future PR quietly retrofitting a `created_at`
parameter onto the verb signatures (which would defeat the Postgres
`now()` boundary).
"""

import dataclasses
import inspect

from memman.store.backend import Backend, EdgeStore, NodeStore, Oplog
from memman.store.model import Edge, Insight, OpLogEntry


def test_node_insert_has_no_timestamp_param():
    """NodeStore.insert(insight) -- backends stamp server-side."""
    params = inspect.signature(NodeStore.insert).parameters
    assert 'created_at' not in params
    assert 'updated_at' not in params
    assert 'now' not in params


def test_edge_upsert_has_no_timestamp_param():
    """EdgeStore.upsert(edge) -- backends stamp server-side."""
    params = inspect.signature(EdgeStore.upsert).parameters
    assert 'created_at' not in params
    assert 'now' not in params


def test_oplog_log_has_no_timestamp_param():
    """Oplog.log(operation, insight_id, detail) -- backend stamps now.
    """
    params = inspect.signature(Oplog.log).parameters
    assert 'created_at' not in params
    assert 'ts' not in params
    assert 'now' not in params


def test_node_stamp_linked_has_no_timestamp_param():
    """NodeStore.stamp_linked(id) -- backend stamps `linked_at` now."""
    params = inspect.signature(NodeStore.stamp_linked).parameters
    assert 'ts' not in params
    assert 'created_at' not in params


def test_node_stamp_enriched_has_no_timestamp_param():
    """NodeStore.stamp_enriched(id) -- backend stamps now."""
    params = inspect.signature(NodeStore.stamp_enriched).parameters
    assert 'ts' not in params
    assert 'created_at' not in params


def test_backend_write_lock_exists():
    """Backend.write_lock(name) is a Protocol verb (Phase 2.5 hook)."""
    assert hasattr(Backend, 'write_lock')
    sig = inspect.signature(Backend.write_lock)
    assert 'name' in sig.parameters


def test_backend_transaction_exists():
    """Backend.transaction() is a Protocol verb (nesting contract)."""
    assert hasattr(Backend, 'transaction')


def test_backend_readonly_context_exists():
    """Backend.readonly_context() is a Protocol verb (autocommit-on-PG).
    """
    assert hasattr(Backend, 'readonly_context')


def test_backend_recall_session_exists():
    """Backend.recall_session() is a Protocol verb."""
    assert hasattr(Backend, 'recall_session')


def test_insight_timestamp_fields_have_no_default_factory():
    """Insight.created_at and Insight.updated_at: no default_factory.
    """
    fields = {f.name: f for f in dataclasses.fields(Insight)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)
    assert (
        fields['updated_at'].default_factory is dataclasses.MISSING)


def test_edge_created_at_has_no_default_factory():
    """Edge.created_at: no default_factory."""
    fields = {f.name: f for f in dataclasses.fields(Edge)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)


def test_oplog_entry_created_at_is_required():
    """OpLogEntry.created_at: no default (DB-stamped on read)."""
    fields = {f.name: f for f in dataclasses.fields(OpLogEntry)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)
    assert fields['created_at'].default is dataclasses.MISSING


def test_node_iter_embeddings_as_vecs_exists():
    """Phase 1b: NodeStore.iter_embeddings_as_vecs Protocol verb."""
    assert hasattr(NodeStore, 'iter_embeddings_as_vecs')


def test_node_get_many_exists():
    """Phase 1b: NodeStore.get_many Protocol verb (bfs hydration)."""
    assert hasattr(NodeStore, 'get_many')
    sig = inspect.signature(NodeStore.get_many)
    assert 'ids' in sig.parameters


def test_edge_get_neighborhood_exists():
    """Phase 1b: EdgeStore.get_neighborhood Protocol verb."""
    assert hasattr(EdgeStore, 'get_neighborhood')
    sig = inspect.signature(EdgeStore.get_neighborhood)
    assert 'seed_id' in sig.parameters
    assert 'depth' in sig.parameters
    assert 'edge_filter' in sig.parameters


def test_node_update_embedding_takes_vec_not_blob():
    """Phase 1b: update_embedding signature flipped from bytes to list[float].
    """
    sig = inspect.signature(NodeStore.update_embedding)
    assert 'vec' in sig.parameters
    assert 'blob' not in sig.parameters
