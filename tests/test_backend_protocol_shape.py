"""Backend integrity-check and introspection behavior tests.

Asserts the architectural commitments that static typing cannot
verify on its own:

1. `Insight.created_at`, `Insight.updated_at`, `Edge.created_at`
   carry no `default_factory` -- backends stamp these server-side.
   A PR adding `default_factory=lambda: datetime.now(UTC)` would
   silently break Postgres `now()` parity at the verb boundary.
2. `NodeStore.update_embedding` takes a `vec` parameter (the
   blob->vec migration must not be reverted).
"""

import dataclasses
import inspect

import pytest
from memman.store.backend import NodeStore
from memman.store.model import Edge, Insight, OpLogEntry


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


def test_node_update_embedding_takes_vec_not_blob():
    """update_embedding signature uses vec, not blob (list[float])."""
    sig = inspect.signature(NodeStore.update_embedding)
    assert 'vec' in sig.parameters
    assert 'blob' not in sig.parameters


class TestBackendIntrospection:
    """Backend.integrity_check and Backend.introspect_columns behavior."""

    def test_integrity_check_returns_ok_on_fresh_store(self, backend):
        """integrity_check returns {'ok': True, ...} on a healthy fresh store."""
        result = backend.integrity_check()
        assert isinstance(result, dict)
        assert result.get('ok') is True
        assert 'detail' in result

    def test_introspect_columns_returns_insights_schema(self, backend):
        """introspect_columns('insights') returns the expected core columns."""
        cols = backend.introspect_columns('insights')
        assert isinstance(cols, set)
        expected_core = {
            'id', 'content', 'category', 'importance',
            'entities', 'source', 'created_at', 'updated_at',
            'embedding'}
        assert expected_core.issubset(cols), (
            f'missing core columns: {sorted(expected_core - cols)}; '
            f'got: {sorted(cols)}')

    def test_introspect_columns_unknown_table_returns_empty(self, backend):
        """introspect_columns on an unknown table returns an empty set."""
        cols = backend.introspect_columns('definitely_not_a_real_table')
        assert cols == set()

    def test_introspect_columns_rejects_unsafe_identifier(self, backend):
        """introspect_columns rejects names that are not valid SQL identifiers.

        SQL injection guard: PRAGMA / DDL identifier slots cannot be
        parameterized; both backends must validate the identifier before
        interpolation.
        """
        from memman.store.errors import ConfigError
        with pytest.raises(ConfigError):
            backend.introspect_columns('insights); DROP TABLE insights; --')
