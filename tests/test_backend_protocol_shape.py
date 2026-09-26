"""Backend integrity-check and introspection behavior tests.

Asserts the architectural commitments that static typing cannot
verify on its own:

1. `Insight.created_at`, `Insight.updated_at` carry no
   `default_factory` -- backends stamp these server-side. A PR
   adding `default_factory=lambda: datetime.now(UTC)` would
   silently break Postgres `now()` parity at the verb boundary.
2. `NodeStore.update_embedding` takes a `vec` parameter (the
   blob->vec migration must not be reverted).
"""

import dataclasses
import inspect

from memman.store.backend import NodeStore
from memman.store.model import Insight, OpLogEntry


def test_insight_timestamp_fields_have_no_default_factory():
    """Insight.created_at and Insight.updated_at: no default_factory.
    """
    fields = {f.name: f for f in dataclasses.fields(Insight)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)
    assert (
        fields['updated_at'].default_factory is dataclasses.MISSING)


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
    """Backend.integrity_check behavior."""

    def test_integrity_check_returns_ok_on_fresh_store(self, backend):
        """integrity_check returns {'ok': True, ...} on a healthy fresh store."""
        result = backend.integrity_check()
        assert isinstance(result, dict)
        assert result.get('ok') is True
        assert 'detail' in result
