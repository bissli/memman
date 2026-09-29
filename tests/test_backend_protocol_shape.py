"""Backend integrity-check and introspection tests.

Pins the commitments that static typing cannot verify:

1. `Insight.created_at`, `Insight.updated_at` carry no
   `default_factory` -- backends stamp these server-side. A PR
   adding `default_factory=lambda: datetime.now(UTC)` would
   silently break Postgres `now()` parity at the verb boundary.
2. `NodeStore.update_embedding` takes a `vec` parameter (the
   blob->vec migration must not be reverted).
"""

import dataclasses
import inspect

from memman.store import db
from memman.store.backend import NodeStore
from memman.store.model import Insight, OpLogEntry


def test_insight_timestamp_fields_have_no_default_factory():
    """Verify Insight created_at and updated_at have no default_factory.

    Mutation: adding default_factory=datetime.now to either field, which
        would stamp client-side and break server-side now() parity.
    Oracle: dataclasses.fields metadata, compared to MISSING.
    """
    fields = {f.name: f for f in dataclasses.fields(Insight)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)
    assert (
        fields['updated_at'].default_factory is dataclasses.MISSING)


def test_oplog_entry_created_at_is_required():
    """Verify OpLogEntry.created_at has neither a default nor a factory.

    Mutation: giving created_at a default value or default_factory, so the
        entry is stamped client-side instead of by the database.
    Oracle: dataclasses.fields metadata, compared to MISSING.
    """
    fields = {f.name: f for f in dataclasses.fields(OpLogEntry)}
    assert (
        fields['created_at'].default_factory is dataclasses.MISSING)
    assert fields['created_at'].default is dataclasses.MISSING


def test_node_update_embedding_takes_vec_not_blob():
    """Verify NodeStore.update_embedding takes vec and no blob parameter.

    Mutation: reverting the signature to the older blob parameter.
    Oracle: inspect.signature of the Protocol method.
    """
    sig = inspect.signature(NodeStore.update_embedding)
    assert 'vec' in sig.parameters
    assert 'blob' not in sig.parameters


def test_no_layer_keeps_storage_summary(backend):
    """Verify no layer keeps `storage_summary`, which nothing called.

    Mutation: deleting the Protocol method but keeping the SQLite or
        Postgres binding, or the SQLite helper in store/db.py.
    Oracle: the attribute list on a live backend of each kind, which
        covers the Protocol defaults as well as each binding.
    """
    assert not hasattr(backend, 'storage_summary')
    assert not hasattr(db, 'storage_summary')


class TestBackendIntrospection:
    """Backend.integrity_check behavior.
    """

    def test_integrity_check_returns_ok_on_fresh_store(self, backend):
        """Verify integrity_check reports ok with a detail on a fresh store.

        Mutation: integrity_check returning a failure or omitting the detail
            key on a healthy store.
        Oracle: the literal result keys and True for a new backend.
        """
        result = backend.integrity_check()
        assert isinstance(result, dict)
        assert result.get('ok') is True
        assert 'detail' in result
