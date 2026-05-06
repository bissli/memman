"""SqliteBackend facade smoke tests.

Verifies the thin facade delegates to the legacy free functions and
produces identical results. Runs against a fresh SQLite store created
by the test fixture.
"""

import pathlib

import pytest
from memman import config
from memman.store.db import open_db
from memman.store.errors import ConfigError
from memman.store.factory import drop_store, list_stores, open_backend
from memman.store.model import Edge, Insight
from memman.store.sqlite import SqliteBackend, drop_sqlite_store
from memman.store.sqlite import open_sqlite_backend


@pytest.fixture
def backend(tmp_path) -> SqliteBackend:
    """Open a SQLite store and wrap it in a SqliteBackend."""
    sdir = tmp_path / 'store'
    sdir.mkdir()
    db = open_db(str(sdir))
    return SqliteBackend(db)


def test_backend_path_exposes_db_path(backend):
    """SqliteBackend.path returns the DB file path."""
    assert backend.path.endswith('memman.db')


def test_node_insert_roundtrip(backend):
    """nodes.insert + nodes.get returns the same content."""
    ins = Insight(id='abc', content='hello', category='fact', importance=4)
    backend.nodes.insert(ins)
    fetched = backend.nodes.get('abc')
    assert fetched is not None
    assert fetched.content == 'hello'
    assert fetched.created_at is not None


def test_node_insert_stamps_created_at_when_absent(backend):
    """Backend stamps created_at when the dataclass omits it."""
    ins = Insight(id='ts', content='no ts')
    assert ins.created_at is None
    backend.nodes.insert(ins)
    fetched = backend.nodes.get('ts')
    assert fetched.created_at is not None


def test_edge_upsert_roundtrip(backend):
    """edges.upsert + edges.by_node returns the edge."""
    backend.nodes.insert(Insight(id='a', content='A'))
    backend.nodes.insert(Insight(id='b', content='B'))
    edge = Edge(source_id='a', target_id='b', edge_type='semantic',
                weight=0.7)
    backend.edges.upsert(edge)
    edges = backend.edges.by_node('a')
    assert len(edges) == 1
    assert edges[0].weight == 0.7
    assert edges[0].created_at is not None


def test_meta_get_set_roundtrip(backend):
    """meta.set + meta.get returns the same value."""
    backend.meta.set('schema_version', '1')
    assert backend.meta.get('schema_version') == '1'
    assert backend.meta.get('absent_key') is None


def test_oplog_log_and_recent(backend):
    """oplog.log persists; oplog.recent returns it."""
    backend.nodes.insert(Insight(id='x', content='entry'))
    backend.oplog.log(operation='add', insight_id='x', detail='entry')
    recent = backend.oplog.recent(limit=10)
    assert any(r.insight_id == 'x' for r in recent)


def test_transaction_commit(backend):
    """transaction() context commits on clean exit."""
    with backend.transaction():
        backend.nodes.insert(Insight(id='c', content='committed'))
    assert backend.nodes.get('c').content == 'committed'


def test_transaction_rollback_on_exception(backend):
    """transaction() rolls back when the block raises."""
    try:
        with backend.transaction():
            backend.nodes.insert(Insight(id='r', content='will roll back'))
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert backend.nodes.get('r') is None


def test_write_lock_is_no_op_on_sqlite(backend):
    """SQLite write_lock() is a no-op context manager."""
    with backend.write_lock('test'):
        backend.nodes.insert(Insight(id='wl', content='in lock'))
    assert backend.nodes.get('wl') is not None


def test_readonly_context_yields_separate_backend(backend):
    """readonly_context() yields a Backend whose writes would fail."""
    backend.nodes.insert(Insight(id='ro', content='read me'))
    with backend.readonly_context() as ro:
        assert ro is not backend
        assert ro.nodes.get('ro').content == 'read me'


def test_open_sqlite_backend_returns_sqlite_backend(tmp_path):
    """open_sqlite_backend(store, data_dir) returns a SqliteBackend.
    """
    bk = open_sqlite_backend('default', str(tmp_path))
    assert isinstance(bk, SqliteBackend)
    bk.close()


def test_list_stores_sqlite(tmp_path):
    """`list_stores` returns sorted SQLite store names."""
    bk = open_sqlite_backend('alpha', str(tmp_path))
    bk.close()
    bk = open_sqlite_backend('beta', str(tmp_path))
    bk.close()
    assert list_stores(str(tmp_path)) == ['alpha', 'beta']


def test_drop_sqlite_store_removes_dir(tmp_path):
    """drop_sqlite_store removes the store directory."""
    bk = open_sqlite_backend('gone', str(tmp_path))
    bk.close()
    drop_sqlite_store('gone', str(tmp_path))
    assert (
        not pathlib.Path(tmp_path / 'data' / 'gone').exists())


def test_open_backend_unknown_kind_raises_configerror(env_file, tmp_path):
    """Unknown per-store backend value yields ConfigError with hint."""
    import os
    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND_FOR('weird'), 'plutonium')
    with pytest.raises(ConfigError, match='unknown backend'):
        open_backend('weird', data_dir)


def test_drop_store_dispatches_to_sqlite(tmp_path):
    """factory.drop_store removes a SQLite store dir."""
    bk = open_sqlite_backend('gone2', str(tmp_path))
    bk.close()
    drop_store('gone2', str(tmp_path))
    assert (
        not pathlib.Path(tmp_path / 'data' / 'gone2').exists())
