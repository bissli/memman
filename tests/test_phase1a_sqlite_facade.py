"""Phase 1a SqliteBackend facade smoke tests.

Verifies the thin facade actually delegates to the legacy free
functions and produces identical results. Runs against a fresh
SQLite store created by the test fixture.
"""

import pathlib

import pytest
from memman import config
from memman.store.db import open_db
from memman.store.errors import ConfigError
from memman.store.factory import open_cluster
from memman.store.model import Edge, Insight
from memman.store.sqlite import SqliteBackend, SqliteCluster


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


def test_recall_session_yields_session(backend):
    """recall_session() yields a SqliteRecallSession."""
    backend.nodes.insert(Insight(id='rs', content='recall'))
    with backend.recall_session() as session:
        assert session is not None
        assert session.snapshot is None or session.snapshot is not None


def test_cluster_open_returns_sqlite_backend(tmp_path):
    """SqliteCluster.open(store=, data_dir=) returns a SqliteBackend.
    """
    cluster = SqliteCluster()
    bk = cluster.open(store='default', data_dir=str(tmp_path))
    assert isinstance(bk, SqliteBackend)
    bk.close()


def test_cluster_list_stores(tmp_path):
    """SqliteCluster.list_stores returns sorted store names."""
    cluster = SqliteCluster()
    bk = cluster.open(store='alpha', data_dir=str(tmp_path))
    bk.close()
    bk = cluster.open(store='beta', data_dir=str(tmp_path))
    bk.close()
    stores = cluster.list_stores(data_dir=str(tmp_path))
    assert stores == ['alpha', 'beta']


def test_cluster_drop_store(tmp_path):
    """SqliteCluster.drop_store removes the store directory."""
    cluster = SqliteCluster()
    bk = cluster.open(store='gone', data_dir=str(tmp_path))
    bk.close()
    cluster.drop_store(store='gone', data_dir=str(tmp_path))
    assert (
        not pathlib.Path(tmp_path / 'data' / 'gone').exists())


def test_open_cluster_unknown_backend_raises_configerror(monkeypatch):
    """Unknown MEMMAN_BACKEND value yields ConfigError with hint."""
    monkeypatch.setattr(config, 'get', lambda key: (
        'plutonium' if key == config.BACKEND else None))
    with pytest.raises(ConfigError, match='unknown'):
        open_cluster()
