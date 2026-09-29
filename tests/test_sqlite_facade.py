"""SqliteBackend facade smoke tests.

Verifies the thin facade delegates to the legacy free functions and
produces identical results. Runs against a fresh SQLite store created
by the test fixture.
"""

import os
import pathlib

import pytest
from memman import config
from memman.store.db import open_db, open_read_only
from memman.store.errors import BackendError, ConfigError
from memman.store.factory import drop_store, list_stores, open_backend
from memman.store.model import Insight
from memman.store.sqlite import SqliteBackend, drop_sqlite_store
from memman.store.sqlite import open_sqlite_backend


@pytest.fixture
def backend(tmp_path) -> SqliteBackend:
    """Open a SQLite store and wrap it in a SqliteBackend.
    """
    sdir = tmp_path / 'store'
    sdir.mkdir()
    db = open_db(str(sdir))
    return SqliteBackend(db)


def test_backend_path_exposes_db_path(backend):
    """SqliteBackend.path returns the DB file path.

    Mutation: `path` returning the store directory or a different file
    name.
    Oracle: the literal `memman.db` suffix.
    """
    assert backend.path.endswith('memman.db')


def test_node_insert_roundtrip(backend):
    """nodes.insert + nodes.get returns the same content.

    Mutation: `insert` dropping the content column or `get` reading
    the wrong row.
    Oracle: the hand-supplied `hello` read back by id.
    """
    ins = Insight(id='abc', content='hello', category='fact')
    backend.nodes.insert(ins)
    fetched = backend.nodes.get('abc')
    assert fetched is not None
    assert fetched.content == 'hello'
    assert fetched.created_at is not None


def test_node_insert_stamps_created_at_when_absent(backend):
    """Backend stamps created_at when the dataclass omits it.

    Mutation: `insert` storing a NULL `created_at` when the caller
    leaves it unset.
    Oracle: a non-None `created_at` read back for an Insight built
    without one.
    """
    ins = Insight(id='ts', content='no ts')
    assert ins.created_at is None
    backend.nodes.insert(ins)
    fetched = backend.nodes.get('ts')
    assert fetched.created_at is not None


def test_meta_get_set_roundtrip(backend):
    """meta.set + meta.get returns the same value.

    Mutation: `meta.get` returning a default string for an absent key
    instead of None, or `set` not persisting.
    Oracle: the hand-supplied `'1'` read back, and None for an absent key.
    """
    backend.meta.set('schema_version', '1')
    assert backend.meta.get('schema_version') == '1'
    assert backend.meta.get('absent_key') is None


def test_oplog_log_and_recent(backend):
    """oplog.log persists; oplog.recent returns it.

    Mutation: `oplog.log` not writing the row, or `recent` filtering
    out the row's insight id.
    Oracle: a logged entry for id `x` found among the recent rows.
    """
    backend.nodes.insert(Insight(id='x', content='entry'))
    backend.oplog.log(operation='add', insight_id='x', detail='entry')
    recent = backend.oplog.recent(limit=10)
    assert any(r.insight_id == 'x' for r in recent)


def test_transaction_commit(backend):
    """transaction() context commits on clean exit.

    Mutation: `transaction` skipping the commit on a clean exit.
    Oracle: the inserted row read back after the block.
    """
    with backend.transaction():
        backend.nodes.insert(Insight(id='c', content='committed'))
    assert backend.nodes.get('c').content == 'committed'


def test_transaction_rollback_on_exception(backend):
    """transaction() rolls back when the block raises.

    Mutation: `transaction` committing in a `finally`, or swallowing
    the error without rollback.
    Oracle: `nodes.get` returns None for the row inserted in the block.
    """
    with pytest.raises(RuntimeError, match='boom'), backend.transaction():
        backend.nodes.insert(Insight(id='r', content='will roll back'))
        raise RuntimeError('boom')
    assert backend.nodes.get('r') is None


def test_transaction_rollback_failure_does_not_mask_original(backend):
    """A failing rollback must not mask the original exception.

    Reproduces the worker-log crash: an inner op ends the transaction,
    then the body raises. The cleanup `rollback` then finds no active
    transaction and raises 'cannot rollback - no transaction is
    active'. The caller must still see the real error, not that one.

    Mutation: letting the cleanup `rollback` error propagate from the
    `except` leg of `transaction`, replacing the body's exception.
    Oracle: `pytest.raises(RuntimeError, match='boom')`.
    """
    with pytest.raises(RuntimeError, match='boom'), backend.transaction():
        backend._db._conn.execute('commit')
        raise RuntimeError('boom')


def test_open_sqlite_backend_returns_sqlite_backend(tmp_path):
    """open_sqlite_backend(store, data_dir) returns a SqliteBackend.

    Mutation: the factory returning the raw `Database` object.
    Oracle: `isinstance` against `SqliteBackend`.
    """
    bk = open_sqlite_backend('default', str(tmp_path))
    assert isinstance(bk, SqliteBackend)
    bk.close()


def test_list_stores_sqlite(tmp_path):
    """`list_stores` returns sorted SQLite store names.

    Mutation: `list_stores` returning names unsorted or omitting a
    store.
    Oracle: the hand-listed `['alpha', 'beta']`.
    """
    bk = open_sqlite_backend('alpha', str(tmp_path))
    bk.close()
    bk = open_sqlite_backend('beta', str(tmp_path))
    bk.close()
    assert list_stores(str(tmp_path)) == ['alpha', 'beta']


def test_drop_sqlite_store_removes_dir(tmp_path):
    """drop_sqlite_store removes the store directory.

    Mutation: `drop_sqlite_store` removing only `memman.db`, leaving
    the directory.
    Oracle: the store directory no longer exists.
    """
    bk = open_sqlite_backend('gone', str(tmp_path))
    bk.close()
    drop_sqlite_store('gone', str(tmp_path))
    assert (
        not pathlib.Path(tmp_path / 'data' / 'gone').exists())


def test_open_backend_unknown_kind_raises_configerror(env_file, tmp_path):
    """Unknown per-store backend value yields ConfigError with hint.

    Mutation: `open_backend` falling back to SQLite for an unknown
    backend value.
    Oracle: `ConfigError` matching `unknown backend`.
    """
    data_dir = os.environ[config.DATA_DIR]
    env_file(config.BACKEND_FOR('weird'), 'plutonium')
    with pytest.raises(ConfigError, match='unknown backend'):
        open_backend('weird', data_dir)


def test_drop_store_dispatches_to_sqlite(tmp_path):
    """factory.drop_store removes a SQLite store dir.

    Mutation: `drop_store` dispatching a SQLite store to the Postgres
    drop, or to nothing.
    Oracle: the store directory no longer exists.
    """
    bk = open_sqlite_backend('gone2', str(tmp_path))
    bk.close()
    drop_store('gone2', str(tmp_path))
    assert (
        not pathlib.Path(tmp_path / 'data' / 'gone2').exists())


def test_open_read_only_reports_a_missing_database(tmp_path):
    """`open_read_only` on a deleted store file raises `BackendError`.

    Mutation: reverting `open_read_only`'s missing-file leg to
        `FileNotFoundError`. Every caller wraps the call in
        `except Exception` and degrades silently, so a direct call is
        the only place the type is observable.
    Oracle: `BackendError` sits outside the `OSError` hierarchy, so
        `pytest.raises(BackendError)` discriminates the two.
    """
    bk = open_sqlite_backend('gone', str(tmp_path))
    store_dir = pathlib.Path(tmp_path) / 'data' / 'gone'
    try:
        (store_dir / 'memman.db').unlink()
        with pytest.raises(BackendError):
            open_read_only(str(store_dir))
    finally:
        bk.close()
