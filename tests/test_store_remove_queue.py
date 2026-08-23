"""A failed store drop must not destroy the store's queued writes.

`factory.drop_store` purges the store's queue rows so the worker
stops retrying them against storage that is gone. Running that purge
in a `finally` applied it to the failure case too: a drop that raised
-- an unreachable Postgres, a name the backend rejects -- left the
store fully intact and its pending memories deleted.
"""

import pytest
from memman import queue as _queue
from memman.store import factory
from memman.store.errors import ConfigError
from tests.conftest import _set_env_file_value, invoke


@pytest.fixture
def runner(mm_runner):
    return mm_runner


def _queue_rows(data_dir, store):
    """Count queue rows currently held for `store`."""
    with _queue.queue_db(data_dir) as conn:
        row = conn.execute(
            'select count(*) from queue where store = ?',
            (store,)).fetchone()
    return row[0]


def _enqueue(data_dir, store):
    """Put one pending row on the queue for `store`."""
    with _queue.queue_db(data_dir) as conn:
        _queue.enqueue(conn, store=store, content='pending memory')


def test_failed_drop_keeps_queue_rows(tmp_path, monkeypatch):
    """A backend that refuses the drop leaves the queued writes alone.

    Mutation: purging in a `finally` rather than after a clean drop,
    which deletes the rows for a store that still exists.
    Oracle: the row count before the failed drop equals the count
    after it.
    """
    import memman.store.sqlite as sqlite_backend

    data_dir = str(tmp_path)
    (tmp_path / 'data' / 'shop').mkdir(parents=True)
    _enqueue(data_dir, 'shop')
    before = _queue_rows(data_dir, 'shop')

    def _refuse(store, data_dir):
        raise ConfigError('backend refused the drop')

    monkeypatch.setattr(sqlite_backend, 'drop_sqlite_store', _refuse)

    with pytest.raises(ConfigError):
        factory.drop_store('shop', data_dir)

    assert before == 1
    assert _queue_rows(data_dir, 'shop') == before


def test_successful_drop_still_purges_queue_rows(tmp_path):
    """A clean drop takes the store's queued writes with it.

    Without this the worker keeps re-attempting rows against a data
    dir that no longer exists, which is why the purge exists at all.

    Mutation: deleting the purge, or gating it on a condition that is
    never true, so rows outlive the store.
    Oracle: one row before, zero after, and the store dir gone.
    """
    data_dir = str(tmp_path)
    sdir = tmp_path / 'data' / 'shop'
    sdir.mkdir(parents=True)
    _enqueue(data_dir, 'shop')

    assert _queue_rows(data_dir, 'shop') == 1

    factory.drop_store('shop', data_dir)

    assert _queue_rows(data_dir, 'shop') == 0
    assert not sdir.exists()


def test_purge_failure_does_not_fail_a_clean_drop(tmp_path, monkeypatch):
    """A queue error after a successful drop stays a warning.

    The store is already gone at that point, so raising would report
    failure for work that succeeded and invite a retry that cannot
    help.

    Mutation: dropping the inner try/except around the purge, which
    turns a queue hiccup into a hard failure after irreversible work.
    Oracle: `drop_store` returns normally and the store dir is gone.
    """
    data_dir = str(tmp_path)
    sdir = tmp_path / 'data' / 'shop'
    sdir.mkdir(parents=True)

    def _boom(conn, store):
        raise RuntimeError('queue is busy')

    monkeypatch.setattr(_queue, 'purge_store', _boom)

    factory.drop_store('shop', data_dir)

    assert not sdir.exists()


def test_store_remove_reports_a_failed_drop_without_traceback(runner,
                                                              monkeypatch):
    """`store remove` surfaces a backend refusal as a CLI error.

    Mutation: leaving ConfigError unhandled in `store_remove`, so the
    operator gets a raw traceback instead of a message.
    Oracle: the escaped exception is not a ConfigError, and the store
    name appears in the output.
    """
    import memman.store.sqlite as sqlite_backend

    _, data_dir = runner
    invoke(runner, ['store', 'create', 'shop'])

    def _refuse(store, data_dir):
        raise ConfigError('backend refused the drop')

    monkeypatch.setattr(sqlite_backend, 'drop_sqlite_store', _refuse)

    result = invoke(runner, ['store', 'remove', 'shop', '--yes'])

    assert not isinstance(result.exception, ConfigError), result.output
    assert result.exit_code != 0
    assert 'shop' in result.output


def test_store_remove_keeps_env_keys_when_the_drop_fails(runner,
                                                         monkeypatch):
    """A failed drop leaves the store's routing keys in place.

    Removing them would strand a store that still exists: it would
    fall back to the default backend and its rows would be
    unreachable at the DSN that actually holds them.

    Mutation: writing the env-key removal before the drop, or
    unconditionally after it.
    Oracle: `MEMMAN_BACKEND_shop` still reads back after the failure.
    """
    import memman.store.sqlite as sqlite_backend
    from memman import config

    _, data_dir = runner
    invoke(runner, ['store', 'create', 'shop'])
    _set_env_file_value('MEMMAN_BACKEND_shop', 'sqlite')

    def _refuse(store, data_dir):
        raise ConfigError('backend refused the drop')

    monkeypatch.setattr(sqlite_backend, 'drop_sqlite_store', _refuse)

    invoke(runner, ['store', 'remove', 'shop', '--yes'])

    config.reset_file_cache()
    assert config.get('MEMMAN_BACKEND_shop') == 'sqlite'


def test_postgres_drop_translates_driver_errors():
    """An unreachable server surfaces as BackendError, not psycopg.

    `store remove` catches `BackendError`; a raw `psycopg.Error`
    would sail past that handler and reach the operator as a
    traceback, which is how an unreachable Postgres used to look.

    Mutation: re-raising the driver error unchanged instead of
    wrapping it.
    Oracle: `pytest.raises(BackendError)` against a DSN pointing at a
    closed port, plus `__cause__` preserving the driver error.
    """
    pytest.importorskip('psycopg')
    from memman.store.errors import BackendError
    from memman.store.postgres import drop_postgres_store

    with pytest.raises(BackendError) as caught:
        drop_postgres_store(
            'goodname', 'postgresql://u:p@127.0.0.1:1/nodb')

    assert caught.value.__cause__ is not None
