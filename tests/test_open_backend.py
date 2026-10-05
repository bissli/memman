"""Tests for `open_backend(store, data_dir)` dispatch.

Per-store backend dispatch reads `MEMMAN_BACKEND_<store>`
and falls back to `MEMMAN_DEFAULT_BACKEND`. Two stores in the same
process can pick different backends.
"""

import os
from pathlib import Path

import pytest
from memman import config
from memman.store.errors import ConfigError
from memman.store.factory import list_stores, open_backend
from memman.store.sqlite import SqliteBackend


def test_open_backend_uses_per_store_key_for_sqlite(tmp_path, env_file):
    """Verify `MEMMAN_BACKEND_<store>=sqlite` opens a SqliteBackend.

    Mutation: open_backend ignoring the per-store key and using another
        backend.
    Oracle: isinstance check against SqliteBackend.
    """
    env_file('MEMMAN_BACKEND_sqlite_only', 'sqlite')
    backend = open_backend(
        'sqlite_only', os.environ[config.DATA_DIR], create=True)
    try:
        assert isinstance(backend, SqliteBackend)
    finally:
        backend.close()


def test_open_backend_falls_back_to_default(tmp_path, env_file):
    """Verify a store with no per-store key uses `MEMMAN_DEFAULT_BACKEND`.

    Mutation: open_backend hard-coding a backend instead of reading the
        default key.
    Oracle: isinstance check against SqliteBackend with only the default set.
    """
    env_file('MEMMAN_DEFAULT_BACKEND', 'sqlite')
    backend = open_backend(
        'fresh_store', os.environ[config.DATA_DIR], create=True)
    try:
        assert isinstance(backend, SqliteBackend)
    finally:
        backend.close()


def test_open_backend_raises_for_unknown_backend_value(
        tmp_path, env_file):
    """Verify an unknown backend kind raises ConfigError.

    Mutation: falling back silently to sqlite for an unrecognized kind.
    Oracle: pytest.raises(ConfigError, match='unknown') for the value 'mongo'.
    """
    env_file('MEMMAN_BACKEND_weird', 'mongo')
    with pytest.raises(ConfigError, match='unknown'):
        open_backend('weird', os.environ[config.DATA_DIR])


@pytest.mark.postgres
def test_open_backend_routes_two_stores_to_two_backends(
        tmp_path, env_file, pg_dsn):
    """Verify a sqlite store and a postgres store open their own backends.

    Mutation: dispatching every store in a process to the first backend
        opened, such as a cached backend kind.
    Oracle: isinstance checks against SqliteBackend and PostgresBackend.
    """
    import psycopg
    from memman.store.postgres import PostgresBackend, _store_schema

    env_file('MEMMAN_BACKEND_local_one', 'sqlite')
    env_file('MEMMAN_BACKEND_pg_one', 'postgres')
    env_file('MEMMAN_POSTGRES_DSN_pg_one', pg_dsn)

    data_dir = os.environ[config.DATA_DIR]

    schema = _store_schema('pg_one')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    try:
        local = open_backend('local_one', data_dir, create=True)
        pg = open_backend('pg_one', data_dir, create=True)
        try:
            assert isinstance(local, SqliteBackend)
            assert isinstance(pg, PostgresBackend)
        finally:
            local.close()
            pg.close()
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_list_stores_returns_local_sqlite_dirs(tmp_path, env_file):
    """Verify list_stores returns the local SQLite store names.

    Mutation: list_stores skipping directories that hold a `memman.db`.
    Oracle: the two hand-made store directories 'one' and 'two'.
    """
    data_dir = Path(os.environ[config.DATA_DIR])
    (data_dir / 'data' / 'one').mkdir(parents=True, exist_ok=True)
    (data_dir / 'data' / 'one' / 'memman.db').write_bytes(b'')
    (data_dir / 'data' / 'two').mkdir(parents=True, exist_ok=True)
    (data_dir / 'data' / 'two' / 'memman.db').write_bytes(b'')

    env_file('MEMMAN_DEFAULT_BACKEND', 'sqlite')
    names = list_stores(str(data_dir))
    assert set(names) >= {'one', 'two'}
