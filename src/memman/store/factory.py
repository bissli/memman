"""Cluster factory dispatch on `MEMMAN_BACKEND`.

Mirrors `embed/__init__.py`'s registry shape: a name -> factory
mapping plus a single `open_cluster()` entry point.
"""

import os
from collections.abc import Callable

from memman import config
from memman.store.backend import Backend, Cluster
from memman.store.config import validate_for
from memman.store.errors import ConfigError


def _sqlite_factory() -> Cluster:
    """Lazy import to avoid pulling sqlite3 wiring at module load."""
    from memman.store.sqlite import SqliteCluster
    return SqliteCluster()


def _postgres_factory() -> Cluster:
    """Lazy import: psycopg + pgvector are an optional dependency."""
    from memman.store.postgres import PostgresCluster
    return PostgresCluster()


BACKENDS: dict[str, Callable[[], Cluster]] = {
    'sqlite': _sqlite_factory,
    'postgres': _postgres_factory,
    }


def open_cluster() -> Cluster:
    """Return the Cluster instance for `MEMMAN_BACKEND`.

    Defaults to 'sqlite' when unset (matches install wizard default).
    Validates the active backend's `MEMMAN_<NS>_*` namespace before
    instantiation so a typo'd key (e.g. `MEMMAN_PG_DSL`) surfaces
    with a `did you mean` hint instead of a connection error.
    """
    raw = config.get(config.BACKEND) or 'sqlite'
    name = raw.lower()
    factory = BACKENDS.get(name)
    if factory is None:
        known = ', '.join(sorted(BACKENDS))
        raise ConfigError(
            f'unknown {config.BACKEND}={name!r}; registered: {known}')
    merged = dict(os.environ)
    merged.update(
        config.parse_env_file(config.env_file_path()))
    validate_for(name, merged)
    return factory()


def _resolve_store_backend(store: str, data_dir: str) -> str:
    """Return the backend kind for `store`: per-store, then default.

    Reads `MEMMAN_BACKEND_<store>`, falling back to
    `MEMMAN_DEFAULT_BACKEND`, then to `'sqlite'`. Centralized so each
    of `open_backend` / `list_stores` / `drop_store` agrees on
    resolution order.
    """
    file_values = config.parse_env_file(
        config.env_file_path(data_dir))
    raw = (
        file_values.get(config.BACKEND_FOR(store))
        or file_values.get(config.DEFAULT_BACKEND)
        or 'sqlite')
    return raw.lower()


def _resolve_store_pg_dsn(store: str, data_dir: str) -> str | None:
    """Return the DSN for `store`: per-store key, then default key.
    """
    file_values = config.parse_env_file(
        config.env_file_path(data_dir))
    return (
        file_values.get(config.PG_DSN_FOR(store))
        or file_values.get(config.DEFAULT_PG_DSN)
        or file_values.get(config.PG_DSN)
        or None)


def open_backend(
        store: str, data_dir: str, *,
        read_only: bool = False) -> Backend:
    """Open the per-store backend for `store`.

    Resolves the backend kind from `MEMMAN_BACKEND_<store>` (with
    fallback to `MEMMAN_DEFAULT_BACKEND`), validates the namespaced
    env keys for that backend, and dispatches to the matching free
    function. Two stores in one process can pick distinct backends.
    """
    name = _resolve_store_backend(store, data_dir)
    if name not in BACKENDS:
        known = ', '.join(sorted(BACKENDS))
        raise ConfigError(
            f'unknown backend {name!r} for store {store!r};'
            f' registered: {known}')
    merged = dict(os.environ)
    merged.update(config.parse_env_file(config.env_file_path(data_dir)))
    validate_for(name, merged)
    if name == 'sqlite':
        from memman.store.sqlite import open_sqlite_backend
        return open_sqlite_backend(
            store, data_dir, read_only=read_only)
    if name == 'postgres':
        from memman.store.postgres import open_postgres_backend
        dsn = _resolve_store_pg_dsn(store, data_dir)
        if not dsn:
            raise ConfigError(
                f'no DSN for postgres-backed store {store!r};'
                f' set {config.PG_DSN_FOR(store)} or'
                f' {config.DEFAULT_PG_DSN}')
        return open_postgres_backend(store, dsn, read_only=read_only)
    raise ConfigError(f'no open path for backend {name!r}')


def list_stores(data_dir: str) -> list[str]:
    """Union of stores reachable via SQLite dirs and Postgres schemas.

    SQLite stores are enumerated from `<data_dir>/data/<name>/memman.db`.
    Postgres stores are enumerated from `pg_namespace` for any DSN
    discoverable via `MEMMAN_DEFAULT_PG_DSN` or per-store DSN keys.
    Stores present in both sources de-duplicate by name.
    """
    from memman.store import db as _db

    names: set[str] = set(_db.list_stores(data_dir))
    file_values = config.parse_env_file(
        config.env_file_path(data_dir))
    dsns: set[str] = set()
    for key, value in file_values.items():
        if not value:
            continue
        if key == config.DEFAULT_PG_DSN or key == config.PG_DSN:
            dsns.add(value)
        elif key.startswith('MEMMAN_PG_DSN_'):
            dsns.add(value)
    if dsns:
        try:
            from memman.store.postgres import _connection
            for dsn in dsns:
                try:
                    with _connection(dsn, autocommit=True) as conn, \
                            conn.cursor() as cur:
                        cur.execute(
                            "select nspname from pg_namespace"
                            " where nspname like 'store_%'"
                            ' order by nspname')
                        for row in cur.fetchall():
                            names.add(row[0][len('store_'):])
                except Exception:
                    continue
        except ImportError:
            pass
    return sorted(names)


def drop_store(store: str, data_dir: str) -> None:
    """Drop the storage for `store` from its resolved backend.

    Also purges queue rows for the store from the local SQLite queue.
    """
    from memman import queue as _queue

    name = _resolve_store_backend(store, data_dir)
    if name == 'sqlite':
        from memman.store.sqlite import SqliteCluster
        SqliteCluster().drop_store(store=store, data_dir=data_dir)
    elif name == 'postgres':
        from memman.store.postgres import drop_postgres_store
        dsn = _resolve_store_pg_dsn(store, data_dir)
        if dsn:
            drop_postgres_store(store, dsn)
    try:
        conn = _queue.open_queue_db(data_dir)
        try:
            _queue.purge_store(conn, store)
        finally:
            conn.close()
    except Exception:
        pass
