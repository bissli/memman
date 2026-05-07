"""Per-store backend factory dispatch.

`open_backend(store, data_dir)` reads `MEMMAN_BACKEND_<store>` (with
fallback to `MEMMAN_DEFAULT_BACKEND`) and dispatches to the matching
free function. `list_stores` and `drop_store` apply the same
resolution so dispatch is one place.
"""

import logging
import os

from memman import config
from memman.store.backend import Backend
from memman.store.config import validate_all
from memman.store.errors import ConfigError

logger = logging.getLogger('memman')

KNOWN_BACKENDS = frozenset({'sqlite', 'postgres'})


def resolve_store_backend(store: str, data_dir: str) -> str:
    """Return the backend kind for `store`: per-store, then default.

    Reads `MEMMAN_BACKEND_<store>`, falling back to
    `MEMMAN_DEFAULT_BACKEND`, then to `'sqlite'`. Centralized so each
    of `open_backend` / `list_stores` / `drop_store` agrees on
    resolution order. Public name -- doctor and other module-level
    callers may import this directly.
    """
    raw = (
        config.get_store_backend(store, data_dir)
        or config.get(config.DEFAULT_BACKEND)
        or 'sqlite')
    return raw.lower()


def resolve_store_pg_dsn(store: str, data_dir: str) -> str | None:
    """Return the DSN for `store`: per-store key, then default key.
    """
    return (
        config.get_store_pg_dsn(store, data_dir)
        or config.get(config.DEFAULT_PG_DSN)
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
    name = resolve_store_backend(store, data_dir)
    if name not in KNOWN_BACKENDS:
        known = ', '.join(sorted(KNOWN_BACKENDS))
        raise ConfigError(
            f'unknown backend {name!r} for store {store!r};'
            f' registered: {known}')
    merged = dict(os.environ)
    merged.update(config.parse_env_file(config.env_file_path(data_dir)))
    validate_all(merged)
    if name == 'sqlite':
        from memman.store.sqlite import open_sqlite_backend
        return open_sqlite_backend(
            store, data_dir, read_only=read_only)
    if name == 'postgres':
        from memman.store.postgres import open_postgres_backend
        dsn = resolve_store_pg_dsn(store, data_dir)
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

    names: set[str] = set(_db.list_local_store_dirs(data_dir))
    file_values = config.parse_env_file(
        config.env_file_path(data_dir))
    dsns: set[str] = set()
    for key, value in file_values.items():
        if not value:
            continue
        if key == config.DEFAULT_PG_DSN:
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
                        names.update(row[0][len('store_'):] for row in cur.fetchall())
                except Exception as exc:
                    logger.warning(
                        'postgres store probe failed for dsn %r: %s',
                        dsn, exc)
                    continue
        except ImportError:
            pass
    return sorted(names)


def drop_store(store: str, data_dir: str) -> None:
    """Drop the storage for `store` from its resolved backend.

    Also purges queue rows for the store from the local SQLite queue.
    """
    from memman import queue as _queue

    name = resolve_store_backend(store, data_dir)
    try:
        if name == 'sqlite':
            from memman.store.sqlite import drop_sqlite_store
            drop_sqlite_store(store, data_dir)
        elif name == 'postgres':
            from memman.store.postgres import drop_postgres_store
            dsn = resolve_store_pg_dsn(store, data_dir)
            if dsn:
                drop_postgres_store(store, dsn)
    finally:
        try:
            with _queue.queue_db(data_dir) as conn:
                _queue.purge_store(conn, store)
        except Exception as exc:
            logger.warning(
                'failed to purge queue rows for store %r: %s', store, exc)
