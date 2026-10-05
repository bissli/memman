"""Per-store backend factory dispatch driven by a static registry.

`BACKENDS` is the single source of truth for the registered storage
backends. Adding an N+1 backend means adding one entry and a
descriptor builder; the existing dispatch in `open_backend`,
`list_stores`, and `drop_store` does not change.

`open_backend(store, data_dir)` reads `MEMMAN_BACKEND_<store>` (with
fallback to `MEMMAN_DEFAULT_BACKEND`) and dispatches via the registry.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from memman import config
from memman import queue as _queue
from memman.store import db as _db
from memman.store.backend import Backend
from memman.store.config import validate_all
from memman.store.errors import ConfigError

logger = logging.getLogger('memman')


@dataclass(frozen=True)
class BackendDescriptor:
    """Registry record for one storage backend.

    Attributes
    ----------
    name : str
        Registry key.
    open_backend : Callable[..., Backend]
        Opens a live `Backend`.
    store_exists_fn : Callable[[str, str], bool]
        True when one store's storage exists, given the store and the
        data directory.
    list_stores_keys : Callable[[str, dict[str, str]], set[str]]
        Store names this backend knows, given the env file values.
    drop_store_fn : Callable[[str, str], None]
        Removes one store's storage.
    extras_packages : tuple[str, ...]
        Packages of the optional extra this backend needs.
    """

    name: str
    open_backend: Callable[..., Backend]
    store_exists_fn: Callable[[str, str], bool]
    list_stores_keys: Callable[[str, dict[str, str]], set[str]]
    drop_store_fn: Callable[[str, str], None]
    extras_packages: tuple[str, ...]


def resolve_store_backend(store: str, data_dir: str) -> str:
    """Return the backend kind for `store`, lower-cased.

    Reads `MEMMAN_BACKEND_<store>`, then `MEMMAN_DEFAULT_BACKEND`,
    then falls back to `'sqlite'`.
    """
    raw = (
        config.get_store_backend(store, data_dir)
        or config.get_scoped(config.DEFAULT_BACKEND, data_dir)
        or 'sqlite')
    return raw.lower()


def resolve_store_pg_dsn(store: str, data_dir: str) -> str | None:
    """Return the DSN for `store`: per-store key, then default key.
    """
    return (
        config.get_store_pg_dsn(store, data_dir)
        or config.get_scoped(config.DEFAULT_PG_DSN, data_dir)
        or None)


def _build_sqlite_descriptor() -> BackendDescriptor:
    """Build the sqlite descriptor.
    """

    def _open(
            store: str, data_dir: str, *,
            read_only: bool = False, create: bool = False) -> Backend:
        from memman.store.sqlite import open_sqlite_backend
        return open_sqlite_backend(
            store, data_dir, read_only=read_only, create=create)

    def _exists(store: str, data_dir: str) -> bool:
        return _db.store_exists(data_dir, store)

    def _list(
            data_dir: str,
            env_values: dict[str, str]) -> set[str]:
        return set(_db.list_local_store_dirs(data_dir))

    def _drop(store: str, data_dir: str) -> None:
        from memman.store.sqlite import drop_sqlite_store
        drop_sqlite_store(store, data_dir)

    return BackendDescriptor(
        name='sqlite',
        open_backend=_open,
        store_exists_fn=_exists,
        list_stores_keys=_list,
        drop_store_fn=_drop,
        extras_packages=())


def _build_postgres_descriptor() -> BackendDescriptor:
    """Build the postgres descriptor.

    The postgres extra is needed only once a postgres store is
    addressed.
    """

    def _dsn(store: str, data_dir: str) -> str:
        dsn = resolve_store_pg_dsn(store, data_dir)
        if not dsn:
            raise ConfigError(
                f'no DSN for postgres-backed store {store!r};'
                f' set {config.POSTGRES_DSN_FOR(store)} or'
                f' {config.DEFAULT_PG_DSN}')
        return dsn

    def _open(
            store: str, data_dir: str, *,
            read_only: bool = False, create: bool = False) -> Backend:
        from memman.store.postgres import open_postgres_backend
        return open_postgres_backend(
            store, _dsn(store, data_dir), read_only=read_only,
            create=create)

    def _exists(store: str, data_dir: str) -> bool:
        from memman.store.postgres import postgres_store_exists
        return postgres_store_exists(store, _dsn(store, data_dir))

    def _list(
            data_dir: str,
            env_values: dict[str, str]) -> set[str]:
        names: set[str] = set()
        dsns: set[str] = set()
        for key, value in env_values.items():
            if not value:
                continue
            if (key == config.DEFAULT_PG_DSN
                    or key.startswith('MEMMAN_POSTGRES_DSN_')):
                dsns.add(value)
        if not dsns:
            return names
        from memman.store.postgres import _connection
        for dsn in dsns:
            try:
                with _connection(
                        dsn, autocommit=True) as conn, \
                        conn.cursor() as cur:
                    cur.execute(
                        "select nspname from pg_namespace"
                        " where starts_with(nspname, 'store_')"
                        ' order by nspname')
                    names.update(
                        row[0][len('store_'):]
                        for row in cur.fetchall())
            except Exception as exc:
                logger.warning(
                    'postgres store probe failed for dsn %r: %s',
                    dsn, exc)
        return names

    def _drop(store: str, data_dir: str) -> None:
        from memman.store.postgres import drop_postgres_store
        dsn = resolve_store_pg_dsn(store, data_dir)
        if dsn:
            drop_postgres_store(store, dsn)

    return BackendDescriptor(
        name='postgres',
        open_backend=_open,
        store_exists_fn=_exists,
        list_stores_keys=_list,
        drop_store_fn=_drop,
        extras_packages=('psycopg', 'psycopg-pool', 'pgvector'))


BACKENDS: dict[str, BackendDescriptor] = {
    'sqlite': _build_sqlite_descriptor(),
    'postgres': _build_postgres_descriptor(),
    }


def descriptor(name: str) -> BackendDescriptor:
    """Return the descriptor for `name`; raise on unknown.
    """
    if name not in BACKENDS:
        known = ', '.join(sorted(BACKENDS.keys()))
        raise ConfigError(
            f'unknown backend {name!r}; registered: {known}')
    return BACKENDS[name]


def known_backends() -> frozenset[str]:
    """Return the set of registered backend names.
    """
    return frozenset(BACKENDS.keys())


def all_descriptors() -> list[BackendDescriptor]:
    """Return descriptors in registration order.
    """
    return list(BACKENDS.values())


def open_backend(
        store: str, data_dir: str, *,
        read_only: bool = False,
        create: bool = False) -> Backend:
    """Open the backend that `store` resolves to.

    Parameters
    ----------
    store : str
        Store name.
    data_dir : str
        Base memman data directory. The store's files live under it.
    read_only : bool
        Open without write access.
    create : bool, default False
        Create the store's storage when it is missing.

    Returns
    -------
    Backend
        A live backend. Two stores in one process can use distinct
        backends.

    Raises
    ------
    ConfigError
        On an unknown backend name, a bad `MEMMAN_POSTGRES_*` key, or
        a postgres store with no DSN.
    StoreMissingError
        The store does not exist and `create` is False. Nothing is
        written.
    """
    if not _db.valid_store_name(store):
        raise ConfigError(f'invalid store name {store!r}')
    name = resolve_store_backend(store, data_dir)
    desc = descriptor(name)
    merged = dict(os.environ)
    merged.update(config.parse_env_file(config.env_file_path(data_dir)))
    validate_all(merged)
    return desc.open_backend(
        store, data_dir, read_only=read_only, create=create)


def store_exists(store: str, data_dir: str) -> bool:
    """True when the storage of `store` exists on its resolved backend.

    Raises
    ------
    ConfigError
        On an invalid store name, a postgres store with no DSN, or a
        backend whose extra is not installed.
    BackendError
        On a postgres connection failure, which answers neither way.
    """
    if not _db.valid_store_name(store):
        raise ConfigError(f'invalid store name {store!r}')
    desc = descriptor(resolve_store_backend(store, data_dir))
    extras = {pkg.replace('-', '_') for pkg in desc.extras_packages}
    try:
        return desc.store_exists_fn(store, data_dir)
    except ImportError as exc:
        if (exc.name or '').split('.')[0] not in extras:
            raise
        raise ConfigError(
            f'store {store!r} uses the {desc.name} backend, which needs'
            f' the memman[{desc.name}] extra: {exc}') from exc


def list_stores(data_dir: str) -> list[str]:
    """Sorted, de-duplicated store names across all backends.

    A backend whose probe fails (missing extra, unreachable DSN)
    contributes nothing.
    """
    file_values = config.parse_env_file(
        config.env_file_path(data_dir))
    names: set[str] = set()
    for desc in all_descriptors():
        try:
            names |= desc.list_stores_keys(data_dir, file_values)
        except ImportError:
            continue
    return sorted(names)


def drop_store(store: str, data_dir: str) -> None:
    """Drop the storage for `store` from its resolved backend.

    Purges the store's rows from the local SQLite queue once the
    backend has dropped the storage.

    Parameters
    ----------
    store : str
        Store name.
    data_dir : str
        Base memman data directory. The store's files live under it.

    Notes
    -----
    - A drop that fails part way can leave queue rows behind. A
      repeated `memman store remove` clears them.
    - A purge failure after a clean drop logs a warning. Raising
      would report failure for finished work.
    """
    if not _db.valid_store_name(store):
        raise ConfigError(f'invalid store name {store!r}')
    name = resolve_store_backend(store, data_dir)
    desc = descriptor(name)
    # Purge only after a clean drop: an unconditional purge would
    # delete the queued writes of a store that still exists.
    desc.drop_store_fn(store, data_dir)
    try:
        with _queue.queue_db(data_dir) as conn:
            _queue.purge_store(conn, store)
    except Exception as exc:
        logger.warning(
            'failed to purge queue rows for store %r: %s',
            store, exc)
