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
from typing import TYPE_CHECKING

from memman import config
from memman.store.backend import Backend
from memman.store.config import validate_all
from memman.store.errors import ConfigError

if TYPE_CHECKING:
    pass

logger = logging.getLogger('memman')


@dataclass(frozen=True)
class EnvKey:
    """Env-file key contract for a backend.

    `name` is the suffix appended after `MEMMAN_<BACKEND>_` (e.g.
    `'DSN'` for `MEMMAN_POSTGRES_DSN_<store>`). `secret=True` flags
    the value for masked CLI display + `~/.memman/env` mode-600
    storage. `required=False` when a backend-default fallback key
    is acceptable.
    """

    name: str
    secret: bool
    required: bool


@dataclass(frozen=True)
class BackendDescriptor:
    """Registry record for one storage backend.

    `open_backend` opens a live `Backend`. `list_stores_keys`
    enumerates store names this backend knows about given the env
    file values. `drop_store_fn` removes one store's storage.
    `migrator_cls` is the class itself (not an instance) so
    callers can read class-level fields like `snapshot_features`
    without constructing.
    """

    name: str
    open_backend: Callable[..., Backend]
    list_stores_keys: Callable[[str, dict[str, str]], set[str]]
    drop_store_fn: Callable[[str, str], None]
    migrator_cls: type
    env_keys: tuple[EnvKey, ...]
    extras_packages: tuple[str, ...]


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


def _build_sqlite_descriptor() -> BackendDescriptor:
    """Lazy-import sqlite implementations into a descriptor."""

    def _open(
            store: str, data_dir: str, *,
            read_only: bool = False) -> Backend:
        from memman.store.sqlite import open_sqlite_backend
        return open_sqlite_backend(
            store, data_dir, read_only=read_only)

    def _list(
            data_dir: str,
            env_values: dict[str, str]) -> set[str]:
        from memman.store import db as _db
        return set(_db.list_local_store_dirs(data_dir))

    def _drop(store: str, data_dir: str) -> None:
        from memman.store.sqlite import drop_sqlite_store
        drop_sqlite_store(store, data_dir)

    from memman.store.sqlite import SqliteMigrator
    return BackendDescriptor(
        name='sqlite',
        open_backend=_open,
        list_stores_keys=_list,
        drop_store_fn=_drop,
        migrator_cls=SqliteMigrator,
        env_keys=(),
        extras_packages=())


def _build_postgres_descriptor() -> BackendDescriptor:
    """Lazy-import postgres implementations into a descriptor.

    The `psycopg` import lives inside `_open`/`_list`/`_drop`
    bodies so the postgres extra is only required when a postgres
    backend is actually addressed -- pure-sqlite users do not need
    `psycopg` installed.
    """

    def _open(
            store: str, data_dir: str, *,
            read_only: bool = False) -> Backend:
        from memman.store.postgres import open_postgres_backend
        dsn = resolve_store_pg_dsn(store, data_dir)
        if not dsn:
            raise ConfigError(
                f'no DSN for postgres-backed store {store!r};'
                f' set {config.PG_DSN_FOR(store)} or'
                f' {config.DEFAULT_PG_DSN}')
        return open_postgres_backend(
            store, dsn, read_only=read_only)

    def _list(
            data_dir: str,
            env_values: dict[str, str]) -> set[str]:
        names: set[str] = set()
        dsns: set[str] = set()
        for key, value in env_values.items():
            if not value:
                continue
            if key == config.DEFAULT_PG_DSN:
                dsns.add(value)
            elif key.startswith('MEMMAN_PG_DSN_'):
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
                        " where nspname like 'store_%'"
                        ' order by nspname')
                    names.update(
                        row[0][len('store_'):]
                        for row in cur.fetchall())
            except Exception as exc:
                logger.warning(
                    'postgres store probe failed for dsn %r: %s',
                    dsn, exc)
                continue
        return names

    def _drop(store: str, data_dir: str) -> None:
        from memman.store.postgres import drop_postgres_store
        dsn = resolve_store_pg_dsn(store, data_dir)
        if dsn:
            drop_postgres_store(store, dsn)

    from memman.store.postgres import PostgresMigrator
    return BackendDescriptor(
        name='postgres',
        open_backend=_open,
        list_stores_keys=_list,
        drop_store_fn=_drop,
        migrator_cls=PostgresMigrator,
        env_keys=(EnvKey('DSN', secret=True, required=False),),
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
    """Return the set of registered backend names."""
    return frozenset(BACKENDS.keys())


def all_descriptors() -> list[BackendDescriptor]:
    """Return descriptors in registration order."""
    return list(BACKENDS.values())


def open_backend(
        store: str, data_dir: str, *,
        read_only: bool = False) -> Backend:
    """Open the per-store backend for `store`.

    Resolves the backend kind from `MEMMAN_BACKEND_<store>` (with
    fallback to `MEMMAN_DEFAULT_BACKEND`), validates the namespaced
    env keys for that backend, and dispatches via the static
    registry. Two stores in one process can pick distinct backends.
    """
    name = resolve_store_backend(store, data_dir)
    desc = descriptor(name)
    merged = dict(os.environ)
    merged.update(config.parse_env_file(config.env_file_path(data_dir)))
    validate_all(merged)
    return desc.open_backend(store, data_dir, read_only=read_only)


def list_stores(data_dir: str) -> list[str]:
    """Union of stores reachable across registered backends.

    Each descriptor's `list_stores_keys` enumerates the names it
    knows about; results de-duplicate by name. SQLite enumerates
    directory entries; Postgres enumerates `pg_namespace` for any
    discoverable DSN. Backends whose probe fails (missing extras,
    unreachable DSN) log a warning and contribute nothing.
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

    Also purges queue rows for the store from the local SQLite queue.
    """
    from memman import queue as _queue

    name = resolve_store_backend(store, data_dir)
    desc = descriptor(name)
    try:
        desc.drop_store_fn(store, data_dir)
    finally:
        try:
            with _queue.queue_db(data_dir) as conn:
                _queue.purge_store(conn, store)
        except Exception as exc:
            logger.warning(
                'failed to purge queue rows for store %r: %s',
                store, exc)
