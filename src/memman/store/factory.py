"""Cluster factory dispatch on `MEMMAN_BACKEND`.

Mirrors `embed/__init__.py`'s registry shape: a name -> factory
mapping plus a single `open_cluster()` entry point.
"""

import os
from collections.abc import Callable

from memman import config
from memman.store.backend import Cluster
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
