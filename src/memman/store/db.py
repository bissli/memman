"""Database connection, schema migration, and store management."""

import logging
import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger('memman')

DEFAULT_STORE_NAME = 'default'

_VALID_STORE_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')


def valid_store_name(name: str) -> bool:
    """Return True if name matches [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    return bool(_VALID_STORE_NAME_RE.match(name))


def default_data_dir() -> str:
    """Return ~/.memman."""
    home = Path.home()
    return str(home / '.memman')


def store_dir(base_dir: str, name: str) -> str:
    """Return <base_dir>/data/<name>."""
    return os.path.join(base_dir, 'data', name)


def active_file(base_dir: str) -> str:
    """Return path to <base_dir>/active."""
    return os.path.join(base_dir, 'active')


def read_active(base_dir: str) -> str:
    """Read the active store name from <base_dir>/active."""
    try:
        data = Path(active_file(base_dir)).read_text()
    except (OSError, FileNotFoundError):
        return DEFAULT_STORE_NAME
    name = data.strip()
    return name or DEFAULT_STORE_NAME


def write_active(base_dir: str, name: str) -> None:
    """Write the active store name to <base_dir>/active."""
    Path(base_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    Path(active_file(base_dir)).write_text(name + '\n')


def list_stores(base_dir: str) -> list[str]:
    """Return sorted names of all stores under <base_dir>/data/."""
    data_dir = os.path.join(base_dir, 'data')
    if not Path(data_dir).is_dir():
        return []
    names = sorted(
        e.name for e in os.scandir(data_dir) if e.is_dir())
    return names


def store_exists(base_dir: str, name: str) -> bool:
    """Check whether the named store directory exists."""
    path = store_dir(base_dir, name)
    return Path(path).is_dir()


class DB:
    """Wraps a SQLite database connection."""

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._in_tx = False
        self.path = path

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the underlying connection."""
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _exec(
            self, sql: str,
            params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a write SQL statement."""
        return self._conn.execute(sql, params)

    def _query(
            self, sql: str,
            params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Query SQL using the transaction cursor or connection."""
        return self._conn.execute(sql, params)

    def in_transaction(self, fn: Callable[[], Any]) -> Any:
        """Run fn inside a single SQL transaction, returning its result."""
        if self._in_tx:
            raise RuntimeError('nested transactions not supported')
        self._in_tx = True
        try:
            self._conn.execute('begin immediate')
            result = fn()
            self._conn.execute('commit')
            return result
        except Exception:
            self._conn.execute('rollback')
            raise
        finally:
            self._in_tx = False


def get_meta(db: 'DB', key: str) -> str | None:
    """Read a value from the meta key-value table."""
    row = db._query(
        'select value from meta where key = ?', (key,)).fetchone()
    return row[0] if row else None


def storage_summary(db: 'DB') -> dict[str, Any]:
    """Return backend-specific storage information for the active DB.

    SQLite-specific: {'db_path': <file path>, 'db_size_bytes': <int>}.
    Used by the `memman status` command.
    """
    summary: dict[str, Any] = {'db_path': db.path}
    try:
        summary['db_size_bytes'] = Path(db.path).stat().st_size
    except OSError:
        summary['db_size_bytes'] = 0
    return summary


def set_meta(db: 'DB', key: str, value: str) -> None:
    """Write a value to the meta key-value table."""
    db._exec(
        'insert or replace into meta (key, value) values (?, ?)',
        (key, value))


def open_db(data_dir: str) -> DB:
    """Open (or create) the SQLite database at the given directory.

    Runs idempotent baseline schema + versioned migrations. Does NOT
    trigger the edge-constants reindex — callers that want that (the
    CLI `_open_db`) invoke `reindex_if_constants_changed(db)` after
    open. Keeping the graph-reindex out of this module avoids a
    backward import edge from `memman.store` to `memman.graph`.
    """
    Path(data_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    db_path = os.path.join(data_dir, 'memman.db')
    is_new_db = not Path(db_path).exists()
    conn = sqlite3.connect(db_path, isolation_level=None)
    if is_new_db:
        conn.execute('pragma auto_vacuum=incremental')
    conn.execute('pragma journal_mode=wal')
    conn.execute('pragma foreign_keys=on')
    conn.execute('pragma busy_timeout=5000')
    db = DB(conn, db_path)
    _migrate(db)
    return db


def open_read_only(data_dir: str) -> DB:
    """Open the SQLite database in read-only mode."""
    db_path = os.path.join(data_dir, 'memman.db')
    if not Path(db_path).exists():
        raise FileNotFoundError(f'database not found: {db_path}')
    uri = f'file:{db_path}?mode=ro'
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.execute('pragma journal_mode=wal')
    conn.execute('pragma foreign_keys=on')
    return DB(conn, db_path)


_BASELINE_SCHEMA = """
create table if not exists insights (
    id          text primary key,
    content     text not null,
    category    text default 'general',
    importance  integer default 3,
    entities    text default '[]',
    source      text default 'user',
    access_count integer default 0,
    keywords    text,
    summary     text,
    semantic_facts text,
    last_accessed_at text,
    embedding   blob,
    embedding_pending blob,
    effective_importance real default 0.5,
    linked_at   text,
    enriched_at text,
    created_at  text not null,
    updated_at  text not null,
    deleted_at  text,
    prompt_version text,
    model_id    text,
    embedding_model text
);

create table if not exists edges (
    source_id   text not null,
    target_id   text not null,
    edge_type   text not null check(edge_type in ('temporal','semantic','causal','entity')),
    weight      real default 1.0,
    metadata    text default '{}',
    created_at  text not null,
    primary key (source_id, target_id, edge_type),
    foreign key (source_id) references insights(id) on delete cascade,
    foreign key (target_id) references insights(id) on delete cascade
);

create index if not exists idx_insights_category on insights(category);
create index if not exists idx_insights_importance on insights(importance);
create index if not exists idx_insights_created on insights(created_at);
create index if not exists idx_insights_deleted on insights(deleted_at);
create index if not exists idx_insights_source on insights(source);
create index if not exists idx_insights_effective_imp on insights(effective_importance);
create index if not exists idx_prune_candidates on insights(deleted_at, importance, access_count, effective_importance);
create index if not exists idx_insights_pending_link
    on insights(linked_at)
    where linked_at is null and deleted_at is null;
create index if not exists idx_edges_source on edges(source_id);
create index if not exists idx_edges_target on edges(target_id);
create index if not exists idx_edges_type on edges(edge_type);
create index if not exists idx_edges_source_type on edges(source_id, edge_type);
create index if not exists idx_edges_target_type on edges(target_id, edge_type);

create table if not exists oplog (
    id          integer primary key autoincrement,
    operation   text not null,
    insight_id  text,
    detail      text default '',
    created_at  text not null,
    before      text,
    after       text
);
create index if not exists idx_oplog_created on oplog(created_at);

create table if not exists meta (
    key   text primary key,
    value text not null
);
"""


def _migrate(db: DB) -> None:
    """Apply the canonical schema to the database.

    Single-user tool: one authoritative schema (`_BASELINE_SCHEMA`),
    always the latest. `create table if not exists` creates a fresh
    database; pre-existing databases must already match the canonical
    shape -- wipe and recreate on schema change rather than carrying
    `alter` migrations.
    """
    db._conn.executescript(_BASELINE_SCHEMA)
