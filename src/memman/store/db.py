"""Database connection, schema migration, and store management."""

import logging
import os
import re
import sqlite3
from pathlib import Path

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

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a write SQL statement."""
        return self._conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Query SQL using the transaction cursor or connection."""
        return self._conn.execute(sql, params)

    def in_transaction(self, fn: callable):
        """Run fn inside a single SQL transaction, returning its result."""
        if self._in_tx:
            raise RuntimeError('nested transactions not supported')
        self._in_tx = True
        try:
            self._conn.execute('BEGIN IMMEDIATE')
            result = fn()
            self._conn.execute('COMMIT')
            return result
        except Exception:
            self._conn.execute('ROLLBACK')
            raise
        finally:
            self._in_tx = False


def get_meta(db: 'DB', key: str) -> str | None:
    """Read a value from the meta key-value table."""
    row = db._query(
        'SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
    return row[0] if row else None


def set_meta(db: 'DB', key: str, value: str) -> None:
    """Write a value to the meta key-value table."""
    db._exec(
        'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
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
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
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
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return DB(conn, db_path)


_BASELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    importance  INTEGER DEFAULT 3,
    tags        TEXT DEFAULT '[]',
    entities    TEXT DEFAULT '[]',
    source      TEXT DEFAULT 'user',
    access_count INTEGER DEFAULT 0,
    keywords    TEXT,
    summary     TEXT,
    semantic_facts TEXT,
    last_accessed_at TEXT,
    embedding   BLOB,
    effective_importance REAL DEFAULT 0.5,
    linked_at   TEXT,
    enriched_at TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted_at  TEXT,
    prompt_version TEXT,
    model_id    TEXT,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL CHECK(edge_type IN ('temporal','semantic','causal','entity')),
    weight      REAL DEFAULT 1.0,
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES insights(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES insights(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
CREATE INDEX IF NOT EXISTS idx_insights_importance ON insights(importance);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at);
CREATE INDEX IF NOT EXISTS idx_insights_deleted ON insights(deleted_at);
CREATE INDEX IF NOT EXISTS idx_insights_source ON insights(source);
CREATE INDEX IF NOT EXISTS idx_insights_effective_imp ON insights(effective_importance);
CREATE INDEX IF NOT EXISTS idx_prune_candidates ON insights(deleted_at, importance, access_count, effective_importance);
CREATE INDEX IF NOT EXISTS idx_insights_pending_link
    ON insights(linked_at)
    WHERE linked_at IS NULL AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target_type ON edges(target_id, edge_type);

CREATE TABLE IF NOT EXISTS oplog (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    operation   TEXT NOT NULL,
    insight_id  TEXT,
    detail      TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oplog_created ON oplog(created_at);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _migrate(db: DB) -> None:
    """Apply the canonical schema to the database.

    Single-user tool: one authoritative schema (`_BASELINE_SCHEMA`),
    always the latest. `CREATE TABLE IF NOT EXISTS` creates fresh
    databases; existing databases are expected to already match.

    When a column is added to the schema here, the author also ALTERs
    their own `~/.memman/data/*/memman.db` files once (a one-off
    maintenance step); after that, every open through this function
    is a no-op.
    """
    db._conn.executescript(_BASELINE_SCHEMA)
