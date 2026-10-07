"""Database connection, schema migration, and store management.
"""

import logging
import os
import re
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

from memman.store.errors import BackendError

logger = logging.getLogger('memman')

DEFAULT_STORE_NAME = 'default'

_VALID_STORE_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')


def valid_store_name(name: str) -> bool:
    """True if name matches `[a-zA-Z0-9][a-zA-Z0-9_-]*`.
    """
    return bool(_VALID_STORE_NAME_RE.fullmatch(name))


def portable_store_name(name: str) -> str:
    """Rewrite `name` into a form every backend can host.

    Parameters
    ----------
    name : str
        Any store name, including one no rule has yet checked.
        Callers pass raw `list_local_store_dirs` output, so a
        directory name carrying a dot or a space arrives here.

    Returns
    -------
    str
        `name` with every character outside `[A-Za-z0-9_]` replaced
        by an underscore, prefixed with `s_` unless it then opens on
        a letter.

    Notes
    -----
    - Suggestion text only; the caller renames nothing.
    - Not unique: `a-b` and `a_b` both yield `a_b`, which may name a
      different live store.
    """
    portable = re.sub(r'[^A-Za-z0-9_]', '_', name)
    # The result must pass `_check_identifier` and `valid_store_name`,
    # since an operator types it into `memman store create`. A leading
    # digit or underscore takes the prefix for that reason.
    if not portable[:1].isalpha():
        portable = f's_{portable}'
    return portable


def default_data_dir() -> str:
    """Return ~/.memman.
    """
    home = Path.home()
    return str(home / '.memman')


def store_dir(base_dir: str, name: str) -> str:
    """Return <base_dir>/data/<name>.
    """
    return os.path.join(base_dir, 'data', name)


def active_file(base_dir: str) -> str:
    """Return path to <base_dir>/active.
    """
    return os.path.join(base_dir, 'active')


def read_active(base_dir: str) -> str:
    """Read the active store name from <base_dir>/active.
    """
    try:
        data = Path(active_file(base_dir)).read_text()
    except OSError:
        return DEFAULT_STORE_NAME
    name = data.strip()
    return name or DEFAULT_STORE_NAME


def write_active(base_dir: str, name: str) -> None:
    """Write the active store name to <base_dir>/active.
    """
    Path(base_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    Path(active_file(base_dir)).write_text(name + '\n')


def list_local_store_dirs(base_dir: str) -> list[str]:
    """Sorted names of the SQLite store dirs under `<base_dir>/data/`.

    Lists SQLite stores only. `memman.store.factory.list_stores`
    covers every backend.
    """
    data_dir = os.path.join(base_dir, 'data')
    if not Path(data_dir).is_dir():
        return []
    return sorted(e.name for e in os.scandir(data_dir) if e.is_dir())


def store_exists(base_dir: str, name: str) -> bool:
    """Check whether the named store directory exists.
    """
    path = store_dir(base_dir, name)
    return Path(path).is_dir()


class DB:
    """Wraps a SQLite database connection.
    """

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._in_tx = False
        self.path = path

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the underlying connection.
        """
        return self._conn

    def close(self) -> None:
        """Close the database connection.
        """
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        self.close()

    def _exec(
            self, sql: str,
            params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a write SQL statement.
        """
        return self._conn.execute(sql, params)

    def _query(
            self, sql: str,
            params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a read SQL statement.
        """
        return self._conn.execute(sql, params)


def get_meta(db: 'DB', key: str) -> str | None:
    """Read a value from the meta key-value table.
    """
    row = db._query(
        'select value from meta where key = ?', (key,)).fetchone()
    return row[0] if row else None


def set_meta(db: 'DB', key: str, value: str) -> None:
    """Write a value to the meta key-value table.
    """
    db._exec(
        'insert or replace into meta (key, value) values (?, ?)',
        (key, value))


def open_db(data_dir: str) -> DB:
    """Open (or create) the SQLite database for one store.

    Applies the baseline schema idempotently.

    Parameters
    ----------
    data_dir : str
        The STORE directory, not the base data directory: every
        caller passes `store_dir(base_dir, name)` output, and the
        database is read from `<data_dir>/memman.db`. Created, with
        any missing parent, when absent.

    Returns
    -------
    DB
        An open handle the caller owns and closes, by `DB.close()`
        or the `with` form.

    Raises
    ------
    BackendError
        When the store directory cannot be created, when
        `<data_dir>/memman.db` cannot be opened or read as a
        database, or when the store predates the current schema.

    Notes
    -----
    - A zero-length `memman.db` raises nothing: SQLite reads it as a
      fresh database, so the baseline schema is recreated in it.
      Only a partially truncated file reads as malformed.
    """
    try:
        Path(data_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    except OSError as exc:
        raise BackendError(
            f'cannot create store directory {data_dir}: {exc}') from exc
    db_path = os.path.join(data_dir, 'memman.db')
    try:
        is_new_db = not Path(db_path).exists()
    except OSError as exc:
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    try:
        conn = sqlite3.connect(db_path, isolation_level=None)
    except sqlite3.Error as exc:
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    try:
        if is_new_db:
            conn.execute('pragma auto_vacuum=incremental')
        conn.execute('pragma journal_mode=wal')
        conn.execute('pragma foreign_keys=on')
        conn.execute('pragma busy_timeout=5000')
        db = DB(conn, db_path)
        _migrate(db)
    except sqlite3.Error as exc:
        conn.close()
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    except Exception:
        conn.close()
        raise
    return db


def open_read_only(data_dir: str) -> DB:
    """Open one store's SQLite database in read-only mode.

    Parameters
    ----------
    data_dir : str
        The STORE directory, exactly as `open_db` takes it. The
        database is read from `<data_dir>/memman.db` and is never
        created.

    Returns
    -------
    DB
        An open read-only handle the caller owns and closes, by
        `DB.close()` or the `with` form.

    Raises
    ------
    BackendError
        When `<data_dir>/memman.db` is absent, or cannot be opened or
        read as a database. Never a bare `OSError` or `sqlite3.Error`.
    """
    db_path = os.path.join(data_dir, 'memman.db')
    try:
        found = Path(db_path).exists()
    except OSError as exc:
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    if not found:
        raise BackendError(f'database not found: {db_path}')
    # Percent-encode: SQLite cuts a URI at the first `?`, so a raw `#`
    # or `?` in the path truncates the filename and demotes `mode=ro`
    # to an unrecognized parameter, which opens (and creates) a
    # different file read-write.
    uri = f'file:{quote(db_path)}?mode=ro'
    try:
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    except sqlite3.Error as exc:
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    try:
        conn.execute('pragma foreign_keys=on')
        # Notes:
        # - Forces the header read, so a malformed file fails here
        #   rather than at the caller's first query. `pragma
        #   foreign_keys` alone returns cleanly on corrupt bytes.
        # - Must stay a read. A write pragma (`journal_mode`) fails
        #   on `mode=ro` against any store not already in WAL.
        conn.execute('pragma schema_version')
    except sqlite3.Error as exc:
        conn.close()
        raise BackendError(
            f'cannot open database {db_path}: {exc}') from exc
    except Exception:
        conn.close()
        raise
    return DB(conn, db_path)


_BASELINE_SCHEMA = """
create table if not exists insights (
    id          text primary key,
    content     text not null,
    summary     text,
    embedding   blob,
    embedding_pending blob,
    enrich_attempted_at text,
    enriched_at text,
    created_at  text not null,
    updated_at  text not null,
    deleted_at  text,
    embedding_model text,
    queue_uuid  text,
    replaced_by text,
    author      text,
    summary_model text
);

create index if not exists idx_insights_created on insights(created_at);
create index if not exists idx_insights_deleted on insights(deleted_at);
create index if not exists idx_insights_queue_uuid on insights(queue_uuid);
-- `created_at` rides along so the scheduler's pending-enrich scan
-- takes its order from the index; without it the planner prefers
-- the listing index below and sorts every current row per tick.
-- Also the schema canary (see _migrate): it is the first statement
-- naming `enrich_attempted_at` and `replaced_by`.
create index if not exists idx_insights_pending_enrich
    on insights(enrich_attempted_at, created_at)
    where enrich_attempted_at is null and deleted_at is null
      and replaced_by is null;
-- Load-bearing as the carrier of `query_insights`' whole predicate
-- and sort order, so
-- `recall --basic` honors its limit from the index instead of
-- reading every current row into a temp b-tree. Declared as a
-- plain composite, not a partial index, so the planner searches the
-- two leading null columns as equalities.
create index if not exists idx_insights_current_listing
    on insights(deleted_at, replaced_by, created_at);

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


# Keyword channel index, applied by `_migrate` in one transaction
# rather than from `_BASELINE_SCHEMA`.
# Notes:
# - External content: FTS5 holds the terms, the text stays in
#   `insights`.
# - Every row is indexed, soft-deleted and replaced ones included.
#   The active predicate is applied by joining `insights` at read.
# - An active-only index would need conditional delete triggers, and
#   a 'delete' whose old values are not exactly what was indexed
#   corrupts the index silently.
_FTS_STATEMENTS = (
    """
create virtual table insights_fts using fts5(
    content,
    content='insights',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 0"
)
""",
    # Scoped to the indexed column: a bare `after update` would make
    # `update_enrichment` and every stamp update write to the index.
    """
create trigger insights_fts_insert after insert on insights begin
    insert into insights_fts(rowid, content) values (new.rowid, new.content);
end
""",
    """
create trigger insights_fts_delete after delete on insights begin
    insert into insights_fts(insights_fts, rowid, content)
    values ('delete', old.rowid, old.content);
end
""",
    """
create trigger insights_fts_update after update of content on insights begin
    insert into insights_fts(insights_fts, rowid, content)
    values ('delete', old.rowid, old.content);
    insert into insights_fts(rowid, content) values (new.rowid, new.content);
end
""",
    "insert into insights_fts(insights_fts) values('rebuild')",
    )


def _migrate(db: DB) -> None:
    """Apply the canonical schema to the database.

    Single-user tool: one authoritative schema (`_BASELINE_SCHEMA`),
    always the latest. `create table if not exists` creates a fresh
    database; pre-existing databases must already match the canonical
    shape -- a schema change is applied to each live store by hand,
    once, rather than carried here as an `alter` migration.

    Raises
    ------
    BackendError
        When a baseline statement names a column the store lacks.

    Notes
    -----
    - `create table if not exists` no-ops on an existing table, so a
      store missing a baseline column fails only through a baseline
      `create index` that names the column. This is the primary
      schema diagnostic.
    - SQLite skips an index name the store already has, so only a
      new index name catches a store the hand DDL missed. A renamed
      column keeps its index names, so a store the rename missed
      opens without error and fails at its first read of the column.
    """
    try:
        db._conn.executescript(_BASELINE_SCHEMA)
    except sqlite3.OperationalError as exc:
        if 'no such column' in str(exc):
            name = Path(db.path).parent.name
            raise BackendError(
                f'store {name} predates the current schema ({exc});'
                ' add or rename the missing column in the live store,'
                ' drop by name every index whose definition changed,'
                ' then reopen') from exc
        raise
    has_fts = db._conn.execute(
        "select 1 from sqlite_master"
        " where type = 'table' and name = 'insights_fts'").fetchone()
    if has_fts:
        return
    # Notes:
    # - The triggers carry only rows written after the table exists,
    #   so creation also backfills the index, or a restored store
    #   opens with an empty keyword channel.
    # - One transaction: `executescript` commits first and the
    #   connection is autocommit, so an interrupted backfill would
    #   leave an empty index durable, and the absence check above
    #   would read it as migrated.
    db._conn.execute('begin immediate')
    try:
        for statement in _FTS_STATEMENTS:
            db._conn.execute(statement)
    except Exception:
        db._conn.execute('rollback')
        raise
    db._conn.execute('commit')
