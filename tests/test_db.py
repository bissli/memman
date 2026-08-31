"""Tests for the DB class context-manager protocol."""

import sqlite3
from pathlib import Path

import pytest
from memman.store.db import open_db, open_read_only
from memman.store.errors import BackendError


def test_db_context_manager_closes_on_exit(tmp_path):
    """Using `with open_db()` closes the underlying connection on exit."""
    with open_db(str(tmp_path)) as db:
        underlying = db.conn
        assert underlying.execute('select 1').fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError):
        underlying.execute('select 1')


def test_db_context_manager_closes_on_exception(tmp_path):
    """Exception inside the with-block still closes the connection."""
    with pytest.raises(RuntimeError), open_db(str(tmp_path)) as db:
        underlying = db.conn
        raise RuntimeError('boom')
    with pytest.raises(sqlite3.ProgrammingError):
        underlying.execute('select 1')


def test_open_db_wraps_unreadable_file_as_backend_error(tmp_path):
    """A non-database file fails as `BackendError`, never raw sqlite3.

    Mutation: dropping the `sqlite3.Error` translation in `open_db`, so
        a raw `sqlite3.DatabaseError` leaves this function. The CLI no
        longer prints a traceback for that -- the root group catches
        `sqlite3.Error` too -- but it would report the generic `sqlite
        query failed` in place of the store path, and every non-CLI
        caller would see the driver type.
    Oracle: `BackendError` sits outside the `sqlite3.Error` hierarchy,
        so the raised type discriminates translated from untranslated.
    """
    (tmp_path / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError) as excinfo:
        open_db(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_db_wraps_unopenable_path_as_backend_error(tmp_path):
    """A directory at the db path fails as `BackendError`.

    Mutation: dropping the `sqlite3.Error` translation around the
        `sqlite3.connect` call specifically. `connect` is lazy, so the
        corrupt-bytes test above never reaches that handler and cannot
        catch this.
    Oracle: a directory at `<data_dir>/memman.db` makes `connect`
        itself raise `unable to open database file`.
    """
    (tmp_path / 'memman.db').mkdir()
    with pytest.raises(BackendError) as excinfo:
        open_db(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_db_wraps_uncreatable_store_dir_as_backend_error(tmp_path):
    """A plain file where the store dir belongs fails as `BackendError`.

    Mutation: leaving the `Path.mkdir` call outside the translation, so
        `OSError` escapes untranslated and the CLI prints a traceback.
        `OSError` is not a `sqlite3.Error`, so the handlers below the
        mkdir cannot catch it.
    Oracle: `mkdir(exist_ok=True)` still raises `FileExistsError` when
        the path exists and is not a directory.
    """
    occupied = tmp_path / 'store'
    occupied.write_text('not a directory')
    with pytest.raises(BackendError) as excinfo:
        open_db(str(occupied))
    assert 'store' in str(excinfo.value)


def test_open_db_closes_the_connection_when_open_fails(tmp_path, monkeypatch):
    """A failed open leaves no connection behind.

    Mutation: dropping either `conn.close()` from the failure handlers,
        so every failed open leaks a file handle. Both the corrupt-file
        tests above pass with the connection left open.
    Oracle: the captured connection is independently probed after the
        raise. `ProgrammingError` means closed; a leaked handle raises
        `DatabaseError` on these bytes instead, so the probe
        discriminates on the type, not on the query succeeding.
    """
    from memman.store import db as db_mod

    captured = []
    real_connect = db_mod.sqlite3.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr(db_mod.sqlite3, 'connect', spy_connect)
    (tmp_path / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError):
        open_db(str(tmp_path))
    assert captured
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute('select 1')


def test_open_read_only_wraps_missing_database_as_backend_error(tmp_path):
    """A store directory with no `memman.db` fails as `BackendError`.

    Mutation: leaving the missing-file case raising `FileNotFoundError`,
        which the CLI seam does not catch, so one stray store directory
        prints a traceback out of `embed reembed --dry-run`.
    Oracle: `BackendError` sits outside the `OSError` hierarchy, so the
        raised type discriminates translated from untranslated.
    """
    with pytest.raises(BackendError) as excinfo:
        open_read_only(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_read_only_wraps_unreadable_file_as_backend_error(tmp_path):
    """A non-database file fails as `BackendError`, never raw sqlite3.

    Mutation: dropping the `sqlite3.Error` translation around the pragma
        block in `open_read_only`.
    Oracle: `sqlite3.connect` is lazy, so garbage bytes surface at the
        first pragma as `sqlite3.DatabaseError`, a type outside
        `BackendError`.
    """
    (tmp_path / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError) as excinfo:
        open_read_only(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_read_only_wraps_unopenable_path_as_backend_error(tmp_path):
    """A directory at the db path fails as `BackendError`.

    Mutation: dropping the translation around the `sqlite3.connect` call
        specifically. `connect` is lazy, so the corrupt-bytes test above
        never reaches that handler and cannot catch this.
    Oracle: a directory at `<data_dir>/memman.db` makes the `mode=ro`
        connect itself raise `disk I/O error`.
    """
    (tmp_path / 'memman.db').mkdir()
    with pytest.raises(BackendError) as excinfo:
        open_read_only(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_read_only_closes_the_connection_when_open_fails(
        tmp_path, monkeypatch):
    """A failed read-only open leaves no connection behind.

    Mutation: dropping `conn.close()` from the pragma failure handler,
        so every failed read-only open leaks a file handle. The
        corrupt-bytes test above passes with the connection left open.
    Oracle: the captured connection is independently probed after the
        raise. `ProgrammingError` means closed; a leaked handle raises
        `DatabaseError` on these bytes instead, so the probe
        discriminates on the type, not on the query succeeding.
    """
    from memman.store import db as db_mod

    captured = []
    real_connect = db_mod.sqlite3.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr(db_mod.sqlite3, 'connect', spy_connect)
    (tmp_path / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(BackendError):
        open_read_only(str(tmp_path))
    assert captured
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute('select 1')


def test_open_read_only_wraps_an_unstattable_path_as_backend_error(tmp_path):
    """A path the filesystem refuses to stat fails as `BackendError`.

    Mutation: leaving the `Path.exists()` probe outside the handlers.
        `Path.exists()` swallows only ENOENT, ENOTDIR, EBADF and
        ELOOP, so every other errno escapes as a raw `OSError` -- past
        the CLI seam, which catches `BackendError` alone.
    Oracle: a 300-character path component raises ENAMETOOLONG, which
        is not on that swallow list; `BackendError` sits outside the
        `OSError` hierarchy, so the raised type discriminates.
    """
    too_long = str(tmp_path / ('a' * 300))
    with pytest.raises(BackendError) as excinfo:
        open_read_only(too_long)
    assert 'memman.db' in str(excinfo.value)


def test_open_db_wraps_an_unstattable_path_as_backend_error(
        tmp_path, monkeypatch):
    """A store dir that cannot be searched fails as `BackendError`.

    Mutation: leaving `open_db`'s `Path.exists()` probe outside the
        handlers. Its `mkdir` guard cannot cover it -- `mkdir` with
        `exist_ok=True` succeeds on an existing unsearchable
        directory and hands the failure to the probe below it.
    Oracle: `Path.exists` is stubbed to raise the `PermissionError`
        an unsearchable parent produces, which reproduces by hand
        under `chmod 000` but is unreachable when the suite runs as
        root; `BackendError` is outside the `OSError` hierarchy.
    """
    real_exists = Path.exists

    def refuse(self):
        if self.name == 'memman.db':
            raise PermissionError(13, 'Permission denied')
        return real_exists(self)

    monkeypatch.setattr(Path, 'exists', refuse)
    with pytest.raises(BackendError) as excinfo:
        open_db(str(tmp_path))
    assert 'memman.db' in str(excinfo.value)


def test_open_read_only_opens_a_store_that_is_not_in_wal_mode(tmp_path):
    """A `journal_mode=DELETE` store opens and reads.

    Mutation: setting a write pragma (`journal_mode=wal`) on the
        read-only connection. It fails with `attempt to write a
        readonly database` on any store not already in WAL -- which
        is every store `backup restore` lays down, since the snapshot
        is stamped DELETE on purpose.
    Oracle: a store whose journal mode is switched to DELETE on disk,
        then read back through `open_read_only` for a real row count.
    """
    open_db(str(tmp_path)).close()
    conn = sqlite3.connect(str(tmp_path / 'memman.db'), isolation_level=None)
    conn.execute('pragma journal_mode=delete')
    conn.close()
    with open_read_only(str(tmp_path)) as db:
        assert db._query('select count(*) from insights').fetchone() == (0,)


def test_open_read_only_keeps_a_path_holding_a_uri_delimiter(tmp_path):
    """A store dir containing `#` opens that store, and creates nothing.

    Mutation: interpolating the raw path into the `file:...?mode=ro`
        URI. SQLite cuts the URI at the first `?`, so a `#` truncates
        the filename AND demotes `mode=ro` to an unrecognized
        parameter -- the open then silently succeeds read-write
        against a different file, raising nothing.
    Oracle: `pragma database_list` names the file actually attached,
        plus the absence of the truncated sibling the unescaped form
        creates.
    """
    sdir = tmp_path / 'da#ta'
    sdir.mkdir()
    open_db(str(sdir)).close()
    with open_read_only(str(sdir)) as db:
        attached = db._query('pragma database_list').fetchone()[2]
    assert attached == str(sdir / 'memman.db')
    assert not (tmp_path / 'da').exists()
