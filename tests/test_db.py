"""Tests for the DB class context-manager protocol."""

import sqlite3

import pytest
from memman.store.db import open_db
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
        a raw `sqlite3.DatabaseError` reaches the CLI seam and prints a
        Python traceback instead of a clean message.
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
        raise -- a live handle answers `select 1`, a closed one raises
        `ProgrammingError`.
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
