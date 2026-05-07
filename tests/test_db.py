"""Tests for the DB class context-manager protocol."""

import sqlite3

import pytest
from memman.store.db import open_db


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
