"""Driver errors must reach callers as `BackendError`.

memman's contract is that a backend raises `BackendError` and the CLI
catches it. The Postgres layer had four `psycopg.Error` handlers
against fourteen `_connection` blocks, so most driver failures
surfaced as raw tracebacks.

Translating inside `_connection` covers every statement run in its
scope. A caller that branches on a driver type nests its handler at
the statement -- `_read_stored_dim` returns None when the schema is
absent -- which runs first and keeps that branch intact.
"""

import pytest
from memman.store.errors import BackendError

psycopg = pytest.importorskip('psycopg')

DEAD_DSN = 'postgresql://u:p@127.0.0.1:1/nodb'


def test_connection_scope_translates_a_statement_error(monkeypatch):
    """A statement that fails inside the block becomes BackendError.

    The connection must succeed first, or the error comes from
    `_open_connection` and this never exercises the scope at all.

    Mutation: dropping `_connection`'s `except _psycopg.Error`
    clause, so the driver error escapes the scope untranslated.
    Oracle: `BackendError` out of a block whose statement raised
    `psycopg.OperationalError`, with `__cause__` preserved.
    """
    from memman.store import postgres as pg

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(pg, '_open_connection', lambda *a, **kw: _Conn())

    with pytest.raises(BackendError) as caught:
        with pg._connection(DEAD_DSN):
            raise psycopg.OperationalError('statement blew up')

    assert isinstance(caught.value.__cause__, psycopg.OperationalError)


def test_read_stored_dim_translates_a_non_missing_schema_error(
        monkeypatch):
    """Only "schema absent" is special-cased; other errors translate.

    An earlier version passed `translate=False` for the whole block,
    which let every driver error out raw -- a permission failure on
    `meta` reached `open_postgres_backend` as a psycopg exception,
    past the CLI's `except BackendError`.

    Mutation: widening the statement handler to `psycopg.Error`, or
    restoring a block-wide opt-out.
    Oracle: `InsufficientPrivilege` from the statement arrives as
    `BackendError`, while `UndefinedTable` still yields None.
    """
    from memman.store import postgres as pg

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **kw):
            raise psycopg.errors.InsufficientPrivilege('denied')

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(pg, '_open_connection', lambda *a, **kw: _Conn())

    with pytest.raises(BackendError):
        pg._read_stored_dim(DEAD_DSN, 'shop')


def test_read_stored_dim_still_returns_none_for_a_missing_schema(
        monkeypatch):
    """The branching caller keeps its documented None result.

    `_read_stored_dim` returns None when the schema does not
    exist, which it detects by catching `psycopg.errors.UndefinedTable`
    around its `_connection` block.

    Mutation: moving the handler from the statement to around the
    block, where `_connection` translates first and the probe raises
    instead of reporting "no schema".
    Oracle: a stubbed connection raising UndefinedTable yields None.
    """
    from memman.store import postgres as pg

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **kw):
            raise psycopg.errors.UndefinedTable('no such table')

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(
        pg, '_open_connection', lambda *a, **kw: _Conn())

    assert pg._read_stored_dim(DEAD_DSN, 'shop') is None


def test_connect_failure_outside_a_scope_translates():
    """`_open_connection` translates a refused connection too.

    The lock-holding paths call it directly rather than through
    `_connection`, so an unreachable server there would still raise a
    driver error.

    Mutation: wrapping only `_connection` and leaving
    `_open_connection` raw.
    Oracle: `pytest.raises(BackendError)` against a closed port.
    """
    from memman.store.postgres import _open_connection

    with pytest.raises(BackendError):
        _open_connection(DEAD_DSN, autocommit=True)


def test_open_postgres_backend_reports_a_dead_server_as_backend_error():
    """The public entry point an operator hits is covered end to end.

    Mutation: translating only in helpers that this path does not
    use, leaving `memman --store X status` on a down server as a
    traceback.
    Oracle: `pytest.raises(BackendError)` opening a store against a
    closed port.
    """
    from memman.store.postgres import open_postgres_backend

    with pytest.raises(BackendError):
        open_postgres_backend('shop', DEAD_DSN)
