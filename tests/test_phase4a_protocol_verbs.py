"""Phase 4a slice 1 -- Backend Protocol verb additions.

Adds two top-level driver-specific probes per the Phase 4 spec:
- `Backend.integrity_check() -> dict` -- SQLite runs `PRAGMA
  integrity_check`; Postgres runs a connectivity / schema-presence
  probe.
- `Backend.introspect_columns(table: str) -> set[str]` -- SQLite uses
  `PRAGMA table_info`; Postgres queries `information_schema.columns`.

The doctor refactor (slice 4) consumes both verbs to close the 11
raw `db._query` / `db._exec` sites in `doctor.py`. Tests parametrize
over both backends via the existing `backend` fixture from Phase 3.
"""

import inspect

from memman.store.backend import Backend


def test_integrity_check_protocol_signature():
    """Backend.integrity_check is a Protocol verb with no arguments."""
    assert hasattr(Backend, 'integrity_check')
    sig = inspect.signature(Backend.integrity_check)
    assert list(sig.parameters) == ['self']


def test_introspect_columns_protocol_signature():
    """Backend.introspect_columns takes a `table` argument."""
    assert hasattr(Backend, 'introspect_columns')
    sig = inspect.signature(Backend.introspect_columns)
    assert 'table' in sig.parameters


def test_integrity_check_returns_ok_on_fresh_store(backend):
    """integrity_check returns {'ok': True, ...} on a healthy fresh store."""
    result = backend.integrity_check()
    assert isinstance(result, dict)
    assert result.get('ok') is True
    assert 'detail' in result


def test_introspect_columns_returns_insights_schema(backend):
    """introspect_columns('insights') returns the expected core columns."""
    cols = backend.introspect_columns('insights')
    assert isinstance(cols, set)
    expected_core = {
        'id', 'content', 'category', 'importance',
        'entities', 'source', 'created_at', 'updated_at',
        'embedding'}
    assert expected_core.issubset(cols), (
        f'missing core columns: {sorted(expected_core - cols)}; '
        f'got: {sorted(cols)}')


def test_introspect_columns_unknown_table_returns_empty(backend):
    """introspect_columns on an unknown table returns an empty set."""
    cols = backend.introspect_columns('definitely_not_a_real_table')
    assert cols == set()
