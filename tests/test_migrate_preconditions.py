"""Source-side precondition tests for `migrate_store`.

Slice 1.2: source SQLite is opened read-only and an empty source
(zero insights and no fingerprint) is rejected with a clear error
before any destination work happens.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from memman.migrate import MigrateError, migrate_store
from memman.store.db import _BASELINE_SCHEMA


def _empty_store(store_dir: Path) -> None:
    """Build a SQLite store with the baseline schema but no rows.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _seed_with_fingerprint_only(store_dir: Path, dim: int = 512) -> None:
    """Build a SQLite store with a fingerprint but no insights.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_dir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?)',
            ('embed_fingerprint',
             '{"provider":"fixture","model":"fixture","dim":'
             + str(dim) + '}'))
        conn.commit()
    finally:
        conn.close()


def test_migrate_rejects_truly_empty_source(tmp_path):
    """Source with no insights AND no fingerprint raises MigrateError.
    """
    sdir = tmp_path / 'empty_store'
    _empty_store(sdir)
    with pytest.raises(MigrateError, match='empty'):
        migrate_store(
            source_dir=str(sdir), dsn='postgresql://unused',
            store='empty_store')


def test_migrate_opens_source_in_readonly_mode(tmp_path):
    """`sqlite3.connect` is called with the read-only URI form.
    """
    sdir = tmp_path / 'ro_check'
    _seed_with_fingerprint_only(sdir)
    seen_uris: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def spy(conn_str, *args, **kwargs):
        seen_uris.append((conn_str, kwargs.get('uri', False)))
        return real_connect(conn_str, *args, **kwargs)

    with patch('sqlite3.connect', side_effect=spy):
        try:
            migrate_store(
                source_dir=str(sdir), dsn='postgresql://unused',
                store='ro_check')
        except MigrateError:
            pass
        except Exception:
            pass

    assert any('mode=ro' in s and uri for s, uri in seen_uris), (
        f'expected sqlite3.connect to be called with read-only URI;'
        f' got {seen_uris}')
