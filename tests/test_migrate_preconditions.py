"""Source-side precondition tests for sqlite -> postgres migration.

The source SQLite is opened read-only, and an empty source (zero
insights and no fingerprint) is rejected with a clear error before any
destination work happens.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import memman.store.sqlite as sqlite_mod
import pytest
from memman.migrate import MigrateError
from memman.store.db import _BASELINE_SCHEMA
from memman.store.sqlite import SqliteMigrator


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
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             '{"model":"fixture","dim":'
             + str(dim) + '}'))
        conn.commit()
    finally:
        conn.close()


def test_migrate_rejects_truly_empty_source(tmp_path):
    """Verify a source with no insights and no fingerprint is rejected.

    Mutation: preflight_source accepting an empty store, so migrate
        creates an empty destination with no dim to resolve.
    Oracle: MigrateError matching 'empty' for a baseline-schema store.
    """
    sdir = tmp_path / 'data' / 'empty_store'
    _empty_store(sdir)
    src_mig = SqliteMigrator(str(tmp_path))
    with pytest.raises(MigrateError, match='empty'):
        src_mig.preflight_source('empty_store')


def test_migrate_opens_source_in_readonly_mode(tmp_path):
    """Verify the source is opened with the read-only URI form.

    Mutation: opening the source with a plain writable connect, so
        migrate can modify or lock the source store.
    Oracle: a spy on sqlite3.connect recording a `mode=ro` URI call.
    """
    sdir = tmp_path / 'data' / 'ro_check'
    _seed_with_fingerprint_only(sdir)
    seen_uris: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def spy(conn_str, *args, **kwargs):
        seen_uris.append((conn_str, kwargs.get('uri', False)))
        return real_connect(conn_str, *args, **kwargs)

    with patch('sqlite3.connect', side_effect=spy):
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source('ro_check')
        payload = src_mig.gather('ro_check')

    assert payload.insights == []

    assert any('mode=ro' in s and uri for s, uri in seen_uris), (
        f'expected sqlite3.connect to be called with read-only URI;'
        f' got {seen_uris}')


def test_preflight_source_reports_a_store_with_no_schema(tmp_path):
    """Verify a store that opens but holds no tables fails as `MigrateError`.

    Mutation: dropping `preflight_source`'s own `sqlite3.Error`
        handler. `_connect_ro` cannot cover this -- the file is a
        valid database, so it opens and probes clean, and the failure
        lands on the `insights` select inside the block.
    Oracle: a zero-length file, which SQLite reads as a fresh empty
        database, so `select count(*) from insights` raises `no such
        table`; `MigrateError` is outside the `sqlite3.Error`
        hierarchy.
    """
    sdir = tmp_path / 'data' / 'schemaless'
    sdir.mkdir(parents=True)
    (sdir / 'memman.db').write_bytes(b'')
    with pytest.raises(MigrateError) as excinfo:
        SqliteMigrator(str(tmp_path)).preflight_source('schemaless')
    assert 'schemaless' in str(excinfo.value)


def test_gather_reports_an_unreadable_store(tmp_path):
    """Verify a corrupt source store fails as `MigrateError` from `gather`.

    Mutation: `gather` using a bare `sqlite3.connect`.
        `connect` is lazy, so the failure would land on the first
        select inside a long `with` block and escape
        untranslated; `preflight_source` above cannot catch that,
        because `migrate --all` reaches `gather` on stores it already
        preflighted.
    Oracle: `MigrateError` is outside the `sqlite3.Error` hierarchy.
    """
    sdir = tmp_path / 'data' / 'broken'
    sdir.mkdir(parents=True)
    (sdir / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(MigrateError) as excinfo:
        SqliteMigrator(str(tmp_path)).gather('broken')
    assert 'broken' in str(excinfo.value)


def test_connect_ro_closes_the_connection_when_the_probe_fails(
        tmp_path, monkeypatch):
    """Verify a failed migration read leaves no connection behind.

    Mutation: dropping `conn.close()` from `_connect_ro`'s handler.
        `sqlite3.connect` is lazy, so it returns a live handle and the
        corrupt file only surfaces at the probe -- every refused
        migration read would then leak a file handle, and
        `migrate --all` walks every store.
    Oracle: the captured connection is probed after the raise.
        `ProgrammingError` means closed; a leaked handle raises
        `DatabaseError` on these bytes instead, so the probe
        discriminates on the type, not on the query succeeding.
    """
    captured = []
    real_connect = sqlite_mod.sqlite3.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr(sqlite_mod.sqlite3, 'connect', spy_connect)
    sdir = tmp_path / 'data' / 'broken'
    sdir.mkdir(parents=True)
    (sdir / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(MigrateError):
        SqliteMigrator(str(tmp_path)).gather('broken')
    assert captured
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute('select 1')


def test_preflight_source_refuses_a_store_mid_swap(tmp_path):
    """Verify a store with an embed swap in flight is refused.

    Mutation: dropping the swap check from `preflight_source`. The
        migrators do not carry `embed_swap_state`, so a store migrated
        mid-swap would restart its swap on the target with the cursor
        reset.
    Oracle: `run_swap` leaves `embed_swap_state` set from its first
        backfill batch until cutover clears it, so its presence means a
        swap is in flight.
    """
    sdir = tmp_path / 'data' / 'mid_swap'
    _seed_with_fingerprint_only(sdir)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_swap_state', 'backfilling'))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(MigrateError, match='embed swap'):
        SqliteMigrator(str(tmp_path)).preflight_source('mid_swap')


def test_preflight_source_offers_only_resume_for_a_swap_at_cutover(tmp_path):
    """Verify a store at swap cutover is pointed at --resume alone.

    Mutation: the refusal offering `--abort` in every swap state, so an
        operator at cutover follows advice that `abort_swap` refuses.
    Oracle: `abort_swap` raises at the cutover state, so the one remedy
        that works there is `--resume`.
    """
    sdir = tmp_path / 'data' / 'at_cutover'
    _seed_with_fingerprint_only(sdir)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_swap_state', 'cutover'))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(MigrateError) as exc:
        SqliteMigrator(str(tmp_path)).preflight_source('at_cutover')

    assert '--resume' in str(exc.value)
    assert '--abort' not in str(exc.value)
