"""Regression suite for the recall hot-path lock-safety guarantees.

These tests pin three properties:

1. Recall must not call `reindex_auto_edges` on its own
   (`reindex_on_open=False`) — the drainer's maintenance pass is the
   sole driver of reindex after this change.
2. `reindex_auto_edges` runs in sub-second chunked transactions and
   is idempotent across retries (constants_hash stamped only after
   all chunks succeed).
3. Maintenance reindexes every on-disk store on drift, including
   stores that received no queue traffic in the current drain.
4. Recall's optional bookkeeping write (access_count + oplog) never
   propagates `database is locked` to the caller — it skips quietly.
"""

import json
import sqlite3

import pytest
from memman.cli import cli
from memman.graph import engine as engine_mod
from memman.graph.engine import compute_constants_hash, reindex_auto_edges
from memman.session import active_store
from memman.store.db import open_db, store_dir
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteBackend
from tests.conftest import invoke, make_insight


def _seed_store_with_stale_hash(
        data_dir: str, store: str, count: int = 6) -> None:
    """Create a store with `count` insights and a known-stale hash.

    Uses `'STALE-HASH-MARKER'` so the value is recognisable but does
    not match `compute_constants_hash()` (which returns a 16-char
    SHA prefix).
    """
    from memman.embed.fingerprint import Fingerprint, write_fingerprint

    sdir = store_dir(data_dir, store)
    db = open_db(str(sdir))
    backend = SqliteBackend(db)
    write_fingerprint(backend, Fingerprint(
        provider='voyage', model='voyage-3-lite', dim=512))
    backend.meta.set('constants_hash', 'STALE-HASH-MARKER')
    for i in range(count):
        insert_insight(db, make_insight(
            id=f'{store}-{i}',
            content=f'{store} insight number {i} about widgets'))
    db.close()


class TestRecallSkipsReindex:
    """Recall must not invoke reindex_auto_edges on open."""

    def test_recall_does_not_trigger_reindex_with_stale_hash(
            self, mm_runner, monkeypatch):
        """A stale constants_hash on the recall path must not fire reindex."""
        r, data_dir = mm_runner
        invoke(mm_runner, [
            'remember', 'Go uses SQLite for persistent storage',
            '--no-reconcile'])
        from memman.embed.fingerprint import Fingerprint, write_fingerprint
        sdir = store_dir(data_dir, 'default')
        db = open_db(str(sdir))
        b = SqliteBackend(db)
        write_fingerprint(b, Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        b.meta.set('constants_hash', 'STALE-HASH-MARKER')
        db.close()

        calls = []

        def _spy(*args, **kwargs):
            calls.append(1)
            return {}

        monkeypatch.setattr(engine_mod, 'reindex_auto_edges', _spy)

        result = r.invoke(cli, [
            '--data-dir', data_dir,
            'recall', '--basic', 'Go SQLite'])
        assert result.exit_code == 0, result.output
        assert not calls, (
            f'recall path triggered reindex_auto_edges {len(calls)} times'
            f' (expected 0)')

        sdir = store_dir(data_dir, 'default')
        db = open_db(str(sdir))
        b = SqliteBackend(db)
        assert b.meta.get('constants_hash') == 'STALE-HASH-MARKER', (
            'recall must NOT stamp the constants_hash')
        db.close()


class TestChunkedReindex:
    """reindex_auto_edges chunks transactions and is idempotent."""

    def test_chunked_reindex_is_idempotent(self, backend):
        """Two consecutive reindexes leave edge counts unchanged."""
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'idem-{i}',
                content=f'idempotent insight {i} about widgets'))

        reindex_auto_edges(backend, chunk_size=2)
        first_edges = sorted(
            (e.source_id, e.target_id, e.edge_type)
            for e in backend.edges.all())

        reindex_auto_edges(backend, chunk_size=2)
        second_edges = sorted(
            (e.source_id, e.target_id, e.edge_type)
            for e in backend.edges.all())

        assert first_edges == second_edges, (
            'second reindex changed the edge set; per-insight'
            ' delete-then-create must be idempotent')

    def test_chunked_reindex_crash_recovery(self, backend, monkeypatch):
        """Crash mid-chunk leaves hash stale so the next pass retries cleanly."""
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'crash-{i}',
                content=f'crash test insight {i} about widgets'))

        backend.meta.set('constants_hash', 'STALE-HASH-MARKER')

        from memman.graph import semantic as semantic_mod
        real_create = semantic_mod.create_semantic_edges
        crash_calls = {'n': 0}

        def _flaky(*args, **kwargs):
            crash_calls['n'] += 1
            if crash_calls['n'] == 3:
                raise RuntimeError('simulated mid-chunk crash')
            return real_create(*args, **kwargs)

        monkeypatch.setattr(
            'memman.graph.engine.create_semantic_edges', _flaky)

        with pytest.raises(RuntimeError, match='simulated mid-chunk crash'):
            reindex_auto_edges(backend, chunk_size=2)

        assert backend.meta.get('constants_hash') == 'STALE-HASH-MARKER', (
            'constants_hash must NOT be stamped after a mid-chunk crash')

        monkeypatch.undo()
        from memman.graph.engine import reindex_if_constants_changed
        stats = reindex_if_constants_changed(backend)
        assert stats is not None, (
            'second pass must run the reindex since the hash is still stale')
        assert backend.meta.get('constants_hash') == compute_constants_hash()


class TestRecallBookkeepingLockSafety:
    """Recall must never propagate `database is locked` from bookkeeping."""

    def test_recall_swallows_operational_error_on_bookkeep(
            self, mm_runner, monkeypatch, caplog):
        """An OperationalError raised by the bookkeeping transaction
        must be logged at debug and not surface as a CLI failure.

        Simulates the lock-busy condition by patching the backend's
        `transaction` context manager to raise on entry, mirroring
        what SQLite's BEGIN IMMEDIATE does once the 5s busy_timeout
        is exhausted.
        """
        import logging
        from contextlib import contextmanager

        r, data_dir = mm_runner
        invoke(mm_runner, [
            'remember', 'Go uses SQLite for persistent storage',
            '--no-reconcile'])

        from memman.store import sqlite as sqlite_mod

        @contextmanager
        def _boom_transaction(self):
            raise sqlite3.OperationalError('database is locked')
            yield  # pragma: no cover

        monkeypatch.setattr(
            sqlite_mod.SqliteBackend, 'transaction', _boom_transaction)

        with caplog.at_level(logging.DEBUG, logger='memman'):
            result = r.invoke(cli, [
                '--data-dir', data_dir, '--debug',
                'recall', '--basic', 'Go SQLite'])

        assert result.exit_code == 0, result.output
        assert '"results"' in result.output, result.output
        assert any(
            'recall_bookkeep_skipped' in rec.getMessage()
            for rec in caplog.records), (
            'recall must log recall_bookkeep_skipped at debug when'
            ' the bookkeeping write loses the lock race')


class TestActiveStoreReindexOnOpenKwarg:
    """Regression guard for the new reindex_on_open kwarg."""

    def test_default_true_still_reindexes(
            self, tmp_path, monkeypatch):
        """Default behavior (no kwarg) preserves the on-open reindex."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv('MEMMAN_DATA_DIR', data_dir)
        _seed_store_with_stale_hash(data_dir, 'default', count=3)

        calls = []

        def _spy(*args, **kwargs):
            calls.append(1)

        monkeypatch.setattr(
            'memman.graph.engine.reindex_if_constants_changed', _spy)

        with active_store(data_dir=data_dir, store='default') as backend:
            _ = backend.nodes.get_all_active()
        assert calls, (
            'default active_store(reindex_on_open=True) must call'
            ' reindex_if_constants_changed')

    def test_false_skips_reindex(self, tmp_path, monkeypatch):
        """reindex_on_open=False suppresses the on-open reindex call."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv('MEMMAN_DATA_DIR', data_dir)
        _seed_store_with_stale_hash(data_dir, 'default', count=3)

        calls = []

        def _spy(*args, **kwargs):
            calls.append(1)

        monkeypatch.setattr(
            'memman.graph.engine.reindex_if_constants_changed', _spy)

        with active_store(
                data_dir=data_dir, store='default',
                reindex_on_open=False) as backend:
            _ = backend.nodes.get_all_active()
        assert not calls, (
            'active_store(reindex_on_open=False) must NOT call'
            ' reindex_if_constants_changed')


class TestMaintenanceReindexesAllStores:
    """Maintenance must reindex every store with drift, not just touched ones."""

    def test_maintenance_repairs_quiet_store(self, mm_runner):
        """A store with no queue traffic still has its hash repaired."""
        r, data_dir = mm_runner
        invoke(mm_runner, [
            'remember', 'first store insight',
            '--no-reconcile'])
        invoke(mm_runner, [
            '--store', 'quiet',
            'remember', 'quiet store insight',
            '--no-reconcile'])

        for store_name in ('default', 'quiet'):
            sdir = store_dir(data_dir, store_name)
            db = open_db(str(sdir))
            b = SqliteBackend(db)
            b.meta.set('constants_hash', 'STALE-HASH-MARKER')
            db.close()

        result = r.invoke(cli, [
            '--data-dir', data_dir,
            'scheduler', 'drain'])
        assert result.exit_code == 0, result.output

        expected = compute_constants_hash()
        for store_name in ('default', 'quiet'):
            sdir = store_dir(data_dir, store_name)
            db = open_db(str(sdir))
            b = SqliteBackend(db)
            actual = b.meta.get('constants_hash')
            db.close()
            assert actual == expected, (
                f'store {store_name!r} hash {actual!r} did not converge'
                f' to {expected!r}; quiet-store reindex regression')
