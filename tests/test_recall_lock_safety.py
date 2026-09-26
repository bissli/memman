"""Recall's bookkeeping write never propagates a lock error to the caller."""

import sqlite3

from memman.cli import cli
from tests.conftest import invoke


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
            'remember', 'Go uses SQLite for persistent storage'])

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
        assert 'SQLite' in result.output, result.output
        assert any(
            'recall_bookkeep_skipped' in rec.getMessage()
            for rec in caplog.records), (
            'recall must log recall_bookkeep_skipped at debug when'
            ' the bookkeeping write loses the lock race')
