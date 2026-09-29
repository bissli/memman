"""Recall's bookkeeping write never propagates a lock error to the caller.
"""

import logging
import sqlite3
from contextlib import contextmanager

from memman.cli import cli
from memman.store import sqlite as sqlite_mod
from tests.conftest import invoke


class TestRecallBookkeepingLockSafety:
    """Recall must never propagate `database is locked` from bookkeeping.
    """

    def test_recall_swallows_operational_error_on_bookkeep(
            self, mm_runner, monkeypatch, caplog):
        """A bookkeeping OperationalError is logged, not a CLI failure.

        The patched `transaction` raises on entry, as SQLite's BEGIN
        IMMEDIATE does once the busy_timeout is exhausted.

        Mutation: dropping the `except` around the bookkeeping write in
        recall, so a lock error surfaces as a CLI failure.
        Oracle: exit code 0, the stored row in the output, and a
        `recall_bookkeep_skipped` debug record.
        """
        r, data_dir = mm_runner
        invoke(mm_runner, [
            'remember', 'Go uses SQLite for persistent storage'])

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
