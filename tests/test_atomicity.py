"""Atomicity tests for CLI sites that wrap multi-statement work.

After autocommit=True landed on PostgresBackend, every statement
commits independently unless explicitly grouped under
`backend.transaction()`. These tests assert that the wrapping is in
place for the three CLI paths fixed in the cleanup pass:

- `insights protect` (boost_retention + refresh_effective_importance + oplog.log)
- `recall --basic` (per-result increment_access_count + oplog.log)
- `graph link` (existence checks + reverse-edge upsert + oplog.log)

Each test runs over SQLite (the default backend); the wrapping is
backend-agnostic so the same Protocol contract holds on Postgres.
"""

import pathlib
from contextlib import contextmanager

from click.testing import CliRunner
from memman.cli import cli


def _insert_seed_insight(data_dir: str, store: str, id: str, content: str) -> None:
    """Insert one insight directly via SQL (no LLM dependency)."""
    from memman.store.db import open_db, store_dir
    from memman.store.node import insert_insight
    from tests.conftest import make_insight

    sdir = store_dir(data_dir, store)
    db = open_db(sdir)
    try:
        insert_insight(db, make_insight(id=id, content=content))
    finally:
        db.close()


def _wrap_backend_open(monkeypatch):
    """Patch `active_store` to count `transaction()` entries.

    Returns a list capturing one entry per `with backend.transaction():`
    block opened during the wrapped command. Length == number of
    distinct top-level transactions.
    """
    from memman import session as session_mod
    entries: list[str] = []
    orig_active = session_mod.active_store

    @contextmanager
    def wrapped(*, data_dir: str, store: str, unchecked: bool = False):
        with orig_active(
                data_dir=data_dir, store=store, unchecked=unchecked) as backend:
            orig_tx = backend.transaction

            @contextmanager
            def counted_tx():
                entries.append('begin')
                with orig_tx() as v:
                    yield v
                entries.append('commit')

            backend.transaction = counted_tx
            yield backend

    monkeypatch.setattr(session_mod, 'active_store', wrapped)
    return entries


def test_insights_protect_runs_in_one_transaction(tmp_path, monkeypatch):
    """`insights protect` opens exactly one transaction."""
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    data_dir = str(tmp_path)
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    _insert_seed_insight(data_dir, 'default', 'p-1', 'protect target')

    entries = _wrap_backend_open(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'insights', 'protect', 'p-1'])
    assert result.exit_code == 0, result.output
    assert entries.count('begin') == 1, entries
    assert entries.count('commit') == 1, entries


def test_recall_basic_mutations_run_in_one_transaction(tmp_path, monkeypatch):
    """`recall --basic` wraps the per-result increments + oplog in one tx."""
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    data_dir = str(tmp_path)
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    _insert_seed_insight(data_dir, 'default', 'r-1', 'alpha bravo charlie')
    _insert_seed_insight(data_dir, 'default', 'r-2', 'alpha delta echo')

    entries = _wrap_backend_open(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'recall', 'alpha', '--basic'])
    assert result.exit_code == 0, result.output
    assert entries.count('begin') == 1, entries


def test_link_check_and_upsert_run_in_one_transaction(tmp_path, monkeypatch):
    """`graph link` opens exactly one transaction covering checks + writes."""
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    data_dir = str(tmp_path)
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    _insert_seed_insight(data_dir, 'default', 'l-a', 'left side')
    _insert_seed_insight(data_dir, 'default', 'l-b', 'right side')

    entries = _wrap_backend_open(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'graph', 'link', 'l-a', 'l-b'])
    assert result.exit_code == 0, result.output
    assert entries.count('begin') == 1, entries
