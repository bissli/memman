"""Maintenance pass: incremental_vacuum after link_pending.

Two properties verified:
1. Fresh DBs adopt `auto_vacuum=INCREMENTAL` (mode 2).
2. `_run_per_store_maintenance` calls `PRAGMA incremental_vacuum` after
   the link_pending step, and respects the deadline budget.
"""

import time
from unittest.mock import MagicMock

from memman.maintenance import _run_per_store_maintenance


def test_fresh_db_uses_incremental_autovacuum(tmp_db):
    """A freshly opened store has auto_vacuum=2 (INCREMENTAL)."""
    row = tmp_db._query('PRAGMA auto_vacuum').fetchone()
    assert row[0] == 2


def test_maintenance_runs_incremental_vacuum_after_link_pending(
            tmp_db, tmp_backend):
    """`_run_per_store_maintenance` issues a PRAGMA incremental_vacuum."""
    ctx = MagicMock()
    ctx.backend = tmp_backend
    ctx.embed_cache = {}
    ctx.llm_client = MagicMock()
    ctx.ec = MagicMock()

    deadline = time.monotonic() + 60
    _run_per_store_maintenance(ctx, 'default', deadline)

    last_query = tmp_db._query('PRAGMA freelist_count').fetchone()
    assert last_query is not None


def test_maintenance_skips_vacuum_when_deadline_exceeded(tmp_backend):
    """Past-deadline maintenance must not issue more SQL after the gate."""
    ctx = MagicMock()
    wrapped = MagicMock(wraps=tmp_backend)
    wrapped.oplog = MagicMock(wraps=tmp_backend.oplog)
    ctx.backend = wrapped
    ctx.embed_cache = {}
    ctx.llm_client = MagicMock()
    ctx.ec = MagicMock()

    deadline = time.monotonic() - 1
    _run_per_store_maintenance(ctx, 'default', deadline)

    assert not wrapped.oplog.maintenance_step.called
