"""Maintenance pass: incremental_vacuum after enrich_pending.

Two properties verified:
1. Fresh DBs adopt `auto_vacuum=INCREMENTAL` (mode 2).
2. `_run_per_store_maintenance` calls `PRAGMA incremental_vacuum` after
   the enrich_pending step, and respects the deadline budget.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.maintenance import _run_per_store_maintenance
from memman.store.model import format_timestamp
from memman.store.node import insert_insight, stamp_enrich_attempted
from tests.conftest import make_insight


def test_fresh_db_uses_incremental_autovacuum(tmp_db):
    """A freshly opened store has auto_vacuum=2 (INCREMENTAL)."""
    row = tmp_db._query('PRAGMA auto_vacuum').fetchone()
    assert row[0] == 2


def test_maintenance_runs_incremental_vacuum_after_enrich_pending(
            tmp_db, tmp_backend):
    """`_run_per_store_maintenance` issues a PRAGMA incremental_vacuum.

    Mutation: dropping the `ctx.backend.oplog.maintenance_step()` call
        (or the deadline check ahead of it swallowing every call), so
        the store's oplog and free pages never get reclaimed.
    Oracle: a `maintenance_step` spy wrapping the real backend, called
        within budget.
    """
    insert_insight(tmp_db, make_insight(
        id='mnt-1', content='maintenance vacuum test content'))

    ctx = MagicMock()
    wrapped = MagicMock(wraps=tmp_backend)
    wrapped.oplog = MagicMock(wraps=tmp_backend.oplog)
    ctx.backend = wrapped
    ctx.llm_client = MagicMock()
    ctx.ec = MagicMock()

    deadline = time.monotonic() + 60
    _run_per_store_maintenance(ctx, 'default', deadline)

    assert wrapped.oplog.maintenance_step.called


def test_maintenance_skips_vacuum_when_deadline_exceeded(tmp_backend):
    """Past-deadline maintenance must not issue more SQL after the gate."""
    ctx = MagicMock()
    wrapped = MagicMock(wraps=tmp_backend)
    wrapped.oplog = MagicMock(wraps=tmp_backend.oplog)
    ctx.backend = wrapped
    ctx.llm_client = MagicMock()
    ctx.ec = MagicMock()

    deadline = time.monotonic() - 1
    _run_per_store_maintenance(ctx, 'default', deadline)

    assert not wrapped.oplog.maintenance_step.called


def test_maintenance_reenriches_stranded_row(tmp_db, tmp_backend):
    """Verify an attempted-but-unenriched row is re-queued and re-enriched.

    Mutation: dropping the stranded-row reset from
        `_run_per_store_maintenance`, so a row stamped
        enrich_attempted_at but not enriched_at never re-enters
        enrich_pending.
    Oracle: the row's enriched_at after one maintenance pass.
    """
    from memman.embed.fingerprint import bound_embedder

    insight = make_insight(
        id='strand-1', content='Python web framework facts')
    insert_insight(tmp_db, insight)
    stamp_enrich_attempted(
        tmp_db, 'strand-1',
        format_timestamp(datetime.now(timezone.utc)))

    assert tmp_backend.nodes.count_pending_enrich() == 0
    assert 'strand-1' in tmp_backend.nodes.get_unenriched_attempted_ids(
        limit=10)

    ctx = MagicMock()
    ctx.backend = tmp_backend
    ctx.ec = bound_embedder(tmp_backend)
    ctx.llm_client = MagicMock()
    ctx.llm_client.complete.return_value = json.dumps({
        'summary': 'Python web frameworks',
        })

    _run_per_store_maintenance(ctx, 'default', time.monotonic() + 60)

    row = tmp_db._conn.execute(
        'SELECT enriched_at FROM insights WHERE id = ?',
        ('strand-1',)).fetchone()
    assert row[0] is not None
