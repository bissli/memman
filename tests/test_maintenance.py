"""Maintenance pass: oplog maintenance step after enrich_pending.

- Fresh DBs adopt `auto_vacuum=INCREMENTAL` (mode 2).
- `_run_per_store_maintenance` runs the oplog maintenance step and
  respects the deadline budget.
- A row stamped attempted but not enriched is re-enriched.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.embed.fingerprint import bound_embedder
from memman.maintenance import _run_per_store_maintenance
from memman.store.model import format_timestamp
from memman.store.node import insert_insight, stamp_enrich_attempted
from tests.conftest import make_insight


def test_fresh_db_uses_incremental_autovacuum(tmp_db):
    """Verify a freshly opened store has auto_vacuum=2 (INCREMENTAL).

    Mutation: dropping the auto_vacuum pragma from the baseline schema, so
        freed pages are never returned to the file.
    Oracle: sqlite's own `pragma auto_vacuum` value, 2.
    """
    row = tmp_db._query('pragma auto_vacuum').fetchone()
    assert row[0] == 2


def test_maintenance_runs_incremental_vacuum_after_enrich_pending(
            tmp_db, tmp_backend):
    """Verify _run_per_store_maintenance calls the oplog maintenance step.

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
    """Verify maintenance past its deadline skips the oplog maintenance step.

    Mutation: dropping the deadline gate, so a pass that is out of budget
        keeps issuing SQL.
    Oracle: a maintenance_step spy that stays uncalled with a deadline one
        second in the past.
    """
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
        'select enriched_at from insights where id = ?',
        ('strand-1',)).fetchone()
    assert row[0] is not None


def test_maintenance_caps_the_oplog_when_nothing_awaits_enrichment(
        tmp_backend):
    """Verify the oplog maintenance step runs on a store with no pending rows.

    Mutation: the step placed after the `pending == 0` early return, so a
        store whose rows are all enriched never has its oplog capped.
    Oracle: a `maintenance_step` spy on an empty store, where
        `count_pending_enrich` is 0.
    """
    ctx = MagicMock()
    wrapped = MagicMock(wraps=tmp_backend)
    wrapped.oplog = MagicMock(wraps=tmp_backend.oplog)
    ctx.backend = wrapped

    assert tmp_backend.nodes.count_pending_enrich() == 0
    _run_per_store_maintenance(ctx, 'default', time.monotonic() + 60)

    assert wrapped.oplog.maintenance_step.called
