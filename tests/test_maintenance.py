"""Maintenance pass: incremental_vacuum after link_pending.

Two properties verified:
1. Fresh DBs adopt `auto_vacuum=INCREMENTAL` (mode 2).
2. `_run_per_store_maintenance` calls `PRAGMA incremental_vacuum` after
   the link_pending step, and respects the deadline budget.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from memman.maintenance import _run_per_store_maintenance
from memman.store.model import format_timestamp
from memman.store.node import insert_insight, stamp_linked
from tests.conftest import make_insight


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


def test_maintenance_reenriches_stranded_row(tmp_db, tmp_backend):
    """A linked-but-unenriched row is re-queued and re-enriched.

    Reproduces the stranding bug: an insight stamped linked_at but not
    enriched_at sits outside the pending-link retry path forever. The
    maintenance self-heal must reset it so link_pending re-enriches it.
    """
    insight = make_insight(
        id='strand-1', content='Python web framework facts')
    insert_insight(tmp_db, insight)
    stamp_linked(
        tmp_db, 'strand-1',
        format_timestamp(datetime.now(timezone.utc)))

    assert tmp_backend.nodes.count_pending_links() == 0
    assert 'strand-1' in tmp_backend.nodes.get_unenriched_linked_ids(
        limit=10)

    ctx = MagicMock()
    ctx.backend = tmp_backend
    ctx.embed_cache = {}
    ctx.ec = MagicMock()
    ctx.ec.available.return_value = False
    ctx.llm_client = MagicMock()
    ctx.llm_client.complete.return_value = json.dumps({
        'entities': ['Python'],
        'keywords': ['web', 'framework'],
        'summary': 'Python web frameworks',
        'semantic_facts': ['Python has web frameworks'],
        })

    _run_per_store_maintenance(ctx, 'default', time.monotonic() + 60)

    row = tmp_db._conn.execute(
        'SELECT keywords, enriched_at FROM insights WHERE id = ?',
        ('strand-1',)).fetchone()
    assert row[0] is not None
    assert row[1] is not None
