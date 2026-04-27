"""Worker maintenance phase, run once per drain after the row loop.

Steps:
1. `queue.purge_done` -- drop completed queue rows.
2. `queue.purge_worker_runs` -- prune the heartbeat ledger.
3. Per touched store with rows_processed > 0:
   - `trim_oplog_by_age` (once per drain, not per row).
   - `link_pending` with a small batch cap so a backlog of pending
     enrichments cannot blow the maintenance budget.
   - Recall snapshot write (delegated to the cli helper).

Each step is bounded by the remaining drain timeout; if less than
30 s remains the entire maintenance phase is skipped and rolled to
the next drain.
"""

import logging
import time

logger = logging.getLogger('memman')

MAINTENANCE_MIN_BUDGET_SECONDS = 30
MAINTENANCE_LINK_PENDING_MAX = 3


def run_maintenance(
        queue_conn,
        data_dir: str,
        touched_stores: set[str],
        store_contexts: dict[str, object],
        deadline_monotonic: float,
        snapshot_writer) -> None:
    """Execute the post-drain maintenance pass.

    `snapshot_writer(data_dir, store_name)` is the cli helper that
    materializes a recall snapshot. Passed in to avoid a circular
    import between maintenance and cli.
    """
    if time.monotonic() + MAINTENANCE_MIN_BUDGET_SECONDS > deadline_monotonic:
        logger.debug(
            'maintenance: skipped, less than'
            f' {MAINTENANCE_MIN_BUDGET_SECONDS}s of budget remains')
        return

    from memman.queue import purge_done, purge_worker_runs

    try:
        dropped = purge_done(queue_conn)
        if dropped:
            logger.debug(f'maintenance: purged {dropped} done queue rows')
    except Exception:
        logger.exception('maintenance: purge_done failed')

    try:
        dropped = purge_worker_runs(queue_conn)
        if dropped:
            logger.debug(
                f'maintenance: pruned {dropped} stale worker_runs rows')
    except Exception:
        logger.exception('maintenance: purge_worker_runs failed')

    for store_name in touched_stores:
        if time.monotonic() >= deadline_monotonic:
            logger.debug(
                'maintenance: deadline reached mid-store loop')
            break
        ctx = store_contexts.get(store_name)
        if ctx is None:
            continue
        _run_per_store_maintenance(
            ctx, store_name, deadline_monotonic)
        try:
            snapshot_writer(data_dir, store_name)
        except Exception:
            logger.exception(
                f'maintenance: snapshot write failed for {store_name!r}')


def _run_per_store_maintenance(
        ctx, store_name: str, deadline_monotonic: float) -> None:
    """Run oplog trim + bounded link_pending for one store."""
    from memman.graph.engine import link_pending
    from memman.store.node import count_pending_links
    from memman.store.oplog import trim_oplog_by_age

    try:
        pruned = trim_oplog_by_age(ctx.db)
        if pruned:
            logger.debug(
                f'maintenance: trimmed {pruned} oplog rows in'
                f' {store_name!r}')
    except Exception:
        logger.exception(
            f'maintenance: trim_oplog_by_age failed for {store_name!r}')

    if time.monotonic() >= deadline_monotonic:
        return

    try:
        pending = count_pending_links(ctx.db)
    except Exception:
        logger.exception(
            f'maintenance: count_pending_links failed for {store_name!r}')
        return
    if pending == 0:
        return

    try:
        processed = link_pending(
            ctx.db,
            embed_cache=ctx.embed_cache,
            llm_client=ctx.llm_client,
            embed_client=ctx.ec,
            max_batch=MAINTENANCE_LINK_PENDING_MAX)
        if processed:
            logger.debug(
                f'maintenance: link_pending processed {processed} insights'
                f' in {store_name!r} (capped at {MAINTENANCE_LINK_PENDING_MAX})')
    except Exception:
        logger.exception(
            f'maintenance: link_pending failed for {store_name!r}')
