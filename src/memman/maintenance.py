"""Worker maintenance phase, run once per drain after the row loop.

Steps:
1. `queue.purge_done` -- drop completed queue rows.
2. `queue.purge_worker_runs` -- prune the heartbeat ledger.
3. All-stores pass: `reindex_if_constants_changed` for every store
   on disk (touched or not). Hash matches -> O(1) no-op; drift
   triggers the chunked reindex here, asynchronously from any user
   recall on the hot path.
4. Per touched store with rows_processed > 0:
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
from typing import Any

logger = logging.getLogger('memman')

MAINTENANCE_MIN_BUDGET_SECONDS = 30
MAINTENANCE_LINK_PENDING_MAX = 3
MAINTENANCE_REENRICH_MAX = 3


def run_maintenance(
        queue_conn: Any,
        data_dir: str,
        touched_stores: set[str],
        store_contexts: dict[str, Any],
        deadline_monotonic: float,
        snapshot_writer: Any) -> None:
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

    from memman.queue import purge_done, purge_worker_runs, retry_stale

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

    try:
        requeued = retry_stale(queue_conn)
        if requeued:
            logger.debug(
                f'maintenance: re-queued {requeued} stale rows back to pending')
    except Exception:
        logger.exception('maintenance: retry_stale failed')

    _reindex_all_stores_if_drift(
        data_dir, store_contexts, deadline_monotonic)

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
            snapshot_writer(data_dir, store_name, ctx._stored_fp)
        except Exception:
            logger.exception(
                f'maintenance: snapshot write failed for {store_name!r}')


def _relink_pending_if_any(
        backend: Any, store_name: str,
        deadline_monotonic: float, *,
        embed_client: Any = None,
        llm_client: Any = None) -> None:
    """Drain a bounded slice of a store's pending-link backlog.

    Quiet stores whose `linked_at` was cleared by a constants-hash
    reindex are otherwise never relinked. Already-enriched rows relink
    without an LLM pass (the `enriched_at` guard in `link_pending`), so
    the batch is the full `MAX_LINK_BATCH`. Bounded by the deadline and
    gated on a cheap `count_pending_links()` so stores with nothing
    pending pay O(1).
    """
    if time.monotonic() >= deadline_monotonic:
        return
    from memman.graph.engine import MAX_LINK_BATCH, link_pending
    try:
        if backend.nodes.count_pending_links() == 0:
            return
    except Exception:
        logger.exception(
            f'maintenance: count_pending_links failed for {store_name!r}')
        return
    if embed_client is None or llm_client is None:
        from memman.embed.fingerprint import bound_embedder
        from memman.llm.client import get_llm_client
        if embed_client is None:
            try:
                embed_client = bound_embedder(backend)
            except Exception:
                embed_client = None
        if llm_client is None:
            try:
                llm_client = get_llm_client('slow_canonical')
            except Exception:
                logger.exception(
                    f'maintenance: llm client unavailable for relink of'
                    f' {store_name!r}')
                return
    try:
        processed = link_pending(
            backend, llm_client=llm_client, embed_client=embed_client,
            max_batch=MAX_LINK_BATCH, store_name=store_name)
        if processed:
            logger.debug(
                f'maintenance: relinked {processed} pending rows in'
                f' {store_name!r}')
    except Exception:
        logger.exception(
            f'maintenance: relink failed for {store_name!r}')


def _reindex_all_stores_if_drift(
        data_dir: str,
        store_contexts: dict[str, Any],
        deadline_monotonic: float) -> None:
    """Reindex auto-edges for every on-disk store whose constants hash drifted.

    Touched stores reuse the open `ctx.backend`; untouched stores are
    opened transiently with `unchecked=True` so a missing fingerprint
    (fresh store with no data) is not fatal. Hash-compare is O(1)
    when nothing has drifted; the chunked reindex only fires on the
    rare drift event (e.g. after a constants-table edit on deploy).
    """
    from memman.graph.engine import reindex_if_constants_changed
    from memman.session import active_store
    from memman.store.factory import list_stores

    try:
        stores = list_stores(data_dir)
    except Exception:
        logger.exception('maintenance: list_stores failed')
        return

    for store_name in stores:
        if time.monotonic() >= deadline_monotonic:
            logger.debug(
                'maintenance: deadline reached mid all-stores reindex pass')
            return
        ctx = store_contexts.get(store_name)
        if ctx is not None:
            try:
                reindex_if_constants_changed(
                    ctx.backend, store_name=store_name)
                _relink_pending_if_any(
                    ctx.backend, store_name, deadline_monotonic,
                    embed_client=ctx.ec, llm_client=ctx.llm_client)
            except Exception:
                logger.exception(
                    f'maintenance: reindex_if_constants_changed failed'
                    f' for touched store {store_name!r}')
            continue
        try:
            with active_store(
                    data_dir=data_dir, store=store_name,
                    unchecked=True) as backend:
                reindex_if_constants_changed(
                    backend, store_name=store_name)
                _relink_pending_if_any(
                    backend, store_name, deadline_monotonic)
        except Exception:
            logger.exception(
                f'maintenance: reindex_if_constants_changed failed'
                f' for quiet store {store_name!r}')


def _run_per_store_maintenance(
        ctx: Any, store_name: str,
        deadline_monotonic: float) -> None:
    """Run oplog trim + bounded link_pending for one store."""
    from memman.graph.engine import link_pending

    try:
        pruned = ctx.backend.oplog.trim_by_age()
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
        stranded = ctx.backend.nodes.get_unenriched_linked_ids(
            limit=MAINTENANCE_REENRICH_MAX)
        if stranded:
            ctx.backend.nodes.reset_for_rebuild(stranded)
            logger.debug(
                f'maintenance: re-queued {len(stranded)} stranded'
                f' (linked-but-unenriched) rows in {store_name!r}')
    except Exception:
        logger.exception(
            f'maintenance: re-queue stranded rows failed for'
            f' {store_name!r}')

    try:
        pending = ctx.backend.nodes.count_pending_links()
    except Exception:
        logger.exception(
            f'maintenance: count_pending_links failed for {store_name!r}')
        return
    if pending == 0:
        return

    try:
        processed = link_pending(
            ctx.backend,
            embed_cache=ctx.embed_cache,
            llm_client=ctx.llm_client,
            embed_client=ctx.ec,
            max_batch=MAINTENANCE_LINK_PENDING_MAX,
            store_name=store_name)
        if processed:
            logger.debug(
                f'maintenance: link_pending processed {processed} insights'
                f' in {store_name!r} (capped at {MAINTENANCE_LINK_PENDING_MAX})')
    except Exception:
        logger.exception(
            f'maintenance: link_pending failed for {store_name!r}')

    if time.monotonic() >= deadline_monotonic:
        return

    try:
        ctx.backend.oplog.maintenance_step()
    except Exception:
        logger.exception(
            f'maintenance: incremental_vacuum failed for {store_name!r}')
