"""Embedding swap orchestration shared by both backends.

State machine: pending -> backfilling -> cutover -> done.

Per-store progress is persisted in `meta.embed_swap_*` keys so a
crashed swap can resume from the recorded cursor without re-embedding
completed rows. Backend-specific DDL (lazy `embedding_pending` column,
cutover transaction, abort cleanup) lives on each backend instance:

- `backend.swap_prepare(target_dim)` -- idempotent ADD COLUMN
- `backend.iter_for_swap(cursor, batch)` -- rows where
  `embedding_pending is null`, ordered by id, after `cursor`
- `backend.write_swap_batch(items)` -- batch update of
  `embedding_pending` for each (id, vec)
- `backend.swap_cutover(provider, model, dim)` -- atomic switch
  (drop+rename on Postgres, copy+null on SQLite); writes the new
  fingerprint
- `backend.swap_abort()` -- drop/clear `embedding_pending` and clear
  swap meta

The `embed_swap:<store>` advisory lock is acquired by the caller
(typically the `embed swap` CLI), not the orchestrator -- the lock
must be held continuously across multiple orchestrator calls when
resuming.
"""

import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from memman.embed.fingerprint import Fingerprint, write_fingerprint

STATE_BACKFILLING = 'backfilling'
STATE_CUTOVER = 'cutover'
STATE_DONE = 'done'

VALID_STATES = frozenset({STATE_BACKFILLING, STATE_CUTOVER, STATE_DONE})

DEFAULT_BATCH_SIZE = 200

META_STATE = 'embed_swap_state'
META_CURSOR = 'embed_swap_cursor'
META_PROVIDER = 'embed_swap_target_provider'
META_MODEL = 'embed_swap_target_model'
META_DIM = 'embed_swap_target_dim'

_META_KEYS = (
    META_STATE, META_CURSOR, META_PROVIDER, META_MODEL, META_DIM)


@dataclass
class SwapPlan:
    """Target embedder fingerprint for a swap.
    """
    target_provider: str
    target_model: str
    target_dim: int


@dataclass
class SwapProgress:
    """Snapshot of swap meta keys read from a backend.
    """
    state: str
    cursor: str
    target_provider: str
    target_model: str
    target_dim: int


def batch_size_from_env() -> int:
    """Read `MEMMAN_EMBED_SWAP_BATCH_SIZE` (default 200).
    """
    raw = os.environ.get('MEMMAN_EMBED_SWAP_BATCH_SIZE')
    if not raw:
        return DEFAULT_BATCH_SIZE
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_BATCH_SIZE
    return n if n > 0 else DEFAULT_BATCH_SIZE


def read_progress(backend: Any) -> SwapProgress:
    """Snapshot the swap meta keys from a backend.
    """
    meta = backend.meta
    dim_raw = meta.get(META_DIM) or '0'
    try:
        dim = int(dim_raw)
    except ValueError:
        dim = 0
    return SwapProgress(
        state=meta.get(META_STATE) or '',
        cursor=meta.get(META_CURSOR) or '',
        target_provider=meta.get(META_PROVIDER) or '',
        target_model=meta.get(META_MODEL) or '',
        target_dim=dim)


def _begin_swap(backend: Any, plan: SwapPlan) -> None:
    """Add `embedding_pending` column and write swap meta.
    """
    backend.swap_prepare(plan.target_dim)
    with backend.transaction():
        backend.meta.set(META_STATE, STATE_BACKFILLING)
        backend.meta.set(META_CURSOR, '')
        backend.meta.set(META_PROVIDER, plan.target_provider)
        backend.meta.set(META_MODEL, plan.target_model)
        backend.meta.set(META_DIM, str(plan.target_dim))


def _backfill_step(
        backend: Any, ec_new: Any, batch_size: int) -> int:
    """Embed and write one batch into `embedding_pending`. Return rows
    processed; zero return signals backfill complete.
    """
    cursor = backend.meta.get(META_CURSOR) or ''
    rows = backend.iter_for_swap(cursor, batch_size)
    if not rows:
        return 0
    texts = [content for (_id, content) in rows]
    vecs = ec_new.embed_batch(texts)
    if len(vecs) != len(rows):
        raise RuntimeError(
            f'embed_batch returned {len(vecs)} vectors for'
            f' {len(rows)} inputs')
    items = [(rid, vecs[i]) for i, (rid, _) in enumerate(rows)]
    last_id = items[-1][0]
    with backend.transaction():
        backend.write_swap_batch(items)
        backend.meta.set(META_CURSOR, last_id)
    return len(items)


def _commit_cutover(backend: Any, plan: SwapPlan) -> None:
    """Mark cutover-in-progress, run backend swap_cutover, drop swap meta.
    """
    with backend.transaction():
        backend.meta.set(META_STATE, STATE_CUTOVER)
    target = Fingerprint(
        provider=plan.target_provider,
        model=plan.target_model,
        dim=plan.target_dim)
    backend.swap_cutover(target)
    with backend.transaction():
        write_fingerprint(backend, target)
        for key in _META_KEYS:
            backend.meta.delete(key)


def abort_swap(backend: Any) -> None:
    """Drop `embedding_pending`/null shadow values and clear all swap meta.
    """
    backend.swap_abort()
    with backend.transaction():
        for key in _META_KEYS:
            backend.meta.delete(key)


def run_swap(
        backend: Any, ec_new: Any, plan: SwapPlan, *,
        batch_size: int | None = None,
        progress_cb: Callable[[int], None] | None = None
        ) -> SwapProgress:
    """Run the full swap workflow end-to-end. Idempotent + resumable.

    Reads existing progress; if no swap is in flight starts a new
    one. Continues backfilling until exhausted, then cuts over.
    """
    if batch_size is None:
        batch_size = batch_size_from_env()
    progress = read_progress(backend)
    if progress.state == '':
        from memman.embed.fingerprint import stored_fingerprint
        stored = stored_fingerprint(backend)
        target = Fingerprint(
            provider=plan.target_provider,
            model=plan.target_model,
            dim=plan.target_dim)
        if stored == target:
            return SwapProgress(
                state=STATE_DONE,
                cursor='',
                target_provider=plan.target_provider,
                target_model=plan.target_model,
                target_dim=plan.target_dim)
        _begin_swap(backend, plan)
    elif progress.state == STATE_BACKFILLING:
        if (progress.target_provider != plan.target_provider
                or progress.target_model != plan.target_model
                or progress.target_dim != plan.target_dim):
            raise RuntimeError(
                f'in-flight swap target'
                f' ({progress.target_provider}/'
                f'{progress.target_model}/dim={progress.target_dim})'
                f' does not match requested target'
                f' ({plan.target_provider}/{plan.target_model}/'
                f'dim={plan.target_dim}); abort first or pass --resume'
                ' without --to')
    elif progress.state == STATE_CUTOVER:
        pass
    else:
        raise RuntimeError(
            f'unknown swap state {progress.state!r}; manual cleanup'
            ' required')

    if progress.state in {'', STATE_BACKFILLING}:
        total_filled = 0
        while True:
            n = _backfill_step(backend, ec_new, batch_size)
            total_filled += n
            if progress_cb is not None:
                progress_cb(total_filled)
            if n == 0:
                break

    _commit_cutover(backend, plan)
    return SwapProgress(
        state=STATE_DONE,
        cursor='',
        target_provider=plan.target_provider,
        target_model=plan.target_model,
        target_dim=plan.target_dim)
