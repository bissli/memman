"""Store branches: an empty local SQLite store over a live parent.

A branch takes one thread's writes while a pasted instruction line
routes them there. `store.overlay.OverlayBackend` reads the parent
through it, so recall on a branch ranks both stores. `merge_branch`
replays the branch's writes into the parent and `drop_branch` discards
them.

Notes
-----
- A store is a branch exactly when its meta holds `branch_parent`,
  whose value names the parent. No code reads either fact from the
  name.
- The parent holds `branch_token:<branch>`, and the branch holds the
  same value under `branch_token`.
- A branch row whose id the parent holds is a copy, made when the
  branch retired that parent row. Every other branch row is
  branch-only.
"""

import logging
import os
import secrets
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from memman import config, drain_lock
from memman.embed.fingerprint import META_KEY, Fingerprint, swap_command
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.migrate import MigrateInsight
from memman.queue import purge_store, queue_db
from memman.store import db as _db
from memman.store import factory
from memman.store.backend import Backend
from memman.store.errors import BackendError, StoreMissingError
from memman.store.model import Insight, format_timestamp, insight_to_delta_dict
from memman.store.overlay import BRANCH_MERGING, BRANCH_PARENT
from memman.store.sqlite import open_sqlite_backend

logger = logging.getLogger('memman')

BRANCH_CREATED_AT = 'branch_created_at'
BRANCH_TOKEN = 'branch_token'

_INSTRUCTION = (
    'Use memman store {branch} for this thread: pass --store {branch} to'
    ' every memman recall, remember, replace, forget and insights show'
    ' call.')


def parent_token_key(branch: str) -> str:
    """The parent meta key that holds `branch`'s token.
    """
    return f'{BRANCH_TOKEN}:{branch}'


def read_branch_info(store: str, data_dir: str) -> dict[str, str] | None:
    """`{parent, created_at}` of a branch, or None for any other store.

    Raises
    ------
    BackendError
        The store directory exists but its database cannot be read.
    """
    if not _db.store_exists(data_dir, store):
        return None
    try:
        with open_sqlite_backend(store, data_dir, read_only=True) as backend:
            parent = backend.meta.get(BRANCH_PARENT)
            created_at = backend.meta.get(BRANCH_CREATED_AT)
    except sqlite3.Error as exc:
        raise BackendError(f'cannot read store {store!r}: {exc}') from exc
    if parent is None:
        return None
    return {'parent': parent, 'created_at': created_at}


def _require_existing(store: str, data_dir: str) -> None:
    """Refuse unless `store` exists on its resolved backend.

    Raises
    ------
    click.ClickException
        The store is missing, its name or routing is invalid, or the
        existence check cannot connect.
    """
    try:
        exists = factory.store_exists(store, data_dir)
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(
            f'cannot check store {store!r}: {exc}') from exc
    if not exists:
        raise click.ClickException(str(StoreMissingError(store)))


def _acquire_drain_lock(data_dir: str) -> int:
    """Take the drain lock, refusing while a drain runs.

    Raises
    ------
    click.ClickException
        Another process holds the lock.
    """
    try:
        return drain_lock.acquire(data_dir)
    except drain_lock.DrainLockBusy as exc:
        raise click.ClickException(
            'a drain is running; re-run in a moment') from exc


def _remove_env_keys(store: str, data_dir: str) -> None:
    """Delete every per-store env key of `store`.
    """
    from memman.setup.scheduler import _write_env_keys_with_flock
    per_store_keys = {
        f'{prefix}{store}' for prefix, _ in config.PER_STORE_KEY_SPECS}
    stale = per_store_keys & set(
        config.parse_env_file(config.env_file_path(data_dir)))
    if stale:
        _write_env_keys_with_flock({}, removes=stale, data_dir=data_dir)


def create_branch(data_dir: str, parent: str, label: str) -> dict[str, Any]:
    """Create `<parent>__<label>_<id>`, an empty branch of `parent`.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    parent : str
        Store to branch. Any backend. Gains only the branch's token.
    label : str
        Name part a person reads. Must pass `valid_store_name` and hold
        no `__`.

    Returns
    -------
    dict[str, Any]
        `action: branched`, `store`, `parent`, `path`, `created_at` and
        `instruction`, the line a session pastes to route its memory
        verbs to the branch.

    Raises
    ------
    click.ClickException
        On every refusal, before anything is written: a bad label, a
        missing parent, a parent that is a branch, has no fingerprint,
        holds a corrupt one, or has an embed swap or re-embed in
        progress, or a parent that cannot be read. A failed step after
        the first write removes the branch directory and env keys before
        raising.
    """
    if not _db.valid_store_name(label) or '__' in label:
        raise click.ClickException(
            f'invalid branch label {label!r}: use letters, digits, dashes'
            ' and single underscores')
    _require_existing(parent, data_dir)
    try:
        with factory.open_backend(parent, data_dir) as backend:
            parent_meta = {
                key: backend.meta.get(key)
                for key in backend.meta.keys()  # noqa: SIM118
                }
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(
            f'cannot read store {parent!r}: {exc}') from exc
    if BRANCH_PARENT in parent_meta:
        raise click.ClickException(
            f'store {parent!r} is a branch of'
            f' {parent_meta[BRANCH_PARENT]!r}; branch a store that is'
            ' not a branch')
    if not parent_meta.get(META_KEY):
        raise click.ClickException(
            f'store {parent!r} has no {META_KEY} meta key; it holds no'
            ' embed model to share with a branch')
    try:
        Fingerprint.from_json(parent_meta[META_KEY])
    except EmbedFingerprintError as exc:
        raise click.ClickException(f'store {parent!r}: {exc}') from exc
    if parent_meta.get('embed_swap_state'):
        raise click.ClickException(
            f'store {parent!r} has an embed swap in progress;'
            ' finish or abort it first')
    if parent_meta.get('embed_reembed_state') == 'in_progress':
        raise click.ClickException(
            f'store {parent!r} has a re-embed in progress; finish it first')

    name = f'{parent}__{label}_{secrets.token_hex(2)}'
    while _db.store_exists(data_dir, name):
        name = f'{parent}__{label}_{secrets.token_hex(2)}'
    created_at = format_timestamp(datetime.now(timezone.utc))
    token = secrets.token_hex(16)

    from memman.setup.scheduler import _write_env_keys_with_flock

    branch_dir = Path(_db.store_dir(data_dir, name))
    # exist_ok=False keeps a concurrent branch's directory safe.
    try:
        branch_dir.mkdir(mode=0o755, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise click.ClickException(
            f'branch name {name!r} was taken meanwhile; re-run') from exc
    except OSError as exc:
        raise click.ClickException(
            f'cannot create branch directory {str(branch_dir)!r}: {exc}'
            ) from exc
    try:
        # Not `_ensure_store_backend_key`: the default may be postgres.
        env_updates = {config.BACKEND_FOR(name): 'sqlite'}
        parent_rerank = config.parse_env_file(
            config.env_file_path(data_dir)).get(
                config.RERANK_ENABLED_FOR(parent))
        if parent_rerank is not None:
            env_updates[config.RERANK_ENABLED_FOR(name)] = parent_rerank
        _write_env_keys_with_flock(env_updates, data_dir=data_dir)
        with open_sqlite_backend(name, data_dir, create=True) as backend, \
                backend.transaction():
            backend.meta.set(META_KEY, parent_meta[META_KEY])
            backend.meta.set(BRANCH_PARENT, parent)
            backend.meta.set(BRANCH_CREATED_AT, created_at)
            backend.meta.set(BRANCH_TOKEN, token)
        with factory.open_backend(parent, data_dir) as backend:
            backend.meta.set(parent_token_key(name), token)
    except BaseException as exc:
        shutil.rmtree(branch_dir, ignore_errors=True)
        _remove_env_keys(name, data_dir)
        if isinstance(exc, (ConfigError, BackendError)):
            raise click.ClickException(f'branch failed: {exc}') from exc
        raise

    return {
        'action': 'branched',
        'store': name,
        'parent': parent,
        'path': str(branch_dir),
        'created_at': created_at,
        'instruction': _INSTRUCTION.format(branch=name),
        }


def _queued_refusal(conn: sqlite3.Connection, branch: str) -> str | None:
    """Refusal text when queue.db holds a pending or failed row for `branch`.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open queue.db connection.
    branch : str
        Branch store name.

    Returns
    -------
    str or None
        The refusal, naming the command that clears each kind of row,
        or None when no such row exists. A claimed row counts as
        pending.
    """
    from memman.setup.scheduler import STATE_STOPPED, read_state
    sql = """
select status, count(*)
from queue
where store = ? and status in ('pending', 'failed')
group by status
"""
    counts = dict(conn.execute(sql, (branch,)).fetchall())
    if not counts:
        return None
    parts = []
    if counts.get('pending'):
        drain = (
            'memman scheduler start' if read_state() == STATE_STOPPED
            else 'memman scheduler trigger')
        parts.append(
            f'{counts["pending"]} pending (run {drain} and wait for the'
            ' drain)')
    if counts.get('failed'):
        parts.append(
            f'{counts["failed"]} failed (see memman scheduler queue'
            ' failed, then memman scheduler queue retry)')
    return (
        f'store {branch!r} has queued writes: {"; ".join(parts)}; re-run'
        ' after the queue is clear')


def _read_branch_meta(branch: str, data_dir: str) -> dict[str, str]:
    """Meta of an existing branch, refusing a missing store or a non-branch.

    Raises
    ------
    click.ClickException
        The store is missing or unreadable, or its meta holds no
        `branch_parent`.
    """
    try:
        with factory.open_backend(branch, data_dir, read_only=True) as backend:
            meta = {
                key: backend.meta.get(key)
                for key in backend.meta.keys()  # noqa: SIM118
                }
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(str(exc)) from exc
    if BRANCH_PARENT not in meta:
        raise click.ClickException(
            f'store {branch!r} is not a branch; merge and drop act only on'
            ' stores made by memman store branch')
    return meta


def _set_merging(data_dir: str, branch: str, value: str | None) -> None:
    """Write the branch's `branch_merging` flag, or clear it for None.
    """
    with open_sqlite_backend(branch, data_dir) as backend:
        if value is None:
            backend.meta.delete(BRANCH_MERGING)
        else:
            backend.meta.set(BRANCH_MERGING, value)


def _remove_branch(
        conn: sqlite3.Connection, branch: str, data_dir: str) -> None:
    """Delete the branch's directory and queue rows in one queue transaction.

    Raises
    ------
    click.ClickException
        A pending or failed row for the branch reached the queue since
        the caller's check. Nothing is removed.
    """
    conn.execute('begin immediate')
    try:
        refusal = _queued_refusal(conn, branch)
        if refusal:
            raise click.ClickException(refusal)
        shutil.rmtree(_db.store_dir(data_dir, branch))
        purge_store(conn, branch)
    except BaseException:
        conn.execute('rollback')
        raise
    conn.execute('commit')
    _remove_env_keys(branch, data_dir)


def _parent_state(backend: Backend, row: Insight) -> str:
    """The state of a copied row in the parent.

    Returns
    -------
    str
        `replaced`, else `deleted`, `current`, or
        `current_with_live_predecessor`. A replaced row counts as
        replaced whether or not it is also deleted.
    """
    if row.replaced_by:
        return 'replaced'
    if row.deleted_at is not None:
        return 'deleted'
    if any(p.deleted_at is None for p in backend.nodes.predecessors(row.id)):
        return 'current_with_live_predecessor'
    return 'current'


def _retire_copied(
        backend: Backend, branch: str, ins: MigrateInsight, *,
        write: bool) -> dict[str, Any]:
    """Repeat the branch's retirement of one copied row in the parent.

    Parameters
    ----------
    backend : Backend
        The open parent, after the branch-only rows are inserted. Inside
        merge's transaction when `write` is True.
    branch : str
        Branch name, for the oplog detail.
    ins : MigrateInsight
        The branch's copy. `replaced_by` set means `replaced`, whatever
        `deleted_at` holds, else `deleted`.
    write : bool
        False reads the parent only: a row the table would retire is
        reported as a conflict.

    Returns
    -------
    dict[str, Any]
        `{'result': 'retired'}` or `{'result': 'done'}`, or the conflict
        entry `{id, branch_state, parent_state, branch_successor,
        parent_head, branch_content, parent_content}`. `parent_head` is
        the current row ending the parent's chain from `id`, `id` itself
        while current, None when the chain ends in a forgotten row. Each
        content is the text of the row beside it, None with it.
    """
    branch_state = 'replaced' if ins.replaced_by else 'deleted'
    for _ in range(2):
        before = backend.nodes.get_include_deleted(ins.id)
        parent_state = _parent_state(backend, before)
        if not write:
            break
        if branch_state == 'replaced' and parent_state.startswith('current'):
            if backend.nodes.mark_replaced(ins.id, ins.replaced_by):
                successor = backend.nodes.get_include_deleted(ins.replaced_by)
                backend.oplog.log(
                    operation='replace', insight_id=ins.id,
                    detail=f'merged from {branch}',
                    before=insight_to_delta_dict(before),
                    after=(insight_to_delta_dict(successor)
                           if successor is not None else None))
                return {'result': 'retired'}
            continue
        if branch_state == 'deleted' and parent_state == 'current':
            if backend.nodes.soft_delete_current(ins.id):
                backend.oplog.log(
                    operation='forget', insight_id=ins.id,
                    detail=f'merged from {branch}',
                    before=insight_to_delta_dict(before))
                return {'result': 'retired'}
            continue
        break
    if (branch_state == 'replaced' and parent_state == 'replaced'
            and before.replaced_by == ins.replaced_by):
        return {'result': 'done'}
    if branch_state == 'deleted' and parent_state == 'deleted':
        return {'result': 'done'}
    head = before if parent_state.startswith('current') else None
    if parent_state == 'replaced':
        head, seen = before, {before.id}
        while (head is not None and head.replaced_by
               and head.replaced_by not in seen):
            seen.add(head.replaced_by)
            head = backend.nodes.get_include_deleted(head.replaced_by)
        if head is not None and (head.replaced_by or head.deleted_at):
            head = None
    successor = (
        backend.nodes.get_include_deleted(ins.replaced_by)
        if ins.replaced_by else None)
    return {
        'id': ins.id,
        'branch_state': branch_state,
        'parent_state': parent_state,
        'branch_successor': ins.replaced_by,
        'parent_head': head.id if head is not None else None,
        'branch_content': successor.content if successor is not None else None,
        'parent_content': head.content if head is not None else None,
        }


def merge_branch(data_dir: str, branch: str) -> dict[str, Any]:
    """Replay a branch into its parent, then delete the branch.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    branch : str
        A store whose meta holds `branch_parent`. The target is that
        parent, never the active or `--store` store.

    Returns
    -------
    dict[str, Any]
        `action: merged`, `store`, `parent`, `copied` (branch-only rows
        inserted by this run), `retired` (retirements this run applied
        in the parent) and `conflicts`, the copied rows whose branch and
        parent states disagree, left as the parent holds them (see
        `_retire_copied`). A resumed run reports the conflicts as the
        parent holds them now.

    Raises
    ------
    click.ClickException
        On every refusal, before the parent is written. A failure
        inside the parent transaction rolls it back and says to re-run.

    Notes
    -----
    - Retirement table, for a copied row the branch retired:

          branch    parent                         result
          replaced  current (with or without a     mark_replaced
                    live predecessor)
          replaced  replaced by the same row       done
          replaced  replaced by another, deleted   conflict
          deleted   current                        soft_delete_current
          deleted   current_with_live_predecessor  conflict
          deleted   deleted                        done
          deleted   replaced                       conflict

    - The parent transaction inserts the branch-only rows, applies the
      retirements and removes the parent's token together. A re-run
      that finds the flag set, the token gone and every branch id in
      the parent writes nothing to the parent, then removes the branch.
    """
    lock_fd = _acquire_drain_lock(data_dir)
    try:
        meta = _read_branch_meta(branch, data_dir)
        parent = meta[BRANCH_PARENT]
        if _db.read_active(data_dir) == branch:
            raise click.ClickException(
                f'store {branch!r} is the active store; run memman store'
                f' use {parent} first')
        _require_existing(parent, data_dir)
        try:
            with factory.open_backend(parent, data_dir) as backend:
                parent_meta = {
                    key: backend.meta.get(key)
                    for key in backend.meta.keys()  # noqa: SIM118
                    }
                parent_ids = backend.nodes.get_all_ids()
        except (ConfigError, BackendError) as exc:
            raise click.ClickException(
                f'cannot read store {parent!r}: {exc}') from exc
        token_refusal = (
            f'store {parent!r} does not hold the token of branch'
            f' {branch!r}: it was removed and recreated, points at another'
            ' database, or was restored from a backup older than the'
            ' branch')
        resume = (BRANCH_MERGING in meta
                  and parent_token_key(branch) not in parent_meta)
        if not resume:
            if meta.get(META_KEY) != parent_meta.get(META_KEY):
                raise click.ClickException(
                    f'store {branch!r} and its parent {parent!r} use'
                    f' different embed models; run {swap_command(branch)}'
                    " with the parent's model first")
            for name, store_meta in ((branch, meta), (parent, parent_meta)):
                if (store_meta.get('embed_swap_state')
                        or store_meta.get('embed_reembed_state')
                        == 'in_progress'):
                    raise click.ClickException(
                        f'store {name!r} has an embed swap or re-embed in'
                        ' progress; finish it first')
            if parent_meta.get(parent_token_key(branch)) != meta.get(
                    BRANCH_TOKEN):
                raise click.ClickException(token_refusal)
            if BRANCH_MERGING not in meta:
                _set_merging(
                    data_dir, branch,
                    format_timestamp(datetime.now(timezone.utc)))

        with open_sqlite_backend(branch, data_dir, read_only=True) as backend:
            rows = [
                backend.nodes.get_raw(row_id)
                for row_id in sorted(backend.nodes.get_all_ids())
                ]
        if resume and not {row.id for row in rows} <= parent_ids:
            raise click.ClickException(token_refusal)
        retired_ids = {
            row.id for row in rows
            if row.replaced_by or row.deleted_at is not None
            }
        # In `begin immediate`, so a write enqueue let past the flag
        # check has committed before this read.
        with queue_db(data_dir) as conn:
            conn.execute('begin immediate')
            try:
                refusal = _queued_refusal(conn, branch)
                pending_targets = {
                    row[0] for row in conn.execute(
                        'select replaced_id from queue where store = ?'
                        " and status in ('pending', 'failed')"
                        ' and replaced_id is not null',
                        (parent,)).fetchall()
                    }
            finally:
                conn.execute('commit')
        if not refusal and not resume and pending_targets:
            # A retry follows its target's chain in the parent, so a
            # target replaced since it was queued counts by its chain.
            chained = set()
            with factory.open_backend(
                    parent, data_dir, read_only=True) as backend:
                for row_id in pending_targets:
                    while row_id is not None and row_id not in chained:
                        chained.add(row_id)
                        row = backend.nodes.get_include_deleted(row_id)
                        row_id = row.replaced_by if row is not None else None
            if retired_ids & chained:
                refusal = (
                    f'store {parent!r} has a pending or failed replace of a'
                    ' row the branch retired, directly or by its chain; run'
                    ' memman scheduler trigger and wait for the drain, or see'
                    ' memman scheduler queue failed, then re-run')
        if refusal:
            if not resume:
                _set_merging(data_dir, branch, None)
            raise click.ClickException(refusal)

        copied = retired = 0
        conflicts = []
        if not resume:
            try:
                with factory.open_backend(parent, data_dir) as backend, \
                        backend.transaction():
                    if backend.meta.get(META_KEY) != meta.get(META_KEY):
                        raise click.ClickException(
                            f'store {parent!r} moved to another embed model'
                            f' during the merge; run {swap_command(branch)}'
                            " with the parent's model, then re-run memman"
                            f' store merge {branch}')
                    held = backend.nodes.get_all_ids()
                    for row in rows:
                        if row.id not in held:
                            copied += backend.nodes.insert_raw(row)
                    for row in rows:
                        if row.id not in held or row.id not in retired_ids:
                            continue
                        outcome = _retire_copied(
                            backend, branch, row, write=True)
                        if outcome.get('result') == 'retired':
                            retired += 1
                        elif 'result' not in outcome:
                            conflicts.append(outcome)
                    backend.meta.delete(parent_token_key(branch))
            except click.ClickException:
                raise
            except Exception as exc:
                raise click.ClickException(
                    f'merge of {branch!r} stopped and wrote nothing to'
                    f' {parent!r}: {exc}; re-run memman store merge'
                    f' {branch}') from exc
        else:
            with factory.open_backend(
                    parent, data_dir, read_only=True) as backend:
                for row in rows:
                    if row.id not in retired_ids:
                        continue
                    outcome = _retire_copied(backend, branch, row, write=False)
                    if 'result' not in outcome:
                        conflicts.append(outcome)
        with queue_db(data_dir) as conn:
            _remove_branch(conn, branch, data_dir)
    finally:
        drain_lock.release(lock_fd)

    return {
        'action': 'merged',
        'store': branch,
        'parent': parent,
        'copied': copied,
        'retired': retired,
        'conflicts': conflicts,
        }


def drop_branch(data_dir: str, branch: str) -> dict[str, Any]:
    """Delete a branch and list the rows written in it.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    branch : str
        A store whose meta holds `branch_parent`.

    Returns
    -------
    dict[str, Any]
        `action: dropped`, `store`, `parent` and `dropped`, a list of
        `{id, content, replaces}` for each current branch row. A copy
        is always retired, so none is listed.
        `replaces` names the parent row at the root of the row's chain
        in the branch, else None. It is None for every row when the
        parent cannot answer.

    Raises
    ------
    click.ClickException
        On every refusal, before anything is removed: a missing store or
        a non-branch, the active branch, a running drain, queued writes
        for the branch, a merge stopped after its parent commit, the
        state a re-run of merge finishes, or a merge started while the
        parent exists and cannot answer.

    Notes
    -----
    - The parent's `branch_token:<branch>` is removed after the branch,
      when the parent holds it and answers. A missing parent, or an
      unreachable one when no merge started, leaves it, and the drop
      goes ahead. A failed removal leaves it and still returns the
      listing.
    """
    lock_fd = _acquire_drain_lock(data_dir)
    try:
        meta = _read_branch_meta(branch, data_dir)
        parent = meta[BRANCH_PARENT]
        if _db.read_active(data_dir) == branch:
            raise click.ClickException(
                f'store {branch!r} is the active store; run memman store'
                f' use {parent} first')
        with queue_db(data_dir) as conn:
            refusal = _queued_refusal(conn, branch)
        if refusal:
            raise click.ClickException(refusal)
        with open_sqlite_backend(branch, data_dir, read_only=True) as backend:
            branch_ids = backend.nodes.get_all_ids()
            current = backend.nodes.get_all_active()
            roots = {}
            for ins in current:
                root, seen = ins, {ins.id}
                while earlier := [
                        p for p in backend.nodes.predecessors(root.id)
                        if p.id not in seen]:
                    root = earlier[0]
                    seen.add(root.id)
                if root.id != ins.id:
                    roots[ins.id] = root.id
        # Drop holds the drain lock; the libpq default can block for
        # minutes on a host that drops packets.
        os.environ.setdefault('PGCONNECT_TIMEOUT', '3')
        parent_ids = None
        token_held = False
        try:
            if factory.store_exists(parent, data_dir):
                with factory.open_backend(
                        parent, data_dir, read_only=True) as backend:
                    parent_ids = backend.nodes.get_all_ids()
                    token_held = (
                        backend.meta.get(parent_token_key(branch)) is not None)
        except (ConfigError, BackendError, sqlite3.Error) as exc:
            if BRANCH_MERGING in meta:
                raise click.ClickException(
                    f'a merge of {branch!r} started, and {parent!r} cannot'
                    f' say whether it holds the branch: {exc}; re-run memman'
                    f' store merge {branch} or this drop when it answers'
                    ) from exc
            logger.debug(f'drop of {branch!r} cannot read {parent!r}: {exc}')
        if (BRANCH_MERGING in meta and parent_ids is not None
                and not token_held and branch_ids <= parent_ids):
            raise click.ClickException(
                f'a merge of {branch!r} stopped part way after writing'
                f' {parent!r}; re-run memman store merge {branch} to finish'
                ' it')
        with queue_db(data_dir) as conn:
            _remove_branch(conn, branch, data_dir)
        if token_held:
            # The branch is gone, so no error here may cost the caller
            # the listing of its rows.
            try:
                with factory.open_backend(parent, data_dir) as backend:
                    backend.meta.delete(parent_token_key(branch))
            except Exception as exc:
                logger.warning(
                    f'drop of {branch!r} left {parent_token_key(branch)} in'
                    f' {parent!r}: {exc}')
    finally:
        drain_lock.release(lock_fd)

    return {
        'action': 'dropped',
        'store': branch,
        'parent': parent,
        'dropped': [
            {
                'id': ins.id,
                'content': ins.content,
                'replaces': (
                    roots.get(ins.id)
                    if roots.get(ins.id) in (parent_ids or ()) else None),
                }
            for ins in current
            ],
        }
