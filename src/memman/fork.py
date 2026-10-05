"""Experiment forks: a temporary local SQLite copy of a store.

A fork starts as a copy of its parent's current rows, takes a
session's writes while a pasted instruction line routes them there,
and ends with `memman store merge` or `memman store drop`.

Notes
-----
- A store is a fork exactly when its meta holds `fork_parent`, whose
  value names the parent. No code reads either fact from the name.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from memman import config, drain_lock
from memman.embed.fingerprint import swap_command
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.migrate import MigrateError, MigrateInsight, MigrationPayload
from memman.migrate import Migrator
from memman.queue import purge_store, queue_db
from memman.store import db as _db
from memman.store import factory
from memman.store.backend import Backend
from memman.store.errors import BackendError, StoreMissingError
from memman.store.model import format_timestamp, insight_to_delta_dict
from memman.store.sqlite import SqliteMigrator, open_sqlite_backend

FORK_PARENT = 'fork_parent'
FORK_CREATED_AT = 'fork_created_at'
FORK_ROWS = 'fork_rows'
FORK_MERGING = 'fork_merging'

_INSTRUCTION = (
    'Use memman store {fork} for this thread: pass --store {fork} to every'
    ' memman recall, remember, replace, forget and insights show call. For'
    ' rows store {parent} gained since the fork, run memman recall'
    ' --store {parent} "<query>" and use only rows dated on or after'
    ' {created_date}.')


def read_fork_info(store: str, data_dir: str) -> dict[str, str] | None:
    """`{parent, created_at}` of a SQLite fork, or None for any other store.

    Raises
    ------
    BackendError
        The store directory exists but its database cannot be read.
    """
    if not _db.store_exists(data_dir, store):
        return None
    try:
        with open_sqlite_backend(store, data_dir, read_only=True) as backend:
            parent = backend.meta.get(FORK_PARENT)
            created_at = backend.meta.get(FORK_CREATED_AT)
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


def _migrator_for(store: str, data_dir: str) -> Migrator:
    """The migrator of the backend `store` resolves to.
    """
    if factory.resolve_store_backend(store, data_dir) == 'postgres':
        from memman.store.postgres import PostgresMigrator
        return PostgresMigrator(
            dsn=factory.resolve_store_pg_dsn(store, data_dir))
    return SqliteMigrator(data_dir)


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


def create_fork(data_dir: str, parent: str, label: str) -> dict[str, Any]:
    """Create `<parent>__<label>_<id>` from the parent's current rows.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    parent : str
        Store to copy. Any backend. Only read.
    label : str
        Name part a person reads. Must pass `valid_store_name` and hold
        no `__`.

    Returns
    -------
    dict[str, Any]
        `action: forked`, `store`, `parent`, `path`, `rows`,
        `created_at` and `instruction`, the line a session pastes to
        route its memory verbs to the fork.

    Raises
    ------
    click.ClickException
        On every refusal, before anything is written: a bad label, a
        missing parent, a parent that is a fork, has no fingerprint, or
        has an embed swap or re-embed in progress, a running drain, or
        a parent that cannot be read. A failed copy removes the fork
        directory and env keys before raising.
    """
    if not _db.valid_store_name(label) or '__' in label:
        raise click.ClickException(
            f'invalid fork label {label!r}: use letters, digits, dashes'
            ' and single underscores')
    _require_existing(parent, data_dir)

    lock_fd = _acquire_drain_lock(data_dir)
    try:
        try:
            payload = _migrator_for(parent, data_dir).gather(parent)
        except (MigrateError, ConfigError, BackendError,
                EmbedFingerprintError) as exc:
            raise click.ClickException(
                f'cannot fork {parent!r}: {exc}') from exc
        if FORK_PARENT in payload.meta:
            raise click.ClickException(
                f'store {parent!r} is a fork of'
                f' {payload.meta[FORK_PARENT]!r}; fork a store that is'
                ' not a fork')
        if payload.meta.get('embed_swap_state'):
            raise click.ClickException(
                f'store {parent!r} has an embed swap in progress;'
                ' finish or abort it first')
        if payload.meta.get('embed_reembed_state') == 'in_progress':
            raise click.ClickException(
                f'store {parent!r} has a re-embed in progress;'
                ' finish it first')

        name = f'{parent}__{label}_{secrets.token_hex(2)}'
        while _db.store_exists(data_dir, name):
            name = f'{parent}__{label}_{secrets.token_hex(2)}'
        created_at = format_timestamp(datetime.now(timezone.utc))
        current = [
            ins for ins in payload.insights
            if ins.deleted_at is None and ins.replaced_by is None
            ]
        fork_payload = MigrationPayload(
            fingerprint=payload.fingerprint,
            embedding_dim=payload.embedding_dim,
            insights=current,
            oplog=[],
            meta={
                'embed_fingerprint': payload.meta['embed_fingerprint'],
                FORK_PARENT: parent,
                FORK_CREATED_AT: created_at,
                FORK_ROWS: str(len(current)),
                })

        from memman.setup.scheduler import _write_env_keys_with_flock

        fork_dir = Path(_db.store_dir(data_dir, name))
        # exist_ok=False keeps a concurrent fork's directory safe.
        try:
            fork_dir.mkdir(mode=0o755, parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise click.ClickException(
                f'fork name {name!r} was taken meanwhile; re-run') from exc
        except OSError as exc:
            raise click.ClickException(
                f'cannot create fork directory {str(fork_dir)!r}: {exc}'
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
            SqliteMigrator(data_dir).apply(name, fork_payload)
        except BaseException:
            shutil.rmtree(fork_dir, ignore_errors=True)
            _remove_env_keys(name, data_dir)
            raise
    except MigrateError as exc:
        raise click.ClickException(f'fork failed: {exc}') from exc
    finally:
        drain_lock.release(lock_fd)

    return {
        'action': 'forked',
        'store': name,
        'parent': parent,
        'path': str(fork_dir),
        'rows': len(current),
        'created_at': created_at,
        'instruction': _INSTRUCTION.format(
            fork=name, parent=parent, created_date=created_at[:10]),
        }


def _queued_refusal(conn: sqlite3.Connection, fork: str) -> str | None:
    """Refusal text when queue.db holds a pending or failed row for `fork`.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open queue.db connection.
    fork : str
        Fork store name.

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
    counts = dict(conn.execute(sql, (fork,)).fetchall())
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
        f'store {fork!r} has queued writes: {"; ".join(parts)}; re-run'
        ' after the queue is clear')


def _read_fork_meta(fork: str, data_dir: str) -> dict[str, str]:
    """Meta of an existing fork, refusing a missing store or a non-fork.

    Raises
    ------
    click.ClickException
        The store is missing or unreadable, or its meta holds no
        `fork_parent`.
    """
    try:
        with factory.open_backend(fork, data_dir, read_only=True) as backend:
            meta = {key: backend.meta.get(key) for key in backend.meta.keys()}
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(str(exc)) from exc
    if FORK_PARENT not in meta:
        raise click.ClickException(
            f'store {fork!r} is not a fork; merge and drop act only on'
            ' stores made by memman store fork')
    return meta


def _remove_fork(conn: sqlite3.Connection, fork: str, data_dir: str) -> None:
    """Delete the fork's directory and queue rows in one queue transaction.

    Raises
    ------
    click.ClickException
        A pending or failed row for the fork reached the queue since the
        caller's check. Nothing is removed.
    """
    conn.execute('begin immediate')
    try:
        refusal = _queued_refusal(conn, fork)
        if refusal:
            raise click.ClickException(refusal)
        shutil.rmtree(_db.store_dir(data_dir, fork))
        purge_store(conn, fork)
    except BaseException:
        conn.execute('rollback')
        raise
    conn.execute('commit')
    _remove_env_keys(fork, data_dir)


def _parent_state(backend: Backend, row_id: str) -> tuple[str, str | None]:
    """The state of an inherited row in the parent, with its successor.

    Returns
    -------
    tuple[str, str or None]
        `replaced` with the successor id, else `deleted`, `current`, or
        `current_with_live_predecessor` with None. A replaced row
        counts as replaced whether or not it is also deleted.
    """
    row = backend.nodes.get_include_deleted(row_id)
    if row.replaced_by:
        return 'replaced', row.replaced_by
    if row.deleted_at is not None:
        return 'deleted', None
    if any(p.deleted_at is None for p in backend.nodes.predecessors(row_id)):
        return 'current_with_live_predecessor', None
    return 'current', None


def _retire_inherited(
        backend: Backend, fork: str, ins: MigrateInsight) -> dict[str, Any]:
    """Repeat the fork's retirement of one inherited row in the parent.

    Parameters
    ----------
    backend : Backend
        The open parent, inside its transaction.
    fork : str
        Fork name, for the oplog detail.
    ins : MigrateInsight
        The fork's row. `replaced_by` set means `replaced`, whatever
        `deleted_at` holds, else `deleted`.

    Returns
    -------
    dict[str, Any]
        `{'result': 'retired'}` or `{'result': 'done'}`, or the conflict
        entry `{id, fork_state, parent_state, fork_successor,
        parent_successor}`, each successor present only for a
        `replaced` state.
    """
    fork_state = 'replaced' if ins.replaced_by else 'deleted'
    for _ in range(2):
        before = backend.nodes.get_include_deleted(ins.id)
        parent_state, parent_successor = _parent_state(backend, ins.id)
        if fork_state == 'replaced' and parent_state.startswith('current'):
            if backend.nodes.mark_replaced(ins.id, ins.replaced_by):
                successor = backend.nodes.get_include_deleted(ins.replaced_by)
                backend.oplog.log(
                    operation='replace', insight_id=ins.id,
                    detail=f'merged from {fork}',
                    before=insight_to_delta_dict(before),
                    after=(insight_to_delta_dict(successor)
                           if successor is not None else None))
                return {'result': 'retired'}
            continue
        if fork_state == 'deleted' and parent_state == 'current':
            if backend.nodes.soft_delete_current(ins.id):
                backend.oplog.log(
                    operation='forget', insight_id=ins.id,
                    detail=f'merged from {fork}',
                    before=insight_to_delta_dict(before))
                return {'result': 'retired'}
            continue
        break
    if (fork_state == 'replaced' and parent_state == 'replaced'
            and parent_successor == ins.replaced_by):
        return {'result': 'done'}
    if fork_state == 'deleted' and parent_state == 'deleted':
        return {'result': 'done'}
    conflict = {
        'id': ins.id, 'fork_state': fork_state, 'parent_state': parent_state,
        }
    if fork_state == 'replaced':
        conflict['fork_successor'] = ins.replaced_by
    if parent_state == 'replaced':
        conflict['parent_successor'] = parent_successor
    return conflict


def merge_fork(data_dir: str, fork: str) -> dict[str, Any]:
    """Replay a fork into its parent, then delete the fork.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    fork : str
        A store whose meta holds `fork_parent`. The target is that
        parent, never the active or `--store` store.

    Returns
    -------
    dict[str, Any]
        `action: merged`, `store`, `parent`, `copied` (fork-only rows
        inserted), `retired` (retirements applied in the parent) and
        `conflicts`, the inherited rows whose fork and parent states
        disagree, left as the parent holds them.

    Raises
    ------
    click.ClickException
        On every refusal, before anything is written. A failure after
        the copy says to re-run, which finishes the merge.

    Notes
    -----
    - Retirement table, for an inherited row the fork retired:

          fork      parent                         result
          replaced  current (with or without a     mark_replaced
                    live predecessor)
          replaced  replaced by the same row       done
          replaced  replaced by another, deleted   conflict
          deleted   current                        soft_delete_current
          deleted   current_with_live_predecessor  conflict
          deleted   deleted                        done
          deleted   replaced                       conflict
    """
    meta = _read_fork_meta(fork, data_dir)
    parent = meta[FORK_PARENT]
    _require_existing(parent, data_dir)
    if _db.read_active(data_dir) == fork:
        raise click.ClickException(
            f'store {fork!r} is the active store; run memman store use'
            f' {parent} first')
    try:
        with factory.open_backend(parent, data_dir) as backend:
            parent_meta = {
                key: backend.meta.get(key) for key in backend.meta.keys()}
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(
            f'cannot read store {parent!r}: {exc}') from exc
    if meta.get('embed_fingerprint') != parent_meta.get('embed_fingerprint'):
        raise click.ClickException(
            f'store {fork!r} and its parent {parent!r} use different'
            f' embed models; run {swap_command(fork)} with the parent\'s'
            ' model first')
    for name, store_meta in ((fork, meta), (parent, parent_meta)):
        if (store_meta.get('embed_swap_state')
                or store_meta.get('embed_reembed_state') == 'in_progress'):
            raise click.ClickException(
                f'store {name!r} has an embed swap or re-embed in'
                ' progress; finish it first')

    lock_fd = _acquire_drain_lock(data_dir)
    try:
        payload = SqliteMigrator(data_dir).gather(fork)
        retired_ids = {
            ins.id for ins in payload.insights
            if ins.replaced_by or ins.deleted_at is not None
            }
        with queue_db(data_dir) as conn:
            refusal = _queued_refusal(conn, fork)
            if refusal:
                raise click.ClickException(refusal)
            pending_targets = {
                row[0] for row in conn.execute(
                    "select replaced_id from queue where store = ?"
                    " and status = 'pending' and replaced_id is not null",
                    (parent,)).fetchall()
                }
        if retired_ids & pending_targets:
            raise click.ClickException(
                f'store {parent!r} has a queued replace of a row the fork'
                ' retired; run memman scheduler trigger, wait for the'
                ' drain, then re-run')
        with factory.open_backend(parent, data_dir) as backend:
            parent_ids = backend.nodes.get_all_ids()
        fork_ids = {ins.id for ins in payload.insights}
        if len(fork_ids & parent_ids) < int(meta[FORK_ROWS]):
            raise click.ClickException(
                f'store {parent!r} lacks rows the fork copied: it was'
                ' removed and recreated, points at another database, or'
                ' was restored from a backup older than the fork')

        with open_sqlite_backend(fork, data_dir) as fork_backend:
            fork_backend.meta.set(
                FORK_MERGING, format_timestamp(datetime.now(timezone.utc)))
        fork_only = [
            ins for ins in payload.insights if ins.id not in parent_ids]
        try:
            _migrator_for(parent, data_dir).apply(parent, MigrationPayload(
                fingerprint=payload.fingerprint,
                embedding_dim=payload.embedding_dim,
                insights=fork_only,
                oplog=[],
                meta={}))
            retired = 0
            conflicts = []
            with factory.open_backend(parent, data_dir) as backend, \
                    backend.transaction():
                for ins in payload.insights:
                    if ins.id not in parent_ids or ins.id not in retired_ids:
                        continue
                    outcome = _retire_inherited(backend, fork, ins)
                    if outcome.get('result') == 'retired':
                        retired += 1
                    elif 'result' not in outcome:
                        conflicts.append(outcome)
        except Exception as exc:
            raise click.ClickException(
                f'merge of {fork!r} stopped part way: {exc}; re-run'
                f' memman store merge {fork} to finish it') from exc
        with queue_db(data_dir) as conn:
            _remove_fork(conn, fork, data_dir)
    finally:
        drain_lock.release(lock_fd)

    return {
        'action': 'merged',
        'store': fork,
        'parent': parent,
        'copied': len(fork_only),
        'retired': retired,
        'conflicts': conflicts,
        }


def drop_fork(data_dir: str, fork: str) -> dict[str, Any]:
    """Delete a fork and list the rows written in it.

    Parameters
    ----------
    data_dir : str
        Base memman data directory.
    fork : str
        A store whose meta holds `fork_parent`.

    Returns
    -------
    dict[str, Any]
        `action: dropped`, `store`, `parent` and `dropped`, a list of
        `{id, content}` for each current fork row the parent lacks.
        Every current fork row is listed when the parent no longer
        exists.

    Raises
    ------
    click.ClickException
        On every refusal, before anything is removed: a missing store or
        a non-fork, a fork a merge started, the active fork, a running
        drain, queued writes for the fork, or a parent whose existence
        check cannot connect.
    """
    lock_fd = _acquire_drain_lock(data_dir)
    try:
        meta = _read_fork_meta(fork, data_dir)
        parent = meta[FORK_PARENT]
        if FORK_MERGING in meta:
            raise click.ClickException(
                f'a merge of {fork!r} stopped part way and part of it is'
                f' already in {parent!r}; re-run memman store merge {fork}')
        if _db.read_active(data_dir) == fork:
            raise click.ClickException(
                f'store {fork!r} is the active store; run memman store use'
                f' {parent} first')
        with queue_db(data_dir) as conn:
            refusal = _queued_refusal(conn, fork)
        if refusal:
            raise click.ClickException(refusal)
        # Drop holds the drain lock; the libpq default can block for
        # minutes on a host that drops packets.
        os.environ.setdefault('PGCONNECT_TIMEOUT', '3')
        try:
            parent_exists = factory.store_exists(parent, data_dir)
        except (ConfigError, BackendError) as exc:
            raise click.ClickException(
                f'cannot check parent {parent!r}, so the rows to list are'
                f' unknown: {exc}') from exc
        with open_sqlite_backend(fork, data_dir, read_only=True) as backend:
            current = backend.nodes.get_all_active()
        if parent_exists:
            try:
                with factory.open_backend(parent, data_dir) as backend:
                    parent_ids = backend.nodes.get_all_ids()
            except (ConfigError, BackendError) as exc:
                raise click.ClickException(
                    f'cannot read parent {parent!r}: {exc}') from exc
            current = [ins for ins in current if ins.id not in parent_ids]
        with queue_db(data_dir) as conn:
            _remove_fork(conn, fork, data_dir)
    finally:
        drain_lock.release(lock_fd)

    return {
        'action': 'dropped',
        'store': fork,
        'parent': parent,
        'dropped': [{'id': ins.id, 'content': ins.content} for ins in current],
        }
