"""Store branches: an empty local SQLite store over a live parent.

A branch takes one thread's writes while a pasted instruction line
routes them there. `store.overlay.OverlayBackend` reads the parent
through it, so recall on a branch ranks both stores.

Notes
-----
- A store is a branch exactly when its meta holds `branch_parent`,
  whose value names the parent. No code reads either fact from the
  name.
- The parent holds `branch_token:<branch>`, and the branch holds the
  same value under `branch_token`.
"""

import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from memman import config
from memman.embed.fingerprint import META_KEY, Fingerprint
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.store import db as _db
from memman.store import factory
from memman.store.errors import BackendError, StoreMissingError
from memman.store.model import format_timestamp
from memman.store.overlay import BRANCH_PARENT
from memman.store.sqlite import open_sqlite_backend

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
