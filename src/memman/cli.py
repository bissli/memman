"""Click CLI for memman.

This module is the entry point and argument-parsing surface only. Core
write-path orchestration lives in `memman.pipeline.remember`, and
re-enrichment in `memman.pipeline.enrich`. Storage, search, embed, and
LLM primitives live under their own packages.
"""

import functools
import json
import logging
import logging.handlers
import math
import os
import pathlib
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from importlib.resources import files as pkg_files
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote

import click
import memman
from memman import branch as branch_mod
from memman import config
from memman.drain_lock import DrainLockBusy, acquire, release
from memman.embed import fingerprint, get_client
from memman.embed import registry as _ec_registry
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.exceptions import ConfigError, EmbedCredentialError
from memman.exceptions import EmbedFingerprintError
from memman.migrate import MigrateError, SchemaState
from memman.migrate import _verify_destination_counts, held_drain_lock
from memman.queue import STATUS_FAILED, claim, enqueue, find_pending
from memman.queue import find_pending_replace, finish_worker_run, get_row
from memman.queue import last_worker_run, list_rows, mark_done, mark_failed
from memman.queue import purge_done, queue_db, queue_db_path, retry_row
from memman.queue import start_worker_run
from memman.queue import stats as queue_stats
from memman.setup.archive import archive_postgres_schema
from memman.store import factory
from memman.store.db import default_data_dir, get_meta, list_local_store_dirs
from memman.store.db import open_db, open_read_only, portable_store_name
from memman.store.db import read_active, store_dir, store_exists
from memman.store.db import valid_store_name, write_active
from memman.store.errors import BackendError
from memman.store.errors import ConfigError as StoreConfigError
from memman.store.errors import StoreMissingError
from memman.store.factory import known_backends, list_stores
from memman.store.factory import resolve_store_backend, resolve_store_pg_dsn
from memman.store.model import Insight, format_timestamp
from memman.store.model import insight_to_delta_dict, insight_to_full_dict
from memman.store.model import insight_to_recall_line
from memman.store.node import count_active_insights, get_stats
from memman.store.node import iter_for_reembed
from memman.store.overlay import BRANCH_MERGING, BRANCH_PARENT, OverlayBackend
from memman.store.sqlite import SqliteBackend, SqliteMigrator
from memman.store.sqlite import open_sqlite_backend
from tqdm import tqdm

if TYPE_CHECKING:
    from memman.embed import EmbeddingProvider
    from memman.llm.client import MemmanLLMClient
    from memman.queue import QueueRow
    from memman.store.backend import Backend

_BACKEND_CHOICES = sorted(known_backends())

logger = logging.getLogger('memman')

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
_WORKER_LOG_MAX_BYTES = 5 * 1024 * 1024
_WORKER_LOG_BACKUPS = 3
_MAX_CONTENT_BYTES = 1000
_LINE_WORD_RE = re.compile(r'\blines? \d+\b', re.IGNORECASE)
_LINE_BREAK_RE = re.compile(r'[\r\n\v\f\x85\u2028\u2029]')
# Notes:
# - A label is at most three words before a colon and a space, so
#   `Fix:`, `**Fix:**`, `- AWS gotcha:` and `User decision
#   2026-09-17:` refuse.
# - A longer run before the colon is as often a sentence ("The rule
#   is simple:") as a label, so it passes.
# - The space after the colon keeps `localhost:6379`, `14:18` and
#   `https://` from reading as a label.
# - A quote or backtick ends the match, so a memory may open on a
#   quoted error such as `fatal: not a git repository`.
_LEADING_LABEL_RE = re.compile(
    r'^[\s*_#>-]*[^\s:`"\']+(?: [^\s:`"\']+){0,2}:[*_`]*(?=\s)')
# Notes:
# - Only a source, config or doc extension marks a locator: a bare
#   dot-letter run would also match a dotted host such as
#   `db.example.com:5432` and refuse its port.
# - `localhost:8080`, `192.0.2.1:8000`, `14:18`, `python:3.11` and
#   `code:404` carry no such extension before the colon.
# - `-N` is grep's context-line form and `~N` an approximate line.
_FILE_LINE_RE = re.compile(
    r'\b[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|md|rst|txt|html|htm|css|scss'
    r'|json|jsonl|yaml|yml|toml|ini|cfg|conf|sql|sh|bash|zsh|ps1|rs|go'
    r'|java|kt|c|h|cc|cpp|hpp|cs|rb|php|swift|lua|xml|csv|tsv|ipynb'
    r'|drawio|tf|vue|svelte|proto|mk|cmake)(?:[:-]|\s+~)\d{1,5}\b',
    re.IGNORECASE)
# Notes:
# - A bare `:N` or `L123` names a line of a file the text named
#   earlier. Only a start, space, `(`, `,` or `;` may precede it, so a
#   slice `[:80]`, `DISPLAY=:99` and `14:18` pass.
# - `L` takes two digits or more, so an `L1` or `L2` cache passes.
_BARE_LINE_RE = re.compile(r'(?<![^\s(,;])(?::\d{1,5}|~?L\d{2,5})\b')
_CALL_LINE_RE = re.compile(
    r'(?P<started_at>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
    r'\|(?P<verb>[a-z][a-z -]*)\|(?:[a-zA-Z0-9][\w-]*|\?)\|\d+\|\d+')


def _line_locator_refusal_message(content: str) -> str | None:
    """Return the refusal for text that names a line number, or None.

    Parameters
    ----------
    content : str
        The write text, as `remember` or `replace` received it.

    Returns
    -------
    str or None
        The refusal quoting the first `path.ext:N`, `path.ext-N` or
        `path.ext ~N` locator, else the first `line N` or `lines N`
        phrase, else the first bare `:N` or `L123`; None when `content`
        holds none of them. A bare `:N` also refuses a port written
        without its host, such as `:9222`; `localhost:9222` passes.
    """
    match = (_FILE_LINE_RE.search(content)
             or _LINE_WORD_RE.search(content)
             or _BARE_LINE_RE.search(content))
    if match is None:
        return None
    return (
        f'content names a line number ({match.group(0)!r}), which goes'
        ' stale on the next edit; name the file and the function or'
        ' symbol instead')


def _author_refusal_message(content: str) -> str | None:
    """Return the refusal for text that opens with its author, or None.

    Parameters
    ----------
    content : str
        The write text, as `remember` or `replace` received it.

    Returns
    -------
    str or None
        The refusal message when the first word of `content`, past any
        leading whitespace or punctuation, is the explicitly set
        `MEMMAN_AUTHOR`, compared case-insensitively; None otherwise.
        The `getpass.getuser()` fallback never refuses, so an OS login
        such as `ubuntu` cannot refuse text about that system.
    """
    author = os.environ.get(config.AUTHOR, '')
    if not author:
        return None
    pattern = re.compile(
        r'^\W*' + re.escape(author) + r'(?:\W|$)', re.IGNORECASE)
    if not pattern.match(content):
        return None
    return (
        f'content starts with the author name {author!r};'
        ' the author field already records who wrote this --'
        ' start with the subject instead')


def _content_refusal_message(
        content: str, verb: str = 'remember') -> str | None:
    """Return the refusal for text not shaped as one memory, or None.

    Parameters
    ----------
    content : str
        The write text, as `remember` or `replace` received it.
    verb : str, default 'remember'
        The command refused. For `replace`, the size refusal keeps the
        corrected claim in the replace, since a `remember` retires
        nothing.

    Returns
    -------
    str or None
        The refusal of the first check `content` fails, in the order
        size, line number, author, line break, leading label; None
        when it passes all five. The size cap counts UTF-8 bytes. No
        check rewrites, so the stored row is the agent's own words.
    """
    content_bytes = len(content.encode('utf-8'))
    if content_bytes > _MAX_CONTENT_BYTES:
        if verb == 'replace':
            advice = (
                'keep the corrected claim in the replace, and store'
                ' the other claims with remember, one thought each')
        else:
            advice = 'split it into several remember calls, one thought each'
        return (
            f'content too long ({content_bytes} bytes, max'
            f' {_MAX_CONTENT_BYTES}); {advice}')
    refusal = (_line_locator_refusal_message(content)
               or _author_refusal_message(content))
    if refusal:
        return refusal
    if _LINE_BREAK_RE.search(content):
        return (
            'content spans several lines; write one thought as one'
            ' paragraph, and give each further thought its own call')
    label = _LEADING_LABEL_RE.match(content)
    if label:
        return (
            f'content opens with a label ({label.group(0)!r});'
            ' open on the subject and write the thought as a sentence')
    return None


def _configure_logging(data_dir: str, verbose: bool, debug: bool) -> None:
    """Configure the memman logger once per process.

    Runs on every CLI invocation including `memman install` (before
    the env file exists). The literal `'WARNING'` fall-through must
    equal `INSTALL_DEFAULTS[LOG_LEVEL]`; a unit test enforces that.

    Parameters
    ----------
    data_dir : str
        Base data directory. Under the worker, `logs/memman.log` lives
        here.
    verbose : bool
        INFO level. Overridden by `debug`.
    debug : bool
        DEBUG level.

    Notes
    -----
    - Levels are set per handler. The stream handler carries the
      configured level, so an interactive caller sees no change. Under
      the worker the file handler takes DEBUG and the logger opens to
      DEBUG to feed it, which keeps a stack the stream never prints.
    - Rotation bounds the worker log at `_WORKER_LOG_MAX_BYTES` x
      (`_WORKER_LOG_BACKUPS` + 1). Routine drain DEBUG traffic shares
      that budget, so a stack can rotate away while the pointer naming
      it sits in the unrotated `enrich.err`.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        raw = config.get(config.LOG_LEVEL) or 'WARNING'
        level = getattr(logging, raw.upper(), logging.WARNING)

    logger.setLevel(level)

    stream = next(
        (h for h in logger.handlers
         if isinstance(h, logging.StreamHandler)
         and not isinstance(h, logging.FileHandler)
         and getattr(h, '_memman', False)),
        None)
    if stream is None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_LOG_FORMAT))
        stream._memman = True
        logger.addHandler(stream)
    stream.setLevel(level)

    if config.is_worker():
        handler = next(
            (h for h in logger.handlers
             if isinstance(h, logging.handlers.RotatingFileHandler)
             and getattr(h, '_memman', False)),
            None)
        if handler is None:
            log_dir = pathlib.Path(data_dir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_dir / 'memman.log',
                maxBytes=_WORKER_LOG_MAX_BYTES,
                backupCount=_WORKER_LOG_BACKUPS)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            handler._memman = True
            logger.addHandler(handler)
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


def _json_out(obj: object) -> None:
    """Write JSON to stdout as one line with sorted keys.

    One line keeps every key on the line `| tail -1` shows. Under
    `--pretty` the JSON indents by two spaces instead.
    """
    pretty = click.get_current_context().meta.get('memman.pretty')
    click.echo(json.dumps(obj, sort_keys=True, indent=2 if pretty else None))


def _require_started(action: str) -> None:
    """Reject the current CLI invocation when the scheduler is stopped.

    When the scheduler is stopped, memman is recall-only: a gated
    command exits 1 with a fixed message that points the operator at
    `memman scheduler start`.
    """
    from memman.setup.scheduler import STATE_STOPPED, read_state
    if read_state() == STATE_STOPPED:
        raise click.ClickException(
            f"Scheduler is stopped; cannot {action}."
            " Run 'memman scheduler start' to enable.")


def _require_stopped(action: str) -> None:
    """Reject the current CLI invocation when the scheduler is started.

    Inverse of `_require_started`, for a command that cannot run while
    the worker may be claiming queued `remember` rows mid-sweep.
    """
    from memman.setup.scheduler import STATE_STOPPED, read_state
    if read_state() != STATE_STOPPED:
        raise click.ClickException(
            f"Scheduler is started; cannot {action}."
            " Run 'memman scheduler stop' first.")


def _resolve_store_name(data_dir: str, store_flag: str) -> str:
    """Validate the name from the flag, environment, or active-store file.
    """
    name = (store_flag or os.environ.get(config.STORE, '')
            or read_active(data_dir))
    if not valid_store_name(name):
        raise click.ClickException(
            f'invalid store name {name!r}'
            ' (start with an alphanumeric character; use only letters,'
            ' digits, dashes, and underscores)')
    return name


def _ensure_store_backend_key(store_name: str, data_dir: str) -> None:
    """Hot-path: write `MEMMAN_BACKEND_<store>` from the default if missing.

    Two-process safe via `_write_env_keys_with_flock`. No-op when the
    per-store key is already present. Single-machine only -- shared
    filesystems (NFS) are out of scope.
    """
    from memman.setup.scheduler import _write_env_keys_with_flock

    file_values = config.parse_env_file(config.env_file_path(data_dir))
    if config.BACKEND_FOR(store_name) in file_values:
        return
    default_kind = (file_values.get(config.DEFAULT_BACKEND)
                    or 'sqlite').lower()
    updates: dict[str, str] = {
        config.BACKEND_FOR(store_name): default_kind,
        }
    if default_kind == 'postgres':
        default_dsn = file_values.get(config.DEFAULT_PG_DSN)
        if default_dsn:
            updates[config.POSTGRES_DSN_FOR(store_name)] = default_dsn
    _write_env_keys_with_flock(updates, data_dir=data_dir)


def _enqueue_into_existing_store(
        conn: sqlite3.Connection, data_dir: str, name: str, content: str,
        *, replaced_id: str | None = None,
        author: str | None = None) -> tuple[int, str]:
    """Queue a write for `name` after checking the store exists.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open queue.db connection in autocommit mode.
    data_dir : str
        Base data directory.
    name : str
        Resolved store name.
    content : str
        The memory text.
    replaced_id : str or None, default None
        Id the write replaces, as `memman.queue.enqueue` takes it.
    author : str or None, default None
        Resolved author.

    Returns
    -------
    tuple[int, str]
        `enqueue`'s `(row_id, queue_uuid)`.

    Raises
    ------
    click.ClickException
        The store does not exist, its config cannot reach it (no DSN,
        no postgres extra), or it is a branch a merge started. Nothing
        is queued. A Postgres connection error is no answer, so that
        write queues.
    """
    if resolve_store_backend(name, data_dir) == 'sqlite':
        # Pairs with the queue checks of `branch._remove_branch` and
        # `branch.merge_branch`, so a write cannot outlive a drop or
        # slip past a merge.
        conn.execute('begin immediate')
        try:
            if not store_exists(data_dir, name):
                raise click.ClickException(str(StoreMissingError(name)))
            try:
                with open_sqlite_backend(
                        name, data_dir, read_only=True) as backend:
                    merging = backend.meta.get(BRANCH_MERGING) is not None
            except BackendError as exc:
                logger.debug(f'merge flag check for {name!r} failed: {exc}')
                merging = False
            if merging:
                raise click.ClickException(
                    f'store {name!r} is merging into its parent; re-run'
                    f' memman store merge {name} to finish it')
            queued = enqueue(
                conn, store=name, content=content,
                replaced_id=replaced_id, author=author)
        except BaseException:
            conn.execute('rollback')
            raise
        conn.execute('commit')
        return queued
    # libpq reads this at each connect and a DSN's own connect_timeout
    # wins. psycopg's default outlasts an agent's tool timeout, which
    # then retries the write.
    os.environ.setdefault('PGCONNECT_TIMEOUT', '3')
    try:
        exists = factory.store_exists(name, data_dir)
    except StoreConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    except BackendError as exc:
        logger.debug(f'store check for {name!r} failed, queuing: {exc}')
        exists = True
    if not exists:
        raise click.ClickException(str(StoreMissingError(name)))
    return enqueue(
        conn, store=name, content=content,
        replaced_id=replaced_id, author=author)


def _get_llm_client_or_fail() -> 'MemmanLLMClient':
    """Return the LLM client, re-wrapping ConfigError as ClickException.

    Keeps `memman.llm` free of `click` - the CLI boundary is the only
    place that should know how to surface a user-facing config error.
    """
    from memman.llm.client import get_llm_client
    try:
        return get_llm_client()
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _active_backend(
        ctx: click.Context, *,
        unchecked: bool = False) -> AbstractContextManager['Backend']:
    """Click adapter around `memman.session.active_store`.

    Resolves data_dir and the active store name from the click context
    and yields the standard "active Backend" context manager. Use as:

        with _active_backend(ctx) as backend:
            ...

    Pass `unchecked=True` from diagnostics (`doctor`, `embed status`)
    that must run against a stale or fresh store without tripping the
    fingerprint assert.
    """
    from memman.session import active_store
    data_dir = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir, ctx.obj['store'])
    return active_store(
        data_dir=data_dir, store=name, unchecked=unchecked)


def _parse_since(since: str) -> str:
    """Parse a relative time string (e.g. '7d', '24h') to ISO timestamp.
    """
    m = re.match(r'^(\d+)([dhm])$', since)
    if not m:
        raise click.ClickException(
            f'Invalid --since format: {since} (use e.g. 7d, 24h, 30m)')
    val, unit = int(m.group(1)), m.group(2)
    delta = {'d': timedelta(days=val), 'h': timedelta(hours=val),
             'm': timedelta(minutes=val)}[unit]
    cutoff = datetime.now(timezone.utc) - delta
    return format_timestamp(cutoff)


class MemmanGroup(click.Group):
    """Root group that reports a backend failure as a clean CLI error.

    It also takes `--pretty` at any position before `--`, which indents
    every JSON reply. The flag shows in no `--help`.

    Notes
    -----
    - One `invoke` override covers the whole command tree: a group
      runs its subcommands inside its own `invoke`, so a
      `BackendError` raised at any depth passes through here.
    - The caught type is `BackendError` and its subclasses, so
      `store.errors.ConfigError` comes here too. A constraint
      violation exits as one line like any other. `--debug` recovers
      the stack of any of them.
    - `session.active_store` keeps its own earlier catch. This seam
      covers the queue, the read-only opens, and every mid-command
      failure the Postgres backend translates.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Pull every `--pretty` before `--` out of args, then parse the rest.

        Parameters
        ----------
        ctx : click.Context
            Root context. `ctx.meta['memman.pretty']`, which every
            subcommand context shares, records whether the flag appeared.
        args : list[str]
            The full command line after `memman`, subcommand included.

        Returns
        -------
        list[str]
            What `click.Group.parse_args` returns for the remaining args.
        """
        end = args.index('--') if '--' in args else len(args)
        ctx.meta['memman.pretty'] = '--pretty' in args[:end]
        kept = [arg for arg in args[:end] if arg != '--pretty'] + args[end:]
        return super().parse_args(ctx, kept)

    def invoke(self, ctx: click.Context) -> Any:
        """Run the subcommand, reporting a backend failure as a message.
        """
        # Notes:
        # - Catching here keeps the callers that branch on a driver
        #   type intact (`queue.claim`'s stale-claim reclaim,
        #   `recall`'s bookkeeping skip). Their handlers sit deeper and
        #   run first, which translation in `DB._query` / `DB._exec`
        #   would preempt.
        # - The `exc_info=True` calls stay in the arms: outside a
        #   lexical handler the formatter reads them as dead and
        #   strips the keyword.
        # - `logger.exception` is the wrong level for these arms. The
        #   stream handler is always attached, so an ERROR record
        #   prints its traceback to interactive users and undoes the
        #   one-line exit.
        try:
            return super().invoke(ctx)
        except BackendError as exc:
            logger.debug('backend error reached the CLI seam', exc_info=True)
            raise click.ClickException(self._name_the_stack(str(exc))) from exc
        # The SQLite backend translates no statement failure such as
        # `database is locked`, so without this arm it exits as a raw
        # traceback where Postgres exits as one line.
        except sqlite3.Error as exc:
            logger.debug('sqlite error reached the CLI seam', exc_info=True)
            raise click.ClickException(
                self._name_the_stack(
                    f'sqlite query failed: {exc}')) from exc

    @staticmethod
    def _name_the_stack(user_message: str) -> str:
        """Append the log file carrying the stack, when one does.

        Parameters
        ----------
        user_message : str
            The one-line message Click prints.

        Returns
        -------
        str
            `user_message`, with the log file appended when a rotating
            file handler is attached to carry the stack.

        Notes
        -----
        - The pointer rides the message instead of a log record, so no
          log level (`MEMMAN_LOG_LEVEL=ERROR`) can suppress it.
        - The worker's stack goes to the rotated `logs/memman.log`,
          which `memman log worker --stack` reads. Its stderr is an
          unrotated systemd `append:` redirect.
        """
        # An attached handler is the test, since `scheduler serve` sets
        # the worker flag after the root callback configured logging,
        # so `is_worker()` can read true while no file holds a stack.
        for handler in logger.handlers:
            if (isinstance(handler, logging.handlers.RotatingFileHandler)
                    and getattr(handler, '_memman', False)):
                stack_file = pathlib.Path(handler.baseFilename)
                writer_data_dir = str(stack_file.parent.parent)
                # The command carries `--data-dir` before the
                # subcommand, the only position Click accepts. The
                # worker's `MEMMAN_DATA_DIR` never reaches the
                # operator's shell, so a bare command would tail
                # another install's log.
                pin = ('' if writer_data_dir == default_data_dir()
                       else f' --data-dir {writer_data_dir}')
                return (f'{user_message} (full traceback in'
                        f' {stack_file}; read it with'
                        f" 'memman{pin} log worker --stack')")
        return user_message


@click.group(cls=MemmanGroup)
@click.version_option(version=memman.__version__, prog_name='memman')
@click.option('--data-dir', default=None,
              help='Base data directory (env: MEMMAN_DATA_DIR)')
@click.option('--store', 'store_name', default='', help='Named memory store')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='INFO-level logging to stderr')
@click.option('--debug', is_flag=True, default=False,
              help='DEBUG-level logging to stderr (overrides --verbose)')
@click.pass_context
def cli(ctx: click.Context, data_dir: str | None, store_name: str,
        verbose: bool, debug: bool) -> None:
    """Persistent memory store for LLM agents.
    """
    if data_dir is None:
        data_dir = os.environ.get(config.DATA_DIR, default_data_dir())
    else:
        # config.get(), prime, and every other env_file_path() call
        # without a data_dir read MEMMAN_DATA_DIR, so the flag must
        # land there too.
        os.environ[config.DATA_DIR] = data_dir
    _configure_logging(data_dir, verbose, debug)
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    ctx.obj['store'] = store_name
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug


def _call_log_path(data_dir: str) -> pathlib.Path:
    """Path of the agent-verb call log under `data_dir`.
    """
    return pathlib.Path(data_dir) / 'logs' / 'calls.log'


def claude_callable(
        cmd: click.Command | None = None, *,
        store_option: bool = True) -> Any:
    """Mark a Click command as agent-callable and log each of its calls.

    `memman install` walks the CLI tree and emits a `permissions.allow`
    entry in `~/.claude/settings.json` for every command marked with
    this decorator. Use it bare, or as `@claude_callable(store_option=
    False)`.

    Parameters
    ----------
    cmd : click.Command or None, default None
        The command to mark. Its callback is wrapped in place. None
        returns the decorator.
    store_option : bool, default True
        Add a per-verb `--store`, which routes like the group flag, so
        `memman recall --store X` matches the `memman recall` allow
        rule. Giving both flags with different values is a usage
        error.

    Returns
    -------
    click.Command
        `cmd`, whose every call appends one line to the call log
        (`memman log calls`) when its body finishes, whether it
        succeeds or fails.

    Notes
    -----
    - Line format: `<utc start>|<verb path>|<store>|<exit code>|<ms>`.
      Arguments never enter the line, since they carry memory text. A
      store name `valid_store_name` rejects is written as `?`.
    - The line covers the command body only. A call Click rejects
      while parsing (bad arguments, `--help`) writes no line, and
      `<ms>` leaves out process startup.
    - A body that raises anything other than a Click exit records 1.
      A failed append logs a warning and leaves the call's outcome
      unchanged.
    """
    if cmd is None:
        return functools.partial(claude_callable, store_option=store_option)
    cmd.claude_callable = True
    callback = cmd.callback
    if store_option:
        cmd.params.append(click.Option(
            ['--store', 'verb_store'], default='',
            help='Named memory store, as the group --store flag'))

    @functools.wraps(callback)
    def logged_callback(*args: Any, **kwargs: Any) -> Any:
        ctx = click.get_current_context()
        if store_option:
            verb_store = kwargs.pop('verb_store')
            group_store = ctx.obj['store']
            if verb_store and group_store and verb_store != group_store:
                raise click.UsageError(
                    f'--store given twice: {group_store!r} before the'
                    f' verb and {verb_store!r} after it')
            if verb_store:
                ctx.obj['store'] = verb_store
        started_at = format_timestamp(datetime.now(timezone.utc))
        started = time.monotonic()
        exit_code = 1
        try:
            result = callback(*args, **kwargs)
            exit_code = 0
            return result
        except (click.exceptions.Exit, click.ClickException) as exc:
            exit_code = exc.exit_code
            raise
        finally:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            data_dir = ctx.obj['data_dir']
            try:
                store = _resolve_store_name(data_dir, ctx.obj['store'])
            except click.ClickException:
                store = '?'
            verb = ctx.command_path.split(' ', 1)[1]
            line = f'{started_at}|{verb}|{store}|{exit_code}|{elapsed_ms}\n'
            log_path = _call_log_path(data_dir)
            try:
                log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                # One unbuffered O_APPEND write per call, so concurrent
                # sessions never interleave a line.
                fd = os.open(
                    log_path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                    0o600)
                try:
                    os.write(fd, line.encode())
                finally:
                    os.close(fd)
            except OSError as exc:
                logger.warning(f'call log append failed: {exc}')

    cmd.callback = logged_callback
    return cmd


def list_agent_commands() -> list[tuple[str, ...]]:
    """Subcommand paths of every @claude_callable command, sorted.
    """
    def walk(group: click.Group,
             prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        for name, cmd in group.commands.items():
            path = (*prefix, name)
            if isinstance(cmd, click.Group):
                out.extend(walk(cmd, path))
            elif getattr(cmd, 'claude_callable', False):
                out.append(path)
        return out
    return sorted(walk(cli, ()))


def list_claude_permissions() -> list[str]:
    """Return `permissions.allow` entries for every @claude_callable command.

    Order is stable: alphabetical by full dotted path.
    """
    return [f'Bash(memman {" ".join(path)}:*)'
            for path in list_agent_commands()]


@cli.group(name='embed')
def embed_grp() -> None:
    """Embed operations: status, re-embed, model swap.
    """


@cli.group(no_args_is_help=True)
def scheduler() -> None:
    """Async write pipeline: scheduler state, queue, worker logs.
    """


@scheduler.group('queue', invoke_without_command=True)
@click.pass_context
def queue(ctx: click.Context) -> None:
    """Inspect and manage the deferred-write queue.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(queue_list)


@cli.group()
def insights() -> None:
    """Operations on stored insights (show, review).
    """


@cli.group()
def log() -> None:
    """View memman logs (operation audit + worker output).
    """


@cli.group(name='config')
def config_cmd() -> None:
    """Inspect and modify persisted memman settings.
    """


@config_cmd.command('set')
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Write `KEY=VALUE` to the env file, bypassing the install seed model.

    Use this to change a persistable setting after initial install --
    switching backends, rotating an API key, updating a DSN. Install
    flags remain sticky-seed (they never override an existing file
    value); `config set` is the explicit override path.

    Four shapes of key are accepted:
      * any member of `config.INSTALLABLE_KEYS`
      * `MEMMAN_BACKEND_<store>` (per-store backend routing)
      * `MEMMAN_POSTGRES_DSN_<store>` (per-store DSN)
      * `MEMMAN_RERANK_ENABLED_<store>` (per-store rerank toggle)
    Bare canonicals (`MEMMAN_BACKEND`, `MEMMAN_POSTGRES_DSN`) are
    rejected with hints pointing at `MEMMAN_DEFAULT_*` or the
    per-store form.
    """
    bare_canonicals = {
        'MEMMAN_BACKEND': (
            'use MEMMAN_DEFAULT_BACKEND as the default backend'
            ' or MEMMAN_BACKEND_<store> for a specific store'),
        'MEMMAN_POSTGRES_DSN': (
            'use MEMMAN_DEFAULT_POSTGRES_DSN as the default Postgres DSN'
            ' or MEMMAN_POSTGRES_DSN_<store> for a specific store'),
        }
    if key in bare_canonicals:
        raise click.ClickException(
            f'{key!r} is not accepted under the per-store routing'
            f' model; {bare_canonicals[key]}')

    accepted = key in config.INSTALLABLE_KEYS or any(
        key.startswith(prefix) and valid_store_name(key[len(prefix):])
        for prefix, _ in config.PER_STORE_KEY_SPECS)
    if not accepted:
        shapes = ', '.join(p + '<store>'
                           for p, _ in config.PER_STORE_KEY_SPECS)
        raise click.ClickException(
            f'{key!r} is not a recognized config key. Accepted shapes:'
            f' INSTALLABLE_KEYS members or {shapes}.')
    from memman.setup.scheduler import _write_env_keys
    data_dir = ctx.obj['data_dir']
    _write_env_keys({key: value}, data_dir=data_dir)
    config.reset_file_cache()
    click.echo(f'set {key} in {config.env_file_path(data_dir)}')


@config_cmd.command('set-pg-dsn')
@click.option('--store', 'store', default=None,
              help='Per-store key: writes MEMMAN_POSTGRES_DSN_<store>.')
@click.option('--default', 'is_default', is_flag=True,
              help='Default fallback: writes MEMMAN_DEFAULT_POSTGRES_DSN.')
@click.pass_context
def config_set_pg_dsn(
        ctx: click.Context, store: str | None, is_default: bool) -> None:
    """Build a Postgres DSN from prompts and write it to the env file.

    Frees the operator from hand-typing libpq URIs. Prompts for host,
    port, user, password, and dbname; assembles a `postgresql://...`
    URI (URL-encoding credentials), and persists it under either
    `MEMMAN_DEFAULT_POSTGRES_DSN` (`--default`) or
    `MEMMAN_POSTGRES_DSN_<store>` (`--store NAME`). Exactly one of the two
    flags is required. No connectivity probe; verify with
    `memman doctor` or `memman migrate --dry-run`.
    """
    from memman.setup.scheduler import _write_env_keys
    from memman.trace import redact_dsn

    if is_default == bool(store):
        raise click.ClickException(
            'specify exactly one of --default or --store NAME')
    if store is not None and not valid_store_name(store):
        raise click.ClickException(
            f'invalid store name {store!r}'
            ' (alnum, dash, underscore; 1-64 chars)')

    host = click.prompt('host', default='localhost')
    port = click.prompt('port', type=int, default=5432)
    user = click.prompt('user')
    password = click.prompt(
        'password (leave empty to use ~/.pgpass)',
        hide_input=True, default='', show_default=False)
    dbname = click.prompt('dbname', default='memman')

    auth = quote(user, safe='')
    if password:
        auth = f'{auth}:{quote(password, safe="")}'
    dsn = f'postgresql://{auth}@{host}:{port}/{quote(dbname, safe="")}'

    key = (config.DEFAULT_PG_DSN if is_default
           else config.POSTGRES_DSN_FOR(store))
    data_dir = ctx.obj['data_dir']
    _write_env_keys({key: dsn}, data_dir=data_dir)
    config.reset_file_cache()
    click.echo(f'set {key}={redact_dsn(dsn)} in {config.env_file_path(data_dir)}')


@config_cmd.command('get')
@click.argument('key')
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
    """Print the value of `KEY` from the env file.

    Reads from `<MEMMAN_DATA_DIR>/env` only, matching the runtime
    resolver. A POSTGRES_DSN key is redacted with `redact_dsn`; an
    API_KEY key prints as `***REDACTED***`; every other key prints
    as stored. Exits 1 if the key is unset.
    """
    data_dir = ctx.obj['data_dir']
    parsed = config.parse_env_file(config.env_file_path(data_dir))
    value = parsed.get(key)
    if value is None or value == '':
        raise click.ClickException(f'{key} is not set')
    if 'POSTGRES_DSN' in key:
        from memman.trace import redact_dsn
        click.echo(redact_dsn(value))
    elif 'API_KEY' in key:
        click.echo('***REDACTED***')
    else:
        click.echo(value)


@config_cmd.command('show')
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Dump effective config: env vars + on-disk files + scheduler state.
    """
    effective = config.enumerate_effective_config()
    data_dir = ctx.obj['data_dir']
    parsed = config.parse_env_file(config.env_file_path(data_dir))
    if parsed.get(config.DEFAULT_BACKEND):
        effective[config.DEFAULT_BACKEND] = parsed[config.DEFAULT_BACKEND]
    if parsed.get(config.DEFAULT_PG_DSN):
        effective[config.DEFAULT_PG_DSN] = '***REDACTED***'
    per_store: dict[str, str] = {}
    for key, value in sorted(parsed.items()):
        if not value:
            continue
        for prefix, secret in config.PER_STORE_KEY_SPECS:
            if key.startswith(prefix):
                per_store[key] = '***REDACTED***' if secret else value
                break
    out: dict = {
        'data_dir': data_dir,
        'env': effective,
        'per_store': per_store,
        'files': {},
        'active_store': _resolve_store_name(data_dir, ctx.obj['store']),
        }
    from memman.setup.scheduler import _state_file_path, read_state
    out['files']['scheduler.state'] = {
        'path': str(_state_file_path()),
        'value': read_state(),
        }
    from memman.setup.scheduler import status as scheduler_status_fn
    s = scheduler_status_fn()
    out['scheduler'] = {
        'state': s.get('state'),
        'installed': s.get('installed'),
        'platform': s.get('platform'),
        'interval_seconds': s.get('interval_seconds'),
        }
    _json_out(out)


@claude_callable
@cli.command()
@click.argument('content', nargs=-1, required=True)
@click.pass_context
def remember(ctx: click.Context, content: tuple[str, ...]) -> None:
    """Queue a new memory and list the current rows it may correct.

    \b
    Parameters
    ----------
    content : str
        The memory text, stored as one row exactly as typed.

    \b
    Notes
    -----
    - The JSON reply carries `id`, the id the row takes once the drain
      lands it, and `related`: up to three current rows of at most
      1,000 bytes, as `<id8> <content>`, ranked by shared words
      divided by the square root of the row's distinct words.
    - A store that does not exist is refused before anything queues.
      The write is queued before the store is read. A read that fails
      gives `related_error` in place of `related`, and the command
      still exits 0.
    - Refused when the scheduler is stopped, and for text the
      one-memory shape checks reject. `quality_warnings` never block.

    \b
    Examples
    --------
    memman remember "the retry cap is five"
    """  # noqa: D301, D410, D411
    _require_started('write')
    content_str = ' '.join(content)
    author = config.resolve_author()
    refusal = _content_refusal_message(content_str)
    if refusal:
        raise click.ClickException(refusal)

    from memman.search.quality import check_content_quality
    quality_warnings = check_content_quality(content_str)

    data_dir_val = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir_val, ctx.obj['store'])

    with queue_db(data_dir_val) as conn:
        row_id, queue_uuid = _enqueue_into_existing_store(
            conn, data_dir_val, name, content_str, author=author)
    reply = {
        'action': 'queued',
        'id': queue_uuid,
        'queue_id': row_id,
        'store': name,
        'quality_warnings': quality_warnings,
        }

    from memman.search.keyword import insight_tokens, tokenize

    # The write is already queued, so no failure of this read may fail
    # the command: an agent that sees exit 1 writes the memory again.
    try:
        with factory.open_backend(
                name, data_dir_val, read_only=True) as backend:
            with backend.recall_session() as session:
                counts = session.keyword_counts(tokenize(content_str))
            # A timer drain can land this write before the read, and a
            # row shares every word with itself.
            related_rows = [
                ins for ins in backend.nodes.get_all_active()
                if ins.id in counts and ins.id != queue_uuid
                and len(ins.content.encode('utf-8')) <= _MAX_CONTENT_BYTES
                ]
        # A row can match the store's own index on a word the tokenizer
        # splits otherwise (a decomposed accent), so its length floors
        # at one.
        related_rows.sort(
            key=lambda ins: counts[ins.id] / math.sqrt(
                max(len(insight_tokens(ins)), 1)),
            reverse=True)
        reply['related'] = [
            f"{ins.id[:8]} {' '.join(ins.content.split())}"
            for ins in related_rows[:3]
            ]
    except Exception as exc:
        logger.debug('related read failed', exc_info=True)
        reply['related_error'] = f'{type(exc).__name__}: {exc}'
    _json_out(reply)


_STOP_REQUESTED = False


def _request_stop() -> None:
    """Flip the module-level stop flag.

    Polled by the serve loop and `_drain_queue`'s inner row loop so a
    SIGTERM during a long drain exits within seconds rather than the
    full per-row timeout.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _stop_requested() -> bool:
    """Return whether a stop has been signaled.
    """
    return _STOP_REQUESTED


def _clear_stop() -> None:
    """Clear the stop flag, so a later drain in this process claims rows.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


_LAST_HEARTBEAT_AT: dict[str, float] = {}
HEARTBEAT_MIN_INTERVAL_SECONDS = 60


@scheduler.command('drain', hidden=True)
@click.option('--limit', default=100, type=int,
              help='Max blobs processed per invocation')
@click.option('--timeout', default=300, type=int,
              help='Max wall-clock seconds per invocation')
@click.option('--stores', default='',
              help='Comma-separated store names; default all')
@click.option('--progress', is_flag=True, default=False,
              help='Echo per-blob progress')
@click.option('--trace', is_flag=True, default=False,
              help='Write structured trace to ~/.memman/logs/debug.log')
@click.pass_context
def scheduler_drain(ctx: click.Context, limit: int,
                    timeout: int, stores: str, progress: bool,
                    trace: bool) -> None:
    """Run the worker drain loop. Hidden: invoked by the systemd/launchd
    unit's ExecStart. Operators should use `scheduler trigger` to kick
    the unit, or `scheduler queue list` to inspect pending rows.
    """
    if trace:
        os.environ[config.DEBUG] = '1'
    _drain_queue(ctx, limit, timeout, stores, progress)


def _maybe_fire_backup(
        data_dir: str, now: datetime,
        settle: Callable[[], None] | None = None) -> None:
    """Run a backup in-process when the cron matches this minute.

    Serve mode only. On systemd/launchd hosts the native backup timer
    owns scheduled backups, so this defers to it and cannot double
    fire.

    Parameters
    ----------
    data_dir : str
        Base data directory to back up.
    now : datetime
        Current time; the cron is matched against its minute.
    settle : Callable[[], None], optional
        Drains the queue to empty before the backup, so the snapshot
        captures a settled store.
    """
    from memman.setup.scheduler import SCHEDULER_KIND_SERVE, detect_scheduler
    from memman.setup.scheduler import read_backup_state, write_backup_state
    try:
        if detect_scheduler() != SCHEDULER_KIND_SERVE:
            return
    except RuntimeError:
        return
    cron = config.get(config.BACKUP_CRON)
    if not cron:
        return
    from memman.backup.cron import cron_matches
    if not cron_matches(cron, now):
        return
    minute_key = now.strftime('%Y-%m-%dT%H:%M')
    if read_backup_state() == minute_key:
        return
    if settle is not None:
        settle()
    from memman.backup import run_backup

    # Notes:
    # - No `drain.lock` is taken: the snapshot is online, and the bundle
    #   includes `queue.db`, so pending writes are preserved without
    #   `settle`.
    # - `backup.state` is stamped only after a successful run, so a
    #   transient failure retries on the next iteration.
    # - `build_bundle` is local-disk-bound and cloud sync of the target
    #   is async, so the run stays inline.
    try:
        run_backup(data_dir)
        write_backup_state(minute_key)
    except Exception as exc:
        logger.warning('scheduler serve: backup failed: %s', exc)


@scheduler.command('serve')
@click.option('--interval', default=None, type=int,
              help=('Seconds between drain iterations.'
                    ' Falls back to MEMMAN_INTERVAL, then 60.'))
@click.option('--once', is_flag=True, default=False,
              help='Run a single drain pass and exit')
@click.pass_context
def scheduler_serve(ctx: click.Context, interval: int | None,
                    once: bool) -> None:
    """Run the drain loop continuously as a long-lived process.

    Used as PID 1 in containers and by hosts where systemd/launchd are
    not available (set MEMMAN_SCHEDULER_KIND=serve). On SIGTERM/SIGINT
    the current drain finishes (bounded by the per-iteration timeout)
    and the process exits 0.

    Resolution order for the iteration interval:
      1. `--interval` flag (when passed)
      2. `MEMMAN_INTERVAL` from the env file
      3. 60 (the documented default)
    """
    from memman import trace
    from memman.setup.scheduler import STATE_STOPPED, clear_serve_interval
    from memman.setup.scheduler import read_state, write_serve_interval

    if interval is None:
        raw = config.get(config.INTERVAL)
        if raw is None or raw.strip() == '':
            interval = 60
        else:
            try:
                interval = int(raw)
            except ValueError:
                raise click.ClickException(
                    f'MEMMAN_INTERVAL must be an integer, got {raw!r}')
    if interval < 0:
        raise click.ClickException('--interval must be >= 0')

    os.environ[config.WORKER] = '1'
    # The root callback configured logging before this flag was set, so
    # without a second pass `serve` runs with no worker file handler and
    # drops every stack it was supposed to keep.
    _configure_logging(
        ctx.obj['data_dir'], ctx.obj['verbose'], ctx.obj['debug'])

    def _handle_stop(signum: int, frame: object) -> None:
        logger.info(
            f'scheduler serve: caught signal {signum}, finishing drain')
        _request_stop()

    prior_term = signal.signal(signal.SIGTERM, _handle_stop)
    prior_int = signal.signal(signal.SIGINT, _handle_stop)
    try:
        write_serve_interval(interval)

        data_dir_val = ctx.obj['data_dir']

        trace.setup()
        trace.event(
            'scheduler_serve_start',
            pid=os.getpid(),
            hostname=socket.gethostname(),
            python=sys.version.split()[0],
            memman_version=memman.__version__,
            interval=interval,
            once=once)

        per_drain_timeout = max(10, interval - 10) if interval > 0 else 300

        def _settle_queue() -> None:
            """Drain the queue to empty before a backup (bounded).
            """
            for _ in range(50):
                drained = _drain_queue(
                    ctx, limit=100, timeout=per_drain_timeout,
                    stores_filter='', verbose=False)
                if not drained or drained.get('claimed', 0) == 0:
                    return

        while True:
            config.reset_file_cache()
            if read_state() == STATE_STOPPED:
                logger.info('scheduler serve: state=STOPPED, exiting')
                break
            result = _drain_queue(
                ctx, limit=100, timeout=per_drain_timeout,
                stores_filter='', verbose=False)
            _maybe_fire_backup(
                data_dir_val, datetime.now(), settle=_settle_queue)
            if _stop_requested() or once:
                break
            if interval > 0:
                slept = 0.0
                while slept < interval and not _stop_requested():
                    if read_state() == STATE_STOPPED:
                        break
                    time.sleep(min(1.0, interval - slept))
                    slept += 1.0
            elif result and result.get('claimed', 0) == 0:
                time.sleep(0.1)

        trace.event('scheduler_serve_stop', pid=os.getpid())
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_int)
        _clear_stop()
        try:
            clear_serve_interval()
        except OSError:
            pass


def _drain_queue(ctx: click.Context, limit: int, timeout: int,
                 stores_filter: str, verbose: bool) -> dict | None:
    """Claim and process queue rows until limit, timeout, or empty.

    Parameters
    ----------
    ctx : click.Context
        Carries `data_dir` and the active store selection.
    limit : int
        Maximum rows to claim in this drain.
    timeout : int
        Seconds to keep claiming before returning, whatever remains.
    stores_filter : str
        Comma-separated store names to drain; empty drains all.
    verbose : bool
        Echo a per-row line to stderr as each row completes.

    Returns
    -------
    dict or None
        `{claimed, processed, failed}`, so a caller can detect an
        empty drain. None when another drain already holds the lock.
    """
    from memman import trace
    from memman.llm import usage as llm_usage
    from memman.setup.scheduler import STATE_STOPPED, read_state

    data_dir_val = ctx.obj['data_dir']
    worker_pid = os.getpid()
    deadline = time.monotonic() + timeout
    store_list = [s.strip() for s in stores_filter.split(',') if s.strip()]

    trace.setup()
    trace.event(
        'scheduler_fired',
        pid=worker_pid,
        hostname=socket.gethostname(),
        python=sys.version.split()[0],
        memman_version=memman.__version__,
        env=config.enumerate_effective_config())

    try:
        lock_fd = acquire(data_dir_val)
    except DrainLockBusy:
        logger.info('drain: another drain is in progress, skipping')
        trace.event('drain_skipped_locked', data_dir=data_dir_val)
        _json_out({
            'processed': 0,
            'failed': 0,
            'remaining': {'pending': 0, 'claimed': 0,
                          'failed': 0, 'done': 0},
            'llm_usage': {},
            'skipped': 'another drain in progress',
            })
        return None

    stack = ExitStack()
    try:
        trace.event(
            'drain_start',
            data_dir=data_dir_val,
            queue_db_path=queue_db_path(data_dir_val),
            limit=limit,
            timeout=timeout,
            stores=store_list)

        conn = stack.enter_context(queue_db(data_dir_val))
        processed = 0
        failed = 0
        claimed = 0
        touched_stores: set[str] = set()
        store_contexts: dict[str, _StoreContext] = {}
        run_error: str | None = None

        last_hb = _LAST_HEARTBEAT_AT.get(data_dir_val, 0.0)
        record_run = (time.monotonic() - last_hb) >= HEARTBEAT_MIN_INTERVAL_SECONDS
        run_id = start_worker_run(conn, worker_pid) if record_run else None
        # Snapshot-and-delta, never reset: the ledger is process-wide
        # and row-level deltas below must not clobber the drain total.
        drain_usage_snap = llm_usage.snapshot()
    except Exception:
        stack.close()
        release(lock_fd)
        raise

    try:
        while processed + failed < limit:
            if _stop_requested() or read_state() == STATE_STOPPED:
                logger.info('drain: stop requested, exiting loop')
                trace.event('drain_stop_requested')
                break
            if time.monotonic() >= deadline:
                logger.info(f'enrich: timeout after {timeout}s')
                trace.event('drain_timeout', timeout=timeout)
                break
            row = claim(conn, worker_pid=worker_pid,
                        stores=store_list or None)
            if row is None:
                break
            claimed += 1

            trace.event(
                'queue_claim',
                row_id=row.id,
                store=row.store,
                attempts=row.attempts,
                content_len=len(row.content))

            store_ctx = store_contexts.get(row.store)
            if store_ctx is None:
                try:
                    store_ctx = stack.enter_context(
                        _StoreContext(row.store, data_dir_val))
                except Exception as exc:
                    mark_failed(
                        conn, row.id, f'{type(exc).__name__}: {exc}')
                    failed += 1
                    trace.event(
                        'queue_failed',
                        row_id=row.id,
                        store=row.store,
                        error_class=type(exc).__name__,
                        error_message=str(exc)[:500],
                        llm_usage={})
                    logger.exception(
                        f'enrich row {row.id} failed during store open')
                    continue
                store_contexts[row.store] = store_ctx
                if record_run:
                    store_ctx.begin_drain_run()

            row_usage_snap = llm_usage.snapshot()
            try:
                row_t0 = time.monotonic()
                _process_queue_row(row, store_ctx)
                row_elapsed_ms = int((time.monotonic() - row_t0) * 1000)
                mark_done(conn, row.id)
                processed += 1
                touched_stores.add(row.store)
                store_ctx.beat_drain_run()
                trace.event(
                    'queue_done',
                    row_id=row.id,
                    store=row.store,
                    elapsed_ms=row_elapsed_ms,
                    llm_usage=llm_usage.delta(
                        row_usage_snap, llm_usage.snapshot()))
                if verbose:
                    click.echo(
                        f'[enrich] done id={row.id} store={row.store}',
                        err=True)
            except Exception as exc:
                mark_failed(conn, row.id, f'{type(exc).__name__}: {exc}')
                failed += 1
                trace.event(
                    'queue_failed',
                    row_id=row.id,
                    store=row.store,
                    error_class=type(exc).__name__,
                    error_message=str(exc)[:500],
                    llm_usage=llm_usage.delta(
                        row_usage_snap, llm_usage.snapshot()))
                if isinstance(exc, EmbedCredentialError):
                    trace.event(
                        'embedder_credential_missing',
                        row_id=row.id,
                        store=row.store,
                        model=store_ctx.ec.model,
                        reason=str(exc)[:500])
                if verbose:
                    click.echo(
                        f'[enrich] fail id={row.id} store={row.store}'
                        f' err={exc}', err=True)
                logger.exception(f'enrich row {row.id} failed')
    except Exception as exc:
        run_error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            from memman.maintenance import run_maintenance
            run_maintenance(
                conn, touched_stores, store_contexts, deadline)
        except Exception:
            logger.exception('drain maintenance phase failed')
        import httpx
        from memman.llm import openrouter_models
        try:
            openrouter_models.refresh_model_state(data_dir_val, force=False)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning(f'model check could not read the catalogs: {exc}')
        if processed > 0:
            record_run = True
        if record_run:
            try:
                if run_id is None:
                    run_id = start_worker_run(conn, worker_pid)
                finish_worker_run(
                    conn, run_id, claimed, processed, failed,
                    error=run_error)
                _LAST_HEARTBEAT_AT[data_dir_val] = time.monotonic()
            except Exception:
                logger.exception('failed to stamp worker_runs finish row')
        remaining = queue_stats(conn)
        stack.close()
        release(lock_fd)

    drain_usage = llm_usage.delta(drain_usage_snap, llm_usage.snapshot())
    trace.event('llm_usage_summary', usage=drain_usage)
    trace.event(
        'drain_end',
        processed=processed,
        failed=failed,
        remaining=remaining)
    _json_out({
        'processed': processed,
        'failed': failed,
        'remaining': remaining,
        'llm_usage': drain_usage,
        })
    return {'claimed': claimed, 'processed': processed, 'failed': failed}


class _StoreContext:
    """Per-store drain-scope state hoisted out of the row loop.

    One context per store touched in a drain: the open store
    connection and the store-bound embed client. Reused across every
    row that targets the same store so that store opens and HTTP
    setup amortize.
    """

    def __init__(self, store_name: str, data_dir: str) -> None:
        """Open the store and bind its embed client.

        Parameters
        ----------
        store_name : str
            Store to open.
        data_dir : str
            Data directory holding the store.

        Raises
        ------
        StoreMissingError
            The store does not exist. Its backend key stays unwritten.
        EmbedFingerprintError
            The store holds data but has no embed fingerprint.
        """
        self.store_name = store_name
        self.data_dir = data_dir
        self.backend = factory.open_backend(store_name, data_dir)
        _ensure_store_backend_key(store_name, data_dir)
        fingerprint.seed_if_fresh(self.backend, get_client())
        stored = fingerprint.stored_fingerprint(self.backend)
        if stored is None:
            raise EmbedFingerprintError(
                f'store {store_name!r} has no embed fingerprint and'
                f' contains data; run `{fingerprint.swap_command(store_name)}`'
                ' to re-embed it.')
        self.ec = fingerprint.bound_embedder(self.backend)
        self._stored_fp = stored
        self._run_id: int | None = None

    def begin_drain_run(self) -> None:
        """Open a per-store drain run row (no-op on SQLite).
        """
        if self._run_id is not None:
            return
        try:
            self._run_id = self.backend.start_run()
        except Exception:
            logger.exception(
                f'start_run failed for store {self.store_name!r};'
                ' continuing without heartbeat')

    def beat_drain_run(self) -> None:
        """Advance the per-store drain heartbeat (no-op on SQLite).
        """
        if self._run_id is None:
            return
        try:
            self.backend.beat_run(self._run_id)
        except Exception:
            logger.exception(
                f'beat_run failed for store {self.store_name!r};'
                ' continuing without heartbeat')

    def assert_fingerprint_unchanged(self) -> None:
        """Check the stored fingerprint against the one captured at init.

        A swap that completes mid-drain would otherwise let the cached
        `ec` write vectors of the wrong dim, so a drain loop calls this
        before every embed call.

        Raises
        ------
        EmbedFingerprintError
            The stored fingerprint differs from the captured one.
        """
        current = fingerprint.stored_fingerprint(self.backend)
        if current != self._stored_fp:
            raise EmbedFingerprintError(
                f'store {self.store_name!r} fingerprint changed during'
                f' drain: was {self._stored_fp.model}:{self._stored_fp.dim},'
                f' now {current.model if current else None}:'
                f'{current.dim if current else None};'
                ' row released for retry.')

    def close(self) -> None:
        """Close the active Backend's underlying connection.
        """
        if self._run_id is not None:
            try:
                self.backend.finish_run(self._run_id)
            except Exception:
                logger.exception(
                    f'finish_run failed for store {self.store_name!r}')
            self._run_id = None
        try:
            self.backend.close()
        except Exception:
            logger.exception(
                f'failed closing backend for store {self.store_name!r}')

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self, exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        self.close()


def _process_queue_row(
        row: 'QueueRow',
        ctx: _StoreContext) -> None:
    """Run the full remember pipeline on a claimed queue row.

    Crash-recovery idempotency is enforced unconditionally via
    `row.queue_uuid`.

    Parameters
    ----------
    row : QueueRow
        Claimed queue row to write.
    ctx : _StoreContext
        Hoisted state for the row's store: the backend and the embed
        client.

    Raises
    ------
    EmbedFingerprintError
        The store's fingerprint changed since `ctx` was built.
    """
    from memman import trace

    ctx.assert_fingerprint_unchanged()

    backend = ctx.backend

    trace.event(
        'process_row',
        row_id=row.id,
        store=row.store,
        data_dir=ctx.data_dir)

    if backend.nodes.has_row_with_queue_uuid(row.queue_uuid):
        logger.info(
            f'queue row {row.id} already committed to store'
            f' {row.store!r}; skipping re-processing')
        trace.event(
            'process_row_skipped',
            row_id=row.id,
            reason='already_committed')
        return

    now = datetime.now(timezone.utc)
    replaced_id = row.replaced_id or ''
    redirected_from = ''
    if replaced_id:
        # Notes:
        # - An earlier queued replace may have replaced the target
        #   between enqueue and claim. Following the chain to its
        #   current head leaves the topic with one current row.
        # - A forgotten or missing head passes the original id
        #   through, and `_apply_plan` degrades to a named add.
        # - A branch follows only a successor it holds. Following one
        #   the parent wrote retires it with text that never saw it.
        old = backend.nodes.get_include_deleted(replaced_id)
        seen: set[str] = set()
        while (old is not None and old.replaced_by
                and old.id not in seen
                and (not isinstance(backend, OverlayBackend)
                     or backend.branch_holds(old.replaced_by))):
            seen.add(old.id)
            old = backend.nodes.get_include_deleted(old.replaced_by)
        if (old is not None and old.deleted_at is None
                and old.replaced_by is None
                and old.id != replaced_id):
            redirected_from = replaced_id
            replaced_id = old.id
    # The row takes the write's `queue_uuid` as its id, the id
    # `remember` and `replace` printed, and carries it as `queue_uuid`
    # too: run_remember stores this object directly, so omitting it
    # makes the idempotency check above a silent no-op.
    insight = Insight(
        id=row.queue_uuid, content=row.content,
        created_at=now, updated_at=now,
        queue_uuid=row.queue_uuid, author=row.author)

    from memman.pipeline.remember import run_remember
    result = run_remember(
        backend, insight,
        replaced_id=replaced_id,
        ec=ctx.ec)
    if redirected_from:
        result['redirected_from'] = redirected_from
    _json_out(result)


@claude_callable
@cli.command()
@click.argument('keyword', nargs=-1, required=True)
@click.option('--limit', default=None, type=int,
              help='Max results (default MEMMAN_RECALL_LIMIT, then 20)')
@click.option('--basic', is_flag=True, default=False, help='Simple SQL LIKE matching')
@click.pass_context
def recall(ctx: click.Context, keyword: tuple[str, ...], limit: int | None,
           basic: bool) -> None:
    """Print the insights matching a query, one line each, best first.

    Each line is `<id8> <score> <created_at> <author> | <text>`: the
    first 8 characters of the id, the score to two decimals, the UTC
    date, the author or `-`, then the summary or a content prefix.
    `--basic` prints the same line without a score. An empty page
    prints nothing.

    \b
    Parameters
    ----------
    keyword : tuple[str, ...]
        Query words, joined by single spaces.
    limit : int or None
        Maximum lines printed. None reads `MEMMAN_RECALL_LIMIT` from
        the env file, then falls back to 20.
    basic : bool
        SQL LIKE matching over content, newest first; computes no
        score.

    \b
    Raises
    ------
    click.ClickException
        `MEMMAN_RECALL_LIMIT` is set to a non-integer.

    \b
    Notes
    -----
    - A score compares only against the other rows of its own page,
      never across queries.
    - Recall writes nothing to the store but its oplog row.

    \b
    Examples
    --------
    memman recall "retry cap"
    memman recall "retry cap" --limit 5
    memman recall "retry" --basic
    """  # noqa: D301, D410, D411
    from memman.search.recall import run_recall
    if limit is None:
        raw = config.get(config.RECALL_LIMIT)
        if raw is None or raw.strip() == '':
            limit = 20
        else:
            try:
                limit = int(raw)
            except ValueError:
                raise click.ClickException(
                    f'MEMMAN_RECALL_LIMIT must be an integer, got {raw!r}')
    keyword_str = ' '.join(keyword)
    store_name = _resolve_store_name(ctx.obj['data_dir'], ctx.obj['store'])
    per_store_rerank = config.get_store_rerank_enabled(store_name)
    rerank = (per_store_rerank if per_store_rerank is not None
              else config.get_bool(config.RERANK_ENABLED, default=True))
    with _active_backend(ctx) as backend:
        if basic:
            results = backend.nodes.query(keyword=keyword_str, limit=limit)
            try:
                with backend.transaction():
                    backend.oplog.log(
                        operation='recall:basic', insight_id='',
                        detail=f'q={keyword_str} hits={len(results)}')
            except sqlite3.OperationalError as exc:
                logger.debug(
                    'recall_bookkeep_skipped basic q=%r: %s',
                    keyword_str, exc)
            for ins in results:
                click.echo(insight_to_recall_line(ins, None))
            return

        ec = fingerprint.bound_embedder(backend)
        query_vec = None
        try:
            query_vec = ec.embed(keyword_str)
        except Exception as exc:
            logger.warning(
                'recall query embed failed (%s): %s;'
                ' degrading to keyword path',
                type(exc).__name__, exc)

        resp = run_recall(
            backend, keyword_str, query_vec, limit, rerank=rerank)

        hits = [{'id': r['insight'].id[:8],
                 'score': round(r['score'], 3)}
                for r in resp['results']]
        try:
            with backend.transaction():
                # `limit` is the REQUESTED page size, not len(hits):
                # the returned count cannot distinguish a thin page
                # from a small ask.
                backend.oplog.log(
                    operation='recall-detail', insight_id='',
                    detail=json.dumps({'q': keyword_str[:80],
                                       'limit': limit,
                                       'hits': hits}))
        except sqlite3.OperationalError as exc:
            logger.debug(
                'recall_bookkeep_skipped detail q=%r: %s',
                keyword_str, exc)

        for r in resp['results']:
            click.echo(insight_to_recall_line(r['insight'], r['score']))


def _resolve_queued_or_stored(
        backend: 'Backend', data_dir: str, store: str,
        id: str) -> tuple[str | None, 'Insight | None']:
    """Resolve an id or prefix across a store's queued writes and rows.

    Parameters
    ----------
    backend : Backend
        The open store `store` names.
    data_dir : str
        Data dir whose queue.db holds the queued writes.
    store : str
        Only writes queued for this store match.
    id : str
        A full id or any prefix of one.

    Returns
    -------
    tuple
        `(queued, stored)`: the full id of the pending write that
        matches or None, and the stored row that matches, read
        through `get_include_deleted`, or None.

    Raises
    ------
    click.ClickException
        When the prefix matches two queued writes, two stored rows, or
        a queued write and a different stored row.
    """
    # A write leaves the queue only after it lands, so reading the
    # queue first finds a write that lands mid-command in one place or
    # the other.
    with queue_db(data_dir) as conn:
        try:
            queued = find_pending(conn, store, id)
        except ValueError as exc:
            raise click.ClickException(str(exc))
    try:
        stored = backend.nodes.get_include_deleted(
            backend.nodes.resolve_id(id))
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if queued is not None and stored is not None and stored.id != queued:
        raise click.ClickException(
            f'prefix {id!r} matches a queued write and a stored row')
    return queued, stored


def _forget_insight(backend: 'Backend', id: str) -> None:
    """Soft-delete `id` and write a forget oplog row carrying `before`.

    A replaced row may be forgotten. A missing or already forgotten
    row is refused with the reason, and so is a current row that
    replaced another, with the `replace` that corrects it.
    """
    with backend.transaction():
        before_ins = backend.nodes.get_include_deleted(id)
        if before_ins is None:
            raise click.ClickException(f'insight {id} not found')
        if before_ins.deleted_at is not None:
            raise click.ClickException(f'insight {id} was forgotten')
        # An agent forgetting a wrong correction expects the row it
        # replaced to come back, but that row stays retired, so the
        # topic would leave recall for good.
        if (before_ins.replaced_by is None
                and any(p.deleted_at is None
                        for p in backend.nodes.predecessors(id))):
            raise click.ClickException(
                f'insight {id} replaced an earlier row, and forgetting'
                ' it does not bring that row back; correct it with'
                f' replace {id} "<new text>"')
        if not backend.nodes.soft_delete(id):
            raise click.ClickException(f'insight {id} not found')
        backend.oplog.log(
            operation='forget', insight_id=id, detail='',
            before=insight_to_delta_dict(before_ins))


@claude_callable
@cli.command()
@click.argument('id')
@click.pass_context
def forget(ctx: click.Context, id: str) -> None:
    """Soft-delete an insight. Rejected when the scheduler is stopped.

    \b
    Parameters
    ----------
    id : str
        A full insight id or any unambiguous prefix of one. The id of
        a write still queued is refused as still queued.

    \b
    Notes
    -----
    - A current row that replaced another is refused: the row it
      replaced stays retired, so forgetting it drops the topic from
      recall. `replace` on its id corrects it instead.
    - A replaced row may be forgotten. A missing or already
      forgotten row is refused.

    \b
    Examples
    --------
    memman forget 16c6c667
    """  # noqa: D301, D410, D411
    _require_started('write')

    name = _resolve_store_name(ctx.obj['data_dir'], ctx.obj['store'])
    with _active_backend(ctx) as backend:
        queued, ins = _resolve_queued_or_stored(
            backend, ctx.obj['data_dir'], name, id)
        if ins is None and queued is not None:
            raise click.ClickException(
                f'insight {queued} is still queued; it lands on'
                ' the next drain')
        if ins is not None:
            id = ins.id
        _forget_insight(backend, id)
        _json_out({
            'id': id,
            'status': 'deleted',
            'message': 'Insight soft-deleted successfully',
            })


@claude_callable
@cli.command()
@click.argument('id')
@click.argument('content', nargs=-1, required=True)
@click.pass_context
def replace(ctx: click.Context, id: str, content: tuple[str, ...]) -> None:
    """Correct a stale insight: store new text in its place via the queue.

    \b
    Parameters
    ----------
    id : str
        A full insight id or any unambiguous prefix of one, current or
        still queued, such as the `id` an earlier `remember` printed.
    content : str
        The corrected text, stored as one row exactly as typed.

    \b
    Notes
    -----
    - The target is retired: it keeps its content behind
      `replaced_by` and leaves every recall and listing.
      `insights show <id> --history` reads the chain back.
    - A forgotten or already replaced id is refused, the latter
      naming its successor. So is an id with a replace still queued:
      the refusal quotes that replace, the one to replace instead.
    - The drain holds a replace while its queued target, or an
      earlier replace in the same store, is pending. A target that
      fails releases it, and it lands as a plain add.
    - Enrichment still runs and rebuilds the summary.

    \b
    Examples
    --------
    memman replace 16c6c667 "the retry cap is five"
    """  # noqa: D301, D410, D411
    _require_started('write')

    content_str = ' '.join(content)
    author = config.resolve_author()
    refusal = _content_refusal_message(content_str, verb='replace')
    if refusal:
        raise click.ClickException(refusal)

    from memman.search.quality import check_content_quality
    quality_warnings = check_content_quality(content_str)

    data_dir_val = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir_val, ctx.obj['store'])

    with _active_backend(ctx) as backend:
        queued, old = _resolve_queued_or_stored(
            backend, data_dir_val, name, id)
        if (queued is None and old is not None and old.replaced_by
                and old.deleted_at is None
                and isinstance(backend, OverlayBackend)
                and not backend.branch_holds(old.id)):
            raise click.ClickException(backend.nodes.retired_message(
                old.id, old.replaced_by, verb='replace'))
    if queued is not None:
        id = queued
    elif old is None:
        raise click.ClickException(f'insight {id} not found')
    elif old.deleted_at is not None:
        raise click.ClickException(f'insight {old.id} was forgotten')
    elif old.replaced_by:
        raise click.ClickException(
            f'insight {old.id} was replaced by {old.replaced_by};'
            f' replace {old.replaced_by}, or run'
            f' insights show {old.id} --history')
    else:
        id = old.id

    with queue_db(data_dir_val) as conn:
        # The drain would chain this write behind the pending one and
        # retire it, so the first correction is lost unless this text
        # restates it.
        pending = find_pending_replace(conn, name, id)
        if pending is not None:
            pending_id, pending_text = pending
            raise click.ClickException(
                f'insight {id} already has a replace pending as'
                f' {pending_id}: "{pending_text}"; replace {pending_id}'
                ' with text that keeps it and adds yours')
        row_id, queue_uuid = _enqueue_into_existing_store(
            conn, data_dir_val, name, content_str,
            replaced_id=id, author=author)
    _json_out({
        'action': 'queued',
        'id': queue_uuid,
        'queue_id': row_id,
        'store': name,
        'replaced_id': id,
        'quality_warnings': quality_warnings,
        })


@queue.command('list')
@click.option('--limit', default=50, type=int, help='Max results')
@click.pass_context
def queue_list(ctx: click.Context, limit: int) -> None:
    """List recent queue rows.
    """
    with queue_db(ctx.obj['data_dir']) as conn:
        _json_out({
            'stats': queue_stats(conn),
            'rows': list_rows(conn, limit=limit),
            })


@queue.command('failed')
@click.option('--limit', default=50, type=int, help='Max results')
@click.pass_context
def queue_failed(ctx: click.Context, limit: int) -> None:
    """List failed queue rows.
    """
    with queue_db(ctx.obj['data_dir']) as conn:
        _json_out({
            'stats': queue_stats(conn),
            'rows': list_rows(conn, status=STATUS_FAILED, limit=limit),
            })


@queue.command('show')
@click.argument('row_id', type=int)
@click.pass_context
def queue_show(ctx: click.Context, row_id: int) -> None:
    """Print the full content of a queue row.
    """
    with queue_db(ctx.obj['data_dir']) as conn:
        row = get_row(conn, row_id)
        if row is None:
            raise click.ClickException(f'queue row {row_id} not found')
        _json_out(row)


@queue.command('retry')
@click.argument('row_id', type=int)
@click.pass_context
def queue_retry(ctx: click.Context, row_id: int) -> None:
    """Re-queue a failed row, with its attempt count reset.

    Parameters
    ----------
    row_id : int
        Queue row id, as `scheduler queue failed` lists it. A row not
        in status `failed` is refused.

    Examples
    --------
    memman scheduler queue retry 42
    """
    with queue_db(ctx.obj['data_dir']) as conn:
        if not retry_row(conn, row_id):
            raise click.ClickException(
                f'queue row {row_id} not found or not in failed state')
        _json_out({'action': 'requeued', 'queue_id': row_id})


@queue.command('purge')
@click.option('--done', is_flag=True, default=False,
              help='Delete all rows with status=done')
@click.pass_context
def queue_purge(ctx: click.Context, done: bool) -> None:
    """Remove completed queue rows.

    Parameters
    ----------
    done : bool
        Delete every row in status `done`. Required as confirmation.

    Examples
    --------
    memman scheduler queue purge --done
    """
    if not done:
        raise click.ClickException('pass --done to confirm deletion')
    with queue_db(ctx.obj['data_dir']) as conn:
        _json_out({'deleted': purge_done(conn)})


@scheduler.command('status')
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable output (default: JSON)')
@click.pass_context
def scheduler_status(ctx: click.Context, text_output: bool) -> None:
    """Show scheduler install state, interval, next run, log paths,
    and the most recent worker-drain summary from worker_runs.
    """
    from memman.setup.scheduler import status
    result = status()
    logs_dir = pathlib.Path.home() / '.memman' / 'logs'
    log_path = logs_dir / 'enrich.log'
    err_path = logs_dir / 'enrich.err'
    # The rotated worker log follows --data-dir while the two unit
    # redirects beside it never do, so it cannot be built off logs_dir.
    stack_path = pathlib.Path(ctx.obj['data_dir']) / 'logs' / 'memman.log'
    result['log_path'] = str(log_path)
    result['err_path'] = str(err_path)
    result['stack_path'] = str(stack_path)
    for key, path in (('log_mtime', log_path), ('err_mtime', err_path),
                      ('stack_mtime', stack_path)):
        try:
            result[key] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
        except OSError:
            result[key] = None

    result['last_run'] = None
    try:
        with queue_db(ctx.obj['data_dir']) as conn:
            result['last_run'] = last_worker_run(conn)
    except Exception as exc:
        logger.debug(f'worker_runs lookup failed: {exc}')

    _scheduler_emit(result, text_output)


@scheduler.command('start')
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable output (default: JSON)')
def scheduler_start(text_output: bool) -> None:
    """Start the scheduler. Worker drains; writes are accepted.

    Idempotent.
    """
    from memman.setup.scheduler import start
    try:
        result = start()
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _scheduler_emit(result, text_output)


@scheduler.command('stop')
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable output (default: JSON)')
def scheduler_stop(text_output: bool) -> None:
    """Stop the scheduler. Trigger files stay; memman becomes recall-only.

    Writes (`remember`/`replace`/`forget`) reject until
    `scheduler start` re-arms the worker; `enrich` runs only while
    the scheduler is stopped. Use
    `memman uninstall` to remove trigger files entirely.
    """
    from memman.setup.scheduler import stop
    try:
        result = stop()
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _scheduler_emit(result, text_output)


def _scheduler_emit(result: dict, text_output: bool) -> None:
    """Render a scheduler-command result dict.

    JSON by default for script consumers; `--text` renders flat
    key:value lines and an indented `actions` list when present.
    """
    if not text_output:
        _json_out(result)
        return
    actions = result.get('actions') or []
    for key, value in result.items():
        if key == 'actions':
            continue
        click.echo(f'{key}: {value}')
    if actions:
        click.echo('actions:')
        for action in actions:
            click.echo(f'  - {action}')


@scheduler.command('install')
@click.option('--interval', type=int, default=None,
              help=('Polling interval in seconds (min 60 for'
                    ' systemd/launchd). Default: 60. For sub-minute'
                    ' intervals, use serve mode instead'
                    ' (`memman scheduler serve --interval N`).'))
@click.option('--endpoint', type=str, default=None,
              help='Endpoint URL to seed into the env file.')
@click.pass_context
def scheduler_install(ctx: click.Context, interval: int | None,
                      endpoint: str | None) -> None:
    """Install the scheduler unit only (no agent integration).

    Reads MEMMAN_API_KEY (or, on OpenRouter, OPENROUTER_API_KEY) from
    env and writes it to
    ~/.memman/env (mode 600), then installs the systemd timer or
    launchd plist that runs the worker every interval. For full
    agent-integration setup (hooks, skill, scheduler), use
    `memman install`.
    """
    from memman.setup.claude import _reject_flag_file_conflicts
    from memman.setup.scheduler import DEFAULT_INTERVAL_SECONDS
    from memman.setup.scheduler import _write_env_keys, install

    data_dir = ctx.obj['data_dir']
    _reject_flag_file_conflicts(
        data_dir=data_dir, backend=None, pg_dsn=None, endpoint=endpoint)
    if endpoint:
        _write_env_keys({config.ENDPOINT: endpoint}, data_dir=data_dir)

    try:
        knobs = config.collect_install_knobs(data_dir)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    seconds = interval if interval is not None else DEFAULT_INTERVAL_SECONDS
    if seconds < 60:
        raise click.ClickException(
            '--interval must be at least 60 seconds for systemd/launchd.'
            ' For sub-minute intervals, set MEMMAN_SCHEDULER_KIND=serve'
            ' and run `memman scheduler serve --interval N` instead.')
    try:
        result = install(data_dir, knobs, seconds)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.command('uninstall')
@click.pass_context
def scheduler_uninstall(ctx: click.Context) -> None:
    """Remove the scheduler unit only (leaves agent integration intact).

    Clears scheduler state files and removes the systemd timer/service or
    launchd plist. `memman uninstall` does this AND removes hooks/skill
    integration.
    """
    from memman.setup.scheduler import uninstall
    _json_out(uninstall(data_dir=ctx.obj['data_dir']))


@scheduler.command('interval')
@click.option('--seconds', type=int, default=None,
              help=('New interval in seconds. Omit to show current.'
                    ' min 60 for systemd/launchd; 0 (continuous) or any'
                    ' non-negative value allowed for serve mode.'))
@click.pass_context
def scheduler_interval(ctx: click.Context, seconds: int | None) -> None:
    """Show or set the scheduler interval.
    """
    from memman.setup.scheduler import change_interval, status
    if seconds is None:
        current = status()
        _json_out({
            'platform': current['platform'],
            'interval_seconds': current['interval_seconds'],
            'installed': current['installed'],
            })
        return
    try:
        result = change_interval(ctx.obj['data_dir'], seconds)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.command('trigger')
def scheduler_trigger() -> None:
    """Dispatch a drain now. The command does not wait for it to finish.

    Rejected when the scheduler is stopped. systemd uses
    `systemctl --user start --no-block` and launchd uses `launchctl
    start`, so a `dispatched` response means the run was queued, not
    that it has started or finished. Read `memman log worker` for the
    outcome.
    """
    _require_started('trigger drain')
    from memman.setup.scheduler import trigger
    try:
        result = trigger()
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.group('debug', no_args_is_help=True)
def scheduler_debug() -> None:
    """Toggle persistent JSONL trace state for scheduler-fired runs.

    Writes ~/.memman/debug.state which `trace.is_enabled()` reads as a
    fallback when MEMMAN_DEBUG is unset. Affects future scheduler-fired
    drains and any CLI invocation in a shell that does not export
    MEMMAN_DEBUG. Trace logs land at ~/.memman/logs/debug.log (mode
    600) and include raw LLM request/response bodies - including
    memory content. Turn off when done.
    """


@scheduler_debug.command('on')
def scheduler_debug_on() -> None:
    """Enable persistent debug traces.
    """
    from memman.setup.scheduler import set_debug
    actions = set_debug(True)
    logs_dir = pathlib.Path.home() / '.memman' / 'logs'
    click.echo(
        '[memman] debug traces ENABLED -- raw LLM request/response bodies'
        f' (including memory content) will be written to {logs_dir}/debug.log'
        ' (mode 600). Turn off with: memman scheduler debug off',
        err=True)
    _json_out({'debug': True, 'actions': actions})


@scheduler_debug.command('off')
def scheduler_debug_off() -> None:
    """Disable persistent debug traces; existing debug.log files are kept.
    """
    from memman.setup.scheduler import set_debug
    actions = set_debug(False)
    _json_out({'debug': False, 'actions': actions})


@scheduler_debug.command('status')
def scheduler_debug_status() -> None:
    """Show whether persistent debug traces are enabled.
    """
    from memman.setup.scheduler import get_debug
    logs_dir = pathlib.Path.home() / '.memman' / 'logs'
    debug_log = logs_dir / 'debug.log'
    _json_out({
        'debug': get_debug(),
        'debug_log': str(debug_log),
        'debug_log_exists': debug_log.is_file(),
        })


@cli.group(invoke_without_command=True)
@click.pass_context
def store(ctx: click.Context) -> None:
    """Manage named memory stores.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(store_list)


@store.command('list')
@click.pass_context
def store_list(ctx: click.Context) -> None:
    """List all stores as JSON (stores[], active, branches{name: info}).
    """
    data_dir = ctx.obj['data_dir']
    stores = list_stores(data_dir)
    active = _resolve_store_name(data_dir, ctx.obj['store']) if stores else None
    branches = {}
    for name in list_local_store_dirs(data_dir):
        try:
            info = branch_mod.read_branch_info(name, data_dir)
        except BackendError as exc:
            logger.debug(f'store list skips branch info for {name!r}: {exc}')
            continue
        if info is not None:
            branches[name] = info
    _json_out({'stores': stores, 'active': active, 'branches': branches})


@store.command('create')
@click.argument('name')
@click.pass_context
def store_create(ctx: click.Context, name: str) -> None:
    """Create a new store.
    """
    data_dir = ctx.obj['data_dir']
    if not valid_store_name(name):
        raise click.ClickException(
            f'invalid store name {name!r}')
    if '__' in name:
        raise click.ClickException(
            f'invalid store name {name!r}: `__` is reserved for branches')
    if name in factory.list_stores(data_dir):
        raise click.ClickException(
            f'store "{name}" already exists')
    from memman.session import active_store
    with active_store(data_dir=data_dir, store=name, create=True) as backend:
        path = backend.path
    _ensure_store_backend_key(name, data_dir)
    _json_out({'action': 'created', 'store': name, 'path': path})


@store.command('use')
@click.argument('name')
@click.pass_context
def store_use(ctx: click.Context, name: str) -> None:
    """Switch the active store.
    """
    data_dir = ctx.obj['data_dir']
    if not valid_store_name(name):
        raise click.ClickException(f'invalid store name {name!r}')
    if name not in factory.list_stores(data_dir):
        raise click.ClickException(str(StoreMissingError(name)))
    if branch_mod.read_branch_info(name, data_dir) is not None:
        raise click.ClickException(
            f'store {name!r} is a branch, and the active file routes every'
            f' session on this host; pass --store {name} to each memory'
            ' verb instead')
    write_active(data_dir, name)
    _json_out({'action': 'set', 'store': name})


@store.command('remove')
@click.argument('name')
@click.option('--yes', is_flag=True, default=False,
              help='Skip confirmation prompt (for scripted use).')
@click.pass_context
def store_remove(ctx: click.Context, name: str, yes: bool) -> None:
    """Remove a store (prompts unless --yes).

    Also drops the store's per-store env keys, so a removed store
    leaves no backend selection or Postgres DSN (password included)
    behind in the env file.

    \b
    Parameters
    ----------
    name : str
        Store to remove. Refused when active, when a branch (`memman
        store drop` removes one), or while it holds any
        `branch_token:<branch>` key, a branch on another host included.
        A SQLite store whose file cannot be read is removed unchecked.
    yes : bool
        Skip the confirmation prompt.

    \b
    Notes
    -----
    - A stale key, left by a drop that could not reach the store, is
      deleted by hand from the store's meta table.

    \b
    Examples
    --------
    memman store remove scratch --yes
    """  # noqa: D301, D410, D411
    data_dir = ctx.obj['data_dir']
    if name not in factory.list_stores(data_dir):
        raise click.ClickException(str(StoreMissingError(name)))
    active = read_active(data_dir)
    if name == active:
        raise click.ClickException(
            f"cannot remove the active store \"{name}\""
            f" (switch first with 'memman store use <other>')")
    prefix = f'{branch_mod.BRANCH_TOKEN}:'
    try:
        info = branch_mod.read_branch_info(name, data_dir)
        token_keys = []
        if info is None:
            with factory.open_backend(
                    name, data_dir, read_only=True) as backend:
                token_keys = sorted(
                    key for key in backend.meta.keys()  # noqa: SIM118
                    if key.startswith(prefix))
    except (ConfigError, BackendError, sqlite3.Error) as exc:
        if resolve_store_backend(name, data_dir) != 'sqlite':
            raise click.ClickException(
                f'cannot check store {name!r} for branch tokens: {exc}'
                ) from exc
        # A local file that cannot be read holds nothing to protect, and
        # no other verb can remove it.
        logger.debug(f'store remove of unreadable {name!r}: {exc}')
        info, token_keys = None, []
    if info is not None:
        raise click.ClickException(
            f'store {name!r} is a branch of {info["parent"]!r}; run memman'
            f' store merge {name} to keep its rows, or memman store drop'
            f' {name} to throw them away')
    if token_keys:
        listing = []
        for key in token_keys:
            holder = branch_mod.read_branch_info(
                key.removeprefix(prefix), data_dir)
            local = holder is not None and holder['parent'] == name
            listing.append(key if local else f'{key} (no local branch)')
        raise click.ClickException(
            f'store {name!r} holds branch tokens: {", ".join(listing)}.'
            ' Merge or drop each branch first. A branch on another host may'
            ' hold a key with no local branch; once no host holds it,'
            f' delete the key from the meta table of {name!r} by hand')
    if not yes:
        click.confirm(
            f'Drop store "{name}" (and all of its data)?',
            abort=True)
    # Notes:
    # - A backend refusing the drop (an unreachable Postgres, or a
    #   name it will not accept as an identifier) is an operator
    #   failure.
    # - Raising leaves the env keys in place on purpose. The store
    #   still exists, and dropping its routing would send the next
    #   read to the default backend instead of the one holding the
    #   data.
    try:
        factory.drop_store(name, data_dir)
    except BackendError as exc:
        raise click.ClickException(
            f'could not remove store {name!r}: {exc}')
    from memman.setup.scheduler import _write_env_keys_with_flock
    per_store_keys = {
        f'{prefix}{name}' for prefix, _ in config.PER_STORE_KEY_SPECS}
    stale = per_store_keys & set(
        config.parse_env_file(config.env_file_path(data_dir)))
    if stale:
        _write_env_keys_with_flock({}, removes=stale, data_dir=data_dir)
    _json_out({
        'action': 'removed',
        'store': name,
        'env_keys_removed': sorted(stale),
        })


@claude_callable(store_option=False)
@store.command('branch')
@click.argument('parent')
@click.argument('label')
@click.pass_context
def store_branch(ctx: click.Context, parent: str, label: str) -> None:
    """Start a branch: an empty local store layered over PARENT.

    \b
    Parameters
    ----------
    parent : str
        Store to branch, on any backend. It gains only the branch's
        token.
    label : str
        Name part for the branch, without `__`. The branch is named
        `<parent>__<label>_<4 hex>`.

    \b
    Notes
    -----
    - Only on the user's request. The JSON reply carries
      `instruction`, the line to paste into the notes the thread's next
      session reads. With no such notes, ask the user where it goes.
    - Recall on the branch ranks its own rows with PARENT's live rows.
      A replace or forget of a PARENT row acts on a copy in the branch.
    - Ends with `memman store merge` or `memman store drop`.

    \b
    Examples
    --------
    memman store branch memman rearch
    """  # noqa: D301, D410, D411
    _json_out(branch_mod.create_branch(ctx.obj['data_dir'], parent, label))


@claude_callable(store_option=False)
@store.command('merge')
@click.argument('branch')
@click.pass_context
def store_merge(ctx: click.Context, branch: str) -> None:
    """Keep a branch: replay it into its parent, then delete it.

    \b
    Parameters
    ----------
    branch : str
        A store made by `memman store branch`. Any other store is
        refused.

    \b
    Notes
    -----
    - Only on the user's request. The parent is the branch's own
      `branch_parent`, never the active store or `--store`.
    - Rows written in the branch are copied with their dates, summaries
      and embeddings. A replace or forget the branch made on a parent
      row is repeated in the parent, all in one parent transaction.
    - Each entry of `conflicts` is a parent row the branch and the
      parent retired differently. The parent keeps its own state, and
      `parent_head` names the parent's current row for it. Settle each
      with replace or forget in the parent.
    - Refused while the branch has queued writes, while either store is
      mid embed swap or re-embed, and when the parent does not hold the
      branch's token. A run that stops part way says to re-run, which
      finishes it.

    \b
    Examples
    --------
    memman store merge memman__rearch_7f3a
    """  # noqa: D301, D410, D411
    _json_out(branch_mod.merge_branch(ctx.obj['data_dir'], branch))


@claude_callable(store_option=False)
@store.command('drop')
@click.argument('branch')
@click.pass_context
def store_drop(ctx: click.Context, branch: str) -> None:
    """Throw a branch away, listing the rows written in it.

    \b
    Parameters
    ----------
    branch : str
        A store made by `memman store branch`. Any other store is
        refused.

    \b
    Notes
    -----
    - Only on the user's request. Nothing reaches the parent.
    - `dropped` holds `{id, content, replaces}` for each current branch
      row. `replaces` names the parent row the claim corrects. Re-save
      each claim unrelated to the thread with `memman remember --store
      <parent>`, or `memman replace --store <parent> <replaces>` where
      `replaces` is set, then a closing row saying why the thread was
      dropped.
    - Refused while the branch has queued writes, and after a merge that
      stopped part way, which only a re-run of merge ends.

    \b
    Examples
    --------
    memman store drop memman__rearch_7f3a
    """  # noqa: D301, D410, D411
    _json_out(branch_mod.drop_branch(ctx.obj['data_dir'], branch))


@cli.group(invoke_without_command=True)
@click.pass_context
def backup(ctx: click.Context) -> None:
    """External, scheduled backups of the whole store layout.

    Backups write only to a user-specified external directory, never
    into ~/.memman/ (that dir is per-host and disposable). No
    subcommand is claude-callable: `run` writes an external filesystem
    and `restore` is destructive.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(backup_status)


@backup.command('run')
@click.argument('target', required=False)
@click.pass_context
def backup_run(ctx: click.Context, target: str | None) -> None:
    """Build one backup bundle now (TARGET or MEMMAN_BACKUP_TARGET).
    """
    data_dir = ctx.obj['data_dir']
    target = target or config.get(config.BACKUP_TARGET)
    if not target:
        raise click.ClickException(
            'no target; pass TARGET or set MEMMAN_BACKUP_TARGET'
            " via 'memman backup schedule'")
    from memman.backup import run_backup
    _json_out({'action': 'backed_up', **run_backup(data_dir, target)})


@backup.command('schedule')
@click.argument('cron')
@click.argument('target')
@click.option('--keep', type=int, default=7,
              help='Number of bundles to retain (default 7).')
@click.pass_context
def backup_schedule(ctx: click.Context, cron: str, target: str,
                    keep: int) -> None:
    """Install a scheduled backup: CRON expression writing to TARGET dir.

    CRON is a 5-field expression (e.g. '0 3 * * *' for 03:00 daily).
    TARGET is created if it does not exist. The cron string is
    translated to the host's native scheduler at install time.
    """
    data_dir = ctx.obj['data_dir']
    from memman.backup.cron import cron_to_oncalendar
    try:
        cron_to_oncalendar(cron)
    except ValueError as exc:
        raise click.ClickException(f'invalid cron expression: {exc}')
    fields = cron.split()
    if len(fields) == 5 and fields[2] != '*' and fields[4] != '*':
        from memman.setup.scheduler import detect_scheduler
        try:
            systemd_host = detect_scheduler() == 'systemd'
        except RuntimeError:
            systemd_host = False
        if systemd_host:
            click.echo(
                'Warning: cron restricts BOTH day-of-month and day-of-week.'
                ' systemd OnCalendar evaluates these as AND (cron uses OR),'
                ' so the backup fires only when both match.', err=True)
    target_path = os.path.expanduser(target)
    os.makedirs(target_path, exist_ok=True)
    from memman.setup.scheduler import _write_env_keys, install_backup
    _write_env_keys(
        {config.BACKUP_CRON: cron,
         config.BACKUP_TARGET: target_path,
         config.BACKUP_KEEP: str(keep)},
        data_dir=data_dir)
    result = install_backup(data_dir, cron)
    _json_out({
        'action': 'scheduled', 'cron': cron,
        'target': target_path, 'keep': keep, **result})


@backup.command('unschedule')
def backup_unschedule() -> None:
    """Remove the scheduled backup trigger (keeps the env config).
    """
    from memman.setup.scheduler import uninstall_backup
    _json_out({'action': 'unscheduled', **uninstall_backup()})


@backup.command('list')
@click.argument('target', required=False)
def backup_list(target: str | None) -> None:
    """List bundles at TARGET (or MEMMAN_BACKUP_TARGET) from sidecars.
    """
    target = target or config.get(config.BACKUP_TARGET)
    if not target:
        raise click.ClickException(
            'no target; pass TARGET or set MEMMAN_BACKUP_TARGET')
    target_path = pathlib.Path(os.path.expanduser(target))
    backups: list[dict] = []
    if target_path.is_dir():
        for sidecar in sorted(
                target_path.glob('memman-backup-*.tar.gz.manifest.json')):
            bundle = sidecar.with_name(
                sidecar.name[:-len('.manifest.json')])
            try:
                manifest = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
            backups.append({
                'bundle': str(bundle),
                'created_at_utc': manifest.get('created_at_utc'),
                'host': manifest.get('host'),
                'stores': [
                    s.get('name') for s in manifest.get('stores', [])],
                'size_bytes': (
                    bundle.stat().st_size if bundle.exists() else None),
                })
    _json_out({'target': str(target_path), 'backups': backups})


@backup.command('status')
def backup_status() -> None:
    """Report backup config, schedule, last fire, and latest bundle.
    """
    from memman.setup import scheduler as sched

    cron = config.get(config.BACKUP_CRON)
    target = config.get(config.BACKUP_TARGET)
    keep = config.get(config.BACKUP_KEEP)
    out: dict = {
        'cron': cron,
        'target': target,
        'keep': int(keep) if keep and keep.isdigit() else None,
        'last_fired': sched.read_backup_state(),
        'scheduler': None,
        'installed': False,
        'next_run': None,
        'latest_bundle': None,
        }
    try:
        kind = sched.detect_scheduler()
    except RuntimeError:
        kind = None
    out['scheduler'] = kind
    if kind == 'systemd':
        timer = (sched._systemd_unit_dir()
                 / sched.SYSTEMD_BACKUP_TIMER_NAME)
        out['installed'] = timer.exists()
        if out['installed']:
            try:
                shown = subprocess.run(
                    ['systemctl', '--user', 'show',
                     '--property=NextElapseUSecRealtime', '--value',
                     sched.SYSTEMD_BACKUP_TIMER_NAME],
                    capture_output=True, text=True,
                    check=False, timeout=5)
                out['next_run'] = shown.stdout.strip() or None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    elif kind == 'launchd':
        plist = (sched._launchd_agent_dir()
                 / f'{sched.LAUNCHD_BACKUP_LABEL}.plist')
        out['installed'] = plist.exists()
    if target:
        target_path = pathlib.Path(os.path.expanduser(target))
        if target_path.is_dir():
            bundles = sorted(
                target_path.glob('memman-backup-*.tar.gz'))
            out['latest_bundle'] = str(bundles[-1]) if bundles else None
    _json_out(out)


@backup.command('restore')
@click.argument('bundle')
@click.option('--yes', is_flag=True, default=False,
              help='Skip the overwrite confirmation (for scripted use).')
@click.pass_context
def backup_restore(ctx: click.Context, bundle: str, yes: bool) -> None:
    """Restore stores + non-secret config from BUNDLE into the data dir.

    Overwrites local stores. Postgres stores need their DSN configured
    on this host (the DSN is a secret and is not in the bundle); a
    store with no DSN is skipped and reported.
    """
    data_dir = ctx.obj['data_dir']
    from memman.backup import restore

    needs_pg = False
    try:
        with tarfile.open(bundle, 'r:gz') as tar:
            member = tar.extractfile('./manifest.json')
            if member is not None:
                manifest = json.loads(member.read().decode())
                needs_pg = any(
                    s.get('backend') == 'postgres'
                    and s.get('status') != 'failed'
                    for s in manifest.get('stores', []))
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise click.ClickException(f'cannot read bundle: {exc}')
    if needs_pg and shutil.which('pg_restore') is None:
        raise click.ClickException(
            'bundle contains postgres stores but pg_restore is not on'
            ' PATH; install postgresql-client and retry')

    click.echo(
        f'About to restore {bundle} into {data_dir} (overwrites local'
        ' stores).', err=True)
    if not yes:
        click.confirm(
            'Proceed? This overwrites local stores.',
            default=False, abort=True)
    try:
        with held_drain_lock(data_dir):
            result = restore(bundle, data_dir)
    except (MigrateError, RuntimeError, EmbedFingerprintError) as exc:
        raise click.ClickException(str(exc))
    _json_out({'action': 'restored', **result})


@backup.command('worker', hidden=True)
@click.pass_context
def backup_worker(ctx: click.Context) -> None:
    """Hidden: run one backup now. The scheduler unit's ExecStart target.
    """
    from memman.backup import run_backup
    try:
        run_backup(ctx.obj['data_dir'])
    except RuntimeError as exc:
        raise click.ClickException(str(exc))


@claude_callable
@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show database statistics.
    """
    data_dir = ctx.obj['data_dir']
    store_name = _resolve_store_name(data_dir, ctx.obj['store'])
    with _active_backend(ctx) as backend:
        node_stats = backend.nodes.stats()
        declared = {
            key[len('MEMMAN_BACKEND_'):]
            for key in config.parse_env_file(config.env_file_path(data_dir))
            if key.startswith('MEMMAN_BACKEND_')
            }
        all_stores = set(list_stores(data_dir)) | declared
        backends_in_use = sorted({
            resolve_store_backend(s, data_dir) for s in all_stores
            })
        try:
            from memman.pipeline.remember import compute_prompt_version
            active_pv = compute_prompt_version()
            stale_insights: int | None = backend.nodes.count_stale_insights(
                active_pv)
        except Exception:
            stale_insights = None
        out = {
            'store': store_name,
            'backend': resolve_store_backend(store_name, data_dir),
            'backends_in_use': backends_in_use,
            'total_insights': node_stats.total_insights,
            'replaced_insights': node_stats.replaced_insights,
            'deleted_insights': node_stats.deleted_insights,
            'stale_insights': stale_insights,
            'oplog_count': node_stats.oplog_count,
            'storage_path': backend.path,
            }
        branch_parent = backend.meta.get(BRANCH_PARENT)
        if branch_parent is not None:
            out['branch_parent'] = branch_parent
            out['branch_created_at'] = backend.meta.get(
                branch_mod.BRANCH_CREATED_AT)
        _json_out(out)


@claude_callable
@cli.command()
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable colored output (default: JSON)')
@click.pass_context
def doctor(ctx: click.Context, text_output: bool) -> None:
    """Run health checks on the database, scheduler, and providers.

    Exits 0 on pass/warn, 1 on fail - usable as a CI/scripted gate.
    """
    from memman.doctor import run_all_checks

    data_dir = ctx.obj['data_dir']
    store_name = _resolve_store_name(data_dir, ctx.obj['store'])
    try:
        backend = factory.open_backend(store_name, data_dir)
    except StoreMissingError:
        backend = None
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(str(exc)) from exc
    with backend or nullcontext():
        result = run_all_checks(backend, data_dir=data_dir)
        result['store'] = store_name
        result['db_path'] = backend.path if backend else None
        if text_output:
            _doctor_text_report(result)
        else:
            _json_out(result)
    if result.get('status') == 'fail':
        ctx.exit(1)


def _doctor_text_report(result: dict) -> None:
    """Render a doctor result dict as colored PASS/WARN/FAIL lines.
    """
    colors = {'pass': 'green', 'warn': 'yellow',
              'fail': 'red', 'empty': 'cyan'}
    overall = result.get('status', 'unknown')
    click.secho(
        f'memman doctor: {overall.upper()}',
        fg=colors.get(overall, 'white'), bold=True)
    click.echo(f"store: {result.get('store', '?')}")
    click.echo(f"db:    {result.get('db_path', '?')}")
    click.echo(f"active insights: {result.get('total_active', 0)}")
    click.echo('')
    for check in result.get('checks', []):
        st = check.get('status', 'unknown')
        click.secho(
            f"  [{st.upper():>4}] {check.get('name', '?')}",
            fg=colors.get(st, 'white'))
        detail = check.get('detail') or {}
        if detail and st != 'pass':
            for key, value in detail.items():
                click.echo(f'         {key}: {value}')


@log.command('list')
@click.option('--limit', default=20, type=int, help='Max entries')
@click.option('--since', default='', help='Time window (e.g. 7d, 24h)')
@click.option('--stats', is_flag=True, default=False,
              help='Show summary statistics (grouped by operation)')
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable text table (default: JSON)')
@click.pass_context
def log_list(ctx: click.Context, limit: int, since: str,
             stats: bool, text_output: bool) -> None:
    """Show the operation audit log (default JSON; --text for human view).
    """
    since_ts = ''
    if since:
        since_ts = _parse_since(since)

    with _active_backend(ctx) as backend:
        if stats:
            stats_data = backend.oplog.stats(since=since_ts)
            _json_out({
                'operation_counts': stats_data.operation_counts,
                'total_active': stats_data.total_active,
                })
            return

        entry_objs = backend.oplog.recent(limit=limit, since=since_ts)
        entries = [
            {
                'created_at': format_timestamp(e.created_at),
                'operation': e.operation,
                'insight_id': e.insight_id,
                'detail': e.detail,
                }
            for e in entry_objs
            ]

        if not text_output:
            _json_out({'entries': entries, 'meta': {'count': len(entries)}})
            return

        if not entries:
            click.echo('No operations recorded yet.')
            return

        headers = ['TIME', 'OP', 'INSIGHT', 'DETAIL']
        sep = ['----', '--', '-------', '------']
        rows = []
        for e in entries:
            detail = e['detail']
            if len(detail) > 60:
                detail = detail[:57] + '...'
            rows.append([
                e['created_at'],
                e['operation'],
                e['insight_id'] or '',
                detail,
                ])

        all_rows = [headers, sep] + rows
        widths = [0] * 4
        for row in all_rows:
            for i, col in enumerate(row):
                widths[i] = max(widths[i], len(col))

        for row in all_rows:
            line = '  '.join(
                col.ljust(widths[i]) for i, col in enumerate(row))
            click.echo(line.rstrip())


@log.command('calls')
@click.option('--since', default='', help='Time window (e.g. 7d, 24h)')
@click.pass_context
def log_calls(ctx: click.Context, since: str) -> None:
    """Count agent-verb calls per UTC date and verb, as JSON.

    Reads `<data dir>/logs/calls.log`, which gains one line each time
    an agent-callable verb (`recall`, `remember`, `status`, ...) runs.

    \b
    Parameters
    ----------
    since : str
        Keep calls that started within this window: a count and a
        unit, `7d`, `24h` or `30m`. Empty keeps every call.

    \b
    Notes
    -----
    - `counts` runs newest date first, then most calls first, then
      verb name. `meta.total` sums the calls counted.
    - `meta.malformed` counts lines off the call-log format, such as a
      write a full disk cut short. They count toward no verb, and the
      window does not apply to them.
    - A data dir where no agent verb has run reports no calls.

    \b
    Examples
    --------
    memman log calls
    memman log calls --since 7d
    """  # noqa: D301, D410, D411
    since_ts = _parse_since(since) if since else ''
    log_path = _call_log_path(ctx.obj['data_dir'])
    try:
        lines = log_path.read_text(errors='replace').splitlines()
    except FileNotFoundError:
        lines = []

    calls_by_date_verb: Counter[tuple[str, str]] = Counter()
    malformed_cnt = 0
    for line in lines:
        match = _CALL_LINE_RE.fullmatch(line)
        if match is None:
            malformed_cnt += 1
        elif match['started_at'] >= since_ts:
            date = match['started_at'][:10]
            calls_by_date_verb[(date, match['verb'])] += 1

    ranked = sorted(
        calls_by_date_verb.items(), key=lambda item: (-item[1], item[0][1]))
    ranked.sort(key=lambda item: item[0][0], reverse=True)
    _json_out({
        'counts': [
            {'date': date, 'verb': verb, 'calls': calls}
            for (date, verb), calls in ranked
            ],
        'meta': {
            'total': sum(calls_by_date_verb.values()),
            'malformed': malformed_cnt,
            },
        })


@log.command('worker')
@click.option('--errors', is_flag=True, default=False,
              help='Read enrich.err instead of enrich.log.')
@click.option('--stack', is_flag=True, default=False,
              help='Read the rotated worker log that preserves tracebacks.')
@click.option('--lines', type=int, default=50,
              help='Number of tail lines to print (default 50).')
@click.pass_context
def log_worker(ctx: click.Context, errors: bool, stack: bool,
               lines: int) -> None:
    """Print the tail of one worker log target.

    `enrich.log` and `enrich.err` are the enrichment worker's stdout
    and stderr. `memman.log` is the rotated DEBUG log holding the
    tracebacks a one-line CLI error cannot carry, and the two targets
    do not share a directory.

    \b
    Parameters
    ----------
    errors : bool
        Read `enrich.err` rather than `enrich.log`.
    stack : bool
        Read `memman.log` and its rotation backups. Rejected together
        with `--errors`.
    lines : int
        Tail length; a non-positive value prints everything read.

    \b
    Notes
    -----
    - `enrich.log` and `enrich.err` sit under `~/.memman/logs`
      whatever `--data-dir` says: the systemd unit pins those two
      redirects to `%h/.memman/logs`, and the launchd plist bakes the
      absolute home in at install time. Neither reads the data dir.
    - `memman.log` is the one target that follows `--data-dir`, since
      the rotating handler builds its path from it. Under a
      non-default data dir the targets live in two directories, and
      this command resolves each from its own source.
    - `--stack` reads the rotation backups as well as the live file,
      oldest first. Rotation caps the set at 20 MB over four files, and
      BOTH the enrichment worker and the backup worker write it, so a
      traceback reaches a backup file sooner than its size suggests.
    - Nothing rotates `enrich.err`, so a pointer naming a stack can
      outlive the stack itself once all four files have turned over.

    \b
    Examples
    --------
    memman log worker --errors
    memman log worker --stack --lines 200
    """  # noqa: D301, D410, D411
    if errors and stack:
        raise click.UsageError(
            '--errors and --stack name different files; pass one.')
    if stack:
        named = pathlib.Path(ctx.obj['data_dir']) / 'logs' / 'memman.log'
        # Oldest backup first, so one tail spans a rotation. The
        # traceback a CLI error pointed at is often already in
        # memman.log.1: two workers share this file and rotation keeps
        # only _WORKER_LOG_BACKUPS of it, so reading the live file
        # alone reports no traceback while the stack is still on disk.
        candidates = [
            named.with_name(f'{named.name}.{i}')
            for i in range(_WORKER_LOG_BACKUPS, 0, -1)]
        candidates.append(named)
    else:
        logs_dir = pathlib.Path.home() / '.memman' / 'logs'
        named = logs_dir / ('enrich.err' if errors else 'enrich.log')
        candidates = [named]
    present = [p for p in candidates if p.is_file()]
    if not present:
        click.echo(f'[memman] no log file yet at {named}', err=True)
        return
    content: list[str] = []
    for path in present:
        try:
            content.extend(path.read_text(errors='replace').splitlines())
        except OSError as exc:
            raise click.ClickException(
                f'failed to read {path}: {exc}') from exc
    tail = content[-lines:] if lines > 0 else content
    for line_str in tail:
        click.echo(line_str)


@claude_callable
@insights.command('review')
@click.option('--limit', default=20, type=int, help='Max flagged results')
@click.pass_context
def insights_review(ctx: click.Context, limit: int) -> None:
    """Scan stored insights for content quality issues.

    Flags transient phrasing and low-signal content, so an
    operator can decide whether to `memman forget` a row.
    """
    from memman.search.quality import check_content_quality

    with _active_backend(ctx) as backend:
        all_active = backend.nodes.get_all_active()
        flagged = []
        for ins in all_active:
            warnings = check_content_quality(ins.content)
            if warnings:
                flagged.append({'insight': ins, 'quality_warnings': warnings})
            if len(flagged) >= limit:
                break
        _json_out({
            'review_results': [{
                'id': f['insight'].id,
                'content': f['insight'].content,
                'quality_warnings': f['quality_warnings'],
                } for f in flagged],
            'total_flagged': len(flagged),
            'actions': {'forget': 'memman forget <id>'},
            })


@claude_callable
@insights.command('show')
@click.argument('id')
@click.option('--history', is_flag=True,
              help='Walk the replacement chain through this id.')
@click.pass_context
def insights_show(ctx: click.Context, id: str, history: bool) -> None:
    """Read one insight by id, or walk its replacement chain.

    ID is a full insight id or any unambiguous prefix of one.

    Without `--history`: the full insight, including a replaced
    row (its `replaced_by` names the successor). A forgotten row is
    refused. With `--history`: every row in the chain through this
    id, oldest first, forgotten rows included and marked.

    \b
    Parameters
    ----------
    id : str
        Any stored id. With `--history` a forgotten id is accepted so
        a chain whose oldest row was forgotten stays walkable. The id
        of a write still queued is refused as still queued.

    \b
    Returns
    -------
    JSON
        Without `--history`, the insight dict. With it,
        `{requested, chain}` where each chain entry is `{id,
        created_at, state, replaced_by, content}`; `state` is one of
        `current`, `replaced`, `forgotten`, and a forgotten entry
        carries no `content`.

    \b
    Notes
    -----
    - The walk follows `replaced_by` forward and every row pointing
      at a chain member backward, so a successor with two predecessors,
      which stored rows may hold, lists both.
    - Order is chain order, not timestamp order: rows written within
      one second still list predecessor first.

    \b
    Examples
    --------
    memman insights show 16c6c667-...
    memman insights show 16c6c667-... --history
    """  # noqa: D301, D410, D411
    name = _resolve_store_name(ctx.obj['data_dir'], ctx.obj['store'])
    with _active_backend(ctx) as backend:
        queued, ins = _resolve_queued_or_stored(
            backend, ctx.obj['data_dir'], name, id)
        if ins is None and queued is not None:
            raise click.ClickException(
                f'insight {queued} is still queued; it lands on'
                ' the next drain')
        if ins is None:
            raise click.ClickException(f'insight {id} not found')
        id = ins.id
        if not history:
            if ins.deleted_at is not None:
                raise click.ClickException(f'insight {id} was forgotten')
            _json_out(insight_to_full_dict(ins))
            return
        rows: dict[str, Insight] = {ins.id: ins}
        frontier = [ins]
        while frontier:
            row = frontier.pop()
            found = list(backend.nodes.predecessors(row.id))
            if row.replaced_by and row.replaced_by not in rows:
                successor = backend.nodes.get_include_deleted(
                    row.replaced_by)
                if successor is not None:
                    found.append(successor)
            for other in found:
                if other.id not in rows:
                    rows[other.id] = other
                    frontier.append(other)

    def depth(row: Insight) -> int:
        steps, seen = 0, set()
        while (row.replaced_by in rows and row.id not in seen):
            seen.add(row.id)
            row = rows[row.replaced_by]
            steps += 1
        return steps

    chain = []
    for row in sorted(rows.values(),
                      key=lambda r: (-depth(r), r.created_at or '', r.id)):
        if row.deleted_at is not None:
            state = 'forgotten'
        elif row.replaced_by:
            state = 'replaced'
        else:
            state = 'current'
        entry: dict[str, Any] = {
            'id': row.id,
            'created_at': format_timestamp(row.created_at),
            'state': state,
            'replaced_by': row.replaced_by,
            }
        if state != 'forgotten':
            entry['content'] = row.content
        chain.append(entry)
    _json_out({'requested': id, 'chain': chain})


@cli.command()
@click.option('--claude-code', is_flag=True,
              help='Install into ~/.claude even when Claude Code is not'
                   ' detected.')
@click.option('--codex', is_flag=True,
              help='Install the Codex skill into ~/.agents/skills even'
                   ' when Codex is not detected.')
@click.option('--backend', type=click.Choice(_BACKEND_CHOICES),
              default=None,
              help='Storage backend; bypasses the wizard prompt when set.')
@click.option('--pg-dsn', default=None,
              help='Postgres DSN (postgresql://...); required with'
                   ' --backend postgres in non-interactive mode.')
@click.option('--endpoint', type=str, default=None,
              help='Endpoint URL for the LLM, embed, and rerank paths;'
                   ' bypasses the wizard prompt when set.')
@click.option('--no-wizard', is_flag=True,
              help='Disable interactive prompts; flags + defaults only.')
@click.pass_context
def install(ctx: click.Context, claude_code: bool, codex: bool,
            backend: str | None,
            pg_dsn: str | None, endpoint: str | None,
            no_wizard: bool) -> None:
    """Install memman integration: skill, hooks, scheduler.

    \b
    Parameters
    ----------
    claude_code : bool
        Install into ~/.claude even when the `claude` binary is not on
        PATH.
    codex : bool
        Install the Codex memory skill even when Codex is not detected.
    backend : str or None
        `sqlite` or `postgres`; unset leaves it to the wizard or the
        env file.
    pg_dsn : str or None
        Postgres DSN for `--backend postgres`; a run that cannot
        prompt needs it here or already in the env file.
    endpoint : str or None
        OpenAI-compatible endpoint URL for the LLM, embed, and rerank
        paths.
    no_wizard : bool
        Take flags, the env file, and defaults only; never prompt.

    \b
    Notes
    -----
    - A flag never overrides a value already in ~/.memman/env; it
      refuses and names `memman config set` as the fix.
    - Without agent flags, install all detected integrations. Explicit
      --claude-code and --codex flags select only those integrations.

    \b
    Examples
    --------
    memman install
    memman install --claude-code --no-wizard
    memman install --codex
    memman install --backend postgres --pg-dsn postgresql://host/db
    """  # noqa: D301, D410, D411
    from memman.setup.claude import run_install
    run_install(
        ctx.obj['data_dir'],
        claude_code=claude_code,
        codex=codex,
        backend=backend,
        pg_dsn=pg_dsn,
        endpoint=endpoint,
        no_wizard=no_wizard)


@cli.command()
@click.option('--claude-code', is_flag=True,
              help='Remove from ~/.claude even when Claude Code is not'
                   ' detected.')
@click.option('--codex', is_flag=True,
              help='Remove the Codex skill from ~/.agents/skills.')
@click.pass_context
def uninstall(ctx: click.Context, claude_code: bool, codex: bool) -> None:
    """Remove memman integration (reverse of `memman install`).

    \b
    Parameters
    ----------
    claude_code : bool
        Remove from ~/.claude even when the `claude` binary is not on
        PATH.
    codex : bool
        Remove the Codex memory skill.

    \b
    Notes
    -----
    - The stores stay on disk. When shared services are removed, the
      env file loses secret keys and keeps the other settings.
    - Agent flags select integrations. Keep the shared scheduler and
      settings while another memman integration remains installed.
      Without agent flags, remove all detected integrations and services.

    \b
    Examples
    --------
    memman uninstall
    memman uninstall --claude-code
    memman uninstall --codex
    """  # noqa: D301, D410, D411
    from memman.setup.claude import run_uninstall
    run_uninstall(ctx.obj['data_dir'], claude_code=claude_code, codex=codex)


@cli.command()
@click.option('--store', default='',
              help='Store to migrate. Required unless --all.')
@click.option('--all', 'migrate_all', is_flag=True,
              help='Migrate every store under the data dir.')
@click.option('--to', 'target_backend',
              type=click.Choice(_BACKEND_CHOICES),
              default='postgres',
              help='Target backend for the migration. Default: postgres.')
@click.option('--dry-run', is_flag=True,
              help='Report the plan without writing or prompting.')
@click.option('--yes', is_flag=True, default=False,
              help='Skip the confirmation prompt (for scripted use).')
@click.pass_context
def migrate(
        ctx: click.Context, store: str, migrate_all: bool,
        target_backend: str, dry_run: bool, yes: bool) -> None:
    """Migrate memman stores between SQLite and Postgres backends.

    Migration is symmetric. `--to postgres` (default) copies SQLite
    stores into a Postgres schema, archives the SQLite source, and
    flips MEMMAN_BACKEND_<store>=postgres. `--to sqlite` dumps the
    Postgres schema to archive/, copies rows into a fresh SQLite
    store, drops the postgres schema, and flips
    MEMMAN_BACKEND_<store>=sqlite. Stores already on the target
    backend emit a warning and are skipped (idempotent). The shared
    drain.lock is held throughout so a scheduler-fired drain cannot
    race the migration.
    """
    from memman.migrate import inspect_target_schemas, preflight
    from memman.setup.scheduler import _write_env_keys
    from memman.store.postgres import PostgresMigrator, _connection
    from memman.store.postgres import _store_schema, drop_postgres_store
    from memman.trace import redact_dsn

    data_dir = ctx.obj['data_dir']

    if shutil.which('pg_dump') is None:
        raise click.ClickException(
            'pg_dump not found on PATH. memman migrate requires'
            ' pg_dump regardless of direction so the postgres source'
            ' can be archived before any destructive step.'
            ' Install postgresql-client:'
            '\n  apt: sudo apt install postgresql-client'
            '\n  brew: brew install libpq && brew link --force libpq')

    if not migrate_all and not store:
        raise click.UsageError('pass --store NAME or --all')
    if migrate_all and store:
        raise click.UsageError(
            'pass either --store NAME or --all, not both')
    if dry_run and target_backend == 'sqlite':
        raise click.UsageError(
            '--dry-run is not supported with --to sqlite')

    if migrate_all:
        if target_backend == 'sqlite':
            stores_all = list_stores(data_dir)
        else:
            stores_all = list_local_store_dirs(data_dir)
    else:
        # Notes:
        # - Existence is checked before the naming guard below, which
        #   would otherwise describe a store that is not there.
        # - Only a sqlite-routed store is judged here, from the
        #   filesystem. Deciding a postgres-routed one needs the
        #   server, and during an outage that reports a live store as
        #   missing; those fall through and fail on their own error.
        if (resolve_store_backend(store, data_dir) == 'sqlite'
                and not store_exists(data_dir, store)):
            raise click.ClickException(
                f'store {store!r} does not exist'
                f' (list them with `memman store list`)')
        stores_all = [store]
    if not stores_all:
        click.echo('no stores to migrate', err=True)
        return

    todo: list[str] = []
    skipped: list[str] = []
    for s in stores_all:
        current = resolve_store_backend(s, data_dir)
        if branch_mod.read_branch_info(s, data_dir) is not None:
            click.echo(
                f'Skipping {s!r}: it is a branch, which stays local; end it'
                f' with memman store merge {s} or memman store drop {s}.',
                err=True)
            skipped.append(s)
            continue
        if current == target_backend:
            click.echo(
                f'Store {s!r} is already on {target_backend} backend.'
                f' Nothing to migrate.')
            skipped.append(s)
            continue
        todo.append(s)

    if not todo:
        if migrate_all:
            click.echo(f'migrated=0 skipped={len(skipped)}')
        return

    # Notes:
    # - Both directions touch a postgres schema, so both need a name
    #   that fits one. Guarding only `--to postgres` lets the same
    #   ConfigError escape `except MigrateError` on the way back,
    #   after the plan prints and the drain lock is held.
    # - Store names reach here unchecked: `list_local_store_dirs`
    #   scans the filesystem, and a postgres route can be hand
    #   written into the env file.
    # - `_store_schema` is the oracle, so this gate agrees with the
    #   call that raised.
    # - Runs before the DSN lookup, so a naming fault needs no
    #   reachable server.
    unhostable: list[tuple[str, str]] = []
    for s in list(todo):
        try:
            _store_schema(s)
        except StoreConfigError as exc:
            unhostable.append((s, str(exc)))
            todo.remove(s)

    # Notes:
    # - `_store_schema` refuses on character class and on length, so
    #   the remedy quotes its message. A hardcoded character-class
    #   reason would misdescribe a name refused only for length.
    # - The rewrite is not injective and does not shorten, so it can
    #   return the name just refused, or one already claimed by
    #   another store in this run. The remedy says so.
    # - `known` comes from the filesystem and this run. `list_stores`
    #   reaches the server, and this gate must work without one.
    known = set(list_local_store_dirs(data_dir)) | set(stores_all)
    taken: set[str] = set()

    def _remedy(name: str) -> str:
        suggestion = portable_store_name(name)
        if suggestion == name:
            return 'rename it to a shorter plain-identifier name'
        if suggestion in known | taken:
            return (f'the portable form {suggestion!r} is already'
                    f' taken, so pick another name')
        taken.add(suggestion)
        return f'create {suggestion!r} and migrate that'

    if unhostable and not migrate_all:
        bad, reason = unhostable[0]
        raise click.ClickException(
            f'store {bad!r} cannot be migrated: {reason}.'
            f' A sqlite-backed store of that name stays fully'
            f' usable. To move it, {_remedy(bad)}.')
    for s, reason in unhostable:
        click.echo(
            f'Skipping {s!r}: {reason}. To move it, {_remedy(s)}.',
            err=True)
        skipped.append(s)
    if not todo:
        verb = 'planned' if dry_run else 'migrated'
        click.echo(f'{verb}=0 skipped={len(skipped)}')
        return

    if target_backend == 'postgres':
        if migrate_all:
            dsn = config.get(config.DEFAULT_PG_DSN)
            if not dsn:
                raise click.UsageError(
                    'MEMMAN_DEFAULT_POSTGRES_DSN is not set; --all requires a'
                    ' default DSN. Run `memman config set-pg-dsn'
                    ' --default`, or migrate one store at a time with'
                    ' --store NAME.')
        else:
            dsn = (config.get(config.POSTGRES_DSN_FOR(todo[0]))
                   or config.get(config.DEFAULT_PG_DSN))
            if not dsn:
                raise click.UsageError(
                    f'no DSN for store {todo[0]!r}: set'
                    f' {config.POSTGRES_DSN_FOR(todo[0])} or'
                    f' {config.DEFAULT_PG_DSN} (run `memman config'
                    f' set-pg-dsn --store {todo[0]}` or `--default`).')

        try:
            preflight(dsn)
        except MigrateError as exc:
            raise click.ClickException(str(exc))

        try:
            states = inspect_target_schemas(dsn, todo)
        except MigrateError as exc:
            raise click.ClickException(str(exc))

        populated = [s for s in todo
                     if states[s] == SchemaState.POPULATED]

        click.echo('Migration plan:')
        click.echo(f'  Source:      {data_dir}/data/')
        click.echo(f'  Destination: {redact_dsn(dsn)}')
        click.echo(f'  Stores ({len(todo)}):')
        width = max((len(s) for s in todo), default=0)
        for s in todo:
            st = states[s]
            if st == SchemaState.ABSENT:
                note = 'will create'
            elif st == SchemaState.EMPTY:
                note = 'EMPTY, will recreate'
            else:
                note = 'POPULATED, will DROP CASCADE and recreate'
            click.echo(
                f'    {s.ljust(width)} -> store_{s}    [{note}]')
        if populated:
            click.echo('')
            click.echo(
                f'WARNING: {len(populated)} store(s) will be'
                f' destructively overwritten.')

        if dry_run:
            src_migrator = SqliteMigrator(data_dir)
            for s in todo:
                try:
                    src_migrator.preflight_source(s)
                    payload = src_migrator.gather(s)
                    click.echo(
                        f'{s}: insights={len(payload.insights)}'
                        f' oplog={len(payload.oplog)}'
                        f' meta={len(payload.meta)} (dry-run)')
                # Notes:
                # - `apply` and `_verify_destination_counts` raise
                #   `BackendError` from the Postgres connection scope.
                # - `SqliteMigrator.gather` runs its selects unwrapped
                #   (`_connect_ro` translates only the connect and the
                #   `pragma schema_version` probe), so a store that
                #   fails mid-read raises a bare `sqlite3.Error`.
                # - Missing either type loses the store name, and
                #   `--all` cannot say which store failed.
                except (MigrateError, BackendError, sqlite3.Error) as exc:
                    raise click.ClickException(f'{s}: {exc}')
            # A plan that skipped stores must say so, as the
            # all-skipped branch does.
            if migrate_all and skipped:
                click.echo(
                    f'planned={len(todo)} skipped={len(skipped)}')
            return

        click.echo('')
        click.echo(
            'After successful migrate,'
            ' MEMMAN_BACKEND_<store>=postgres'
            ' and MEMMAN_POSTGRES_DSN_<store>=<dsn> will be written to'
            ' the env file for each migrated store.')
        if not yes:
            click.echo('')
            click.confirm('Proceed?', default=False, abort=True)

        try:
            with held_drain_lock(data_dir):
                src_migrator = SqliteMigrator(data_dir)
                tgt_migrator = PostgresMigrator(dsn=dsn)
                tgt_migrator.preflight_target(todo[0])
                for s in todo:
                    try:
                        src_migrator.preflight_source(s)
                        if states[s] in {
                                SchemaState.EMPTY,
                                SchemaState.POPULATED}:
                            drop_postgres_store(s, dsn)
                        payload = src_migrator.gather(s)
                        tgt_migrator.apply(s, payload)
                        with _connection(dsn, autocommit=True) as conn:
                            _verify_destination_counts(
                                conn, _store_schema(s), s,
                                expected={
                                    'insights': len(payload.insights),
                                    'oplog': len(payload.oplog),
                                    'meta': len(payload.meta),
                                    })
                        click.echo(
                            f'{s}: insights={len(payload.insights)}'
                            f' oplog={len(payload.oplog)}'
                            f' meta={len(payload.meta)} (verified)')
                        _write_env_keys({
                            config.BACKEND_FOR(s): 'postgres',
                            config.POSTGRES_DSN_FOR(s): dsn,
                            }, data_dir=data_dir)
                        click.echo(
                            f'  Wrote {config.BACKEND_FOR(s)}=postgres'
                            f' to {data_dir}/env.')
                        artifact = src_migrator.archive(s, data_dir)
                        if (artifact.kind == 'filesystem'
                                and artifact.location):
                            click.echo(
                                f'  Archived source to'
                                f' {artifact.location}.')
                    except (MigrateError, BackendError,
                            sqlite3.Error) as exc:
                        raise click.ClickException(f'{s}: {exc}')
                    except OSError as exc:
                        click.echo(
                            f'  WARNING: could not archive source'
                            f' for {s!r}: {exc}; leaving in place.'
                            ' Run `memman doctor` to track.',
                            err=True)
        except MigrateError as exc:
            raise click.ClickException(str(exc))

        click.echo('')
        click.echo(
            f'Migration complete: {len(todo)} store(s)'
            f' copied to Postgres.')
        click.echo(
            f'Sources archived to'
            f' {data_dir}/archive/<store>/<YYYYMMDD>_<NN>/.'
            f' Remove with `rm -rf` when no longer needed.')
        if migrate_all and skipped:
            click.echo(
                f'migrated={len(todo)} skipped={len(skipped)}')
        click.echo('')
        click.echo('Recommended next step:')
        click.echo(
            '  memman doctor    # verify the postgres backend health')
        return

    store_dsns: dict[str, str] = {}
    for s in todo:
        dsn = resolve_store_pg_dsn(s, data_dir)
        if not dsn:
            raise click.UsageError(
                f'no postgres DSN for store {s!r}: set'
                f' {config.POSTGRES_DSN_FOR(s)} or'
                f' {config.DEFAULT_PG_DSN}.')
        store_dsns[s] = dsn

    target_paths: dict[str, pathlib.Path] = {
        s: pathlib.Path(store_dir(data_dir, s)) for s in todo
        }
    for s in todo:
        if target_paths[s].exists():
            raise click.ClickException(
                f'target directory {target_paths[s]} already exists;'
                f' move it aside (`mv {target_paths[s]}'
                f' {target_paths[s]}.bak`) and re-run.')

    click.echo('Migration plan (postgres -> sqlite):')
    click.echo(f'  Target: {data_dir}/data/')
    click.echo(f'  Stores ({len(todo)}):')
    width = max((len(s) for s in todo), default=0)
    for s in todo:
        click.echo(
            f'    {s.ljust(width)} <- {redact_dsn(store_dsns[s])}'
            f' (schema store_{s})')
    click.echo('')
    click.echo(
        'After successful migrate, MEMMAN_BACKEND_<store>=sqlite will'
        ' be written and MEMMAN_POSTGRES_DSN_<store> removed for each'
        ' migrated store. Postgres schemas will be archived to'
        f' {data_dir}/archive/<store>/<YYYYMMDD>_<NN>/dump.pgdump'
        ' and dropped.')

    if not yes:
        click.echo('')
        click.confirm('Proceed?', default=False, abort=True)

    try:
        with held_drain_lock(data_dir):
            for s in todo:
                dsn = store_dsns[s]
                target = target_paths[s]
                scratch = pathlib.Path(tempfile.mkdtemp(
                    dir=data_dir, prefix='migrate-'))
                (scratch / 'data').mkdir()
                produced = scratch / 'data' / s
                try:
                    src_migrator = PostgresMigrator(dsn=dsn)
                    src_migrator.preflight_source(s)
                    payload = src_migrator.gather(s)
                    tgt_migrator = SqliteMigrator(str(scratch))
                    tgt_migrator.preflight_target(s)
                    tgt_migrator.apply(s, payload)
                    click.echo(
                        f'{s}: insights={len(payload.insights)}'
                        f' oplog={len(payload.oplog)}'
                        f' meta={len(payload.meta)} (verified)')
                # Notes:
                # - A backend rejecting the store still removes the
                #   scratch dir: an escape would strand a `migrate-*`
                #   directory in the data dir.
                # - BackendError covers ConfigError and the bare
                #   BackendError that `gather` / `apply` /
                #   `_verify_destination_counts` raise from the
                #   Postgres connection scope.
                except (MigrateError, BackendError,
                        sqlite3.Error) as exc:
                    shutil.rmtree(scratch, ignore_errors=True)
                    raise click.ClickException(f'{s}: {exc}')

                shutil.move(str(produced), str(target))
                shutil.rmtree(scratch, ignore_errors=True)
                click.echo(f'  Wrote sqlite store at {target}.')

                try:
                    archive_dest = archive_postgres_schema(
                        data_dir, s, dsn)
                    click.echo(
                        f'  Archived postgres schema to'
                        f' {archive_dest}.')
                except Exception as exc:
                    raise click.ClickException(
                        f'{s}: archive_postgres_schema failed: {exc}')

                _write_env_keys(
                    {config.BACKEND_FOR(s): 'sqlite'},
                    removes={config.POSTGRES_DSN_FOR(s)},
                    data_dir=data_dir)
                click.echo(
                    f'  Wrote {config.BACKEND_FOR(s)}=sqlite to'
                    f' {data_dir}/env (removed'
                    f' {config.POSTGRES_DSN_FOR(s)}).')

                try:
                    drop_postgres_store(s, dsn)
                    click.echo(f'  Dropped postgres schema store_{s}.')
                except Exception as exc:
                    click.echo(
                        f'  WARNING: failed to drop postgres schema'
                        f' for {s!r}: {exc}; remove manually with'
                        f' `psql -c "drop schema store_{s} cascade"`.',
                        err=True)
    except MigrateError as exc:
        raise click.ClickException(str(exc))

    click.echo('')
    click.echo(
        f'Migration complete: {len(todo)} store(s)'
        f' migrated to SQLite.')
    click.echo(
        f'Postgres schemas archived to'
        f' {data_dir}/archive/<store>/<YYYYMMDD>_<NN>/dump.pgdump.'
        f' Replay with `pg_restore -d <dsn> <archive>` if needed.')
    if migrate_all and skipped:
        click.echo(f'migrated={len(todo)} skipped={len(skipped)}')
    click.echo('')
    click.echo('Recommended next step:')
    click.echo('  memman doctor    # verify the sqlite backend health')


@cli.command(hidden=True)
def prime() -> None:
    """Hook shim: emit status + optional compact hint + guide. Invoked by
    the SessionStart hook (claude/prime.sh). Not meant for direct use.
    """
    input_raw = '{}'
    if not sys.stdin.isatty():
        try:
            input_raw = sys.stdin.read()
        except OSError:
            input_raw = '{}'
    try:
        session = json.loads(input_raw) if input_raw.strip() else {}
    except json.JSONDecodeError:
        session = {}

    source = session.get('source', '')
    session_id = session.get('session_id', '')

    status_line = '[memman] Memory active.'
    try:
        data_dir = os.environ.get(config.DATA_DIR, default_data_dir())
        name = _resolve_store_name(data_dir, '')
        backend_name = resolve_store_backend(name, data_dir)
        branch_parent = None
        if backend_name == 'sqlite':
            if not store_exists(data_dir, name):
                raise StoreMissingError(name)
            with open_read_only(store_dir(data_dir, name)) as db:
                stats = get_stats(db)
                branch_parent = get_meta(db, BRANCH_PARENT)
            if branch_parent is None:
                status_line = (f"[memman] Memory active "
                               f"({stats['total_insights']} insights).")
        if backend_name != 'sqlite' or branch_parent is not None:
            with factory.open_backend(name, data_dir) as backend:
                s = backend.nodes.stats()
                status_line = (f'[memman] Memory active '
                               f'({s.total_insights} insights).')
    except StoreMissingError as exc:
        status_line = f'[memman] Memory unavailable: {exc}'
    except Exception as exc:
        logger.debug('prime status fallback: %s', exc)
    click.echo(status_line)
    from memman.llm import openrouter_models
    notice = openrouter_models.read_model_notice(
        os.environ.get(config.DATA_DIR, default_data_dir()))
    if notice:
        click.echo(f'[memman] {notice}')

    if source == 'compact':
        flag = (pathlib.Path.home() / '.memman' / 'compact'
                / f'{session_id}.json')
        trigger = 'auto'
        if flag.is_file():
            try:
                flag_data = json.loads(flag.read_text())
                trigger = flag_data.get('trigger', 'auto') or 'auto'
            except (json.JSONDecodeError, OSError):
                pass
        click.echo(f'[memman] Context was just compacted ({trigger}). '
                   f'Recall critical context now: '
                   f'memman recall "<topic>"')

    shipped = (pkg_files('memman.setup.assets')
               .joinpath('claude/guide.md').read_text())
    click.echo(shipped, nl=False)


def _enrich_stale_only(
        ctx: click.Context, *, dry_run: bool,
        progress_jsonl: bool) -> None:
    """Stale-only branch of `enrich`.

    Filters work to rows whose persisted `prompt_version` no longer
    matches `compute_prompt_version()` -- the enrichment prompt plus
    the LLM model, which is exactly the set this command
    replays -- and to stranded rows, whose enrichment call failed
    after the attempt stamp. Works on SQLite and Postgres. Lock +
    predicate + reset run inside a single `reembed_lock('rebuild')`
    window so a concurrent wholesale rebuild cannot race.
    """
    from memman.pipeline.enrich import MAX_ENRICH_BATCH, enrich_pending
    from memman.pipeline.remember import compute_prompt_version

    if not dry_run:
        _require_stopped('rebuild')

    try:
        active_pv = compute_prompt_version()
    except Exception as exc:
        raise click.ClickException(
            f'cannot resolve active prompt version: {exc}')
    # `active_pv` already folds this model in. The payload names it so
    # a reader can tell which input drifted without un-folding the
    # hash.
    try:
        enrich_model: str | None = config.require(
            config.LLM_MODEL)
    except Exception:
        enrich_model = None

    with _active_backend(ctx) as backend:
        if dry_run:
            stale = backend.nodes.count_stale_insights(active_pv)
            _json_out({
                'mode': 'stale-only', 'total': stale, 'dry_run': 1,
                'active_pv': active_pv,
                'active_enrich_model': enrich_model,
                })
            return

        with backend.reembed_lock('rebuild') as held:
            if not held:
                raise click.ClickException(
                    'another enrich run is in progress on this store')

            stale_ids = backend.nodes.iter_stale_insight_ids(active_pv)
            total_count = len(stale_ids)

            if total_count == 0:
                stats = {
                    'processed': 0, 'remaining': 0,
                    'mode': 'stale-only',
                    'skipped': 'no_stale_rows',
                    'active_pv': active_pv,
                    'active_enrich_model': enrich_model,
                    }
                _json_out(stats)
                return

            llm_client = _get_llm_client_or_fail()
            ec = fingerprint.bound_embedder(backend)

            processed = 0

            bar = tqdm(
                total=total_count, desc='Rebuilding (stale)',
                unit='insight', file=sys.stderr,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty())

            done_count = 0

            def _on_progress(stage: str, insight: Insight) -> None:
                nonlocal done_count
                preview = insight.content[:40].replace('\n', ' ')
                bar.set_description(f'{stage}: {preview}')
                if stage == 'done':
                    bar.update(1)
                    done_count += 1
                    if progress_jsonl:
                        sys.stderr.write(json.dumps({
                            'event': 'progress',
                            'stage': 'done',
                            'n': done_count,
                            'total': total_count,
                            }) + '\n')
                        sys.stderr.flush()

            for i in range(0, total_count, MAX_ENRICH_BATCH):
                batch_ids = stale_ids[i:i + MAX_ENRICH_BATCH]
                backend.nodes.reset_for_rebuild(batch_ids)

                while True:
                    count = enrich_pending(
                        backend,
                        llm_client=llm_client,
                        embed_client=ec,
                        on_progress=_on_progress)
                    processed += count
                    if count == 0:
                        break

            bar.set_description('Done')
            bar.close()

            remaining = backend.nodes.count_pending_enrich()

            stats = {
                'processed': processed, 'remaining': remaining,
                'mode': 'stale-only',
                'active_pv': active_pv,
                'active_enrich_model': enrich_model,
                }
            backend.oplog.log(
                operation='rebuild', insight_id='',
                detail=json.dumps(stats))
            _json_out(stats)


@cli.command('enrich')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show counts without modifying DB')
@click.option('--progress-jsonl', is_flag=True, default=False,
              help='Emit one JSON line per done event to stderr'
                   ' (for parents that capture stderr and need streaming'
                   ' progress while the inner tqdm bar is suppressed).')
@click.option('--stale-only', is_flag=True, default=False,
              help='Re-enrich only rows whose prompt_version no longer'
                   ' matches the active config -- the enrichment prompt'
                   ' plus the LLM model, which is exactly what'
                   ' this command replays -- and stranded rows, whose'
                   ' enrichment call failed. Cross-backend'
                   ' (works on Postgres). An enriched row with NULL'
                   ' provenance is not swept; it needs a separate'
                   ' backfill.')
@click.pass_context
def enrich(ctx: click.Context, dry_run: bool,
           progress_jsonl: bool, stale_only: bool) -> None:
    """Re-enrich all insights through the full LLM pipeline.
    """
    if stale_only:
        _enrich_stale_only(
            ctx, dry_run=dry_run, progress_jsonl=progress_jsonl)
        return

    if not dry_run:
        _require_stopped('rebuild')
    from memman.pipeline.enrich import MAX_ENRICH_BATCH, enrich_pending

    with _active_backend(ctx) as backend:
        llm_client = _get_llm_client_or_fail()
        ec = fingerprint.bound_embedder(backend)

        all_ids = backend.nodes.get_active_ids()
        total_count = len(all_ids)

        if dry_run:
            _json_out({'total': total_count, 'dry_run': 1})
            return

        if total_count == 0:
            _json_out({'processed': 0, 'remaining': 0})
            return

        with backend.reembed_lock('rebuild') as held:
            if not held:
                raise click.ClickException(
                    'another enrich run is in progress on this store')

            processed = 0

            bar = tqdm(
                total=total_count, desc='Rebuilding',
                unit='insight', file=sys.stderr,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty())

            done_count = 0

            def _on_progress(stage: str, insight: Insight) -> None:
                nonlocal done_count
                preview = insight.content[:40].replace('\n', ' ')
                bar.set_description(f'{stage}: {preview}')
                if stage == 'done':
                    bar.update(1)
                    done_count += 1
                    if progress_jsonl:
                        sys.stderr.write(json.dumps({
                            'event': 'progress',
                            'stage': 'done',
                            'n': done_count,
                            'total': total_count,
                            }) + '\n')
                        sys.stderr.flush()

            for i in range(0, total_count, MAX_ENRICH_BATCH):
                batch_ids = all_ids[i:i + MAX_ENRICH_BATCH]
                backend.nodes.reset_for_rebuild(batch_ids)

                while True:
                    count = enrich_pending(
                        backend,
                        llm_client=llm_client,
                        embed_client=ec,
                        on_progress=_on_progress)
                    processed += count
                    if count == 0:
                        break

            bar.set_description('Done')
            bar.close()

            remaining = backend.nodes.count_pending_enrich()

            stats = {'processed': processed, 'remaining': remaining}
            backend.oplog.log(
                operation='rebuild', insight_id='',
                detail=json.dumps(stats))
            _json_out(stats)


@embed_grp.command('status')
@click.pass_context
def embed_status(ctx: click.Context) -> None:
    """Show the store's stored fingerprint, swap state, and whether the
    endpoint serves that fingerprint's model.

    Under per-store embedder sovereignty, the store's stored
    fingerprint is the source of truth -- there is no env-active
    fingerprint to compare against.
    """
    from memman.embed.swap import read_progress

    name = _resolve_store_name(ctx.obj['data_dir'], ctx.obj['store'])
    with _active_backend(ctx, unchecked=True) as backend:
        stored = fingerprint.stored_fingerprint(backend)
        progress = read_progress(backend)

    out: dict = {
        'stored': None if stored is None else {
            'model': stored.model,
            'dim': stored.dim,
            },
        }
    if stored is not None:
        ec = _ec_registry.get_for(stored.model)
        out['credentials_available'] = ec.available()
        if not ec.available():
            out['hint'] = (
                f'{ec.unavailable_message()}; or re-embed onto a served'
                f' model with `{fingerprint.swap_command(name)}`')
    else:
        out['hint'] = (
            'store has no embed fingerprint; run'
            f' `{fingerprint.swap_command(name)}` to record one.')
    if progress.state:
        out['swap'] = {
            'state': progress.state,
            'cursor': progress.cursor,
            'target_model': progress.target_model,
            'target_dim': progress.target_dim,
            }
    _json_out(out)


_REEMBED_BATCH = 50


def _reembed_one_store(
        sdir: str, ec: 'EmbeddingProvider', target: 'Fingerprint',
        dry_run: bool, bar: 'tqdm') -> dict:
    """Re-embed a single store with the active client.

    Walks all active insights, comparing each to `target`. Rows that
    already match are skipped. The per-row blob write and cursor
    advance is one transaction. The final fingerprint write, cursor
    reset and state=idle is another.

    Parameters
    ----------
    sdir : str
        Store directory; its name is the store name.
    ec : EmbeddingProvider
        Client that produces the new vectors.
    target : Fingerprint
        Fingerprint each row is compared against.
    dry_run : bool
        Count rows to re-embed without writing.
    bar : tqdm
        Progress bar to advance per row.

    Returns
    -------
    dict
        Keys `store`, `scanned`, and `reembedded`. A dry run returns
        `would_reembed` in place of `reembedded`.

    Raises
    ------
    click.ClickException
        Another reembed holds the store's lock.
    """
    store_name = pathlib.Path(sdir).name
    with open_db(sdir) as db:
        backend = SqliteBackend(db)
        with backend.reembed_lock('reembed') as held:
            if not held:
                raise click.ClickException(
                    f'another reembed is in progress on {store_name}')

            cur_state = backend.meta.get('embed_reembed_state')
            cursor = backend.meta.get('embed_reembed_cursor') or ''

            scanned = 0
            reembedded = 0

            if not dry_run and cur_state != 'in_progress':
                with backend.transaction():
                    backend.meta.set('embed_reembed_state', 'in_progress')
                    backend.meta.set('embed_reembed_cursor', '')
                cursor = ''

            bar.set_description(f'reembed {store_name}')

            while True:
                rows = iter_for_reembed(db, cursor, _REEMBED_BATCH)
                if not rows:
                    break

                for row_id, content, row_model, blob_len in rows:
                    scanned += 1
                    row_dim = (blob_len // 8) if blob_len else 0
                    matches = (
                        row_model == target.model
                        and row_dim == target.dim
                        and blob_len)
                    if not matches and not dry_run:
                        new_vec = ec.embed(content)
                        with backend.transaction():
                            backend.nodes.update_embedding(
                                row_id, new_vec, target.model)
                            backend.meta.set(
                                'embed_reembed_cursor', row_id)
                        reembedded += 1
                    elif not dry_run:
                        backend.meta.set(
                            'embed_reembed_cursor', row_id)
                    cursor = row_id
                    bar.update(1)

            if dry_run:
                return {
                    'store': store_name,
                    'scanned': scanned,
                    'would_reembed': reembedded,
                    }

            with backend.transaction():
                write_fingerprint(backend, target)
                backend.meta.set('embed_reembed_cursor', '')
                backend.meta.set('embed_reembed_state', 'idle')

            stats = {
                'store': store_name,
                'scanned': scanned,
                'reembedded': reembedded,
                }
            backend.oplog.log(
                operation='embed_reembed', insight_id='',
                detail=json.dumps(stats))
            return stats


@embed_grp.command('reembed')
@click.option(
    '--dry-run', is_flag=True, default=False,
    help='Count rows that would be re-embedded; no DB writes.')
@click.pass_context
def embed_reembed(ctx: click.Context, dry_run: bool) -> None:
    """Sweep every store with the active client; write fingerprints.

    Always global: iterates all stores under the configured
    data_dir. The active embed model is set by a single global env
    var, so a sweep necessarily applies to every store; per-store
    scoping is intentionally not supported. A branch moves only with
    its parent, so one over a Postgres parent is skipped.

    Three cases through one walk per store:
    1. Empty DB - zero rows; only the fingerprint is written.
    2. Existing DB on the same model - rows match; skip re-embed.
    3. Model change - rows mismatch; re-embed each.

    The sweep is resumable per store: progress is tracked in each
    store's `meta.embed_reembed_state` and `meta.embed_reembed_cursor`.
    A crash mid-sweep leaves state='in_progress'; re-running picks
    up from the cursor.
    """
    data_dir = ctx.obj['data_dir']
    active_name = _resolve_store_name(data_dir, ctx.obj['store'])
    if resolve_store_backend(active_name, data_dir) != 'sqlite':
        raise click.ClickException(
            'embed reembed is SQLite-only; Postgres reembed requires'
            ' a separate workflow (track in a follow-up issue)')

    if not dry_run:
        _require_stopped('reembed')

    ec = get_client()
    if not ec.available():
        raise click.ClickException(ec.unavailable_message())

    target = Fingerprint.from_client(ec)
    sqlite_names = [
        n for n in list_local_store_dirs(data_dir)
        if resolve_store_backend(n, data_dir) == 'sqlite']
    names = [
        n for n in sqlite_names
        if (info := branch_mod.read_branch_info(n, data_dir)) is None
        or info['parent'] in sqlite_names]

    grand_total = 0
    for name in names:
        with open_read_only(store_dir(data_dir, name)) as db:
            grand_total += count_active_insights(db)
    bar = tqdm(
        total=grand_total, desc='reembed', unit='row',
        file=sys.stderr, dynamic_ncols=True,
        disable=not sys.stderr.isatty())

    per_store = []
    total_scanned = 0
    total_reembedded = 0
    try:
        for name in names:
            sdir = store_dir(data_dir, name)
            result = _reembed_one_store(
                sdir, ec, target, dry_run, bar=bar)
            per_store.append(result)
            total_scanned += result.get('scanned', 0)
            total_reembedded += result.get(
                'reembedded' if not dry_run else 'would_reembed', 0)
            bar.set_postfix(
                reembedded=total_reembedded, refresh=False)
    finally:
        bar.close()

    out: dict = {
        'fingerprint': {
            'model': target.model,
            'dim': target.dim,
            },
        'stores': per_store,
        'total_scanned': total_scanned,
        }
    if dry_run:
        out['total_would_reembed'] = total_reembedded
        out['dry_run'] = 1
    else:
        out['total_reembedded'] = total_reembedded
    _json_out(out)


@embed_grp.command('swap')
@click.option(
    '--to', 'to_model', default='',
    help="Target embed model on the shared endpoint"
         " (e.g. 'voyageai/voyage-4-lite').")
@click.option(
    '--resume', 'resume', is_flag=True, default=False,
    help='Continue an in-flight swap from the recorded cursor.')
@click.option(
    '--abort', 'abort', is_flag=True, default=False,
    help='Discard the in-flight swap before cutover. Drops'
         ' embedding_pending and clears all swap meta. A swap at'
         ' cutover refuses the abort; --resume finishes it. Cutover is'
         ' one-way, so reverting requires running swap again with the'
         ' old model (full re-embed cost).')
@click.pass_context
def embed_swap(
        ctx: click.Context, to_model: str,
        resume: bool, abort: bool) -> None:
    """Online per-store swap to a new embed model.

    Postgres: shadow `embedding_pending vector(N)` column with HNSW
    built CONCURRENTLY, backfilled `WHERE embedding_pending IS NULL`,
    cut over in one transaction (drop + rename). SQLite: shadow
    `embedding_pending BLOB` column populated under
    `swap_lock()`, cutover is `update insights set
    embedding=embedding_pending, embedding_pending=null`. Recall
    keeps reading `embedding` throughout.

    Rollback note: cutover is one-way -- old embeddings are dropped.
    Reverting requires running swap again with the old model (full
    re-embed cost). Use `--abort` BEFORE cutover to discard the
    in-flight backfill safely.

    `MEMMAN_EMBED_SWAP_BATCH_SIZE` (default 200) tunes the HTTP
    batch size; `MEMMAN_EMBED_SWAP_INDEX_TIMEOUT` (default 0 =
    unlimited) caps the Postgres HNSW build.
    """
    from memman.embed.swap import SwapPlan, abort_swap, read_progress, run_swap

    if abort and resume:
        raise click.ClickException(
            '--abort and --resume are mutually exclusive')

    data_dir = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir, ctx.obj['store'])
    with factory.open_backend(name, data_dir) as backend:
        if abort:
            try:
                abort_swap(backend)
            except RuntimeError as exc:
                raise click.ClickException(f'store {name!r}: {exc}') from exc
            _json_out({'store': name, 'state': 'aborted'})
            return

        progress = read_progress(backend)
        if resume:
            if progress.state == '':
                raise click.ClickException(
                    f'no in-flight swap on store {name!r}')
            target_model = progress.target_model
            target_dim = progress.target_dim
        else:
            if progress.state not in {'', 'done'}:
                raise click.ClickException(
                    f'store {name!r} has an in-flight swap'
                    f' (state={progress.state}); use --resume or'
                    ' --abort')
            if not to_model:
                raise click.ClickException(
                    '--to <model> is required to start a new swap')
            target_model = to_model
            target_dim = 0

        ec_new = _ec_registry.get_for(target_model)
        if not ec_new.available():
            raise click.ClickException(ec_new.unavailable_message())
        if target_dim == 0:
            target_dim = ec_new.dim
        if target_dim <= 0:
            raise click.ClickException(
                f'failed to discover dim for {target_model}; the client'
                ' should expose dim after prepare()')

        plan = SwapPlan(target_model=target_model, target_dim=target_dim)

        with backend.swap_lock() as held:
            if not held:
                raise click.ClickException(
                    f'another swap is in progress on store {name!r}')
            _require_stopped('swap')
            progress = run_swap(backend, ec_new, plan)

        # Notes:
        # - Everything below runs after the one-way cutover, so a
        #   backend failure here must not read as a retryable swap.
        #   `run_swap` already wrote the fingerprint in one
        #   transaction with its own state cleanup, so this block is a
        #   confirming re-read. The generic seam message would read
        #   like a failure to connect before any data moved.
        # - Re-running the same swap is safe: `run_swap` returns DONE
        #   at once when the stored fingerprint already matches the
        #   target, so it re-embeds nothing.
        try:
            target_fp = Fingerprint(
                model=plan.target_model, dim=plan.target_dim)
            fp = fingerprint.stored_fingerprint(backend) or target_fp
            if fp != target_fp:
                write_fingerprint(backend, target_fp)
                fp = target_fp
        # Both types are reachable: the Postgres backend translates at
        # its connection scope, while the SQLite backend translates no
        # statement failure. The same read raises `BackendError` on
        # one and `sqlite3.Error` on the other.
        except (BackendError, sqlite3.Error) as exc:
            raise click.ClickException(
                f'store {name!r}: the vector cutover to'
                f' {plan.target_model} COMPLETED,'
                f' and only the fingerprint check after it failed:'
                f' {exc}. No data is pending and nothing was rolled'
                ' back. Re-run the same `memman embed swap --to` to'
                ' confirm the state; it re-embeds nothing when the'
                ' stored fingerprint already matches.') from exc

        _json_out({
            'store': name,
            'state': progress.state,
            'fingerprint': {
                'model': fp.model,
                'dim': fp.dim,
                },
            })
