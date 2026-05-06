"""Click CLI for memman.

This module is the entry point and argument-parsing surface only. Core
write-path orchestration lives in `memman.pipeline.remember`. Storage,
graph, search, embed, and LLM primitives live under their own packages.
"""

import json
import logging
import logging.handlers
import os
import pathlib
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

import click
import memman
from memman import config
from memman.store.db import default_data_dir, list_stores, open_db
from memman.store.db import read_active, store_dir, store_exists
from memman.store.db import valid_store_name, write_active
from memman.store.model import VALID_CATEGORIES, VALID_EDGE_TYPES, Edge
from memman.store.model import Insight, format_timestamp, is_immune
from memman.store.sqlite import open_ro_db
from tqdm import tqdm

logger = logging.getLogger('memman')

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
_WORKER_LOG_MAX_BYTES = 5 * 1024 * 1024
_WORKER_LOG_BACKUPS = 3


def _configure_logging(data_dir: str, verbose: bool, debug: bool) -> None:
    """Configure the memman logger once per process.

    Runs on every CLI invocation including `memman install` (before
    the env file exists). The literal `'WARNING'` fall-through must
    equal `INSTALL_DEFAULTS[LOG_LEVEL]`; a unit test enforces that.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        raw = config.get(config.LOG_LEVEL) or 'WARNING'
        level = getattr(logging, raw.upper(), logging.WARNING)

    logger.setLevel(level)

    has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, '_memman', False)
        for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT))
        sh._memman = True
        logger.addHandler(sh)

    if config.is_worker():
        has_file = any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            and getattr(h, '_memman', False)
            for h in logger.handlers)
        if not has_file:
            log_dir = pathlib.Path(data_dir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / 'memman.log',
                maxBytes=_WORKER_LOG_MAX_BYTES,
                backupCount=_WORKER_LOG_BACKUPS)
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            fh._memman = True
            logger.addHandler(fh)


def _json_out(obj: object) -> None:
    """Write JSON to stdout with 2-space indent, sorted keys."""
    click.echo(json.dumps(obj, indent=2, sort_keys=True))


def _require_started(action: str) -> None:
    """Reject the current CLI invocation when the scheduler is stopped.

    Single gate for write-producing commands. When the scheduler is
    stopped, memman is recall-only — every write returns exit 1 with a
    fixed message that points the operator at `memman scheduler start`.
    """
    from memman.setup.scheduler import STATE_STOPPED, read_state
    if read_state() == STATE_STOPPED:
        raise click.ClickException(
            f"Scheduler is stopped; cannot {action}."
            " Run 'memman scheduler start' to enable.")


def _require_stopped(action: str) -> None:
    """Reject the current CLI invocation when the scheduler is started.

    Inverse of `_require_started`. Used by `memman embed reembed`,
    which cannot run while the worker may be claiming queued
    `remember` rows mid-sweep.
    """
    from memman.setup.scheduler import STATE_STOPPED, read_state
    if read_state() != STATE_STOPPED:
        raise click.ClickException(
            f"Scheduler is started; cannot {action}."
            " Run 'memman scheduler stop' first.")


def _resolve_store_name(data_dir: str, store_flag: str) -> str:
    """Resolve effective store name."""
    if store_flag:
        return store_flag
    env = os.environ.get(config.STORE, '')
    if env:
        return env
    return read_active(data_dir)


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
            updates[config.PG_DSN_FOR(store_name)] = default_dsn
    _write_env_keys_with_flock(updates, data_dir=data_dir)


def _get_llm_client_or_fail(role: str):
    """Return a per-role LLM client, re-wrapping ConfigError as ClickException.

    Keeps `memman.llm` free of `click` — the CLI boundary is the only
    place that should know how to surface a user-facing config error.
    `role` is `'fast'`, `'slow_canonical'`, or `'slow_metadata'` (worker pipeline,
    operator rebuilds).
    """
    from memman.exceptions import ConfigError
    from memman.llm.client import get_llm_client
    try:
        return get_llm_client(role)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _open_db(ctx: click.Context) -> 'DB':
    """Open the database using context options."""
    data_dir = ctx.obj['data_dir']
    store_flag = ctx.obj['store']
    read_only = ctx.obj['readonly']

    name = _resolve_store_name(data_dir, store_flag)
    sdir = store_dir(data_dir, name)

    if read_only:
        return open_ro_db(sdir)

    db = open_db(sdir)
    from memman.embed import get_client
    from memman.embed.fingerprint import assert_consistent, seed_if_fresh
    from memman.exceptions import ConfigError, EmbedFingerprintError
    from memman.graph.engine import reindex_if_constants_changed
    from memman.store.sqlite import SqliteBackend
    backend = SqliteBackend(db)
    reindex_if_constants_changed(backend)
    try:
        ec = get_client()
        seed_if_fresh(backend, ec)
        assert_consistent(backend, ec)
    except (EmbedFingerprintError, ConfigError) as exc:
        db.close()
        raise click.ClickException(str(exc)) from exc
    return db


def _active_backend(ctx: click.Context, *, unchecked: bool = False):
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
    return active_store(data_dir=data_dir, store=name, unchecked=unchecked)


def _parse_since(since: str) -> str:
    """Parse a relative time string (e.g. '7d', '24h') to ISO timestamp."""
    m = re.match(r'^(\d+)([dhm])$', since)
    if not m:
        raise click.ClickException(
            f'Invalid --since format: {since} (use e.g. 7d, 24h, 30m)')
    val, unit = int(m.group(1)), m.group(2)
    delta = {'d': timedelta(days=val), 'h': timedelta(hours=val),
             'm': timedelta(minutes=val)}[unit]
    cutoff = datetime.now(timezone.utc) - delta
    return format_timestamp(cutoff)


def _insight_to_dict(i: Insight) -> dict:
    """Serialize an Insight for JSON output."""
    d = {
        'id': i.id,
        'content': i.content,
        'category': i.category,
        'importance': i.importance,
        'entities': i.entities,
        'source': i.source,
        'access_count': i.access_count,
        'created_at': format_timestamp(i.created_at),
        'updated_at': format_timestamp(i.updated_at),
        }
    if i.deleted_at:
        d['deleted_at'] = format_timestamp(i.deleted_at)
    if i.summary:
        d['summary'] = i.summary
    return d


def _parse_entities(entities: str) -> list[str]:
    """Parse and validate comma-separated entities."""
    entity_list: list[str] = []
    if not entities:
        return entity_list
    for e in entities.split(','):
        e = e.strip()
        if e:
            if len(e) > 200:
                raise click.ClickException(
                    f'entity too long ({len(e)} chars, max 200):'
                    f' {e[:50]}')
            entity_list.append(e)
    if len(entity_list) > 50:
        raise click.ClickException(
            f'too many entities ({len(entity_list)}, max 50)')
    return entity_list


@click.group()
@click.version_option(version=memman.__version__, prog_name='memman')
@click.option('--data-dir', default=None, help='Base data directory (env: MEMMAN_DATA_DIR)')
@click.option('--store', 'store_name', default='', help='Named memory store')
@click.option('--readonly', is_flag=True, default=False, help='Open database in read-only mode')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='INFO-level logging to stderr')
@click.option('--debug', is_flag=True, default=False,
              help='DEBUG-level logging to stderr (overrides --verbose)')
@click.pass_context
def cli(ctx: click.Context, data_dir: str | None, store_name: str,
        readonly: bool, verbose: bool, debug: bool) -> None:
    """Persistent memory store for LLM agents."""
    if data_dir is None:
        data_dir = os.environ.get(config.DATA_DIR, default_data_dir())
    _configure_logging(data_dir, verbose, debug)
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    ctx.obj['store'] = store_name
    ctx.obj['readonly'] = readonly


@cli.group()
def graph() -> None:
    """Graph operations on insights and edges."""


@cli.group(name='embed')
def embed_grp() -> None:
    """Embed-provider operations: status, re-embed on swap."""


@cli.group(no_args_is_help=True)
def scheduler() -> None:
    """Async write pipeline: scheduler state, queue, worker logs."""


@scheduler.group('queue', invoke_without_command=True)
@click.pass_context
def queue(ctx: click.Context) -> None:
    """Inspect and manage the deferred-write queue."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(queue_list)


@cli.group()
def insights() -> None:
    """Operations on stored insights (read, prune, protect)."""


@cli.group()
def log() -> None:
    """View memman logs (operation audit + worker output)."""


@cli.group(name='config')
def config_cmd() -> None:
    """Inspect and modify persisted memman settings."""


@config_cmd.command('set')
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Write `KEY=VALUE` to the env file, bypassing the install seed model.

    Use this to change an INSTALLABLE_KEYS value after initial install
    -- switching backends, rotating an API key, updating a DSN. Install
    flags remain sticky-seed (they never override an existing file
    value); `config set` is the explicit override path.
    """
    if key not in config.INSTALLABLE_KEYS:
        raise click.ClickException(
            f'{key!r} is not in INSTALLABLE_KEYS; only persistable'
            ' settings can be changed via `config set`')
    from memman.setup.scheduler import _write_env_keys
    data_dir = ctx.obj['data_dir']
    _write_env_keys({key: value}, data_dir=data_dir)
    config.reset_file_cache()
    click.echo(f'set {key} in {config.env_file_path(data_dir)}')


@config_cmd.command('migrate-keys', hidden=True)
@click.pass_context
def config_migrate_keys(ctx: click.Context) -> None:
    """Convert legacy `MEMMAN_BACKEND` / `MEMMAN_PG_DSN` to per-store keys.

    Idempotent. Re-running on a fully-converted install is a no-op.
    Operator-edited per-store keys are not overwritten.
    """
    from memman.setup.per_store_bootstrap import bootstrap_per_store_keys
    data_dir = ctx.obj['data_dir']
    actions = bootstrap_per_store_keys(data_dir)
    if not actions:
        click.echo('no legacy keys found; already on per-store shape')
        return
    for line in actions:
        click.echo(line)


@config_cmd.command('show')
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Dump effective config: env vars + on-disk files + scheduler state."""
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
        if key.startswith('MEMMAN_BACKEND_'):
            per_store[key] = value
        elif key.startswith('MEMMAN_PG_DSN_'):
            per_store[key] = '***REDACTED***'
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
    try:
        from memman.setup.scheduler import status as scheduler_status_fn
        s = scheduler_status_fn()
        out['scheduler'] = {
            'state': s.get('state'),
            'installed': s.get('installed'),
            'platform': s.get('platform'),
            'interval_seconds': s.get('interval_seconds'),
            }
    except Exception:
        pass
    _json_out(out)


@cli.command()
@click.argument('content', nargs=-1, required=True)
@click.option('--cat', default='general', help='Category')
@click.option('--imp', default=3, type=int, help='Importance (1-5)')
@click.option('--source', default='user', help='Source')
@click.option('--entities', default='', help='Comma-separated entities')
@click.option('--no-reconcile', is_flag=True, default=False,
              help='Skip LLM reconciliation')
@click.pass_context
def remember(ctx: click.Context, content: tuple[str, ...], cat: str,
             imp: int, source: str, entities: str,
             no_reconcile: bool) -> None:
    """Store a new insight via the queue.

    Always enqueues. The worker drains the queue (under systemd/launchd
    on a host, or in-process when the trigger is inline). Rejected when
    the scheduler is stopped (memman is recall-only in that state).
    """
    _require_started('write')
    content_str = ' '.join(content)
    content_bytes = len(content_str.encode('utf-8'))
    if content_bytes > 8000:
        raise click.ClickException(
            f'content too long ({content_bytes} bytes, max 8000);'
            ' consider chunking into multiple remember calls')

    if cat not in VALID_CATEGORIES:
        raise click.ClickException(
            f'invalid category {cat!r}; valid: preference, decision,'
            ' fact, insight, context, general')
    if imp < 1 or imp > 5:
        raise click.ClickException(
            f'importance must be 1-5, got {imp}')

    from memman.search.quality import check_content_quality
    quality_warnings = check_content_quality(content_str)

    data_dir_val = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir_val, ctx.obj['store'])

    from memman.queue import enqueue, open_queue_db
    conn = open_queue_db(data_dir_val)
    try:
        cat_hint = cat if cat != 'general' else None
        imp_hint = imp if imp != 3 else None
        row_id = enqueue(
            conn, store=name, content=content_str,
            hint_cat=cat_hint, hint_imp=imp_hint,
            hint_source=source if source != 'user' else None,
            hint_entities=entities or None,
            hint_no_reconcile=no_reconcile,
            priority=0)
    finally:
        conn.close()
    _json_out({
        'action': 'queued',
        'queue_id': row_id,
        'store': name,
        'quality_warnings': quality_warnings,
        })


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
    """Return whether a stop has been signaled."""
    return _STOP_REQUESTED


def _reset_stop_for_tests() -> None:
    """Test-only: clear the stop flag between in-process serve invocations."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


_LAST_HEARTBEAT_AT: dict[str, float] = {}
HEARTBEAT_MIN_INTERVAL_SECONDS = 60


def _reset_heartbeat_state() -> None:
    """Test-only: clear the heartbeat tracking dict.

    `_LAST_HEARTBEAT_AT` is module-level. In-process CliRunner tests
    share it across test invocations, which can cause cross-test
    contamination if data_dir paths are reused. Tests reset between
    invocations via an autouse conftest fixture.
    """
    _LAST_HEARTBEAT_AT.clear()


def _start_postgres_heartbeat(record_run: bool):
    """Start a postgres-side worker_runs row when MEMMAN_BACKEND=postgres.

    Returns `(queue_backend, run_id)` tuple, both `None` on sqlite mode
    or when something goes wrong (postgres heartbeat is best-effort
    monitoring infrastructure, NOT correctness — failures are
    logged but never abort the drain).
    """
    if not record_run:
        return None, None
    try:
        from memman import config
        backend_name = (config.get(config.BACKEND) or 'sqlite').lower()
        if backend_name != 'postgres':
            return None, None
        dsn = config.get(config.PG_DSN)
        if not dsn:
            return None, None
        from memman.store.postgres import PostgresQueueBackend
        pg_queue = PostgresQueueBackend(dsn=dsn)
        pg_run_id = pg_queue.start_run()
        return pg_queue, pg_run_id
    except Exception:
        logger.exception('postgres heartbeat init failed; continuing')
        return None, None


def _beat_postgres_heartbeat(pg_queue, pg_run_id) -> None:
    """Advance last_heartbeat_at on the postgres worker_runs row.

    No-op when sqlite mode or when the postgres heartbeat init failed
    (both yield `pg_queue=None`). Best-effort: failures are logged but
    do not abort the drain row.
    """
    if pg_queue is None or pg_run_id is None:
        return
    try:
        pg_queue.beat_run(pg_run_id)
    except Exception:
        logger.exception(
            'postgres beat_run failed; continuing without heartbeat')


@scheduler.command('drain', hidden=True)
@click.option('--pending', is_flag=True, default=False,
              help='Drain the deferred-write queue')
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
def scheduler_drain(ctx: click.Context, pending: bool, limit: int,
                    timeout: int, stores: str, progress: bool,
                    trace: bool) -> None:
    """Run the worker drain loop. Hidden: invoked by the systemd/launchd
    unit's ExecStart. Operators should use `scheduler trigger` to kick
    the unit, or `scheduler queue list` to inspect pending rows.
    """
    if not pending:
        raise click.ClickException(
            'only --pending mode is supported; pass --pending explicitly')
    if trace:
        os.environ[config.DEBUG] = '1'
    _drain_queue(ctx, limit, timeout, stores, progress)


@scheduler.command('serve')
@click.option('--interval', default=60, type=int,
              help='Seconds between drain iterations (default 60)')
@click.option('--once', is_flag=True, default=False,
              help='Run a single drain pass and exit')
@click.pass_context
def scheduler_serve(ctx: click.Context, interval: int, once: bool) -> None:
    """Run the drain loop continuously as a long-lived process.

    Used as PID 1 in containers and by hosts where systemd/launchd are
    not available (set MEMMAN_SCHEDULER_KIND=serve). On SIGTERM/SIGINT
    the current drain finishes (bounded by the per-iteration timeout)
    and the process exits 0.
    """
    import signal
    import socket
    import sys as _sys
    import time as _time

    from memman import __version__ as _memman_version
    from memman import trace
    from memman.queue import mark_stale_on_resume, open_queue_db
    from memman.setup.scheduler import STATE_STOPPED, clear_serve_interval
    from memman.setup.scheduler import read_state, write_serve_interval

    if interval < 0:
        raise click.ClickException('--interval must be >= 0')

    os.environ[config.WORKER] = '1'
    _reset_stop_for_tests()

    def _handle_stop(signum, frame):
        logger.info(
            f'scheduler serve: caught signal {signum}, finishing drain')
        _request_stop()

    prior_term = signal.signal(signal.SIGTERM, _handle_stop)
    prior_int = signal.signal(signal.SIGINT, _handle_stop)
    try:
        write_serve_interval(interval)

        data_dir_val = ctx.obj['data_dir']
        conn = open_queue_db(data_dir_val)
        try:
            reclaimed = mark_stale_on_resume(conn)
            if reclaimed:
                logger.info(
                    f'scheduler serve: reclaimed {reclaimed} stale rows')
        finally:
            conn.close()

        trace.setup()
        trace.event(
            'scheduler_serve_start',
            pid=os.getpid(),
            hostname=socket.gethostname(),
            python=_sys.version.split()[0],
            memman_version=_memman_version,
            interval=interval,
            once=once)

        per_drain_timeout = max(10, interval - 10) if interval > 0 else 300

        while True:
            config.reset_file_cache()
            if read_state() == STATE_STOPPED:
                logger.info('scheduler serve: state=STOPPED, exiting')
                break
            result = _drain_queue(
                ctx, limit=100, timeout=per_drain_timeout,
                stores_filter='', verbose=False)
            if _stop_requested() or once:
                break
            if interval > 0:
                slept = 0.0
                while slept < interval and not _stop_requested():
                    if read_state() == STATE_STOPPED:
                        break
                    _time.sleep(min(1.0, interval - slept))
                    slept += 1.0
            elif result and result.get('claimed', 0) == 0:
                _time.sleep(0.1)

        trace.event('scheduler_serve_stop', pid=os.getpid())
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_int)
        try:
            clear_serve_interval()
        except OSError:
            pass


def _drain_queue(ctx: click.Context, limit: int, timeout: int,
                 stores_filter: str, verbose: bool) -> dict | None:
    """Claim and process queue rows until limit, timeout, or empty.

    Returns a status dict `{claimed, processed, failed}` so callers
    can detect empty drains. Returns None if the drain was skipped
    because another drain is already running.
    """
    import socket
    import sys as _sys
    import time as _time

    from memman import __version__ as _memman_version
    from memman import trace
    from memman.drain_lock import DrainLockBusy, acquire, release
    from memman.queue import claim, finish_worker_run, mark_done, mark_failed
    from memman.queue import open_queue_db, queue_db_path, start_worker_run
    from memman.queue import stats
    from memman.setup.scheduler import STATE_STOPPED, read_state

    data_dir_val = ctx.obj['data_dir']
    worker_pid = os.getpid()
    deadline = _time.monotonic() + timeout
    store_list = [s.strip() for s in stores_filter.split(',') if s.strip()]

    trace.setup()
    trace.event(
        'scheduler_fired',
        pid=worker_pid,
        hostname=socket.gethostname(),
        python=_sys.version.split()[0],
        memman_version=_memman_version,
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
            'skipped': 'another drain in progress',
            })
        return None

    try:
        trace.event(
            'drain_start',
            data_dir=data_dir_val,
            queue_db_path=queue_db_path(data_dir_val),
            limit=limit,
            timeout=timeout,
            stores=store_list)

        from concurrent.futures import ThreadPoolExecutor

        conn = open_queue_db(data_dir_val)
        processed = 0
        failed = 0
        claimed = 0
        touched_stores: set[str] = set()
        store_contexts: dict[str, _StoreContext] = {}
        executor = ThreadPoolExecutor(max_workers=2)
        run_error: str | None = None

        last_hb = _LAST_HEARTBEAT_AT.get(data_dir_val, 0.0)
        record_run = (_time.monotonic() - last_hb) >= HEARTBEAT_MIN_INTERVAL_SECONDS
        run_id = start_worker_run(conn, worker_pid) if record_run else None

        pg_queue, pg_run_id = _start_postgres_heartbeat(record_run)
    except Exception:
        release(lock_fd)
        raise

    try:
        while processed + failed < limit:
            if _stop_requested() or read_state() == STATE_STOPPED:
                logger.info('drain: stop requested, exiting loop')
                trace.event('drain_stop_requested')
                break
            if _time.monotonic() >= deadline:
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
                priority=row.priority,
                attempts=row.attempts,
                content_len=len(row.content),
                hint_cat=row.hint_cat,
                hint_imp=row.hint_imp,
                hint_source=row.hint_source,
                hint_entities=row.hint_entities)

            ctx = store_contexts.get(row.store)
            if ctx is None:
                try:
                    ctx = _StoreContext(row.store, data_dir_val)
                except Exception as exc:
                    mark_failed(
                        conn, row.id, f'{type(exc).__name__}: {exc}')
                    failed += 1
                    trace.event(
                        'queue_failed',
                        row_id=row.id,
                        store=row.store,
                        error_class=type(exc).__name__,
                        error_message=str(exc)[:500])
                    logger.exception(
                        f'enrich row {row.id} failed during store open')
                    continue
                store_contexts[row.store] = ctx

            embed_snap, insights_snap = ctx.snapshot_caches()
            try:
                row_t0 = _time.monotonic()
                _process_queue_row(row, ctx, executor)
                row_elapsed_ms = int((_time.monotonic() - row_t0) * 1000)
                mark_done(conn, row.id)
                processed += 1
                touched_stores.add(row.store)
                _beat_postgres_heartbeat(pg_queue, pg_run_id)
                trace.event(
                    'queue_done',
                    row_id=row.id,
                    store=row.store,
                    elapsed_ms=row_elapsed_ms)
                if verbose:
                    click.echo(
                        f'[enrich] done id={row.id} store={row.store}',
                        err=True)
            except Exception as exc:
                ctx.restore_caches(embed_snap, insights_snap)
                mark_failed(conn, row.id, f'{type(exc).__name__}: {exc}')
                failed += 1
                trace.event(
                    'queue_failed',
                    row_id=row.id,
                    store=row.store,
                    error_class=type(exc).__name__,
                    error_message=str(exc)[:500])
                from memman.exceptions import EmbedCredentialError
                if isinstance(exc, EmbedCredentialError):
                    trace.event(
                        'embedder_credential_missing',
                        row_id=row.id,
                        store=row.store,
                        provider=getattr(ctx.ec, 'name', None),
                        model=getattr(ctx.ec, 'model', None),
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
        executor.shutdown(wait=True)
        try:
            from memman.maintenance import run_maintenance
            run_maintenance(
                conn, data_dir_val, touched_stores,
                store_contexts, deadline,
                _write_recall_snapshot_for_store)
        except Exception:
            logger.exception('drain maintenance phase failed')
        for ctx in store_contexts.values():
            ctx.close()
        if processed > 0:
            record_run = True
        if record_run:
            try:
                if run_id is None:
                    run_id = start_worker_run(conn, worker_pid)
                finish_worker_run(
                    conn, run_id, claimed, processed, failed,
                    error=run_error)
                _LAST_HEARTBEAT_AT[data_dir_val] = _time.monotonic()
            except Exception:
                logger.exception('failed to stamp worker_runs finish row')
        s = stats(conn)
        conn.close()
        release(lock_fd)

    trace.event(
        'drain_end',
        processed=processed,
        failed=failed,
        remaining=s)
    _json_out({
        'processed': processed,
        'failed': failed,
        'remaining': s,
        })
    return {'claimed': claimed, 'processed': processed, 'failed': failed}


def _write_recall_snapshot_for_store(
        data_dir_val: str, store_name: str) -> None:
    """Materialize the recall snapshot for one store after a successful drain.

    SQLite-only: Postgres reads via live `recall_session` queries and
    has no on-disk snapshot file. Snapshot writes are idempotent and
    bounded (cap at 1000 active insights). Failures here are isolated
    per store and never abort the drain or cause queue rows to retry.
    """
    from memman.store.factory import _resolve_store_backend
    backend_name = _resolve_store_backend(store_name, data_dir_val)
    if backend_name != 'sqlite':
        return

    from memman.embed.fingerprint import active_fingerprint
    from memman.store.db import open_db as _open_store_db
    from memman.store.sqlite import SqliteBackend

    sdir = store_dir(data_dir_val, store_name)
    db = _open_store_db(sdir)
    try:
        SqliteBackend(db).write_snapshot(active_fingerprint())
    finally:
        db.close()


class _StoreContext:
    """Per-store drain-scope state hoisted out of the row loop.

    One context per store touched in a drain: the open store DB
    connection, the embedding+insight cache (built lazily on first
    use), and the slow-role LLM client + embed client. Reused across
    every row that targets the same store so that scans and HTTP
    setup amortize.
    """

    def __init__(self, store_name: str, data_dir: str) -> None:
        from memman.embed import fingerprint as _fp_mod
        from memman.embed import registry as _ec_registry
        from memman.exceptions import EmbedFingerprintError
        from memman.llm.client import get_llm_client
        from memman.store.factory import open_backend

        self.store_name = store_name
        self.data_dir = data_dir
        from memman.embed import get_client
        _ensure_store_backend_key(store_name, data_dir)
        self.backend = open_backend(store_name, data_dir)
        _fp_mod.seed_if_fresh(self.backend, get_client())
        stored = _fp_mod.stored_fingerprint(self.backend)
        if stored is None:
            raise EmbedFingerprintError(
                f"store {store_name!r} has no embed fingerprint and"
                " contains data; run 'memman embed reembed' to converge.")
        self.ec = _ec_registry.get_for(stored.provider, stored.model)
        self._stored_fp = stored
        self.llm_client = get_llm_client('slow_canonical')
        self.embed_cache: dict[str, list[float]] = dict(
            self.backend.nodes.iter_embeddings_as_vecs())
        self.insights_by_id = {
            i.id: i for i in self.backend.nodes.get_all_active()}

    def assert_fingerprint_unchanged(self) -> None:
        """Raise EmbedFingerprintError if the store's stored fingerprint
        diverged from the value captured at context construction.

        Per-row heartbeat: a swap that completes mid-drain would
        otherwise let the cached `ec` write vectors of the wrong dim.
        Callers must invoke this before every embed call inside a
        long-running drain loop.
        """
        from memman.embed import fingerprint as _fp_mod
        from memman.exceptions import EmbedFingerprintError
        current = _fp_mod.stored_fingerprint(self.backend)
        if current != self._stored_fp:
            raise EmbedFingerprintError(
                f'store {self.store_name!r} fingerprint changed during'
                f' drain: was {self._stored_fp.provider}:'
                f'{self._stored_fp.model}:{self._stored_fp.dim},'
                f' now {current.provider if current else None}:'
                f'{current.model if current else None}:'
                f'{current.dim if current else None};'
                ' row released for retry.')

    def snapshot_caches(self) -> tuple[dict, dict]:
        """Return shallow copies of the caches for rollback."""
        return dict(self.embed_cache), dict(self.insights_by_id)

    def restore_caches(self, embed: dict, insights: dict) -> None:
        """Restore caches to a prior snapshot after a failed row."""
        self.embed_cache.clear()
        self.embed_cache.update(embed)
        self.insights_by_id.clear()
        self.insights_by_id.update(insights)

    def close(self) -> None:
        """Close the active Backend's underlying connection."""
        try:
            self.backend.close()
        except Exception:
            logger.exception(
                f'failed closing backend for store {self.store_name!r}')


def _process_queue_row(
        row: 'memman.queue.QueueRow',
        ctx: _StoreContext,
        executor: 'ThreadPoolExecutor') -> None:
    """Run the full remember pipeline on a claimed queue row.

    The insight's `source` is set to `row.hint_source` when provided
    (so the user's `--source` flag survives the queue), falling back
    to `queue:<row.id>`. Crash-recovery idempotency is enforced only
    when `hint_source` is absent, via a `source=queue:<id>` lookup.

    Hoisted state (db, embed_cache, insights_by_id, llm_client, ec,
    executor) comes from `ctx` and the drain-level executor. The
    drain loop snapshots and restores `ctx`'s caches around this call
    so a transaction failure can't pollute the next row's planning.
    """
    from memman import trace as _trace

    ctx.assert_fingerprint_unchanged()

    entity_list = _parse_entities(row.hint_entities or '')
    category = row.hint_cat or 'general'
    importance = row.hint_imp if row.hint_imp is not None else 3
    source = row.hint_source or f'queue:{row.id}'

    if category not in VALID_CATEGORIES:
        category = 'general'
    if importance < 1 or importance > 5:
        importance = 3

    backend = ctx.backend

    _trace.event(
        'process_row',
        row_id=row.id,
        store=row.store,
        data_dir=ctx.data_dir,
        source=source,
        category=category,
        importance=importance)

    if (row.hint_source is None
            and backend.nodes.has_active_with_source(source)):
        logger.info(
            f'queue row {row.id} already committed to store'
            f' {row.store!r}; skipping re-processing')
        _trace.event(
            'process_row_skipped',
            row_id=row.id,
            reason='already_committed')
        return

    now = datetime.now(timezone.utc)
    access_count = 0
    if row.hint_replaced_id:
        old = backend.nodes.get(row.hint_replaced_id)
        if old is not None:
            access_count = old.access_count
    insight = Insight(
        id=str(uuid.uuid4()), content=row.content,
        category=category, importance=importance,
        entities=entity_list, source=source,
        access_count=access_count,
        created_at=now, updated_at=now)

    from memman.pipeline.remember import run_remember
    result = run_remember(
        backend, insight, row.content,
        no_reconcile=row.hint_no_reconcile or bool(row.hint_replaced_id),
        replaced_id=row.hint_replaced_id or '',
        cat_explicit=row.hint_cat is not None,
        imp_explicit=row.hint_imp is not None,
        embed_cache=ctx.embed_cache,
        insights_by_id=ctx.insights_by_id,
        executor=executor,
        llm_client=ctx.llm_client,
        ec=ctx.ec)
    _json_out(result)


@cli.command()
@click.argument('keyword', nargs=-1, required=True)
@click.option('--cat', default='', help='Filter by category')
@click.option('--limit', default=10, type=int, help='Max results')
@click.option('--source', default='', help='Filter by source')
@click.option('--basic', is_flag=True, default=False, help='Simple SQL LIKE matching')
@click.option('--intent', default='', help='Override intent')
@click.option('--expand', 'expand', is_flag=True, default=False,
              help='Run LLM query expansion before retrieval (off by default)')
@click.option('--rerank', 'rerank', is_flag=True, default=False,
              help='Apply Voyage cross-encoder reranker on a 100-doc '
                   'shortlist; auto-skipped on queries of 2 tokens or fewer')
@click.pass_context
def recall(ctx: click.Context, keyword: tuple[str, ...], cat: str,
           limit: int, source: str, basic: bool,
           intent: str, expand: bool, rerank: bool) -> None:
    """Retrieve insights by keyword."""
    from memman.embed import get_client
    from memman.llm.extract import expand_query
    from memman.search.intent import intent_from_string
    from memman.search.recall import intent_aware_recall
    keyword_str = ' '.join(keyword)
    with _active_backend(ctx) as backend:
        if basic:
            results = backend.nodes.query(
                keyword=keyword_str, category=cat,
                source=source, limit=limit)
            with backend.transaction():
                for r in results:
                    backend.nodes.increment_access_count(r.id)
                    r.access_count += 1
                backend.oplog.log(
                    operation='recall:basic', insight_id='',
                    detail=f'q={keyword_str} hits={len(results)}')
            _json_out({
                'results': [_insight_to_dict(r) for r in results],
                'meta': {'basic': True},
                })
            return

        expansion: dict = {}
        if expand:
            llm_client = _get_llm_client_or_fail('fast')
            expansion = expand_query(llm_client, keyword_str)
            keyword_str = expansion['expanded_query']

        intent_override = None
        if intent:
            try:
                intent_override = intent_from_string(intent)
            except ValueError as e:
                raise click.ClickException(str(e))
        elif expansion.get('intent'):
            try:
                intent_override = intent_from_string(
                    expansion['intent'])
            except ValueError:
                pass

        ec = get_client()
        query_vec = None
        try:
            query_vec = ec.embed(keyword_str)
        except Exception:
            pass

        query_entities = list(expansion.get('entities', []))

        fetch_limit = limit * 3 if (cat or source) else limit
        resp = intent_aware_recall(
            backend, keyword_str, query_vec, query_entities,
            fetch_limit, intent_override, rerank=rerank)
        if cat:
            resp['results'] = [
                r for r in resp['results']
                if r['insight'].category == cat][:limit]
        if source:
            resp['results'] = [
                r for r in resp['results']
                if r['insight'].source == source][:limit]

        hits = [{'id': r['insight'].id[:8], 'via': r.get('via', ''),
                 'score': round(r['score'], 3),
                 'kw': round(r['signals']['keyword'], 3),
                 'sim': round(r['signals']['similarity'], 3),
                 'gr': round(r['signals']['graph'], 3),
                 'ent': round(r['signals']['entity'], 3)}
                for r in resp['results']]
        with backend.transaction():
            for r in resp['results']:
                backend.nodes.increment_access_count(r['insight'].id)
                r['insight'].access_count += 1
            backend.oplog.log(
                operation='recall-detail', insight_id='',
                detail=json.dumps({'intent': resp['meta']['intent'],
                                   'q': keyword_str[:80], 'hits': hits}))

        out = {
            'results': [
                {
                    'insight': _insight_to_dict(r['insight']),
                    'score': r['score'],
                    'intent': r['intent'],
                    'signals': r['signals'],
                    **({'via': r['via']} if r.get('via') else {}),
                    }
                for r in resp['results']
                ],
            'meta': resp['meta'],
            }
        _json_out(out)


def _forget_insight(backend, id: str) -> None:
    """Soft-delete `id` and write a forget oplog row carrying `before`.
    """
    from memman.store.model import insight_to_delta_dict
    with backend.transaction():
        before_ins = backend.nodes.get_include_deleted(id)
        before_delta = (
            insight_to_delta_dict(before_ins)
            if before_ins is not None else None)
        deleted = backend.nodes.soft_delete(id, tolerate_missing=True)
        if not deleted:
            raise click.ClickException(f'insight {id!r} not found')
        backend.oplog.log(
            operation='forget', insight_id=id, detail='',
            before=before_delta)


@cli.command()
@click.argument('id')
@click.pass_context
def forget(ctx: click.Context, id: str) -> None:
    """Soft-delete an insight. Rejected when the scheduler is stopped."""
    _require_started('write')

    with _active_backend(ctx) as backend:
        _forget_insight(backend, id)
        _json_out({
            'id': id,
            'status': 'deleted',
            'message': 'Insight soft-deleted successfully',
            })


@cli.command()
@click.argument('id')
@click.argument('content', nargs=-1, required=True)
@click.option('--cat', default='general', help='Category')
@click.option('--imp', default=3, type=int, help='Importance (1-5)')
@click.option('--source', default='user', help='Source')
@click.option('--entities', default='', help='Comma-separated entities')
@click.option('--reconcile/--no-reconcile', 'reconcile', default=False,
              help=('Run LLM reconciliation against existing insights.'
                    ' Default: skip — replace targets a specific id.'))
@click.pass_context
def replace(ctx: click.Context, id: str, content: tuple[str, ...],
            cat: str, imp: int, source: str,
            entities: str, reconcile: bool) -> None:
    """Replace an insight by ID with new content via the queue."""
    _require_started('write')

    content_str = ' '.join(content)
    content_bytes = len(content_str.encode('utf-8'))
    if content_bytes > 8000:
        raise click.ClickException(
            f'content too long ({content_bytes} bytes, max 8000);'
            ' consider chunking into multiple remember calls')

    if cat not in VALID_CATEGORIES:
        raise click.ClickException(
            f'invalid category {cat!r}; valid: preference, decision,'
            ' fact, insight, context, general')
    if imp < 1 or imp > 5:
        raise click.ClickException(
            f'importance must be 1-5, got {imp}')

    from memman.search.quality import check_content_quality
    quality_warnings = check_content_quality(content_str)

    data_dir_val = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir_val, ctx.obj['store'])

    with _active_backend(ctx) as backend:
        old = backend.nodes.get(id)
    if old is None:
        raise click.ClickException(
            f'insight {id} not found or already deleted')

    cat_src = ctx.get_parameter_source('cat')
    imp_src = ctx.get_parameter_source('imp')
    source_src = ctx.get_parameter_source('source')
    entities_src = ctx.get_parameter_source('entities')
    if cat_src != click.core.ParameterSource.COMMANDLINE:
        cat = old.category
    if imp_src != click.core.ParameterSource.COMMANDLINE:
        imp = old.importance
    source_explicit = source_src == click.core.ParameterSource.COMMANDLINE
    if not source_explicit:
        source = old.source
    if entities_src != click.core.ParameterSource.COMMANDLINE:
        entities = ','.join(old.entities) if old.entities else ''

    from memman.queue import enqueue, open_queue_db
    conn = open_queue_db(data_dir_val)
    try:
        row_id = enqueue(
            conn, store=name, content=content_str,
            hint_cat=cat, hint_imp=imp,
            hint_source=source if source_explicit else None,
            hint_entities=entities or None,
            hint_replaced_id=id,
            hint_no_reconcile=not reconcile,
            priority=0)
    finally:
        conn.close()
    _json_out({
        'action': 'queued',
        'queue_id': row_id,
        'store': name,
        'replaced_id': id,
        'quality_warnings': quality_warnings,
        })


@graph.command('link')
@click.argument('source_id')
@click.argument('target_id')
@click.option('--type', 'edge_type', default='semantic', help='Edge type')
@click.option('--weight', default=0.5, type=float, help='Edge weight')
@click.option('--meta', default='', help='JSON metadata')
@click.pass_context
def graph_link(ctx: click.Context, source_id: str, target_id: str,
               edge_type: str, weight: float, meta: str) -> None:
    """Create a manual edge between two insights."""
    _require_started('create edges')

    if edge_type not in VALID_EDGE_TYPES:
        raise click.ClickException(
            f'invalid edge type {edge_type!r}')

    if weight < 0.0 or weight > 1.0:
        raise click.ClickException(
            'weight must be between 0.0 and 1.0')

    metadata: dict[str, str] = {}
    if meta:
        try:
            metadata = json.loads(meta)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f'invalid JSON metadata: {e}')
        if not isinstance(metadata, dict):
            raise click.ClickException(
                'metadata must be a JSON object, not '
                + type(metadata).__name__)
    metadata.setdefault('created_by', 'claude')

    if source_id == target_id:
        raise click.ClickException(
            'cannot link an insight to itself')

    now = datetime.now(timezone.utc)
    with _active_backend(ctx) as backend:
        with backend.transaction():
            if backend.nodes.get(source_id) is None:
                raise click.ClickException(
                    f'insight {source_id} not found')
            if backend.nodes.get(target_id) is None:
                raise click.ClickException(
                    f'insight {target_id} not found')

            existing_weight = backend.edges.get_weight(
                source_id, target_id, edge_type)

            backend.edges.upsert(Edge(
                source_id=source_id, target_id=target_id,
                edge_type=edge_type, weight=weight,
                metadata=metadata, created_at=now))
            backend.edges.upsert(Edge(
                source_id=target_id, target_id=source_id,
                edge_type=edge_type, weight=weight,
                metadata=metadata, created_at=now))
            backend.oplog.log(
                operation='link', insight_id=source_id,
                detail=f'{source_id} <-> {target_id} ({edge_type})')

        actual_weight = (
            backend.edges.get_weight(source_id, target_id, edge_type)
            or weight)
        out = {
            'status': 'linked',
            'source_id': source_id,
            'target_id': target_id,
            'edge_type': edge_type,
            'weight': actual_weight,
            'metadata': metadata,
            }
        if existing_weight is not None and existing_weight > weight:
            out['warning'] = (
                f'existing weight {existing_weight} > requested'
                f' {weight}; kept higher')
        _json_out(out)


@graph.command('related')
@click.argument('id')
@click.option('--edge', default='', help='Filter by edge type')
@click.option('--depth', default=2, type=int, help='Max traversal depth')
@click.pass_context
def graph_related(ctx: click.Context, id: str, edge: str,
                  depth: int) -> None:
    """Find connected insights via graph traversal."""
    from memman.graph.bfs import BFSOptions, bfs

    with _active_backend(ctx) as backend:
        nodes = bfs(backend, id, BFSOptions(
            max_depth=depth, max_nodes=0, edge_filter=edge))
        out = []
        for n in nodes:
            entry: dict = {
                'id': n['insight'].id,
                'content': n['insight'].content,
                'category': n['insight'].category,
                'importance': n['insight'].importance,
                'depth': n['hop'],
                }
            if n.get('via_edge'):
                entry['via_edge_type'] = n['via_edge']
            out.append(entry)
        _json_out(out)


@queue.command('list')
@click.option('--limit', default=50, type=int, help='Max results')
@click.pass_context
def queue_list(ctx: click.Context, limit: int) -> None:
    """List recent queue rows."""
    from memman.queue import list_rows, open_queue_db, stats
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        _json_out({
            'stats': stats(conn),
            'rows': list_rows(conn, limit=limit),
            })
    finally:
        conn.close()


@queue.command('failed')
@click.option('--limit', default=50, type=int, help='Max results')
@click.pass_context
def queue_failed(ctx: click.Context, limit: int) -> None:
    """List failed queue rows."""
    from memman.queue import STATUS_FAILED, list_rows, open_queue_db, stats
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        _json_out({
            'stats': stats(conn),
            'rows': list_rows(conn, status=STATUS_FAILED, limit=limit),
            })
    finally:
        conn.close()


@queue.command('show')
@click.argument('row_id', type=int)
@click.pass_context
def queue_show(ctx: click.Context, row_id: int) -> None:
    """Print the full content of a queue row."""
    from memman.queue import get_row, open_queue_db
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        row = get_row(conn, row_id)
        if row is None:
            raise click.ClickException(f'queue row {row_id} not found')
        _json_out(row)
    finally:
        conn.close()


@queue.command('retry')
@click.argument('row_id', type=int)
@click.pass_context
def queue_retry(ctx: click.Context, row_id: int) -> None:
    """Re-queue a failed row."""
    from memman.queue import open_queue_db, retry_row
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        if not retry_row(conn, row_id):
            raise click.ClickException(
                f'queue row {row_id} not found or not in failed state')
        _json_out({'action': 'requeued', 'queue_id': row_id})
    finally:
        conn.close()


@queue.command('purge')
@click.option('--done', is_flag=True, default=False,
              help='Delete all rows with status=done')
@click.pass_context
def queue_purge(ctx: click.Context, done: bool) -> None:
    """Remove completed queue rows."""
    if not done:
        raise click.ClickException(
            'pass --done to confirm deletion of completed rows')
    from memman.queue import open_queue_db, purge_done
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        deleted = purge_done(conn)
        _json_out({'deleted': deleted})
    finally:
        conn.close()


@scheduler.command('status')
@click.pass_context
def scheduler_status(ctx: click.Context) -> None:
    """Show scheduler install state, interval, next run, log paths,
    and the most recent worker-drain summary from worker_runs.
    """
    from memman.queue import last_worker_run, open_queue_db
    from memman.setup.scheduler import status
    result = status()
    logs_dir = pathlib.Path.home() / '.memman' / 'logs'
    log_path = logs_dir / 'enrich.log'
    err_path = logs_dir / 'enrich.err'
    result['log_path'] = str(log_path)
    result['err_path'] = str(err_path)
    for key, path in (('log_mtime', log_path), ('err_mtime', err_path)):
        try:
            result[key] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
        except OSError:
            result[key] = None

    result['last_run'] = None
    try:
        conn = open_queue_db(ctx.obj['data_dir'])
        try:
            result['last_run'] = last_worker_run(conn)
        finally:
            conn.close()
    except Exception as exc:
        logger.debug(f'worker_runs lookup failed: {exc}')

    _json_out(result)


@scheduler.command('start')
@click.pass_context
def scheduler_start(ctx: click.Context) -> None:
    """Start the scheduler. Worker drains; writes are accepted.

    Idempotent. Sweeps long-stalled queue rows to `stale` so they can
    be retried with `scheduler queue retry`.
    """
    from memman.queue import mark_stale_on_resume, open_queue_db
    from memman.setup.scheduler import start
    try:
        result = start()
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    data_dir_val = ctx.obj['data_dir']
    conn = open_queue_db(data_dir_val)
    try:
        n_stale = mark_stale_on_resume(conn)
    finally:
        conn.close()
    if n_stale:
        result['marked_stale'] = n_stale
        result.setdefault('actions', []).append(
            f"moved {n_stale} long-pending rows to status='stale'"
            ' (retry with `memman scheduler queue retry`)')
    _json_out(result)


@scheduler.command('stop')
@click.pass_context
def scheduler_stop(ctx: click.Context) -> None:
    """Stop the scheduler. Trigger files stay; memman becomes recall-only.

    Writes (`remember`/`replace`/`forget`/`graph link`/`graph rebuild`/
    `insights protect`) reject until `scheduler start` re-arms the
    worker. Use `memman uninstall` to remove trigger files entirely.
    """
    from memman.setup.scheduler import stop
    try:
        result = stop()
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.command('install')
@click.option('--interval', type=int, default=None,
              help=('Polling interval in seconds (min 60 for'
                    ' systemd/launchd). Default: 60. For sub-minute'
                    ' intervals, use serve mode instead'
                    ' (`memman scheduler serve --interval N`).'))
@click.pass_context
def scheduler_install(ctx: click.Context, interval: int | None) -> None:
    """Install the scheduler unit only (no agent integration).

    Reads OPENROUTER_API_KEY and VOYAGE_API_KEY from env and writes them
    to ~/.memman/env (mode 600), then installs the systemd timer or
    launchd plist that runs the worker every interval. For full agent-
    integration setup (hooks, skill, scheduler), use `memman install`.
    """
    from memman.exceptions import ConfigError
    from memman.setup.scheduler import DEFAULT_INTERVAL_SECONDS, install

    try:
        knobs = config.collect_install_knobs(ctx.obj['data_dir'])
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    seconds = interval if interval is not None else DEFAULT_INTERVAL_SECONDS
    if seconds < 60:
        raise click.ClickException(
            '--interval must be at least 60 seconds for systemd/launchd.'
            ' For sub-minute intervals, set MEMMAN_SCHEDULER_KIND=serve'
            ' and run `memman scheduler serve --interval N` instead.')
    try:
        result = install(ctx.obj['data_dir'], knobs, seconds)
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
    """Show or set the scheduler interval."""
    from memman.setup.scheduler import change_interval, status
    if seconds is None:
        s = status()
        _json_out({
            'platform': s['platform'],
            'interval_seconds': s['interval_seconds'],
            'installed': s['installed'],
            })
        return
    try:
        result = change_interval(ctx.obj['data_dir'], seconds)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.command('trigger')
@click.pass_context
def scheduler_trigger(ctx: click.Context) -> None:
    """Run the scheduler's drain job now. Rejected when stopped."""
    _require_started('trigger drain')
    from memman.setup.scheduler import trigger
    try:
        result = trigger()
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    _json_out(result)


@scheduler.group('debug', no_args_is_help=True)
def scheduler_debug() -> None:
    """Toggle persistent JSONL trace state for scheduler-fired runs.

    Writes ~/.memman/debug.state which `trace.is_enabled()` reads as a
    fallback when MEMMAN_DEBUG is unset. Affects future scheduler-fired
    drains and any CLI invocation in a shell that does not export
    MEMMAN_DEBUG. Trace logs land at ~/.memman/logs/debug.log (mode
    600) and include raw LLM request/response bodies — including
    memory content. Turn off when done.
    """


@scheduler_debug.command('on')
@click.pass_context
def scheduler_debug_on(ctx: click.Context) -> None:
    """Enable persistent debug traces."""
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
@click.pass_context
def scheduler_debug_off(ctx: click.Context) -> None:
    """Disable persistent debug traces; existing debug.log files are kept."""
    from memman.setup.scheduler import set_debug
    actions = set_debug(False)
    _json_out({'debug': False, 'actions': actions})


@scheduler_debug.command('status')
@click.pass_context
def scheduler_debug_status(ctx: click.Context) -> None:
    """Show whether persistent debug traces are enabled."""
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
    """Manage named memory stores."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(store_list)


@store.command('list')
@click.pass_context
def store_list(ctx: click.Context) -> None:
    """List all stores as JSON (stores[], active)."""
    data_dir = ctx.obj['data_dir']
    stores = list_stores(data_dir)
    active = _resolve_store_name(data_dir, ctx.obj['store']) if stores else None
    _json_out({'stores': stores, 'active': active})


@store.command('create')
@click.argument('name')
@click.pass_context
def store_create(ctx: click.Context, name: str) -> None:
    """Create a new store."""
    data_dir = ctx.obj['data_dir']
    if not valid_store_name(name):
        raise click.ClickException(
            f'invalid store name {name!r}')
    if store_exists(data_dir, name):
        raise click.ClickException(
            f'store "{name}" already exists')
    from memman.session import active_store
    with active_store(data_dir=data_dir, store=name) as backend:
        path = backend.path
    _json_out({'action': 'created', 'store': name, 'path': path})


@store.command('use')
@click.argument('name')
@click.pass_context
def store_use(ctx: click.Context, name: str) -> None:
    """Switch the active store."""
    data_dir = ctx.obj['data_dir']
    if not store_exists(data_dir, name):
        raise click.ClickException(
            f"store \"{name}\" does not exist"
            f" (use 'memman store create {name}' first)")
    write_active(data_dir, name)
    _json_out({'action': 'set', 'store': name})


@store.command('remove')
@click.argument('name')
@click.option('--yes', is_flag=True, default=False,
              help='Skip confirmation prompt (for scripted use).')
@click.pass_context
def store_remove(ctx: click.Context, name: str, yes: bool) -> None:
    """Remove a store (prompts unless --yes)."""
    import shutil
    data_dir = ctx.obj['data_dir']
    if not store_exists(data_dir, name):
        raise click.ClickException(
            f"store \"{name}\" does not exist"
            f" (use 'memman store create {name}' first)")
    active = read_active(data_dir)
    if name == active:
        raise click.ClickException(
            f"cannot remove the active store \"{name}\""
            f" (switch first with 'memman store use <other>')")
    sdir = store_dir(data_dir, name)
    if not yes:
        click.confirm(
            f'Delete store "{name}" and all data at {sdir}?',
            abort=True)
    from memman.queue import open_queue_db
    from memman.queue import purge_store as queue_purge_store
    qconn = open_queue_db(data_dir)
    try:
        queue_purge_store(qconn, name)
    finally:
        qconn.close()
    shutil.rmtree(sdir)
    _json_out({'action': 'removed', 'store': name})


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show database statistics."""
    from memman.store.factory import _resolve_store_backend, list_stores

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
            _resolve_store_backend(s, data_dir) for s in all_stores
            })
        out = {
            'store': store_name,
            'backend': _resolve_store_backend(store_name, data_dir),
            'backends_in_use': backends_in_use,
            'total_insights': node_stats.total_insights,
            'deleted_insights': node_stats.deleted_insights,
            'edge_count': node_stats.edge_count,
            'oplog_count': node_stats.oplog_count,
            'by_category': node_stats.by_category,
            'top_entities': node_stats.top_entities,
            'storage_path': backend.path,
            }
        _json_out(out)


@cli.command()
@click.option('--text', 'text_output', is_flag=True, default=False,
              help='Human-readable colored output (default: JSON)')
@click.pass_context
def doctor(ctx: click.Context, text_output: bool) -> None:
    """Run health checks on the database, scheduler, and providers.

    Exits 0 on pass/warn, 1 on fail — usable as a CI/scripted gate.
    """
    from memman.doctor import run_all_checks

    with _active_backend(ctx, unchecked=True) as backend:
        result = run_all_checks(
            backend, data_dir=ctx.obj['data_dir'])
        result['store'] = _resolve_store_name(
            ctx.obj['data_dir'], ctx.obj['store'])
        result['db_path'] = backend.path
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
    """Show the operation audit log (default JSON; --text for human view)."""
    since_ts = ''
    if since:
        since_ts = _parse_since(since)

    with _active_backend(ctx) as backend:
        if stats:
            stats_data = backend.oplog.stats(since=since_ts)
            _json_out({
                'operation_counts': stats_data.operation_counts,
                'never_accessed': stats_data.never_accessed,
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


@log.command('worker')
@click.option('--errors', is_flag=True, default=False,
              help='Read enrich.err instead of enrich.log.')
@click.option('--lines', type=int, default=50,
              help='Number of tail lines to print (default 50).')
@click.pass_context
def log_worker(ctx: click.Context, errors: bool, lines: int) -> None:
    """Print the tail of ~/.memman/logs/enrich.{log,err} (worker output)."""
    logs_dir = pathlib.Path.home() / '.memman' / 'logs'
    path = logs_dir / ('enrich.err' if errors else 'enrich.log')
    if not path.is_file():
        click.echo(f'[memman] no log file yet at {path}', err=True)
        return
    try:
        content = path.read_text(errors='replace').splitlines()
    except OSError as exc:
        raise click.ClickException(
            f'failed to read {path}: {exc}') from exc
    tail = content[-lines:] if lines > 0 else content
    for line_str in tail:
        click.echo(line_str)


@insights.command('candidates')
@click.option('--threshold', default=0.5, type=float,
              help='Effective-importance cutoff; insights below are surfaced (default 0.5).')
@click.option('--limit', default=20, type=int, help='Max candidates returned')
@click.pass_context
def insights_candidates(ctx: click.Context, threshold: float,
                        limit: int) -> None:
    """List insights with low effective importance.

    Surfaces candidates only — does NOT delete. Use `memman forget <id>`
    to actually remove an insight, or `memman insights protect <id>` to
    boost its retention.
    """
    from memman.store.model import MAX_INSIGHTS

    with _active_backend(ctx) as backend:
        candidates, total = backend.nodes.get_retention_candidates(
            threshold=threshold, limit=limit)
        out_candidates = []
        for c in candidates:
            ins = c['insight']
            out_candidates.append({
                'id': ins.id,
                'content': ins.content,
                'category': ins.category,
                'importance': ins.importance,
                'access_count': ins.access_count,
                'effective_importance': c['effective_importance'],
                'days_since_access': c['days_since_access'],
                'edge_count': c['edge_count'],
                'immune': c['immune'],
                })
        _json_out({
            'total_insights': total,
            'threshold': threshold,
            'candidates_found': len(candidates),
            'candidates': out_candidates,
            'max_insights': MAX_INSIGHTS,
            'actions': {
                'purge': 'memman forget <id>',
                'protect': 'memman insights protect <id>',
                },
            })


@insights.command('review')
@click.option('--limit', default=20, type=int, help='Max flagged results')
@click.pass_context
def insights_review(ctx: click.Context, limit: int) -> None:
    """Scan stored insights for content quality issues.

    Different criteria than `candidates`: this checks content quality
    (transient phrasing, low signal) rather than retention score.
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
                'importance': f['insight'].importance,
                'quality_warnings': f['quality_warnings'],
                } for f in flagged],
            'total_flagged': len(flagged),
            'actions': {
                'forget': 'memman forget <id>',
                'protect': 'memman insights protect <id>',
                },
            })


@insights.command('protect')
@click.argument('id')
@click.pass_context
def insights_protect(ctx: click.Context, id: str) -> None:
    """Boost retention of an insight.

    Increments access count by 3 and refreshes effective importance,
    keeping the insight off retention-candidate lists.
    """
    _require_started('protect insights')

    with _active_backend(ctx) as backend:
        ins = backend.nodes.get(id)
        if ins is None:
            raise click.ClickException(
                f'insight {id} not found or already deleted')
        with backend.transaction():
            backend.nodes.boost_retention(id)
            ei = backend.nodes.refresh_effective_importance(id)
            backend.oplog.log(
                operation='protect', insight_id=id,
                detail=f'access+3, ei={ei:.4f}')
        new_access = ins.access_count + 3
        _json_out({
            'status': 'retained',
            'id': id,
            'content': ins.content,
            'new_access': new_access,
            'effective_importance': ei,
            'immune': is_immune(ins.importance, new_access),
            })


@insights.command('show')
@click.argument('id')
@click.pass_context
def insights_show(ctx: click.Context, id: str) -> None:
    """Read a single insight by ID with full content and metadata."""
    with _active_backend(ctx) as backend:
        ins = backend.nodes.get(id)
        if ins is None:
            raise click.ClickException(
                f'insight {id} not found or already deleted')
        _json_out(_insight_to_dict(ins))


@cli.command()
@click.option('--target', default='',
              help='Target environment (claude-code | openclaw | nanoclaw)')
@click.option('--backend', type=click.Choice(['sqlite', 'postgres']),
              default=None,
              help='Storage backend; bypasses the wizard prompt when set.')
@click.option('--pg-dsn', default=None,
              help='Postgres DSN (postgresql://...); required with'
                   ' --backend postgres in non-interactive mode.')
@click.option('--no-wizard', is_flag=True,
              help='Disable interactive prompts; flags + defaults only.')
@click.pass_context
def install(ctx: click.Context, target: str, backend: str | None,
            pg_dsn: str | None, no_wizard: bool) -> None:
    """Install memman integration: skill, hooks, scheduler."""
    from memman.setup.claude import run_install
    run_install(
        ctx.obj['data_dir'],
        target=target,
        backend=backend,
        pg_dsn=pg_dsn,
        no_wizard=no_wizard)


@cli.command()
@click.option('--target', default='',
              help='Target environment (claude-code | openclaw | nanoclaw)')
@click.pass_context
def uninstall(ctx: click.Context, target: str) -> None:
    """Remove memman integration (reverse of `memman install`)."""
    from memman.setup.claude import run_uninstall
    run_uninstall(ctx.obj['data_dir'], target=target)


@cli.command()
@click.option('--store', default='',
              help='Store to migrate. Required unless --all.')
@click.option('--all', 'migrate_all', is_flag=True,
              help='Migrate every store under the data dir.')
@click.option('--dry-run', is_flag=True,
              help='Report the plan without writing or prompting.')
@click.option('--yes', is_flag=True, default=False,
              help='Skip the confirmation prompt (for scripted use).')
@click.pass_context
def migrate(
        ctx: click.Context, store: str, migrate_all: bool,
        dry_run: bool, yes: bool) -> None:
    """Migrate memman stores from SQLite to Postgres.

    Echoes a plan (source paths, redacted destination DSN, per-store
    target schema state) and prompts for confirmation. Stores whose
    target schema already exists on the remote are dropped and
    recreated. Holds the shared drain.lock for the duration so a
    scheduler-fired drain cannot race the SQLite reader. On success
    flips `MEMMAN_BACKEND=postgres` in the env file so the next drain
    routes to the new database.
    """
    from memman import config
    from memman.migrate import MigrateError, SchemaState, held_drain_lock
    from memman.migrate import inspect_target_schemas, migrate_store
    from memman.migrate import preflight
    from memman.setup.scheduler import _write_env_keys
    from memman.store.db import list_stores, store_dir
    from memman.trace import redact_dsn

    data_dir = ctx.obj['data_dir']

    if not migrate_all and not store:
        raise click.UsageError('pass --store NAME or --all')
    if migrate_all and store:
        raise click.UsageError(
            'pass either --store NAME or --all, not both')

    dsn = config.get(config.PG_DSN)
    if not dsn:
        raise click.UsageError(
            'MEMMAN_PG_DSN is not set; configure with '
            '`memman config set MEMMAN_PG_DSN <url>` first')

    try:
        preflight(dsn)
    except MigrateError as exc:
        raise click.ClickException(str(exc))

    if migrate_all:
        stores = list_stores(data_dir)
    else:
        stores = [store]
    if not stores:
        click.echo('no stores to migrate', err=True)
        return

    try:
        states = inspect_target_schemas(dsn, stores)
    except MigrateError as exc:
        raise click.ClickException(str(exc))

    populated = [s for s in stores
                 if states[s] == SchemaState.POPULATED]

    click.echo('Migration plan:')
    click.echo(f'  Source:      {data_dir}/data/')
    click.echo(f'  Destination: {redact_dsn(dsn)}')
    click.echo(f'  Stores ({len(stores)}):')
    width = max((len(s) for s in stores), default=0)
    for s in stores:
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
    if not dry_run:
        click.echo('')
        click.echo(
            "After successful migrate, MEMMAN_BACKEND will flip to"
            " 'postgres' in the env file.")

    if dry_run:
        for s in stores:
            source = store_dir(data_dir, s)
            try:
                res = migrate_store(
                    source_dir=source, dsn=dsn, store=s,
                    dry_run=True, state=states[s])
                click.echo(
                    f'{s}: insights={res.insights}'
                    f' edges={res.edges} oplog={res.oplog}'
                    f' meta={res.meta} (dry-run)')
            except MigrateError as exc:
                raise click.ClickException(f'{s}: {exc}')
        return

    if not yes:
        click.echo('')
        click.confirm('Proceed?', default=False, abort=True)

    try:
        with held_drain_lock(data_dir):
            for s in stores:
                source = store_dir(data_dir, s)
                try:
                    res = migrate_store(
                        source_dir=source, dsn=dsn, store=s,
                        dry_run=False, state=states[s])
                    click.echo(
                        f'{s}: insights={res.insights}'
                        f' edges={res.edges} oplog={res.oplog}'
                        f' meta={res.meta} (verified, source cleaned)')
                except MigrateError as exc:
                    raise click.ClickException(f'{s}: {exc}')
            _write_env_keys(
                {config.BACKEND: 'postgres'}, data_dir=data_dir)
    except MigrateError as exc:
        raise click.ClickException(str(exc))

    click.echo('')
    click.echo(
        f'Migration complete: {len(stores)} store(s) copied to Postgres.')
    click.echo("MEMMAN_BACKEND flipped to 'postgres'.")
    click.echo('')
    click.echo('Recommended next step:')
    click.echo('  memman doctor    # verify the postgres backend health')
    click.echo('')
    click.echo('To revert:')
    click.echo('  memman config set MEMMAN_BACKEND sqlite')


def _emit_guide() -> None:
    """Write shipped guide.md to stdout."""
    from importlib.resources import files as pkg_files
    shipped = (pkg_files('memman.setup.assets')
               .joinpath('claude/guide.md').read_text())
    click.echo(shipped, nl=False)


@cli.command(hidden=True)
def guide() -> None:
    """Print the memman behavioral guide. Hidden — called by openclaw bootstrap."""
    _emit_guide()


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
        env_store = os.environ.get(config.STORE, '').strip()
        name = env_store or read_active(data_dir)
        from memman.store.factory import _resolve_store_backend
        backend_name = _resolve_store_backend(name, data_dir)
        if backend_name == 'sqlite':
            from memman.store.node import get_stats
            if store_exists(data_dir, name):
                db = open_ro_db(store_dir(data_dir, name))
                try:
                    stats = get_stats(db)
                finally:
                    db.close()
                status_line = (f"[memman] Memory active "
                               f"({stats['total_insights']} insights, "
                               f"{stats['edge_count']} edges).")
        else:
            from memman.session import active_store
            with active_store(
                    data_dir=data_dir, store=name,
                    unchecked=True) as backend:
                s = backend.nodes.stats()
                status_line = (f'[memman] Memory active '
                               f'({s.total_insights} insights, '
                               f'{s.edge_count} edges).')
    except Exception:
        pass
    click.echo(status_line)

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
                   f'memman recall "<topic>" --limit 5')

    _emit_guide()


@graph.command('rebuild')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show counts without modifying DB')
@click.pass_context
def graph_rebuild(ctx: click.Context, dry_run: bool) -> None:
    """Re-enrich all insights through the full LLM pipeline."""
    from memman.store.factory import _resolve_store_backend
    data_dir = ctx.obj['data_dir']
    store_name = _resolve_store_name(data_dir, ctx.obj['store'])
    backend_name = _resolve_store_backend(store_name, data_dir)
    if backend_name != 'sqlite':
        raise click.ClickException(
            'graph rebuild is SQLite-only; Postgres maintains HNSW'
            ' indexes live and reindexes on constant change')
    if not dry_run:
        _require_started('rebuild')
    from memman.embed import get_client
    from memman.graph.engine import MAX_LINK_BATCH, link_pending
    from memman.store.node import count_pending_links, get_active_insight_ids
    from memman.store.node import reset_for_rebuild
    from memman.store.oplog import log_op

    db = _open_db(ctx)
    try:
        llm_client = _get_llm_client_or_fail('slow_canonical')
        metadata_llm_client = _get_llm_client_or_fail('slow_metadata')
        ec = get_client()

        all_ids = get_active_insight_ids(db)
        total_count = len(all_ids)

        if dry_run:
            _json_out({'total': total_count, 'dry_run': 1})
            return

        if total_count == 0:
            _json_out({'processed': 0, 'remaining': 0})
            return

        from memman.store.sqlite import SqliteBackend
        backend = SqliteBackend(db)
        with backend.reembed_lock('rebuild') as held:
            if not held:
                raise click.ClickException(
                    'another graph rebuild is in progress on this store')

            embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())
            processed = 0

            bar = tqdm(
                total=total_count, desc='Rebuilding',
                unit='insight', file=sys.stderr,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty())

            def _on_progress(stage: str, insight: Insight) -> None:
                preview = insight.content[:40].replace('\n', ' ')
                bar.set_description(f'{stage}: {preview}')
                if stage == 'done':
                    bar.update(1)

            for i in range(0, total_count, MAX_LINK_BATCH):
                batch_ids = all_ids[i:i + MAX_LINK_BATCH]
                reset_for_rebuild(db, batch_ids)

                while True:
                    count = link_pending(
                        backend, embed_cache=embed_cache,
                        llm_client=llm_client,
                        metadata_llm_client=metadata_llm_client,
                        embed_client=ec,
                        on_progress=_on_progress)
                    processed += count
                    if count == 0:
                        break

            bar.set_description('Done')
            bar.close()

            remaining = count_pending_links(db)

            stats = {'processed': processed, 'remaining': remaining}
            log_op(db, 'rebuild', '', json.dumps(stats))
            _json_out(stats)
    finally:
        db.close()


@embed_grp.command('status')
@click.pass_context
def embed_status(ctx: click.Context) -> None:
    """Show active client, stored fingerprint, consistency, swap state.
    """
    from memman.embed.fingerprint import active_fingerprint, stored_fingerprint
    from memman.embed.swap import read_progress

    active = active_fingerprint()
    with _active_backend(ctx, unchecked=True) as backend:
        stored = stored_fingerprint(backend)
        progress = read_progress(backend)

    out: dict = {
        'active': {
            'provider': active.provider,
            'model': active.model,
            'dim': active.dim,
            },
        'stored': None if stored is None else {
            'provider': stored.provider,
            'model': stored.model,
            'dim': stored.dim,
            },
        }
    if progress.state:
        out['swap'] = {
            'state': progress.state,
            'cursor': progress.cursor,
            'target_provider': progress.target_provider,
            'target_model': progress.target_model,
            'target_dim': progress.target_dim,
            }
    if stored is None:
        out['consistent'] = False
        out['hint'] = (
            "DB not initialized. Run 'memman embed reembed'.")
    elif stored != active:
        out['consistent'] = False
        out['hint'] = (
            "Fingerprint mismatch. Run"
            " 'memman scheduler stop && memman embed reembed'.")
    else:
        out['consistent'] = True
    _json_out(out)


_REEMBED_BATCH = 50


def _count_active_rows(sdir: str) -> int:
    """Return the count of non-deleted insights in the given store.
    """
    from memman.store.node import count_active_insights

    db = open_ro_db(sdir)
    try:
        return count_active_insights(db)
    finally:
        db.close()


def _reembed_one_store(
        sdir: str, ec: 'EmbeddingProvider', target: 'Fingerprint',
        dry_run: bool, bar: 'tqdm | None' = None) -> dict:
    """Re-embed a single store with the active client.

    Walk all active insights, comparing each to `target`. Skip rows
    that already match; re-embed rows that differ. Per-row blob +
    cursor advance is one transaction; the final fingerprint write +
    cursor reset + state=idle + edge reindex is another.
    """
    from memman.embed.fingerprint import write_fingerprint
    from memman.graph.engine import reindex_auto_edges
    from memman.store.db import open_db
    from memman.store.node import iter_for_reembed
    from memman.store.sqlite import SqliteBackend

    store_name = pathlib.Path(sdir).name
    db = open_db(sdir)
    backend = SqliteBackend(db)
    try:
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

            if bar is not None:
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
                    else:
                        if not dry_run:
                            backend.meta.set(
                                'embed_reembed_cursor', row_id)
                    cursor = row_id
                    if bar is not None:
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

            edge_stats = reindex_auto_edges(backend)

            stats = {
                'store': store_name,
                'scanned': scanned,
                'reembedded': reembedded,
                'edges': edge_stats,
                }
            backend.oplog.log(
                operation='embed_reembed', insight_id='',
                detail=json.dumps(stats))
            return stats
    finally:
        db.close()


@embed_grp.command('reembed')
@click.option(
    '--dry-run', is_flag=True, default=False,
    help='Count rows that would be re-embedded; no DB writes.')
@click.pass_context
def embed_reembed(ctx: click.Context, dry_run: bool) -> None:
    """Sweep every store with the active client; write fingerprints.

    Always global: iterates all stores under the configured
    data_dir. The active embed provider is set by a single global
    env var, so a swap necessarily applies to every store; per-store
    scoping is intentionally not supported.

    Three cases through one walk per store:
    1. Empty DB - zero rows; only the fingerprint is written.
    2. Existing DB on the same provider - rows match; skip re-embed.
    3. Provider swap - rows mismatch; re-embed each.

    The sweep is resumable per store: progress is tracked in each
    store's `meta.embed_reembed_state` and `meta.embed_reembed_cursor`.
    A crash mid-sweep leaves state='in_progress'; re-running picks
    up from the cursor.
    """
    from memman.embed import get_client
    from memman.embed.fingerprint import Fingerprint
    from memman.store.db import list_stores, store_dir
    from memman.store.factory import _resolve_store_backend

    data_dir = ctx.obj['data_dir']
    active_name = _resolve_store_name(data_dir, ctx.obj['store'])
    if _resolve_store_backend(active_name, data_dir) != 'sqlite':
        raise click.ClickException(
            'embed reembed is SQLite-only; Postgres reembed requires'
            ' a separate workflow (track in a follow-up issue)')

    if not dry_run:
        _require_stopped('reembed')

    ec = get_client()
    if not ec.available():
        raise click.ClickException(ec.unavailable_message())

    target = Fingerprint.from_client(ec)
    names = [
        n for n in list_stores(data_dir)
        if _resolve_store_backend(n, data_dir) == 'sqlite']

    grand_total = sum(
        _count_active_rows(store_dir(data_dir, name)) for name in names)
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
            'provider': target.provider,
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
    help="Target embed model (e.g. 'voyage-3-large'). Resolved with"
         " the active provider unless --provider is given.")
@click.option(
    '--provider', 'to_provider', default='',
    help='Target embed provider (default: active provider from env).')
@click.option(
    '--resume', 'resume', is_flag=True, default=False,
    help='Continue an in-flight swap from the recorded cursor.')
@click.option(
    '--abort', 'abort', is_flag=True, default=False,
    help='Discard the in-flight swap. Drops embedding_pending and'
         ' clears all swap meta. After cutover, this is one-way: the'
         ' old embeddings are gone, so reverting requires running swap'
         ' again with the old model (full re-embed cost).')
@click.pass_context
def embed_swap(
        ctx: click.Context, to_model: str, to_provider: str,
        resume: bool, abort: bool) -> None:
    """Online per-store swap to a new embed model.

    Postgres: shadow `embedding_pending vector(N)` column with HNSW
    built CONCURRENTLY, backfilled `WHERE embedding_pending IS NULL`,
    cut over in one transaction (drop + rename). SQLite: shadow
    `embedding_pending BLOB` column populated under
    `write_lock("embed_swap")`, cutover is `update insights set
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
    from memman.embed import registry as _ec_registry
    from memman.embed.fingerprint import Fingerprint, stored_fingerprint
    from memman.embed.fingerprint import write_fingerprint
    from memman.embed.swap import SwapPlan
    from memman.embed.swap import abort_swap as _abort_swap
    from memman.embed.swap import read_progress, run_swap
    from memman.store.factory import open_backend

    if abort and resume:
        raise click.ClickException(
            '--abort and --resume are mutually exclusive')

    data_dir = ctx.obj['data_dir']
    name = _resolve_store_name(data_dir, ctx.obj['store'])
    backend = open_backend(name, data_dir)
    try:
        if abort:
            _abort_swap(backend)
            _json_out({'store': name, 'state': 'aborted'})
            return

        progress = read_progress(backend)
        if resume:
            if progress.state == '':
                raise click.ClickException(
                    f'no in-flight swap on store {name!r}')
            target_provider = progress.target_provider
            target_model = progress.target_model
            target_dim = progress.target_dim
        else:
            if progress.state and progress.state not in {'', 'done'}:
                raise click.ClickException(
                    f'store {name!r} has an in-flight swap'
                    f' (state={progress.state}); use --resume or'
                    ' --abort')
            if not to_model:
                raise click.ClickException(
                    '--to <model> is required to start a new swap')
            if not to_provider:
                to_provider = (
                    config.get(config.EMBED_PROVIDER) or 'voyage')
            target_provider = to_provider
            target_model = to_model
            target_dim = 0

        ec_new = _ec_registry.get_for(target_provider, target_model)
        if not ec_new.available():
            raise click.ClickException(ec_new.unavailable_message())
        if target_dim == 0:
            target_dim = int(getattr(ec_new, 'dim', 0))
        if target_dim <= 0:
            raise click.ClickException(
                f'failed to discover dim for {target_provider}:'
                f'{target_model}; provider should expose dim after'
                ' prepare()')

        plan = SwapPlan(
            target_provider=target_provider,
            target_model=target_model,
            target_dim=target_dim)

        from memman.store.factory import _resolve_store_backend
        backend_name = _resolve_store_backend(name, data_dir)
        if backend_name == 'postgres':
            with backend.swap_lock() as held:
                if not held:
                    raise click.ClickException(
                        f'another swap is in progress on'
                        f' store {name!r}')
                _require_stopped('swap')
                progress = run_swap(backend, ec_new, plan)
        else:
            _require_stopped('swap')
            progress = run_swap(backend, ec_new, plan)

        fp = stored_fingerprint(backend) or Fingerprint(
            provider=plan.target_provider,
            model=plan.target_model,
            dim=plan.target_dim)
        if fp.provider != plan.target_provider:
            write_fingerprint(
                backend,
                Fingerprint(
                    provider=plan.target_provider,
                    model=plan.target_model,
                    dim=plan.target_dim))
            fp = Fingerprint(
                provider=plan.target_provider,
                model=plan.target_model,
                dim=plan.target_dim)

        _json_out({
            'store': name,
            'state': progress.state,
            'fingerprint': {
                'provider': fp.provider,
                'model': fp.model,
                'dim': fp.dim,
                },
            })
    finally:
        backend.close()
