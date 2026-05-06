"""Health checks for a memman store.

Runs read-only diagnostics and reports per-check pass/warn/fail
with an overall worst-status summary. Each per-store check takes a
`Backend` and routes through Protocol verbs; raw `db._query` /
`db._exec` / `db.path` access is forbidden in this module.
"""

import os
import stat
import statistics
from pathlib import Path
from typing import Any

from memman.store.backend import Backend


def check_integrity(backend: Backend) -> dict[str, Any]:
    """Run the backend's integrity probe."""
    result = backend.integrity_check()
    ok = bool(result.get('ok'))
    return {
        'name': 'integrity',
        'status': 'pass' if ok else 'fail',
        'detail': {'result': result.get('detail')},
        }


def check_enrichment_coverage(backend: Backend) -> dict[str, Any]:
    """Check that embedding, keywords, summary, semantic_facts are populated."""
    cov = backend.nodes.enrichment_coverage()
    total = cov.total_active
    if total == 0:
        return {'name': 'enrichment_coverage', 'status': 'pass',
                'detail': {'total_active': 0, 'coverage_pct': 100.0}}
    missing_any = max(
        cov.missing_embedding, cov.missing_keywords,
        cov.missing_summary, cov.missing_semantic_facts)
    coverage_pct = round((total - missing_any) / total * 100, 1)
    if missing_any == 0:
        status = 'pass'
    elif coverage_pct >= 90:
        status = 'warn'
    else:
        status = 'fail'
    return {
        'name': 'enrichment_coverage',
        'status': status,
        'detail': {
            'total_active': total,
            'missing_embedding': cov.missing_embedding,
            'missing_keywords': cov.missing_keywords,
            'missing_summary': cov.missing_summary,
            'missing_semantic_facts': cov.missing_semantic_facts,
            'coverage_pct': coverage_pct,
            },
        }


def check_oplog_delta_coverage(backend: Backend) -> dict[str, Any]:
    """Check the share of oplog rows that carry before/after deltas.

    Older oplog rows have null `before` / `after`; coverage
    approaches 100% on stores with frequent write traffic. New
    stores report 100% (vacuously covered).
    """
    total, with_delta = backend.oplog.delta_coverage()
    if total == 0:
        return {
            'name': 'oplog_delta_coverage',
            'status': 'pass',
            'detail': {
                'total_oplog_rows': 0,
                'rows_with_delta': 0,
                'coverage_pct': 100.0,
                },
            }
    coverage_pct = round(with_delta / total * 100, 1)
    if coverage_pct >= 90:
        status = 'pass'
    elif coverage_pct >= 50:
        status = 'warn'
    else:
        status = 'warn'
    return {
        'name': 'oplog_delta_coverage',
        'status': status,
        'detail': {
            'total_oplog_rows': total,
            'rows_with_delta': with_delta,
            'coverage_pct': coverage_pct,
            },
        }


def check_orphan_insights(backend: Backend) -> dict[str, Any]:
    """Find active insights with zero edges."""
    orphan_count, total = backend.nodes.count_orphans()
    if total == 0:
        return {'name': 'orphan_insights', 'status': 'pass',
                'detail': {'orphan_count': 0, 'total_active': 0,
                           'orphan_pct': 0.0}}
    orphan_pct = round(orphan_count / total * 100, 1)
    if orphan_count == 0:
        status = 'pass'
    elif orphan_pct <= 5:
        status = 'warn'
    else:
        status = 'fail'
    return {
        'name': 'orphan_insights',
        'status': status,
        'detail': {
            'orphan_count': orphan_count,
            'total_active': total,
            'orphan_pct': orphan_pct,
            },
        }


def check_dangling_edges(backend: Backend) -> dict[str, Any]:
    """Find edges referencing deleted or missing insights."""
    by_type = backend.edges.count_dangling_by_type()
    total = sum(by_type.values())
    status = 'pass' if total == 0 else 'fail'
    return {
        'name': 'dangling_edges',
        'status': status,
        'detail': {'count': total, 'by_type': by_type},
        }


def check_embedding_consistency(backend: Backend) -> dict[str, Any]:
    """Verify all embeddings have the same size (byte length on SQLite,
    pgvector dimension on Postgres).
    """
    dist = backend.nodes.embedding_size_distribution()
    sizes = {str(size): cnt for size, cnt in dist.items()}
    status = 'pass' if len(sizes) <= 1 else 'fail'
    return {
        'name': 'embedding_consistency',
        'status': status,
        'detail': {'sizes': sizes},
        }


def check_edge_degree(backend: Backend) -> dict[str, Any]:
    """Compute degree distribution stats across active insights."""
    active_ids = backend.nodes.get_active_ids()
    if not active_ids:
        return {'name': 'edge_degree', 'status': 'pass',
                'detail': {'min': 0, 'max': 0, 'median': 0, 'mean': 0.0}}
    degree_by_id = backend.edges.degree_distribution()
    degrees = sorted(degree_by_id.get(aid, 0) for aid in active_ids)
    med = statistics.median(degrees)
    avg = round(statistics.mean(degrees), 1)
    if med >= 5:
        status = 'pass'
    elif med >= 2:
        status = 'warn'
    else:
        status = 'fail'
    return {
        'name': 'edge_degree',
        'status': status,
        'detail': {
            'min': degrees[0],
            'max': degrees[-1],
            'median': med,
            'mean': avg,
            },
        }


QUEUE_DEPTH_WARN = 50
QUEUE_DEPTH_FAIL = 100
QUEUE_AGE_WARN_SECONDS = 3600
QUEUE_AGE_FAIL_SECONDS = 86400


_ORPHAN_ARTIFACTS = (
    'memman.db', 'memman.db-wal', 'memman.db-shm',
    'recall_snapshot.v1.bin',
    )


def check_orphan_storage(data_dir: str) -> dict[str, Any]:
    """Flag SQLite artifacts left behind on a Postgres-backed store.

    A successful `memman migrate <store>` deletes the source memman.db
    (plus WAL/SHM and the recall snapshot). If the cleanup step
    crashed between commit and unlink, the file remains on disk but
    no longer represents the truth of the store. This check walks
    every store directory under `<data_dir>/data/` and reports any
    survivor when the resolved backend for the process is `postgres`.
    """
    from memman import config
    backend = (
        config.get(config.DEFAULT_BACKEND) or 'sqlite').lower()
    if backend != 'postgres':
        return {
            'name': 'orphan_storage', 'status': 'pass',
            'detail': {
                'reason': 'check_skipped_for_sqlite_backend',
                },
            }
    data_root = Path(data_dir) / 'data'
    if not data_root.is_dir():
        return {
            'name': 'orphan_storage', 'status': 'pass',
            'detail': {'stores': []},
            }
    orphans: dict[str, list[str]] = {}
    for entry in sorted(os.scandir(data_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        survivors = [
            name for name in _ORPHAN_ARTIFACTS
            if (Path(entry.path) / name).exists()
            ]
        if survivors:
            orphans[entry.name] = survivors
    if not orphans:
        return {
            'name': 'orphan_storage', 'status': 'pass',
            'detail': {'stores': []},
            }
    return {
        'name': 'orphan_storage', 'status': 'fail',
        'detail': (
            f'orphan SQLite files in postgres-backed stores: {orphans}'),
        }


def check_queue_backlog(data_dir: str) -> dict[str, Any]:
    """Report pending/failed counts and oldest-pending age."""
    from memman.queue import open_queue_db
    from memman.queue import stats as queue_stats

    conn = open_queue_db(data_dir)
    try:
        s = queue_stats(conn)
    finally:
        conn.close()

    pending = s['pending']
    failed = s['failed']
    oldest_age = s['oldest_pending_age_seconds']

    status = 'pass'
    if pending >= QUEUE_DEPTH_FAIL:
        status = 'fail'
    elif pending >= QUEUE_DEPTH_WARN:
        status = 'warn'
    if oldest_age is not None:
        if oldest_age >= QUEUE_AGE_FAIL_SECONDS:
            status = 'fail'
        elif oldest_age >= QUEUE_AGE_WARN_SECONDS and status == 'pass':
            status = 'warn'
    if failed > 0 and status == 'pass':
        status = 'warn'

    return {
        'name': 'queue_backlog',
        'status': status,
        'detail': {
            'pending': pending,
            'failed': failed,
            'done': s['done'],
            'oldest_pending_age_seconds': oldest_age,
            'thresholds': {
                'depth_warn': QUEUE_DEPTH_WARN,
                'depth_fail': QUEUE_DEPTH_FAIL,
                'age_warn_seconds': QUEUE_AGE_WARN_SECONDS,
                'age_fail_seconds': QUEUE_AGE_FAIL_SECONDS,
                },
            },
        }


EXPECTED_INSIGHT_COLUMNS = {
    'prompt_version', 'model_id', 'embedding_model',
    'linked_at', 'enriched_at',
    }
EXPECTED_QUEUE_TABLES = {'queue', 'worker_runs'}


def check_schema_columns(backend: Backend) -> dict[str, Any]:
    """Verify the insights table has the canonical provenance columns.

    Single-user canonical-schema policy: missing columns mean the DB
    predates a schema change; the fix is a one-off `alter table`.
    """
    present = backend.introspect_columns('insights')
    missing = sorted(EXPECTED_INSIGHT_COLUMNS - present)
    status = 'pass' if not missing else 'fail'
    return {
        'name': 'schema_columns',
        'status': status,
        'detail': {'missing': missing},
        }


def check_queue_schema(data_dir: str) -> dict[str, Any]:
    """Verify queue.db has the canonical tables (queue + worker_runs).
    """
    from memman.queue import open_queue_db

    conn = open_queue_db(data_dir)
    try:
        rows = conn.execute(
            "select name from sqlite_master where type='table'"
            ).fetchall()
    finally:
        conn.close()
    present = {row[0] for row in rows}
    missing = sorted(EXPECTED_QUEUE_TABLES - present)
    status = 'pass' if not missing else 'fail'
    return {
        'name': 'queue_schema',
        'status': status,
        'detail': {'missing': missing},
        }


def check_optional_extras() -> dict[str, Any]:
    """Report which `memman[extras]` install groups resolve at runtime.

    Always passes; the result is informational. Lets users verify their
    install matches their backend choice (e.g., backend=postgres
    requires the `postgres` extra to be active).
    """
    from memman import extras
    active = extras.detect_active_extras()
    return {
        'name': 'optional_extras',
        'status': 'pass',
        'detail': {'active': active},
        }


def check_env_completeness() -> dict[str, Any]:
    """Verify ~/.memman/env contains every INSTALLABLE_KEYS entry.

    Catches the upgrade case where a new release adds a key the user's
    existing file lacks. Reports the missing keys so the user can run
    `memman install` to repopulate. Optional secrets are not flagged
    when absent (the user may not have configured an alternate provider).

    Per-store dispatch is validated separately by `check_per_store_keys`;
    this check covers cluster-global installable knobs only.
    """
    from memman import config

    path = config.env_file_path()
    parsed = config.parse_env_file(path)

    optional_secrets = {
        config.OPENAI_EMBED_API_KEY,
        }
    missing = [
        key for key in config.INSTALLABLE_KEYS
        if not parsed.get(key) and key not in optional_secrets
        ]

    if not missing:
        return {
            'name': 'env_completeness',
            'status': 'pass',
            'detail': {'env_file': str(path)},
            }

    return {
        'name': 'env_completeness',
        'status': 'warn',
        'detail': {
            'env_file': str(path),
            'missing': missing,
            'fix': ('run `memman install` to populate the env file with'
                    ' the latest INSTALLABLE_KEYS values'),
            },
        }


_KNOWN_BACKENDS = frozenset({'sqlite', 'postgres'})


def check_per_store_keys(data_dir: str) -> dict[str, Any]:
    """Validate per-store backend dispatch keys for every known store.

    For each store enumerated by `list_stores`:
    - resolve `MEMMAN_BACKEND_<store>` (per-store key first, then
      `MEMMAN_DEFAULT_BACKEND`, then 'sqlite');
    - fail when the resolved value is not a registered backend;
    - fail when the resolved kind is `postgres` and no DSN is reachable
      via `MEMMAN_PG_DSN_<store>` or `MEMMAN_DEFAULT_PG_DSN`.

    Empty data dirs (no stores at all) pass with an empty list.
    """
    from memman import config
    from memman.store.factory import list_stores

    file_values = config.parse_env_file(config.env_file_path(data_dir))
    try:
        stores = list_stores(data_dir)
    except Exception as exc:
        return {
            'name': 'per_store_keys',
            'status': 'fail',
            'detail': {'error': f'list_stores: {exc}'},
            }

    entries: list[dict[str, Any]] = []
    worst = 'pass'
    for store in stores:
        per_key = config.BACKEND_FOR(store)
        per_value = (file_values.get(per_key) or '').strip()
        if per_value:
            kind = per_value.lower()
            source = 'per_store'
        else:
            kind = (file_values.get(config.DEFAULT_BACKEND)
                    or 'sqlite').lower()
            source = 'default'
        entry: dict[str, Any] = {
            'store': store,
            'backend': kind,
            'source': source,
            'error': None,
            }
        if kind not in _KNOWN_BACKENDS:
            entry['error'] = (
                f'unknown backend {kind!r}; registered:'
                f' {", ".join(sorted(_KNOWN_BACKENDS))}')
            worst = 'fail'
        elif kind == 'postgres':
            dsn = (
                file_values.get(config.PG_DSN_FOR(store))
                or file_values.get(config.DEFAULT_PG_DSN))
            if not dsn:
                entry['error'] = (
                    f'no DSN: set {config.PG_DSN_FOR(store)} or'
                    f' {config.DEFAULT_PG_DSN}')
                worst = 'fail'
        entries.append(entry)

    return {
        'name': 'per_store_keys',
        'status': worst,
        'detail': {'stores': entries},
        }


def check_env_permissions() -> dict[str, Any]:
    """Verify ~/.memman/env is 0600 and ~/.memman is 0700.

    Relaxed to a PASS when the files don't exist (fresh install, no
    keys yet — that's a separate problem surfaced by other tools).
    """
    home = Path.home()
    mm_dir = home / '.memman'
    env_file = mm_dir / 'env'
    detail: dict[str, Any] = {'mm_dir': str(mm_dir), 'env_file': str(env_file)}

    if not mm_dir.is_dir():
        return {'name': 'env_permissions', 'status': 'pass',
                'detail': {**detail, 'reason': 'no ~/.memman directory'}}

    dir_mode = stat.S_IMODE(os.stat(mm_dir).st_mode)
    detail['dir_mode'] = oct(dir_mode)

    if not env_file.is_file():
        status = 'pass' if dir_mode & 0o077 == 0 else 'warn'
        return {'name': 'env_permissions', 'status': status,
                'detail': {**detail, 'reason': 'no env file yet'}}

    env_mode = stat.S_IMODE(os.stat(env_file).st_mode)
    detail['env_mode'] = oct(env_mode)

    issues = []
    if env_mode & 0o077:
        issues.append('env file not 0600')
    if dir_mode & 0o077:
        issues.append('~/.memman not 0700')

    detail['issues'] = issues
    status = 'pass' if not issues else 'fail'
    return {'name': 'env_permissions', 'status': status, 'detail': detail}


def check_scheduler_state() -> dict[str, Any]:
    """Compare persisted scheduler state against OS install/active truth.
    """
    from memman.setup.scheduler import status as sch_status

    try:
        s = sch_status()
    except Exception as exc:
        return {
            'name': 'scheduler_state',
            'status': 'warn',
            'detail': {'error': f'{type(exc).__name__}: {exc}'},
            }

    installed = bool(s.get('installed'))
    state = s.get('state')
    drift = bool(s.get('drift'))

    if drift:
        status = 'fail'
    elif not installed:
        status = 'warn'
    elif state == 'off':
        status = 'warn'
    else:
        status = 'pass'

    return {
        'name': 'scheduler_state',
        'status': status,
        'detail': {
            'installed': installed,
            'state': state,
            'active': bool(s.get('active')),
            'drift': drift,
            'interval_seconds': s.get('interval_seconds'),
            },
        }


def check_scheduler_heartbeat(data_dir: str) -> dict[str, Any]:
    """Verify the worker fired within max(3 x interval, 180s).

    Cross-references scheduler state: only fails when the scheduler is
    installed AND active but no recent worker_runs row exists. Empty
    drains rate-limit the heartbeat write to once per 60s wall, so a
    floor of 180s (3 x 60s) avoids false fails at sub-minute intervals
    (interval=0, 1, 10, etc. — serve mode only). At intervals >= 60s
    the 3x multiplier dominates: two consecutive misses indicate a
    real problem; one-miss tolerance handles transient delays.
    """
    from memman.queue import last_worker_run, open_queue_db
    from memman.setup.scheduler import STATE_STARTED
    from memman.setup.scheduler import status as sch_status

    try:
        s = sch_status()
        interval = s.get('interval_seconds')
        state = s.get('state')
        installed = s.get('installed', False)
        platform = s.get('platform', '')
    except Exception:
        interval = None
        state = None
        installed = False
        platform = ''

    if not installed or state != STATE_STARTED:
        return {
            'name': 'scheduler_heartbeat',
            'status': 'pass',
            'detail': {
                'reason': f'scheduler state is {state!r}; no drain expected',
                'installed': installed,
                'state': state,
                'platform': platform,
                },
            }

    if platform == 'serve' and interval is None:
        return {
            'name': 'scheduler_heartbeat',
            'status': 'fail',
            'detail': {
                'reason': ('serve interval not recorded;'
                           ' restart `memman scheduler serve` so the'
                           ' interval file is rewritten'),
                'platform': platform,
                'state': state,
                },
            }

    conn = open_queue_db(data_dir)
    try:
        last = last_worker_run(conn)
    finally:
        conn.close()

    if last is None:
        return {
            'name': 'scheduler_heartbeat',
            'status': 'fail',
            'detail': {
                'reason': ('scheduler is active but no drains recorded yet;'
                           ' check ~/.memman/logs/enrich.log'),
                'state': state,
                'platform': platform,
                'interval_seconds': interval,
                },
            }

    import time as _time
    age = int(_time.time()) - int(last['started_at'])
    detail = {
        'started_at': last['started_at'],
        'age_seconds': age,
        'rows_done': last['rows_done'],
        'rows_failed': last['rows_failed'],
        'error': last['error'],
        'interval_seconds': interval,
        'state': state,
        'platform': platform,
        }

    threshold_fail = (
        max(3 * interval, 180) if interval is not None else 180)
    threshold_warn = (
        max(2 * interval, 120) if interval is not None else 120)
    detail['threshold_fail_seconds'] = threshold_fail
    if last['error']:
        status = 'fail'
    elif age > threshold_fail:
        status = 'fail'
        detail['reason'] = (
            'scheduler is active but worker has not fired in '
            f'{age}s (interval={interval}s, threshold={threshold_fail}s);'
            ' check ~/.memman/logs/enrich.log')
    elif age > threshold_warn:
        status = 'warn'
    else:
        status = 'pass'

    return {
        'name': 'scheduler_heartbeat', 'status': status, 'detail': detail}


DRAIN_HEARTBEAT_STALE_SECONDS = 5 * 60


def check_drain_heartbeat(data_dir: str) -> dict[str, Any]:
    """Warn when any in-progress drain run is past 5 minutes without a beat.

    Iterates `list_stores(data_dir)` and queries each Postgres-backed
    store's per-store `worker_runs` table for rows with `ended_at IS
    NULL` whose `last_heartbeat_at` is older than 5 minutes. SQLite-
    backed stores return an empty list (single-process; drain hangs
    are visible at the foreground prompt). The aggregate status warns
    when any store has stale runs.
    """
    from datetime import datetime, timezone

    from memman.store.factory import _resolve_store_backend, list_stores
    from memman.store.factory import open_backend

    try:
        stores = list_stores(data_dir)
    except Exception as exc:
        return {
            'name': 'drain_heartbeat',
            'status': 'fail',
            'detail': {'error': f'{type(exc).__name__}: {exc}'},
            }

    pg_stores = [
        s for s in stores
        if _resolve_store_backend(s, data_dir) == 'postgres']
    if not pg_stores:
        return {
            'name': 'drain_heartbeat',
            'status': 'pass',
            'detail': {
                'skipped_reason': 'no postgres-backed stores',
                'stores_checked': stores,
                },
            }

    now = datetime.now(timezone.utc)
    stale: list[dict[str, Any]] = []
    in_progress = 0
    failures: list[dict[str, str]] = []
    for store in pg_stores:
        try:
            backend = open_backend(store, data_dir, read_only=True)
        except Exception as exc:
            failures.append({
                'store': store,
                'error': f'{type(exc).__name__}: {exc}',
                })
            continue
        try:
            runs = backend.recent_runs(limit=50)
        except Exception as exc:
            failures.append({
                'store': store,
                'error': f'{type(exc).__name__}: {exc}',
                })
            continue
        finally:
            try:
                backend.close()
            except Exception:
                pass
        for r in runs:
            if r.ended_at is not None:
                continue
            in_progress += 1
            if r.last_heartbeat_at is None:
                continue
            age = (now - r.last_heartbeat_at).total_seconds()
            if age > DRAIN_HEARTBEAT_STALE_SECONDS:
                stale.append({
                    'store': store,
                    'run_id': r.id,
                    'age_seconds': int(age),
                    'last_heartbeat_at': r.last_heartbeat_at.isoformat(),
                    })

    detail: dict[str, Any] = {
        'stores_checked': pg_stores,
        'in_progress': in_progress,
        'stale_runs': stale,
        'threshold_seconds': DRAIN_HEARTBEAT_STALE_SECONDS,
        }
    if failures:
        detail['failures'] = failures
    if failures and not stale:
        return {
            'name': 'drain_heartbeat',
            'status': 'fail',
            'detail': detail,
            }
    if stale:
        return {
            'name': 'drain_heartbeat',
            'status': 'warn',
            'detail': detail,
            }
    return {
        'name': 'drain_heartbeat',
        'status': 'pass',
        'detail': detail,
        }


def check_llm_probe() -> dict[str, Any]:
    """Probe the LLM endpoint with the cheapest possible call.

    Verifies API key validity + endpoint reachability. Subsumes what
    used to be `memman keys test`'s LLM check.
    """
    import time as _time

    detail: dict[str, Any] = {
        'model': None,
        'elapsed_ms': None,
        'sample': None,
        'error': None,
        }
    t0 = _time.monotonic()
    try:
        from memman.exceptions import ConfigError
        from memman.llm.client import get_llm_client
        try:
            client = get_llm_client('fast')
        except ConfigError as exc:
            detail['error'] = str(exc)
            detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
            return {'name': 'llm_probe', 'status': 'fail', 'detail': detail}
        out = client.complete('Reply with exactly: ok', 'probe')
        detail['model'] = getattr(client, 'model', None)
        detail['sample'] = (out or '')[:60]
        detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
        if out:
            return {'name': 'llm_probe', 'status': 'pass', 'detail': detail}
        detail['error'] = 'empty response'
        return {'name': 'llm_probe', 'status': 'fail', 'detail': detail}
    except Exception as exc:
        detail['error'] = f'{type(exc).__name__}: {exc}'
        detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
        return {'name': 'llm_probe', 'status': 'fail', 'detail': detail}


def check_embed_probe() -> dict[str, Any]:
    """Probe the embedding endpoint with the cheapest possible call.

    Subsumes what used to be `memman keys test`'s embed check.
    """
    import time as _time

    detail: dict[str, Any] = {
        'provider': None,
        'model': None,
        'elapsed_ms': None,
        'dim': None,
        'error': None,
        }
    t0 = _time.monotonic()
    try:
        from memman.embed import get_client
        ec = get_client()
        detail['provider'] = getattr(ec, 'name', None)
        detail['model'] = getattr(ec, 'model', None)
        if not ec.available():
            detail['error'] = ec.unavailable_message()
            detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
            return {'name': 'embed_probe', 'status': 'fail', 'detail': detail}
        vec = ec.embed('probe')
        detail['dim'] = len(vec) if vec else 0
        detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
        if vec:
            return {'name': 'embed_probe', 'status': 'pass', 'detail': detail}
        detail['error'] = 'empty embedding'
        return {'name': 'embed_probe', 'status': 'fail', 'detail': detail}
    except Exception as exc:
        detail['error'] = f'{type(exc).__name__}: {exc}'
        detail['elapsed_ms'] = int((_time.monotonic() - t0) * 1000)
        return {'name': 'embed_probe', 'status': 'fail', 'detail': detail}


def check_embed_fingerprint(backend: Backend) -> dict[str, Any]:
    """Compare active client fingerprint against `meta.embed_fingerprint`.

    Surfaces the same mismatch that `assert_consistent` enforces at
    runtime, but as a structured doctor check so the operator sees
    the active/stored values explicitly.
    """
    from memman.embed.fingerprint import active_fingerprint, stored_fingerprint

    detail: dict[str, Any] = {
        'active': None,
        'stored': None,
        'error': None,
        }
    try:
        active = active_fingerprint()
        detail['active'] = {
            'provider': active.provider,
            'model': active.model,
            'dim': active.dim,
            }
    except Exception as exc:
        detail['error'] = f'{type(exc).__name__}: {exc}'
        return {
            'name': 'embed_fingerprint', 'status': 'fail',
            'detail': detail}

    stored = stored_fingerprint(backend)
    if stored is not None:
        detail['stored'] = {
            'provider': stored.provider,
            'model': stored.model,
            'dim': stored.dim,
            }

    if stored is None:
        detail['error'] = (
            "DB not initialized."
            " Run 'memman embed reembed' to initialize this store.")
        return {
            'name': 'embed_fingerprint', 'status': 'fail',
            'detail': detail}
    if stored != active:
        detail['error'] = (
            "Active does not match stored. Run"
            " 'memman scheduler stop && memman embed reembed'"
            " to converge.")
        return {
            'name': 'embed_fingerprint', 'status': 'fail',
            'detail': detail}
    return {
        'name': 'embed_fingerprint', 'status': 'pass',
        'detail': detail}


def check_no_stale_swap_meta(backend: Backend) -> dict[str, Any]:
    """Warn when `embed_swap_*` meta keys persist on a non-swapping store.

    Cutover and abort delete the swap meta keys. Any leftover key
    indicates a regression in the cleanup path and is reported as a
    warning so the operator can investigate.
    """
    leftover = sorted(
        k for k in backend.meta.keys()  # noqa: SIM118
        if k.startswith('embed_swap_'))
    if not leftover:
        return {
            'name': 'no_stale_swap_meta', 'status': 'pass',
            'detail': {'leftover_keys': []}}
    return {
        'name': 'no_stale_swap_meta', 'status': 'warn',
        'detail': {'leftover_keys': leftover}}


def check_provenance_drift(backend: Backend) -> dict[str, Any]:
    """Surface rows whose prompt_version or model_id no longer matches active.

    Reads per-row provenance columns directly. No meta-key fingerprint
    is maintained; the data already lives on each insight.
    """
    from memman import config
    from memman.pipeline.remember import compute_prompt_version

    detail: dict[str, Any] = {
        'active_prompt_version': None,
        'active_model_slow': None,
        'stale_rows': 0,
        'breakdown': [],
        }
    try:
        detail['active_prompt_version'] = compute_prompt_version()
    except Exception as exc:
        detail['error'] = f'compute_prompt_version: {exc}'
        return {
            'name': 'provenance_drift', 'status': 'fail',
            'detail': detail}
    try:
        detail['active_model_slow_canonical'] = config.require(
            config.LLM_MODEL_SLOW_CANONICAL)
    except Exception:
        detail['active_model_slow_canonical'] = None
    try:
        detail['active_model_slow_metadata'] = config.require(
            config.LLM_MODEL_SLOW_METADATA)
    except Exception:
        detail['active_model_slow_metadata'] = None

    provenance = backend.nodes.provenance_distribution()

    active_pv = detail['active_prompt_version']
    active_model = detail['active_model_slow_canonical']
    stale_rows = 0
    breakdown: list[dict[str, Any]] = []
    for pc in provenance:
        is_stale = (
            (pc.prompt_version is not None
             and pc.prompt_version != active_pv)
            or (pc.model_id is not None and active_model is not None
                and pc.model_id != active_model))
        breakdown.append({
            'prompt_version': pc.prompt_version,
            'model_id': pc.model_id,
            'count': pc.count,
            'stale': is_stale,
            })
        if is_stale:
            stale_rows += pc.count
    detail['breakdown'] = breakdown
    detail['stale_rows'] = stale_rows

    if stale_rows == 0:
        return {
            'name': 'provenance_drift', 'status': 'pass',
            'detail': detail}
    detail['remediation'] = (
        "Run 'memman graph rebuild' to re-enrich every stale row,"
        " or scope: update insights set linked_at=null,"
        " enriched_at=null where prompt_version=<old> or model_id=<old>;"
        " then drain the scheduler.")
    return {
        'name': 'provenance_drift', 'status': 'warn',
        'detail': detail}


def run_all_checks(
        backend: Backend,
        data_dir: str | None = None) -> dict[str, Any]:
    """Run all health checks and return results with overall status.

    Routes every check through the Backend Protocol so SQLite and
    Postgres are both supported.
    """
    total = backend.nodes.count_active()
    checks = []
    if total > 0:
        checks.extend([
            check_integrity(backend),
            check_schema_columns(backend),
            check_enrichment_coverage(backend),
            check_oplog_delta_coverage(backend),
            check_orphan_insights(backend),
            check_dangling_edges(backend),
            check_embedding_consistency(backend),
            check_embed_fingerprint(backend),
            check_no_stale_swap_meta(backend),
            check_provenance_drift(backend),
            check_edge_degree(backend),
            ])
    else:
        checks.extend([
            check_schema_columns(backend),
            check_embed_fingerprint(backend),
            check_no_stale_swap_meta(backend),
            ])
    if data_dir:
        checks.extend((
            check_queue_schema(data_dir),
            check_queue_backlog(data_dir),
            check_scheduler_heartbeat(data_dir),
            check_drain_heartbeat(data_dir),
            check_env_completeness(),
            check_per_store_keys(data_dir),
            check_env_permissions(),
            check_scheduler_state(),
            check_llm_probe(),
            check_embed_probe(),
            check_optional_extras()))
    if total == 0 and data_dir is None:
        return {'status': 'empty', 'total_active': 0, 'checks': []}
    statuses = [c['status'] for c in checks]
    if 'fail' in statuses:
        overall = 'fail'
    elif 'warn' in statuses:
        overall = 'warn'
    else:
        overall = 'pass'
    return {'status': overall, 'total_active': total, 'checks': checks}
