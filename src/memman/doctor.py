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

from memman.store.backend import Backend


def check_integrity(backend: Backend) -> dict:
    """Run the backend's integrity probe."""
    result = backend.integrity_check()
    ok = bool(result.get('ok'))
    return {
        'name': 'integrity',
        'status': 'pass' if ok else 'fail',
        'detail': {'result': result.get('detail')},
        }


def check_enrichment_coverage(backend: Backend) -> dict:
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


def check_orphan_insights(backend: Backend) -> dict:
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


def check_dangling_edges(backend: Backend) -> dict:
    """Find edges referencing deleted or missing insights."""
    by_type = backend.edges.count_dangling_by_type()
    total = sum(by_type.values())
    status = 'pass' if total == 0 else 'fail'
    return {
        'name': 'dangling_edges',
        'status': status,
        'detail': {'count': total, 'by_type': by_type},
        }


def check_embedding_consistency(backend: Backend) -> dict:
    """Verify all embeddings have the same size (byte length on SQLite,
    pgvector dimension on Postgres)."""
    dist = backend.nodes.embedding_size_distribution()
    sizes = {str(size): cnt for size, cnt in dist.items()}
    status = 'pass' if len(sizes) <= 1 else 'fail'
    return {
        'name': 'embedding_consistency',
        'status': status,
        'detail': {'sizes': sizes},
        }


def check_edge_degree(backend: Backend) -> dict:
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


def check_queue_backlog(data_dir: str) -> dict:
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


def check_schema_columns(backend: Backend) -> dict:
    """Verify the insights table has the canonical provenance columns.

    Single-user canonical-schema policy: missing columns mean the DB
    predates a schema change; the fix is a one-off ALTER TABLE.
    """
    present = backend.introspect_columns('insights')
    missing = sorted(EXPECTED_INSIGHT_COLUMNS - present)
    status = 'pass' if not missing else 'fail'
    return {
        'name': 'schema_columns',
        'status': status,
        'detail': {'missing': missing},
        }


def check_queue_schema(data_dir: str) -> dict:
    """Verify queue.db has the canonical tables (queue + worker_runs).
    """
    from memman.queue import open_queue_db

    conn = open_queue_db(data_dir)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
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


def check_optional_extras() -> dict:
    """Report which `memman[extras]` install groups resolve at runtime.

    Always passes; the result is informational. Lets users verify their
    install matches their `MEMMAN_BACKEND` choice (e.g., backend=postgres
    requires the `postgres` extra to be active).
    """
    from memman import extras
    active = extras.detect_active_extras()
    return {
        'name': 'optional_extras',
        'status': 'pass',
        'detail': {'active': active},
        }


def check_env_completeness() -> dict:
    """Verify ~/.memman/env contains every INSTALLABLE_KEYS entry.

    Catches the upgrade case where a new release adds a key the user's
    existing file lacks. Reports the missing keys so the user can run
    `memman install` to repopulate. Optional secrets are not flagged
    when absent (the user may not have configured an alternate provider).
    """
    from memman import config

    path = config.env_file_path()
    parsed = config.parse_env_file(path)

    optional_secrets = {config.OPENAI_EMBED_API_KEY}
    if parsed.get(config.BACKEND, 'sqlite') != 'postgres':
        optional_secrets |= {config.PG_DSN}
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


def check_env_permissions() -> dict:
    """Verify ~/.memman/env is 0600 and ~/.memman is 0700.

    Relaxed to a PASS when the files don't exist (fresh install, no
    keys yet — that's a separate problem surfaced by other tools).
    """
    home = Path.home()
    mm_dir = home / '.memman'
    env_file = mm_dir / 'env'
    detail: dict = {'mm_dir': str(mm_dir), 'env_file': str(env_file)}

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


def check_scheduler_state() -> dict:
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


def check_scheduler_heartbeat(data_dir: str) -> dict:
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


def check_drain_heartbeat() -> dict:
    """Postgres-only: warn if any in-progress drain run is past 5 minutes
    without a heartbeat.

    Reads `queue.worker_runs` for rows with `ended_at IS NULL` and
    flags any whose `last_heartbeat_at` is older than 5 minutes (the
    documented threshold). No-op on SQLite (single-process; drain
    hangs are visible to the operator at the foreground prompt).

    Phase 4a installs the schema + Protocol verbs. Phase 4b's
    drain-loop dispatch will start populating live rows; until then
    a healthy Postgres deployment returns 'pass' with no in-progress
    rows.
    """
    from datetime import datetime, timezone

    from memman import config

    backend_name = (config.get(config.BACKEND) or 'sqlite').lower()
    if backend_name != 'postgres':
        return {
            'name': 'drain_heartbeat',
            'status': 'pass',
            'detail': {'skipped_reason': 'backend is sqlite'},
            }
    dsn = config.get(config.PG_DSN)
    if not dsn:
        return {
            'name': 'drain_heartbeat',
            'status': 'warn',
            'detail': {'error': 'MEMMAN_PG_DSN not set'},
            }

    try:
        from memman.store.postgres import PostgresQueueBackend
        queue = PostgresQueueBackend(dsn=dsn)
        runs = queue.recent_runs(limit=50)
    except Exception as exc:
        return {
            'name': 'drain_heartbeat',
            'status': 'fail',
            'detail': {'error': f'{type(exc).__name__}: {exc}'},
            }

    now = datetime.now(timezone.utc)
    stale: list[dict] = []
    in_progress = 0
    for r in runs:
        if r.ended_at is not None:
            continue
        in_progress += 1
        if r.last_heartbeat_at is None:
            continue
        age = (now - r.last_heartbeat_at).total_seconds()
        if age > DRAIN_HEARTBEAT_STALE_SECONDS:
            stale.append({
                'run_id': r.id,
                'age_seconds': int(age),
                'last_heartbeat_at': r.last_heartbeat_at.isoformat(),
                })

    detail = {
        'in_progress': in_progress,
        'stale_runs': stale,
        'threshold_seconds': DRAIN_HEARTBEAT_STALE_SECONDS,
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


def check_llm_probe() -> dict:
    """Probe the LLM endpoint with the cheapest possible call.

    Verifies API key validity + endpoint reachability. Subsumes what
    used to be `memman keys test`'s LLM check.
    """
    import time as _time

    detail: dict = {
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


def check_embed_probe() -> dict:
    """Probe the embedding endpoint with the cheapest possible call.

    Subsumes what used to be `memman keys test`'s embed check.
    """
    import time as _time

    detail: dict = {
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


def check_embed_fingerprint(db: 'DB') -> dict:
    """Compare active client fingerprint against `meta.embed_fingerprint`.

    Surfaces the same mismatch that `assert_consistent` enforces at
    runtime, but as a structured doctor check so the operator sees
    the active/stored values explicitly.
    """
    from memman.embed.fingerprint import active_fingerprint, stored_fingerprint

    detail: dict = {
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

    stored = stored_fingerprint(db)
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


def check_provenance_drift(backend: Backend) -> dict:
    """Surface rows whose prompt_version or model_id no longer matches active.

    Reads per-row provenance columns directly. No meta-key fingerprint
    is maintained; the data already lives on each insight.
    """
    from memman import config
    from memman.pipeline.remember import compute_prompt_version

    detail: dict = {
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
    breakdown: list[dict] = []
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
        " or scope: UPDATE insights SET linked_at=NULL,"
        " enriched_at=NULL WHERE prompt_version=<old> OR model_id=<old>;"
        " then drain the scheduler.")
    return {
        'name': 'provenance_drift', 'status': 'warn',
        'detail': detail}


def run_all_checks(
        backend: Backend, db: 'DB',
        data_dir: str | None = None) -> dict:
    """Run all health checks and return results with overall status.

    Takes both `backend` (for the Backend-Protocol-routed checks) and
    `db` (still needed by `check_embed_fingerprint` until
    `stored_fingerprint` is refactored to take a Backend).
    """
    total = backend.nodes.count_active()
    checks = []
    if total > 0:
        checks.extend([
            check_integrity(backend),
            check_schema_columns(backend),
            check_enrichment_coverage(backend),
            check_orphan_insights(backend),
            check_dangling_edges(backend),
            check_embedding_consistency(backend),
            check_embed_fingerprint(db),
            check_provenance_drift(backend),
            check_edge_degree(backend),
            ])
    else:
        checks.extend([
            check_schema_columns(backend),
            check_embed_fingerprint(db),
            ])
    if data_dir:
        checks.extend((
            check_queue_schema(data_dir),
            check_queue_backlog(data_dir),
            check_scheduler_heartbeat(data_dir),
            check_drain_heartbeat(),
            check_env_completeness(),
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
