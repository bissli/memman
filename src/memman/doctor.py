"""Health checks for a memman store.

Runs read-only diagnostics and reports per-check pass/warn/fail
with an overall worst-status summary.
"""

import os
import stat
import statistics
from pathlib import Path


def check_sqlite_integrity(db: 'DB') -> dict:
    """Run PRAGMA integrity_check."""
    row = db._query('PRAGMA integrity_check').fetchone()
    result = row[0] if row else 'unknown'
    status = 'pass' if result == 'ok' else 'fail'
    return {'name': 'sqlite_integrity', 'status': status,
            'detail': {'result': result}}


def check_enrichment_coverage(db: 'DB') -> dict:
    """Check that embedding, keywords, summary, semantic_facts are populated."""
    row = db._query(
        'SELECT COUNT(*) AS total,'
        " SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END),"
        " SUM(CASE WHEN keywords IS NULL OR keywords = '' THEN 1 ELSE 0 END),"
        " SUM(CASE WHEN summary IS NULL OR summary = '' THEN 1 ELSE 0 END),"
        " SUM(CASE WHEN semantic_facts IS NULL OR semantic_facts = ''"
        "   THEN 1 ELSE 0 END)"
        ' FROM insights WHERE deleted_at IS NULL').fetchone()
    total, miss_emb, miss_kw, miss_sum, miss_sf = row
    if total == 0:
        return {'name': 'enrichment_coverage', 'status': 'pass',
                'detail': {'total_active': 0, 'coverage_pct': 100.0}}
    missing_any = max(miss_emb, miss_kw, miss_sum, miss_sf)
    coverage_pct = round((total - missing_any) / total * 100, 1)
    if miss_emb == 0 and miss_kw == 0 and miss_sum == 0 and miss_sf == 0:
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
            'missing_embedding': miss_emb,
            'missing_keywords': miss_kw,
            'missing_summary': miss_sum,
            'missing_semantic_facts': miss_sf,
            'coverage_pct': coverage_pct,
            },
        }


def check_orphan_insights(db: 'DB') -> dict:
    """Find active insights with zero edges."""
    total = db._query(
        'SELECT COUNT(*) FROM insights WHERE deleted_at IS NULL'
        ).fetchone()[0]
    orphan_count = db._query(
        'SELECT COUNT(*) FROM insights i'
        ' WHERE i.deleted_at IS NULL'
        ' AND NOT EXISTS ('
        '  SELECT 1 FROM edges e'
        '  WHERE e.source_id = i.id OR e.target_id = i.id'
        ')').fetchone()[0]
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


def check_dangling_edges(db: 'DB') -> dict:
    """Find edges referencing deleted or missing insights."""
    rows = db._query(
        'SELECT e.edge_type, COUNT(*) FROM edges e'
        ' WHERE NOT EXISTS ('
        '  SELECT 1 FROM insights i'
        '  WHERE i.id = e.source_id AND i.deleted_at IS NULL'
        ') OR NOT EXISTS ('
        '  SELECT 1 FROM insights i'
        '  WHERE i.id = e.target_id AND i.deleted_at IS NULL'
        ') GROUP BY e.edge_type').fetchall()
    total = sum(cnt for _, cnt in rows)
    by_type = dict(rows)
    status = 'pass' if total == 0 else 'fail'
    return {
        'name': 'dangling_edges',
        'status': status,
        'detail': {'count': total, 'by_type': by_type},
        }


def check_embedding_consistency(db: 'DB') -> dict:
    """Verify all embeddings have the same byte size."""
    rows = db._query(
        'SELECT LENGTH(embedding), COUNT(*) FROM insights'
        ' WHERE deleted_at IS NULL AND embedding IS NOT NULL'
        ' GROUP BY LENGTH(embedding)').fetchall()
    sizes = {str(size): cnt for size, cnt in rows}
    if len(sizes) <= 1:
        status = 'pass'
    else:
        status = 'fail'
    return {
        'name': 'embedding_consistency',
        'status': status,
        'detail': {'sizes': sizes},
        }


def check_edge_degree(db: 'DB') -> dict:
    """Compute degree distribution stats across active insights."""
    id_rows = db._query(
        'SELECT id FROM insights WHERE deleted_at IS NULL').fetchall()
    if not id_rows:
        return {'name': 'edge_degree', 'status': 'pass',
                'detail': {'min': 0, 'max': 0, 'median': 0, 'mean': 0.0}}
    degree_rows = db._query(
        'SELECT id, SUM(cnt) AS degree FROM ('
        '  SELECT source_id AS id, COUNT(*) AS cnt'
        '  FROM edges GROUP BY source_id'
        '  UNION ALL'
        '  SELECT target_id AS id, COUNT(*) AS cnt'
        '  FROM edges GROUP BY target_id'
        ') GROUP BY id').fetchall()
    degree_by_id = {row[0]: row[1] for row in degree_rows}
    active_ids = {row[0] for row in id_rows}
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


def check_schema_columns(db: 'DB') -> dict:
    """Verify the insights table has the canonical provenance columns.

    Single-user canonical-schema policy: missing columns mean the DB
    predates a schema change; the fix is a one-off ALTER TABLE.
    """
    rows = db._query('PRAGMA table_info(insights)').fetchall()
    present = {row[1] for row in rows}
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


def check_last_worker_run(data_dir: str) -> dict:
    """Verify the worker fired within 2 x the scheduler interval.

    Cross-references scheduler state: only fails when the scheduler is
    installed AND active but no recent worker_runs row exists. In paused
    or disabled states a stale `last_worker_run` is expected.

    This is the rename-agnostic catch-all that detects ExecStart drift,
    PATH changes, chmod regressions, and other silent-failure modes
    without ever inspecting the unit file's text.
    """
    from memman.queue import last_worker_run, open_queue_db
    from memman.setup.scheduler import STATE_STARTED
    from memman.setup.scheduler import status as sch_status

    try:
        s = sch_status()
        interval = s.get('interval_seconds')
        state = s.get('state')
        installed = s.get('installed', False)
    except Exception:
        interval = None
        state = None
        installed = False

    if not installed or state != STATE_STARTED:
        return {
            'name': 'last_worker_run',
            'status': 'pass',
            'detail': {
                'reason': f'scheduler state is {state!r}; no drain expected',
                'installed': installed,
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
            'name': 'last_worker_run',
            'status': 'fail',
            'detail': {
                'reason': 'scheduler is active but no drains recorded yet;'
                ' check ~/.memman/logs/enrich.err and re-run `memman install`'
                ' if the unit was upgraded',
                'state': state,
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
        }

    if last['error']:
        status = 'fail'
    elif interval and age > 2 * interval:
        status = 'fail'
        detail['reason'] = (
            'scheduler is active but worker has not fired in '
            f'{age}s (interval={interval}s); check '
            '~/.memman/logs/enrich.err')
    elif interval and age > interval + 60:
        status = 'warn'
    else:
        status = 'pass'

    return {'name': 'last_worker_run', 'status': status, 'detail': detail}


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
            client = get_llm_client()
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
    from memman.embed.fingerprint import (
        active_fingerprint, stored_fingerprint)

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


def run_all_checks(db: 'DB', data_dir: str | None = None) -> dict:
    """Run all health checks and return results with overall status."""
    total = db._query(
        'SELECT COUNT(*) FROM insights WHERE deleted_at IS NULL'
        ).fetchone()[0]
    checks = []
    if total > 0:
        checks.extend([
            check_sqlite_integrity(db),
            check_schema_columns(db),
            check_enrichment_coverage(db),
            check_orphan_insights(db),
            check_dangling_edges(db),
            check_embedding_consistency(db),
            check_embed_fingerprint(db),
            check_edge_degree(db),
            ])
    else:
        checks.extend([
            check_schema_columns(db),
            check_embed_fingerprint(db),
            ])
    if data_dir:
        checks.extend((
            check_queue_schema(data_dir),
            check_queue_backlog(data_dir),
            check_last_worker_run(data_dir),
            check_env_permissions(),
            check_scheduler_state(),
            check_llm_probe(),
            check_embed_probe()))
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
