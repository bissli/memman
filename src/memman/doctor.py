"""Health checks for a memman store.

Runs read-only diagnostics and reports per-check pass/warn/fail
with an overall worst-status summary.
"""

import statistics


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


def run_all_checks(db: 'DB', data_dir: str | None = None) -> dict:
    """Run all health checks and return results with overall status."""
    total = db._query(
        'SELECT COUNT(*) FROM insights WHERE deleted_at IS NULL'
        ).fetchone()[0]
    checks = [
        check_sqlite_integrity(db),
        check_enrichment_coverage(db),
        check_orphan_insights(db),
        check_dangling_edges(db),
        check_embedding_consistency(db),
        check_edge_degree(db),
        ] if total > 0 else []
    if data_dir:
        checks.append(check_queue_backlog(data_dir))
    if total == 0 and not checks:
        return {'status': 'empty', 'total_active': 0, 'checks': []}
    statuses = [c['status'] for c in checks]
    if 'fail' in statuses:
        overall = 'fail'
    elif 'warn' in statuses:
        overall = 'warn'
    else:
        overall = 'pass'
    return {'status': overall, 'total_active': total, 'checks': checks}
