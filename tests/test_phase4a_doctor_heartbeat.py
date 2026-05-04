"""Phase 4a slice 5 -- `memman doctor` drain-heartbeat consumer.

Verifies the new `check_drain_heartbeat` consumer warns when any
in-progress `queue.worker_runs` row has not heartbeat-updated in 5+
minutes. Postgres-only check; SQLite mode skips the check with
status pass.

Phase 4 gate item 3: `pytest -k "test_doctor_warns_no_drain_heartbeat_in_5m" -m postgres -v`.
"""

from datetime import datetime, timedelta, timezone

import psycopg
import pytest

pytestmark = pytest.mark.postgres


def _set_backend_postgres(env_file_writer, dsn: str) -> None:
    """Switch the test env to postgres mode for the duration."""
    env_file_writer('MEMMAN_BACKEND', 'postgres')
    env_file_writer('MEMMAN_PG_DSN', dsn)


def _ensure_queue_schema(dsn: str) -> None:
    """Trigger queue schema creation by opening a throwaway store."""
    from memman.store.postgres import PostgresCluster
    cluster = PostgresCluster(dsn=dsn)
    try:
        cluster.drop_store(store='hb_doctor_setup', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hb_doctor_setup', data_dir='')
    try:
        backend.close()
    finally:
        try:
            cluster.drop_store(store='hb_doctor_setup', data_dir='')
        except Exception:
            pass


def test_doctor_skips_drain_heartbeat_on_sqlite():
    """SQLite mode: drain_heartbeat returns pass with skipped_reason."""
    from memman.doctor import check_drain_heartbeat
    result = check_drain_heartbeat()
    assert result['name'] == 'drain_heartbeat'
    assert result['status'] == 'pass'
    assert 'skipped_reason' in result['detail']


def test_doctor_passes_when_no_in_progress_runs(env_file, pg_dsn):
    """Postgres mode with no in-progress runs: status pass."""
    _set_backend_postgres(env_file, pg_dsn)
    _ensure_queue_schema(pg_dsn)
    from memman.doctor import check_drain_heartbeat

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE queue.worker_runs SET ended_at = now()'
                ' WHERE ended_at IS NULL')

    result = check_drain_heartbeat()
    assert result['status'] == 'pass'
    assert result['detail']['in_progress'] == 0
    assert result['detail']['stale_runs'] == []


def test_doctor_warns_no_drain_heartbeat_in_5m(env_file, pg_dsn):
    """Postgres mode: in-progress run with stale heartbeat -> warn.

    Phase 4 gate item 3.
    """
    _set_backend_postgres(env_file, pg_dsn)
    from memman.doctor import check_drain_heartbeat

    stale = datetime.now(timezone.utc) - timedelta(minutes=10)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE queue.worker_runs SET ended_at = now()'
                ' WHERE ended_at IS NULL')
            cur.execute(
                'INSERT INTO queue.worker_runs'
                ' (started_at, ended_at, last_heartbeat_at)'
                ' VALUES (%s, NULL, %s) RETURNING id',
                (stale, stale))
            stale_id = cur.fetchone()[0]
    try:
        result = check_drain_heartbeat()
        assert result['status'] == 'warn'
        stale_runs = result['detail']['stale_runs']
        assert any(s['run_id'] == stale_id for s in stale_runs)
        match = next(
            s for s in stale_runs if s['run_id'] == stale_id)
        assert match['age_seconds'] >= 5 * 60
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE queue.worker_runs SET ended_at = now()'
                    ' WHERE id = %s',
                    (stale_id,))


def test_doctor_no_warn_for_fresh_heartbeat(env_file, pg_dsn):
    """In-progress run with recent heartbeat does NOT warn."""
    _set_backend_postgres(env_file, pg_dsn)
    from memman.doctor import check_drain_heartbeat

    fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE queue.worker_runs SET ended_at = now()'
                ' WHERE ended_at IS NULL')
            cur.execute(
                'INSERT INTO queue.worker_runs'
                ' (started_at, ended_at, last_heartbeat_at)'
                ' VALUES (%s, NULL, %s) RETURNING id',
                (fresh, fresh))
            fresh_id = cur.fetchone()[0]
    try:
        result = check_drain_heartbeat()
        assert result['status'] == 'pass'
        assert result['detail']['stale_runs'] == []
        assert result['detail']['in_progress'] >= 1
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE queue.worker_runs SET ended_at = now()'
                    ' WHERE id = %s',
                    (fresh_id,))
