"""Phase 4b slice 3 -- drain-loop dispatch + inline beat_run per row.

When `MEMMAN_BACKEND=postgres`, the drain loop opens a postgres
`worker_runs` row at start (`start_run`) and advances its
`last_heartbeat_at` once per row processed (`beat_run`). The
existing SQLite-backed drain plumbing stays operationally
canonical; postgres heartbeat tracking is added alongside as
best-effort monitoring infrastructure (failures don't abort the
drain).

The actual drain loop in `cli.py:_drain_queue` is large; these
tests exercise the helpers `_start_postgres_heartbeat` and
`_beat_postgres_heartbeat` directly to keep the smoke focused.
"""

import psycopg
import pytest

pytestmark = pytest.mark.postgres


def _ensure_queue_schema(dsn: str) -> None:
    from memman.store.postgres import PostgresCluster
    cluster = PostgresCluster(dsn=dsn)
    try:
        cluster.drop_store(store='hb_drain_setup', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hb_drain_setup', data_dir='')
    try:
        backend.close()
    finally:
        try:
            cluster.drop_store(store='hb_drain_setup', data_dir='')
        except Exception:
            pass


def test_start_heartbeat_returns_none_on_sqlite_mode(env_file):
    """Sqlite mode: start helper returns (None, None)."""
    env_file('MEMMAN_BACKEND', 'sqlite')
    from memman.cli import _start_postgres_heartbeat
    pg_queue, pg_run_id = _start_postgres_heartbeat(record_run=True)
    assert pg_queue is None
    assert pg_run_id is None


def test_start_heartbeat_returns_none_when_record_run_false(env_file, pg_dsn):
    """`record_run=False` skips even on postgres mode."""
    env_file('MEMMAN_BACKEND', 'postgres')
    env_file('MEMMAN_PG_DSN', pg_dsn)
    from memman.cli import _start_postgres_heartbeat
    pg_queue, pg_run_id = _start_postgres_heartbeat(record_run=False)
    assert pg_queue is None
    assert pg_run_id is None


def test_start_heartbeat_inserts_worker_run_on_postgres_mode(
        env_file, pg_dsn):
    """Postgres mode: helper inserts a worker_runs row + returns id."""
    env_file('MEMMAN_BACKEND', 'postgres')
    env_file('MEMMAN_PG_DSN', pg_dsn)
    _ensure_queue_schema(pg_dsn)
    from memman.cli import _start_postgres_heartbeat
    pg_queue, pg_run_id = _start_postgres_heartbeat(record_run=True)
    assert pg_queue is not None
    assert isinstance(pg_run_id, int)
    assert pg_run_id > 0
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT last_heartbeat_at FROM queue.worker_runs'
                ' WHERE id = %s',
                (pg_run_id,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] is not None


def test_beat_heartbeat_advances_timestamp(env_file, pg_dsn):
    """`_beat_postgres_heartbeat` updates `last_heartbeat_at`."""
    env_file('MEMMAN_BACKEND', 'postgres')
    env_file('MEMMAN_PG_DSN', pg_dsn)
    _ensure_queue_schema(pg_dsn)
    from memman.cli import (
        _beat_postgres_heartbeat,
        _start_postgres_heartbeat,
    )
    import time
    pg_queue, pg_run_id = _start_postgres_heartbeat(record_run=True)
    assert pg_run_id is not None
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT last_heartbeat_at FROM queue.worker_runs'
                ' WHERE id = %s',
                (pg_run_id,))
            first = cur.fetchone()[0]
    time.sleep(0.05)
    _beat_postgres_heartbeat(pg_queue, pg_run_id)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT last_heartbeat_at FROM queue.worker_runs'
                ' WHERE id = %s',
                (pg_run_id,))
            second = cur.fetchone()[0]
    assert second > first


def test_beat_heartbeat_no_op_with_none_args():
    """Beat helper is a no-op when init returned None,None."""
    from memman.cli import _beat_postgres_heartbeat
    _beat_postgres_heartbeat(None, None)
