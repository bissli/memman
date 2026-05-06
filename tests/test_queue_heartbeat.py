"""Heartbeat schema + QueueBackend verb tests.

Covers `queue.worker_runs.last_heartbeat_at` via live DB round-trip:

- `start_run() -> int` -- inserts a fresh worker_runs row with
  `started_at = now()` and `last_heartbeat_at = now()`, returns id.
- `beat_run(run_id: int) -> None` -- advances `last_heartbeat_at`.

Column presence and Protocol shape are proven transitively by the
`start_run` / `beat_run` round-trip tests against a real container.
"""

import time

import psycopg
import pytest

pytestmark = pytest.mark.postgres


def test_start_run_inserts_worker_run_with_heartbeat(pg_dsn):
    """start_run on Postgres inserts a row with last_heartbeat_at set."""
    from memman.store.postgres import PostgresCluster

    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store='hb_test', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hb_test', data_dir='')
    from memman.store.postgres import PostgresQueueBackend
    queue = PostgresQueueBackend(pg_dsn)
    try:
        run_id = queue.start_run()
        assert isinstance(run_id, int)
        assert run_id > 0

        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT started_at, last_heartbeat_at, ended_at'
                    ' FROM queue.worker_runs WHERE id = %s',
                    (run_id,))
                row = cur.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is None
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            cluster.drop_store(store='hb_test', data_dir='')
        except Exception:
            pass


def test_beat_run_updates_heartbeat_timestamp(pg_dsn):
    """beat_run advances last_heartbeat_at on the named run."""
    from memman.store.postgres import PostgresCluster

    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store='hb_beat', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hb_beat', data_dir='')
    from memman.store.postgres import PostgresQueueBackend
    queue = PostgresQueueBackend(pg_dsn)
    try:
        run_id = queue.start_run()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT last_heartbeat_at'
                    ' FROM queue.worker_runs WHERE id = %s',
                    (run_id,))
                first = cur.fetchone()[0]
        time.sleep(0.05)
        queue.beat_run(run_id)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT last_heartbeat_at'
                    ' FROM queue.worker_runs WHERE id = %s',
                    (run_id,))
                second = cur.fetchone()[0]
        assert second > first
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            cluster.drop_store(store='hb_beat', data_dir='')
        except Exception:
            pass


def test_recent_runs_includes_last_heartbeat_at(pg_dsn):
    """recent_runs populates the WorkerRun.last_heartbeat_at field."""
    from memman.store.postgres import PostgresCluster

    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store='hb_recent', data_dir='')
    except Exception:
        pass
    backend = cluster.open(store='hb_recent', data_dir='')
    from memman.store.postgres import PostgresQueueBackend
    queue = PostgresQueueBackend(pg_dsn)
    try:
        run_id = queue.start_run()
        runs = queue.recent_runs(limit=5)
        match = [r for r in runs if r.id == run_id]
        assert match
        assert match[0].last_heartbeat_at is not None
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            cluster.drop_store(store='hb_recent', data_dir='')
        except Exception:
            pass
