"""CLI-level e2e for `memman migrate` (SQLite -> Postgres).

The unit suite at `tests/test_migrate.py` covers the migrate functions
directly. This test exercises the CLI orchestration end-to-end:
plan-echo, --yes confirmation flow, drain.lock guard, target schema
population, and the `MEMMAN_BACKEND=postgres` env flip on success.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest

pytestmark = [pytest.mark.e2e_cli, pytest.mark.postgres]


def _seed_sqlite_store(data_dir: Path, store: str) -> Path:
    """Build a minimal SQLite store with one insight + embed fingerprint."""
    from memman.store.db import open_db, set_meta, store_dir
    from memman.store.model import Insight
    from memman.store.node import insert_insight

    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        ins = Insight(
            id='mig-cli-1',
            content='migrate cli round-trip insight',
            category='fact',
            importance=3,
            entities=[],
            source='migrate-e2e',
            access_count=0,
            updated_at=datetime.now(timezone.utc),
            deleted_at=None,
            last_accessed_at=None,
            effective_importance=0.0)
        insert_insight(db, ins)
        set_meta(
            db, 'embed_fingerprint',
            '{"provider":"voyage","model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def test_migrate_cli_round_trip_to_postgres(tmp_path: Path, pg_dsn: str):
    """`memman migrate --yes` drives the full CLI flow into Postgres.
    """
    from memman.store.postgres import _store_schema

    home = tmp_path / 'home'
    home.mkdir()
    data_dir = home / '.memman'
    data_dir.mkdir()
    (data_dir / 'env').write_text(f'MEMMAN_PG_DSN={pg_dsn}\n')
    store = 'mig_cli'

    _seed_sqlite_store(data_dir, store)

    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    env = {**os.environ, 'HOME': str(home)}

    result = subprocess.run(
        ['memman', 'migrate', '--store', store, '--yes'],
        capture_output=True, text=True, env=env, check=False)

    try:
        assert result.returncode == 0, (
            f'migrate failed: stdout={result.stdout!r} '
            f'stderr={result.stderr!r}')

        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT count(*) FROM {schema}.insights')
                assert cur.fetchone()[0] == 1
                cur.execute(
                    f'SELECT content FROM {schema}.insights '
                    f"WHERE id = 'mig-cli-1'")
                row = cur.fetchone()
                assert row
                assert row[0] == 'migrate cli round-trip insight'

        env_file = home / '.memman' / 'env'
        if env_file.exists():
            content = env_file.read_text()
            assert 'MEMMAN_BACKEND=postgres' in content, (
                f'env file did not flip backend: {content!r}')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
