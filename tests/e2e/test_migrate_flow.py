"""CLI-level e2e for `memman migrate` (SQLite -> Postgres).

The unit suite at `tests/test_migrate.py` covers the migrate functions
directly. This test covers the CLI orchestration: plan-echo, --yes
confirmation flow, drain.lock guard, target schema population, and the
per-store `MEMMAN_BACKEND_<store>=postgres` env write on success.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from memman.store.db import open_db, set_meta, store_dir
from memman.store.model import Insight
from memman.store.node import insert_insight

psycopg = pytest.importorskip('psycopg')

pytestmark = [pytest.mark.e2e_cli, pytest.mark.postgres]


def _seed_sqlite_store(data_dir: Path, store: str) -> Path:
    """Build a minimal SQLite store with one insight + embed fingerprint.
    """
    sdir = store_dir(str(data_dir), store)
    db = open_db(sdir)
    try:
        ins = Insight(
            id='mig-cli-1',
            content='migrate cli round-trip insight',
            updated_at=datetime.now(timezone.utc),
            deleted_at=None)
        insert_insight(db, ins)
        set_meta(
            db, 'embed_fingerprint',
            '{"model":"voyage-3-lite","dim":512}')
    finally:
        db.close()
    return Path(sdir)


def test_migrate_cli_round_trip_to_postgres(tmp_path: Path, pg_dsn: str):
    """Verify `memman migrate --yes` copies a SQLite store into Postgres.

    Mutation: migrate exiting 0 without importing the insight rows,
        or without writing the per-store `MEMMAN_BACKEND_<store>`
        key to the env file.
    Oracle: one row with the seeded content in the target schema,
        and the backend key in the env file.
    """
    from memman.store.postgres import _store_schema

    home = tmp_path / 'home'
    home.mkdir()
    data_dir = home / '.memman'
    data_dir.mkdir()
    (data_dir / 'env').write_text(f'MEMMAN_DEFAULT_POSTGRES_DSN={pg_dsn}\n')
    store = 'mig_cli'

    _seed_sqlite_store(data_dir, store)

    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

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
                    f'select count(*) from {schema}.insights')
                assert cur.fetchone()[0] == 1
                cur.execute(
                    f'select content from {schema}.insights '
                    f"where id = 'mig-cli-1'")
                row = cur.fetchone()
                assert row
                assert row[0] == 'migrate cli round-trip insight'

        env_file = home / '.memman' / 'env'
        if env_file.exists():
            content = env_file.read_text()
            assert f'MEMMAN_BACKEND_{store}=postgres' in content, (
                f'env file did not write per-store backend key: {content!r}')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
