"""Phase 5 e2e -- rolling-deploy schema-version skew refusal.

Drives the `_apply_pending_migrations` `stored > code_version`
refusal: if a newer memman binary writes `meta.pg_schema_version =
N+1`, an older binary that only knows version N must refuse to open
that store. Phase 2.5 cut `application_version` as redundant; this
test confirms `pg_schema_version` alone is enough.
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.errors import BackendError
from memman.store.postgres import (
    _PG_SCHEMA_VERSION,
    PostgresCluster,
    _store_schema,
)

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def _safe(s: str) -> str:
    out = ''.join(c if c.isalnum() else '_' for c in s).lower()
    if out and not out[0].isalpha():
        out = 'p_' + out
    return out[:40] or 'p_test'


def test_open_refuses_when_stored_version_ahead_of_code(pg_dsn, request):
    """Bumped pg_schema_version raises BackendError on open.

    Setup: open a store at the current code version. Manually bump
    `meta.pg_schema_version` to N+1 (simulating a newer binary having
    upgraded the store). Reopening with the current binary must
    raise `BackendError` with an "upgrade memman" hint.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass

    backend = cluster.open(store=store, data_dir='')
    backend.close()

    schema = _store_schema(store)
    ahead = _PG_SCHEMA_VERSION + 1
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'INSERT INTO {schema}.meta (key, value)'
                " VALUES ('pg_schema_version', %s)"
                ' ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value',
                (str(ahead),))

    try:
        with pytest.raises(BackendError) as excinfo:
            cluster.open(store=store, data_dir='')
        msg = str(excinfo.value)
        assert 'schema version' in msg
        assert str(ahead) in msg
        assert 'Upgrade memman' in msg, (
            f'refusal must hint at upgrade; got: {msg}')
    finally:
        cluster.drop_store(store=store, data_dir='')


def test_open_succeeds_at_current_version(pg_dsn, request):
    """Sanity check: stored == code version opens cleanly."""
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass

    backend = cluster.open(store=store, data_dir='')
    try:
        schema = _store_schema(store)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT value FROM {schema}.meta'
                    " WHERE key = 'pg_schema_version'")
                row = cur.fetchone()
        assert row is not None
        assert int(row[0]) == _PG_SCHEMA_VERSION
    finally:
        backend.close()
        cluster.drop_store(store=store, data_dir='')
