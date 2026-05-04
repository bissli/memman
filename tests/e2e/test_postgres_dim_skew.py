"""Final-cleanup -- 512-dim parameterization & dim-skew refusal.

Two related guarantees on top of the historical 512-dim hardcode:

- `_ensure_baseline_schema` honors a caller-supplied `dim` so a
  non-Voyage operator deploying a 1024-dim provider gets a
  `vector(1024)` column on first create.
- `_assert_vector_dim_matches` refuses to open if the active client
  dim differs from the stored column width, with a clear upgrade
  hint pointing at `memman embed reembed`.
"""

from __future__ import annotations

import psycopg
import pytest
from memman.store.errors import BackendError
from memman.store.postgres import (
    _assert_vector_dim_matches,
    _ensure_baseline_schema,
    _store_schema,
)

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def _safe(s: str) -> str:
    out = ''.join(c if c.isalnum() else '_' for c in s).lower()
    if out and not out[0].isalpha():
        out = 'p_' + out
    return out[:40] or 'p_test'


def test_baseline_schema_honors_caller_dim(pg_dsn, request):
    """`_ensure_baseline_schema(dim=N)` builds a vector(N) column.

    Drives the 512-dim parameterization: a non-Voyage operator
    (e.g. openai 1536) gets the right column width on first deploy
    instead of being silently locked to 512.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    _ensure_baseline_schema(pg_dsn, store, dim=1024)

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT atttypmod FROM pg_attribute'
                    " WHERE attrelid = (%s || '.insights')::regclass"
                    "   AND attname = 'embedding'"
                    '   AND NOT attisdropped',
                    (schema,))
                stored_dim = int(cur.fetchone()[0])
        assert stored_dim == 1024, (
            f'expected vector(1024); got vector({stored_dim})')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_dim_mismatch_refused_on_reopen(pg_dsn, request):
    """Reopening a vector(512) store with active=1024 raises BackendError.

    Mirror of the schema-version skew refusal: if the operator
    swaps embedding providers without first running
    `memman embed reembed` against a fresh store, the open is
    refused with a clear upgrade hint.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')

    _ensure_baseline_schema(pg_dsn, store, dim=512)
    try:
        with pytest.raises(BackendError) as excinfo:
            _assert_vector_dim_matches(pg_dsn, store, 1024)
        msg = str(excinfo.value)
        assert 'vector(512)' in msg
        assert 'dim=1024' in msg
        assert 'memman embed reembed' in msg
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')


def test_dim_match_passes_silently(pg_dsn, request):
    """When stored dim matches active, the assertion is a no-op."""
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    _ensure_baseline_schema(pg_dsn, store, dim=512)
    try:
        _assert_vector_dim_matches(pg_dsn, store, 512)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
