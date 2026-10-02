"""Vector dim parameterization and dim-skew refusal.

Two related guarantees:

- `_ensure_baseline_schema` honors a caller-supplied `dim` so a
  non-Voyage operator deploying a 1024-dim embed model gets a
  `vector(1024)` column on first create.
- `_assert_vector_dim_matches` refuses to open if the active client
  dim differs from the stored column width, with a clear upgrade
  hint pointing at `memman embed swap`.
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip('psycopg')

from memman.store.errors import BackendError
from memman.store.postgres import _assert_vector_dim_matches
from memman.store.postgres import _ensure_baseline_schema, _store_schema
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_baseline_schema_honors_caller_dim(pg_dsn, request):
    """Verify `_ensure_baseline_schema(dim=N)` builds a vector(N) column.

    Mutation: ignoring `dim` and creating the column at the default
        512 width.
    Oracle: `pg_attribute.atttypmod` of the `embedding` column,
        1024 for a dim=1024 call.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    _ensure_baseline_schema(pg_dsn, store, dim=1024)

    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'select atttypmod from pg_attribute'
                    " where attrelid = (%s || '.insights')::regclass"
                    "   and attname = 'embedding'"
                    '   and not attisdropped',
                    (schema,))
                stored_dim = int(cur.fetchone()[0])
        assert stored_dim == 1024, (
            f'expected vector(1024); got vector({stored_dim})')
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_dim_mismatch_refused_on_reopen(pg_dsn, request):
    """Verify a vector(512) store refuses an active dim of 1024.

    Mutation: dropping the width comparison in
        `_assert_vector_dim_matches`, or losing the stored width,
        active dim, or upgrade hint from the message.
    Oracle: a store built at dim=512, checked with 1024; the
        `BackendError` message names `vector(512)`, `dim=1024`, and
        `memman embed swap`.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')

    _ensure_baseline_schema(pg_dsn, store, dim=512)
    try:
        with pytest.raises(BackendError) as excinfo:
            _assert_vector_dim_matches(pg_dsn, store, 1024)
        msg = str(excinfo.value)
        assert 'vector(512)' in msg
        assert 'dim=1024' in msg
        assert 'memman embed swap' in msg
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_dim_match_passes_silently(pg_dsn, request):
    """Verify `_assert_vector_dim_matches` passes when widths agree.

    Mutation: refusing on every call, or comparing against a
        constant other than the stored width.
    Oracle: a store built at dim=512 and checked with 512 raises
        nothing.
    """
    store = _safe(request.node.name)
    schema = _store_schema(store)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    _ensure_baseline_schema(pg_dsn, store, dim=512)
    try:
        _assert_vector_dim_matches(pg_dsn, store, 512)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
