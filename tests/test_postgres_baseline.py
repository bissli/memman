"""Baseline contract tests against raw psycopg + pgvector.

Validates the primitives a future Postgres backend depends on:

1. `vector(512)` round-trip with a Voyage-shaped 512-dim list[float].
2. HNSW index correctness (top-5 against an exact seqscan).
3. `pg_try_advisory_lock` contention: only one connection wins.
4. `SET search_path` persists across cursor close in autocommit
   mode (pool-reuse hazard documentation).
5. Advisory lock released on connection close (no explicit unlock
   needed -- the crash-recovery mechanism `reembed_lock` and
   `swap_lock` rely on).

Gated behind `@pytest.mark.postgres` so SQLite-only `make test`
runs are unaffected.
"""


import random
import socket
from typing import Any

import numpy as np
import pytest

psycopg = pytest.importorskip('psycopg')
pytest.importorskip('pgvector')

from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.search.recall import run_recall
from memman.store import postgres as pg_mod
from memman.store.errors import BackendError
from memman.store.model import Insight
from memman.store.postgres import _ensure_baseline_schema, _ensure_hnsw_index
from memman.store.postgres import _read_stored_dim, _store_schema
from memman.store.postgres import drop_postgres_store, open_postgres_backend
from pgvector.psycopg import register_vector
from tests.fixtures.postgres import SCHEMA, connection_pair
from tests.fixtures.postgres import simulate_connection_drop, wait_for

pytestmark = pytest.mark.postgres


def _voyage_shape_vector(seed: int = 0, dim: int = 512) -> list[float]:
    """A deterministic float list shaped like a Voyage embedding.

    Parameters
    ----------
    seed : int
        Seeds the generator; equal seeds give equal vectors.
    dim : int
        Vector length.

    Returns
    -------
    list[float]
        Components in [-1, 1], not unit-normalized. pgvector cosine
        distance normalizes implicitly.
    """
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def test_vector_512_round_trip(pg_conn):
    """Verify a 512-dim list[float] round-trips through pgvector.

    Mutation: A column declared with the wrong dimension or a lossy adapter
        that truncates or reorders components.
    Oracle: The original Python list, compared per component within 1e-5.
    """
    register_vector(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(f'set search_path = {SCHEMA}, public')
        cur.execute(
            'create table vec_test ('
            ' id integer primary key,'
            ' embedding vector(512))')
        original = _voyage_shape_vector(seed=42)
        cur.execute(
            'insert into vec_test (id, embedding) values (%s, %s)',
            (1, original))
        cur.execute('select embedding from vec_test where id = 1')
        roundtripped = list(cur.fetchone()[0])
    assert len(roundtripped) == 512
    for a, b in zip(original, roundtripped):
        assert abs(a - b) < 1e-5, (
            'pgvector float32 truncation should be < 1e-5 per dim')


def test_hnsw_top5_correctness(pg_conn):
    """Verify the HNSW top-5 matches the sequential-scan top-5.

    Mutation: Ordering by a different operator than the index was built for
        (`<->` or `<#>` against `vector_cosine_ops`), or an index that ranks by
        descending distance.
    Oracle: The top-5 ids from the same query with `enable_seqscan = on`, which
        is exact; at least 4 of 5 ids must agree.
    """
    register_vector(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(f'set search_path = {SCHEMA}, public')
        cur.execute(
            'create table corpus ('
            ' id integer primary key,'
            ' embedding vector(512))')
        rows = [
            (i, _voyage_shape_vector(seed=i))
            for i in range(100)
            ]
        cur.executemany(
            'insert into corpus (id, embedding) values (%s, %s)',
            rows)
        cur.execute(
            'create index hnsw_corpus on corpus'
            ' using hnsw (embedding vector_cosine_ops)')
        query_vec = np.asarray(_voyage_shape_vector(seed=7))
        cur.execute('set enable_seqscan = on')
        cur.execute(
            'select id, 1 - (embedding <=> %s) as sim from corpus'
            ' order by embedding <=> %s limit 5',
            (query_vec, query_vec))
        seqscan_top5 = [r[0] for r in cur.fetchall()]
        cur.execute('set enable_seqscan = off')
        cur.execute(
            'select id from corpus'
            ' order by embedding <=> %s limit 5',
            (query_vec,))
        index_top5 = [r[0] for r in cur.fetchall()]
    assert len(seqscan_top5) == 5
    assert len(index_top5) == 5
    overlap = len(set(seqscan_top5) & set(index_top5))
    assert overlap >= 4, (
        f'HNSW top-5 should match seqscan top-5 in >=4 of 5'
        f' (got {overlap}); index={index_top5} seq={seqscan_top5}')


def test_pg_try_advisory_lock_contention(pg_dsn):
    """Verify a second connection is denied an advisory lock the first holds.

    Mutation: A blocking or transaction-scoped lock call, or a key that differs
        per connection, so both connections win.
    Oracle: The literal booleans pg_try_advisory_lock returns before and after
        pg_advisory_unlock on the holder.
    """
    lock_id = 9991
    with connection_pair(pg_dsn) as (conn_a, conn_b):
        with conn_a.cursor() as cur_a:
            cur_a.execute(
                'select pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_a.fetchone()[0] is True, (
                'first connection should win the advisory lock')
        with conn_b.cursor() as cur_b:
            cur_b.execute(
                'select pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_b.fetchone()[0] is False, (
                'second connection should be denied while first holds')
        with conn_a.cursor() as cur_a:
            cur_a.execute(
                'select pg_advisory_unlock(%s)', (lock_id,))
            assert cur_a.fetchone()[0] is True
        with conn_b.cursor() as cur_b:
            cur_b.execute(
                'select pg_try_advisory_lock(%s)', (lock_id,))
            assert cur_b.fetchone()[0] is True, (
                'second connection should now acquire after release')
            cur_b.execute(
                'select pg_advisory_unlock(%s)', (lock_id,))


def test_search_path_persists_across_cursor_close_in_autocommit(pg_dsn):
    """`set search_path` in autocommit mode survives cursor close.

    A pooled connection keeps the schema of its last request, so the recall
    session must reset `search_path` before returning it to the pool.

    Mutation: A driver or session change that resets search_path on cursor
        close, which would make that reset dead code.
    Oracle: `show search_path` on a second cursor of the same connection.
    """
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'create schema if not exists {SCHEMA}')
            cur.execute(f'set search_path = {SCHEMA}, public')
            cur.execute('show search_path')
            assert SCHEMA in cur.fetchone()[0]
        with conn.cursor() as cur2:
            cur2.execute('show search_path')
            after = cur2.fetchone()[0]
            assert SCHEMA in after, (
                f'search_path should persist across cursor close in'
                f' autocommit mode; got {after!r}. Pool reuse without'
                f' explicit reset would leak schema selection.')


def test_advisory_lock_released_on_connection_close(pg_dsn):
    """Verify closing a connection releases its advisory lock.

    This is the crash-recovery path `reembed_lock` and `swap_lock` rely on:
    Postgres frees the lock when the holder session dies.

    Mutation: A lock taken with a transaction-independent key that outlives the
        session, such as a table-backed lock row, so a dead holder blocks every
        later sweep.
    Oracle: A second connection sees the lock denied while the holder is open,
        then acquires it within 5 seconds of the holder dropping.
    """
    lock_id = 9992
    holder = psycopg.connect(pg_dsn, autocommit=True)
    try:
        with holder.cursor() as cur:
            cur.execute(
                'select pg_try_advisory_lock(%s)', (lock_id,))
            assert cur.fetchone()[0] is True
        with psycopg.connect(pg_dsn, autocommit=True) as observer:
            with observer.cursor() as cur:
                cur.execute(
                    'select pg_try_advisory_lock(%s)', (lock_id,))
                assert cur.fetchone()[0] is False, (
                    'lock should still be held by holder')
    finally:
        simulate_connection_drop(holder)
    with psycopg.connect(pg_dsn, autocommit=True) as later:

        def _can_acquire() -> bool:
            with later.cursor() as cur:
                cur.execute(
                    'select pg_try_advisory_lock(%s)', (lock_id,))
                got = cur.fetchone()[0]
                if got:
                    cur.execute(
                        'select pg_advisory_unlock(%s)', (lock_id,))
                return bool(got)

        assert wait_for(_can_acquire, timeout_sec=5.0), (
            'advisory lock should be released within 5s of'
            ' connection close (Postgres detects dead session)')


@pytest.fixture
def _pg_store_backend(pg_dsn):
    """A PostgresBackend bound to a fresh `store_pg_salvage` schema.
    """
    store_name = 'pg_salvage'
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    backend = open_postgres_backend(store_name, pg_dsn)
    try:
        yield backend, pg_dsn, store_name
    finally:
        backend.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


def test_hnsw_partial_index_built_concurrently(
        _pg_store_backend):
    """Verify the store builds a valid partial HNSW index with cosine ops.

    Mutation: Building the index with `vector_l2_ops`, without the `deleted_at`
        predicate, or leaving an invalid index behind.
    Oracle: The catalog: pg_index.indisvalid, pg_am.amname, and pg_get_indexdef
        for the named index.
    """
    backend, pg_dsn, store_name = _pg_store_backend
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select i.indisvalid, am.amname,'
                ' pg_get_indexdef(i.indexrelid)'
                ' from pg_index i'
                ' join pg_class c on c.oid = i.indexrelid'
                ' join pg_am am on am.oid = ('
                '   select relam from pg_class'
                '   where oid = i.indexrelid)'
                ' where c.relname = %s',
                (f'idx_insights_hnsw_{schema}',))
            row = cur.fetchone()
    assert row is not None, 'HNSW index missing'
    assert row[0] is True, 'HNSW index is invalid'
    assert row[1] == 'hnsw'
    indexdef = row[2]
    assert 'vector_cosine_ops' in indexdef
    assert 'deleted_at IS NULL' in indexdef


def test_reindex_drops_invalid_hnsw_remnant(pg_dsn):
    """Verify _ensure_hnsw_index replaces an invalid HNSW remnant.

    An aborted concurrent build leaves an index with `indisvalid = false` under
    the same name.

    Mutation: Dropping the invalid-remnant check, so `create index if not
        exists` sees the name, skips, and leaves the broken index in place.
    Oracle: pg_index.indisvalid read before (False, forced by update) and after
        (True) the call.
    """
    store_name = 'pg_remnant'
    schema = _store_schema(store_name)
    _ensure_baseline_schema(pg_dsn, store_name)
    index_name = f'idx_insights_hnsw_{schema}'
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop index if exists {schema}.{index_name}')
            cur.execute(
                f'create index {index_name}'
                f' on {schema}.insights'
                f' using hnsw (embedding vector_cosine_ops)'
                f' where deleted_at IS NULL')
            cur.execute(
                'update pg_index set indisvalid = false'
                ' where indexrelid = ('
                '  select oid from pg_class where relname = %s)',
                (index_name,))
            cur.execute(
                'select indisvalid from pg_index where indexrelid = ('
                '  select oid from pg_class where relname = %s)',
                (index_name,))
            assert cur.fetchone()[0] is False
    _ensure_hnsw_index(pg_dsn, schema)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select indisvalid from pg_index where indexrelid = ('
                '  select oid from pg_class where relname = %s)',
                (index_name,))
            row = cur.fetchone()
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    assert row is not None
    assert row[0] is True


def test_postgres_recall_issues_pgvector_distance_operator(
        tmp_path, pg_dsn, monkeypatch):
    """Postgres recall issues an HNSW-shaped pgvector anchor query.

    Mutation: replacing the indexed anchor query with a
        sequential-scan equivalent that returns the same rows - the
        exact regression a bare `<=>` substring check cannot see,
        because `RecallSession.similarities` also emits `<=>` in an
        unordered full-table scan that HNSW cannot serve.
    Oracle: the captured SQL, required to carry `<=>` inside an
        `order by` AND a `limit`, which is the only shape the HNSW
        index answers.
    """
    drop_postgres_store('hnsw_smoke', pg_dsn)
    backend = open_postgres_backend('hnsw_smoke', pg_dsn)
    backend.meta.set(META_KEY, seed_default_fingerprint().to_json())

    n_insights = 10
    for i in range(n_insights):
        ins = Insight(
            id=f'hs-{i:02d}',
            content=f'document {i} alpha bravo charlie',
            created_at=None, updated_at=None,
            deleted_at=None)
        backend.nodes.insert(ins)
        backend.nodes.update_embedding(
            ins.id, _voyage_shape_vector(seed=i), 'voyage-3-lite')

    captured_sql: list[str] = []
    real_execute = psycopg.Cursor.execute

    def spy(self: psycopg.Cursor, query: Any, *args: Any, **kwargs: Any) -> Any:
        captured_sql.append(str(query))
        return real_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, 'execute', spy)

    try:
        run_recall(
            backend, query='document alpha',
            query_vec=_voyage_shape_vector(seed=999),
            limit=5)
    finally:
        backend.close()
        drop_postgres_store('hnsw_smoke', pg_dsn)

    pgvector_ops = [s for s in captured_sql if '<=>' in s]
    assert pgvector_ops, (
        f'pgvector distance operator <=> never appeared in '
        f'{len(captured_sql)} SQL statements during Postgres recall. '
        f'HNSW is dead code. Sample SQL: {captured_sql[:5]}')

    def _is_hnsw_shaped(sql: str) -> bool:
        lowered = ' '.join(sql.lower().split())
        after_order = lowered.partition('order by')[2]
        return '<=>' in after_order and 'limit' in after_order

    assert any(_is_hnsw_shaped(s) for s in pgvector_ops), (
        f'`<=>` appears but never inside an `order by ... limit`, so '
        f'no statement can use the HNSW index. Sample: {pgvector_ops[:3]}')


def test_reembed_lock_session_scoped_and_releases_on_close(
        pg_dsn):
    """Verify `reembed_lock` denies a second holder until release.

    Mutation: A lock key that varies per connection, a blocking acquire, or a
        release path that leaves the lock held.
    Oracle: The yielded booleans: True, then False for the competitor, then
        True once the first block exits.
    """
    store_name = 'pg_reembed'
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    a = open_postgres_backend(store_name, pg_dsn)
    b = open_postgres_backend(store_name, pg_dsn)
    try:
        with a.reembed_lock('reembed') as got_a:
            assert got_a is True
            with b.reembed_lock('reembed') as got_b:
                assert got_b is False
        with b.reembed_lock('reembed') as got_b2:
            assert got_b2 is True
    finally:
        a.close()
        b.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')


@pytest.mark.parametrize('lock_name', ['reembed_lock', 'swap_lock'])
def test_lock_connection_sets_client_tcp_keepalive(
        pg_dsn, monkeypatch, lock_name):
    """Verify each advisory-lock connection keeps TCP keepalive at 30s.

    Mutation: `reembed_lock` or `swap_lock` opening its connection
        without `keepalives=True`, or `_open_connection` dropping
        `keepalives_idle`, so a holder on a dead network path keeps
        the lock until the kernel default idle of hours runs out.
    Oracle: `TCP_KEEPIDLE` read from the lock connection's own client
        socket; the server's `show tcp_keepalives_idle` reports the
        server socket and reads the same either way.
    """
    store_name = 'pg_keepalive'
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    original = pg_mod._open_connection
    keepidle = []

    def spy(dsn: str, **kwargs: Any) -> psycopg.Connection:
        conn = original(dsn, **kwargs)
        sock = socket.socket(fileno=conn.pgconn.socket)
        try:
            keepidle.append(sock.getsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPIDLE))
        finally:
            sock.detach()
        return conn

    backend = open_postgres_backend(store_name, pg_dsn)
    try:
        monkeypatch.setattr(pg_mod, '_open_connection', spy)
        lock = (backend.reembed_lock('reembed') if lock_name == 'reembed_lock'
                else backend.swap_lock())
        with lock as got:
            assert got is True
    finally:
        backend.close()
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')
    assert keepidle == [30]


def test_memman_reindex_timeout_caps_hnsw_build(
        pg_dsn, monkeypatch):
    """Verify MEMMAN_REINDEX_TIMEOUT caps the HNSW build.

    Mutation: Ignoring the env var (the default 180s applies), or issuing `set
        statement_timeout` after `create index concurrently`, so a stuck build
        is never capped.
    Oracle: The statements captured from Cursor.execute: a `7s` timeout appears
        at a lower index than the create statement.
    """
    store_name = 'pg_timeout'
    schema = _store_schema(store_name)
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    _ensure_baseline_schema(pg_dsn, store_name)

    monkeypatch.setenv('MEMMAN_REINDEX_TIMEOUT', '7')
    captured: list[str] = []
    real_execute = psycopg.Cursor.execute

    def spy(self: psycopg.Cursor, sql: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(str(sql))
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, 'execute', spy)

    try:
        _ensure_hnsw_index(pg_dsn, schema)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')

    set_idx = next(
        (i for i, s in enumerate(captured)
         if "statement_timeout = '7s'" in s.lower()), None)
    create_idx = next(
        (i for i, s in enumerate(captured)
         if 'create index concurrently' in s.lower()), None)
    assert set_idx is not None, (
        f'expected SET statement_timeout in: {captured}')
    assert create_idx is not None, (
        f'expected CREATE INDEX CONCURRENTLY in: {captured}')
    assert set_idx < create_idx, (
        f'SET statement_timeout must precede CREATE INDEX;'
        f' captured order: {captured}')


def test_read_stored_dim_distinguishes_absent_from_unreachable(
        pg_dsn, monkeypatch):
    """Verify _read_stored_dim tells an absent schema from an outage.

    None for an absent schema; BackendError, with the driver error as
    `__cause__`, for an unreachable server.

    Mutation: A handler that swallows every exception, so a connection outage
        reads as a fresh schema and the next fingerprint check passes on an
        unknown dimension.
    Oracle: A dropped schema for the None case and an unresolvable host name
        for the raise case.
    """
    schema = _store_schema('absent_store')
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'drop schema if exists {schema} cascade')
    assert _read_stored_dim(pg_dsn, 'absent_store') is None

    bad_dsn = 'postgresql://user@nonexistent.invalid:5432/x'
    with pytest.raises(BackendError) as caught:
        _read_stored_dim(bad_dsn, 'absent_store')
    assert isinstance(caught.value.__cause__, psycopg.OperationalError)
