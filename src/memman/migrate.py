"""Bidirectional SQLite <-> Postgres store migration.

Backs the `memman migrate --to <backend>` CLI command. Forward
(SQLite -> Postgres) wraps `scripts/import_sqlite_to_postgres.py`
with CLI orchestration: DSN preflight, drain-lock guard, per-store
transaction, dry-run mode, and an interactive confirmation flow.
Reverse (Postgres -> SQLite) streams the four memman tables back
into a fresh SQLite store directory.

Forward migration uses `ON CONFLICT (id) DO NOTHING` on the
`insights` insert path so an interrupted run can be re-run safely.
When a target schema already exists the operator is shown its
state (EMPTY / POPULATED) and explicit destructive overwrite
happens only on confirmation. The shared drain.lock is held for
the duration of the migrate command so a scheduler-fired drain
cannot race the SQLite reader.

On a successful forward migrate the CLI writes per-store
`MEMMAN_BACKEND_<store>=postgres` and `MEMMAN_PG_DSN_<store>=<dsn>`
keys to the env file. On a successful reverse migrate the CLI
writes `MEMMAN_BACKEND_<store>=sqlite` and removes
`MEMMAN_PG_DSN_<store>`. The drain pipeline dispatches per-store
via `open_backend`, so this write makes the cutover atomic for the
migrated store while leaving sibling stores untouched.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger('memman.migrate')


class MigrateError(Exception):
    """Migration aborted because a precondition or invariant failed."""


class SchemaState(enum.Enum):
    """Target Postgres schema state for a memman store."""

    ABSENT = 'absent'
    EMPTY = 'empty'
    POPULATED = 'populated'


@dataclass
class MigrateResult:
    """One store's migration outcome (counts per table)."""

    store: str
    schema: str
    insights: int = 0
    edges: int = 0
    oplog: int = 0
    meta: int = 0
    dry_run: bool = False
    verified: bool = False


def preflight(dsn: str) -> dict[str, bool]:
    """Verify the target Postgres role can run the migration.

    Returns a dict mapping check name to pass/fail. Raises
    `MigrateError` on the first hard failure (connection refused,
    pgvector missing).
    """
    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True)
    except Exception as exc:
        raise MigrateError(
            f'cannot connect to postgres: {type(exc).__name__}: {exc}'
            ) from exc

    checks: dict[str, bool] = {}
    try:
        with conn.cursor() as cur:
            cur.execute('select 1')
            checks['select_1'] = cur.fetchone()[0] == 1

            sql = """
select 1 from pg_extension
where extname = 'vector'
"""
            cur.execute(sql)
            row = cur.fetchone()
            if row is None:
                raise MigrateError(
                    'pgvector extension is not installed in the target '
                    'database; run `create extension vector;` as a '
                    'superuser first')
            checks['pgvector_installed'] = True

            sql = """
select has_database_privilege(current_user, current_database(), 'CREATE')
"""
            cur.execute(sql)
            checks['create_schema_privilege'] = bool(cur.fetchone()[0])
            if not checks['create_schema_privilege']:
                raise MigrateError(
                    'current postgres role lacks create schema '
                    'privilege on the target database')
    finally:
        conn.close()
    return checks


def inspect_target_schemas(
        dsn: str, stores: list[str]) -> dict[str, SchemaState]:
    """Classify each `store_<name>` schema as ABSENT / EMPTY / POPULATED.

    Single round-trip query joining `pg_namespace` with
    `information_schema.tables` filtered to the four memman tables.
    A schema absent from the result is ABSENT; present with no
    memman tables is EMPTY (likely an aborted prior run); present
    with one or more tables is POPULATED. Raises `MigrateError` on
    connection or permission failures so preflight stays fail-closed.
    """
    from memman.store.postgres import _connection, _store_schema

    schema_to_store = {_store_schema(s): s for s in stores}
    schema_names = list(schema_to_store.keys())

    sql = """
select n.nspname, count(t.table_name)
from pg_namespace n
left join information_schema.tables t
  on t.table_schema = n.nspname
  and t.table_name in ('insights', 'edges', 'oplog', 'meta')
where n.nspname = any(%s)
group by n.nspname
"""
    try:
        with _connection(dsn, autocommit=True) as conn, \
                conn.cursor() as cur:
            cur.execute(sql, (schema_names,))
            rows = cur.fetchall()
    except Exception as exc:
        raise MigrateError(
            f'failed to inspect target schemas: '
            f'{type(exc).__name__}: {exc}') from exc

    seen = {row[0]: int(row[1]) for row in rows}
    result: dict[str, SchemaState] = {}
    for schema, store in schema_to_store.items():
        if schema not in seen:
            result[store] = SchemaState.ABSENT
        elif seen[schema] == 0:
            result[store] = SchemaState.EMPTY
        else:
            result[store] = SchemaState.POPULATED
    return result


def _verify_destination_counts(
        pg_conn, schema: str, store: str,
        expected: dict[str, int]) -> None:
    """Compare destination table counts against captured source counts.

    Idempotent re-runs against an already-populated schema may legitimately
    end with destination counts equal to source counts; the destination's
    `ON CONFLICT DO NOTHING` makes the per-call insert count a lower
    bound but the post-commit absolute count is the authoritative check.
    Raises `MigrateError` on any mismatch with the per-table delta.
    """
    sql = (
        f'select '
        f'  (select count(*) from {schema}.insights),'
        f'  (select count(*) from {schema}.edges),'
        f'  (select count(*) from {schema}.oplog),'
        f'  (select count(*) from {schema}.meta)')
    with pg_conn.cursor() as cur:
        cur.execute(sql)
        ins, edges, oplog, meta = cur.fetchone()
    actual = {
        'insights': int(ins), 'edges': int(edges),
        'oplog': int(oplog), 'meta': int(meta),
        }
    diffs = [
        (table, expected[table], actual[table])
        for table in ('insights', 'edges', 'oplog', 'meta')
        if expected[table] != actual[table]
        ]
    if diffs:
        detail = ', '.join(
            f'{t}: source={s} dest={d}' for t, s, d in diffs)
        raise MigrateError(
            f'verify failed for store {store!r}: {detail}')


@contextmanager
def held_drain_lock(data_dir: str) -> Iterator[int]:
    """Acquire the shared drain.lock for the duration of the block."""
    from memman.drain_lock import DrainLockBusy, acquire, release
    try:
        fd = acquire(data_dir)
    except DrainLockBusy:
        raise MigrateError(
            'drain.lock is held by another process; stop the scheduler '
            'with `memman scheduler stop` before running migrate')
    try:
        yield fd
    finally:
        release(fd)


def migrate_store_to_postgres(
        *, source_dir: str, dsn: str, store: str,
        dry_run: bool = False,
        state: SchemaState = SchemaState.ABSENT) -> MigrateResult:
    """Migrate a single SQLite store into Postgres schema `store_<store>`.

    Returns counts of rows moved per table. On `dry_run=True`, reports
    the counts that would be moved without writing. When `state` is
    `EMPTY` or `POPULATED` the existing schema is dropped and
    recreated; when `ABSENT` the schema is created fresh. All inserts
    run inside one Postgres transaction; on any failure the
    transaction rolls back and `MigrateError` is raised. The caller
    is responsible for the outer drain.lock guard and the
    confirmation prompt.
    """
    import contextlib
    import sqlite3

    from memman.store.postgres import _check_identifier, _store_schema

    _check_identifier(store)
    schema = _store_schema(store)

    sqlite_path = Path(source_dir) / 'memman.db'
    if not sqlite_path.exists():
        raise MigrateError(
            f'source SQLite database not found: {sqlite_path}')

    with contextlib.closing(sqlite3.connect(
            f'file:{sqlite_path}?mode=ro', uri=True)) as sqlite_conn:
        sqlite_conn.row_factory = None
        n_insights = sqlite_conn.execute(
            'select count(*) from insights').fetchone()[0]
        n_edges = sqlite_conn.execute(
            'select count(*) from edges').fetchone()[0]
        n_oplog = sqlite_conn.execute(
            'select count(*) from oplog').fetchone()[0]
        n_meta = sqlite_conn.execute(
            'select count(*) from meta').fetchone()[0]
        fp_row = sqlite_conn.execute(
            "select 1 from meta where key = 'embed_fingerprint'"
            ).fetchone()
        if n_insights == 0 and fp_row is None:
            raise MigrateError(
                f'source store {store!r} is empty (no insights, no'
                f' embed fingerprint); nothing to migrate')
        if dry_run:
            return MigrateResult(
                store=store, schema=schema,
                insights=n_insights, edges=n_edges,
                oplog=n_oplog, meta=n_meta, dry_run=True)

        from scripts.import_sqlite_to_postgres import _ensure_schema
        from scripts.import_sqlite_to_postgres import _import_edges
        from scripts.import_sqlite_to_postgres import _import_insights
        from scripts.import_sqlite_to_postgres import _import_meta
        from scripts.import_sqlite_to_postgres import _import_oplog
        from scripts.import_sqlite_to_postgres import _read_source_dim

        try:
            dim = _read_source_dim(sqlite_conn)
        except SystemExit as exc:
            raise MigrateError(str(exc)) from exc

        from memman.store.postgres import _connection
        with _connection(dsn, autocommit=False) as pg_conn:
            try:
                if state in {SchemaState.EMPTY, SchemaState.POPULATED}:
                    with pg_conn.cursor() as cur:
                        cur.execute(
                            f'drop schema if exists {schema} cascade')
                _ensure_schema(pg_conn, schema, dim)
                ins_count = _import_insights(
                    sqlite_conn, pg_conn, schema, dim)
                edge_count = _import_edges(sqlite_conn, pg_conn, schema)
                oplog_count = _import_oplog(sqlite_conn, pg_conn, schema)
                meta_count = _import_meta(sqlite_conn, pg_conn, schema)
                pg_conn.commit()
                _verify_destination_counts(
                    pg_conn, schema, store,
                    expected={
                        'insights': n_insights,
                        'edges': n_edges,
                        'oplog': n_oplog,
                        'meta': n_meta,
                        })
            except Exception as exc:
                pg_conn.rollback()
                if isinstance(exc, MigrateError):
                    raise
                raise MigrateError(
                    f'migration of store {store!r} failed: '
                    f'{type(exc).__name__}: {exc}') from exc
    return MigrateResult(
        store=store, schema=schema,
        insights=ins_count, edges=edge_count,
        oplog=oplog_count, meta=meta_count, dry_run=False,
        verified=True)


def migrate_store_to_sqlite(
        *, dsn: str, target_dir: str, store: str) -> MigrateResult:
    """Migrate a single Postgres store schema into a fresh SQLite store.

    Streams `insights`, `edges`, `oplog`, `meta` from
    `store_<store>` into the SQLite database at `<target_dir>/memman.db`.
    `worker_runs` is postgres-only operational state and is skipped.
    Oplog ids are recovered via `coalesce(legacy_id, id)`; explicit-id
    inserts that collide fall back to sqlite autoincrement with a
    logged warning. Returns counts of rows moved per table. Raises
    `MigrateError` on any precondition or insert failure.
    """
    import json
    import sqlite3

    from memman.embed.vector import serialize_vector
    from memman.store.db import open_db
    from memman.store.model import format_timestamp
    from memman.store.postgres import (
        _check_identifier, _connection, _store_schema,
        )

    _check_identifier(store)
    schema = _store_schema(store)

    with _connection(dsn, autocommit=True) as pg_conn, \
            pg_conn.cursor() as cur:
        cur.execute(
            'select 1 from pg_namespace where nspname = %s',
            (schema,))
        if cur.fetchone() is None:
            raise MigrateError(
                f'source postgres schema {schema!r} does not exist'
                f' for store {store!r}')

        cur.execute(
            f"select value from {schema}.meta"
            " where key = 'embed_fingerprint'")
        fp_row = cur.fetchone()
        if fp_row is None or not fp_row[0]:
            raise MigrateError(
                f"source schema {schema!r} has no"
                f" meta.embed_fingerprint; run `memman doctor` on the"
                f" source store before migrating")

        cur.execute(f'select count(*) from {schema}.insights')
        n_insights = int(cur.fetchone()[0])
        cur.execute(f'select count(*) from {schema}.edges')
        n_edges = int(cur.fetchone()[0])
        cur.execute(f'select count(*) from {schema}.oplog')
        n_oplog = int(cur.fetchone()[0])
        cur.execute(f'select count(*) from {schema}.meta')
        n_meta = int(cur.fetchone()[0])

        cur.execute(f"""
select id, content, category, importance, entities,
       source, access_count, keywords, summary, semantic_facts,
       last_accessed_at, embedding, effective_importance,
       linked_at, enriched_at, created_at, updated_at,
       deleted_at, prompt_version, model_id, embedding_model
from {schema}.insights
order by id
""")
        insight_rows = cur.fetchall()

        cur.execute(f"""
select source_id, target_id, edge_type, weight,
       metadata, created_at
from {schema}.edges
order by source_id, target_id, edge_type
""")
        edge_rows = cur.fetchall()

        cur.execute(f"""
select coalesce(legacy_id, id) as sqlite_id,
       operation, insight_id, detail, created_at,
       before, after
from {schema}.oplog
order by sqlite_id
""")
        oplog_rows = cur.fetchall()

        cur.execute(f'select key, value from {schema}.meta')
        meta_rows = cur.fetchall()

    Path(target_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    db = open_db(target_dir)
    try:
        conn = db.conn
        try:
            conn.execute('begin')
            insight_out: list[tuple[Any, ...]] = []
            for r in insight_rows:
                emb_blob = (
                    serialize_vector(list(r[11]))
                    if r[11] is not None else None)
                insight_out.append((
                    r[0], r[1], r[2], r[3],
                    json.dumps(r[4]) if r[4] is not None else '[]',
                    r[5], r[6],
                    json.dumps(r[7]) if r[7] is not None else None,
                    r[8],
                    json.dumps(r[9]) if r[9] is not None else None,
                    format_timestamp(r[10]) if r[10] else None,
                    emb_blob,
                    r[12],
                    format_timestamp(r[13]) if r[13] else None,
                    format_timestamp(r[14]) if r[14] else None,
                    format_timestamp(r[15]) if r[15] else None,
                    format_timestamp(r[16]) if r[16] else None,
                    format_timestamp(r[17]) if r[17] else None,
                    r[18], r[19], r[20],
                    ))
            conn.executemany(
                'insert into insights ('
                ' id, content, category, importance, entities,'
                ' source, access_count, keywords, summary,'
                ' semantic_facts, last_accessed_at, embedding,'
                ' effective_importance, linked_at, enriched_at,'
                ' created_at, updated_at, deleted_at,'
                ' prompt_version, model_id, embedding_model)'
                ' values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                ' ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                insight_out)

            edge_out = [(
                r[0], r[1], r[2], r[3],
                json.dumps(r[4]) if r[4] is not None else '{}',
                format_timestamp(r[5]),
                ) for r in edge_rows]
            conn.executemany(
                'insert into edges ('
                ' source_id, target_id, edge_type, weight,'
                ' metadata, created_at)'
                ' values (?, ?, ?, ?, ?, ?)',
                edge_out)

            max_oplog_id = 0
            for r in oplog_rows:
                desired_id = int(r[0])
                row = (
                    desired_id, r[1], r[2], r[3],
                    format_timestamp(r[4]),
                    json.dumps(r[5]) if r[5] is not None else None,
                    json.dumps(r[6]) if r[6] is not None else None,
                    )
                try:
                    conn.execute(
                        'insert into oplog ('
                        ' id, operation, insight_id, detail,'
                        ' created_at, before, after)'
                        ' values (?, ?, ?, ?, ?, ?, ?)',
                        row)
                    max_oplog_id = max(max_oplog_id, desired_id)
                except sqlite3.IntegrityError as exc:
                    logger.warning(
                        'oplog id collision at desired_id=%d for'
                        ' store %r (%s); falling back to autoincrement',
                        desired_id, store, exc)
                    conn.execute(
                        'insert into oplog ('
                        ' operation, insight_id, detail,'
                        ' created_at, before, after)'
                        ' values (?, ?, ?, ?, ?, ?)',
                        row[1:])
            if max_oplog_id > 0:
                conn.execute(
                    "insert or replace into sqlite_sequence"
                    " (name, seq) values ('oplog', ?)",
                    (max_oplog_id,))

            conn.executemany(
                'insert or replace into meta (key, value)'
                ' values (?, ?)',
                list(meta_rows))
            conn.execute('commit')
        except Exception as exc:
            try:
                conn.execute('rollback')
            except sqlite3.Error:
                pass
            if isinstance(exc, MigrateError):
                raise
            raise MigrateError(
                f'migration of store {store!r} to sqlite failed:'
                f' {type(exc).__name__}: {exc}') from exc

        actual_insights = conn.execute(
            'select count(*) from insights').fetchone()[0]
        actual_edges = conn.execute(
            'select count(*) from edges').fetchone()[0]
        actual_oplog = conn.execute(
            'select count(*) from oplog').fetchone()[0]
        actual_meta = conn.execute(
            'select count(*) from meta').fetchone()[0]
        diffs = []
        for table, expected, actual in (
                ('insights', n_insights, actual_insights),
                ('edges', n_edges, actual_edges),
                ('oplog', n_oplog, actual_oplog),
                ('meta', n_meta, actual_meta),
                ):
            if expected != actual:
                diffs.append((table, expected, actual))
        if diffs:
            detail = ', '.join(
                f'{t}: source={s} dest={d}' for t, s, d in diffs)
            raise MigrateError(
                f'verify failed for store {store!r}: {detail}')
    finally:
        db.close()

    return MigrateResult(
        store=store, schema=schema,
        insights=n_insights, edges=n_edges,
        oplog=n_oplog, meta=n_meta, dry_run=False, verified=True)
