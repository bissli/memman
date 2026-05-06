r"""Import a memman SQLite store into a Postgres schema.

Usage:

    python scripts/import_sqlite_to_postgres.py \\
        --source ~/.memman/data/<store-name> \\
        --target 'postgresql://user:pass@host:port/dbname' \\
        --schema staging

Streams the four memman tables (`insights`, `edges`, `oplog`, `meta`)
from a SQLite store directory into the named target schema, applying
type translations:

- `BLOB` embeddings -> pgvector `vector(N)` via the pgvector adapter,
  where `N` is read from the source store's `meta.embed_fingerprint`.
  Source SQLite blobs are float64 (`<Nd`); pgvector stores float32, so
  values are cast to `numpy.float32` before binding (otherwise psycopg
  rounds silently and the parity test top-5 set intersection drops).
- TEXT JSON columns (`entities`, `keywords`, `summary`, `semantic_facts`,
  `metadata`) -> `JSONB`. Empty or NULL values become JSON null.
- ISO TEXT timestamps (`created_at`, `updated_at`, etc.) ->
  `TIMESTAMPTZ` via `datetime.fromisoformat` (with explicit UTC for
  values that arrive without offset).

Doubles as the operator-facing migration path for users who decide
to switch a store from SQLite to Postgres after the
`MEMMAN_BACKEND=postgres` mode lands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

logger = logging.getLogger('memman.import')


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.insights (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    importance  INTEGER DEFAULT 3,
    entities    JSONB DEFAULT '[]'::jsonb,
    source      TEXT DEFAULT 'user',
    access_count INTEGER DEFAULT 0,
    keywords    JSONB,
    summary     TEXT,
    semantic_facts JSONB,
    last_accessed_at TIMESTAMPTZ,
    embedding   vector({dim}),
    effective_importance REAL DEFAULT 0.5,
    linked_at   TIMESTAMPTZ,
    enriched_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    deleted_at  TIMESTAMPTZ,
    prompt_version TEXT,
    model_id    TEXT,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS {schema}.edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    metadata    JSONB DEFAULT '{{}}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS {schema}.oplog (
    id          BIGSERIAL PRIMARY KEY,
    operation   TEXT NOT NULL,
    insight_id  TEXT,
    detail      TEXT DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL,
    before      JSONB,
    after       JSONB,
    legacy_id   BIGINT
);

CREATE TABLE IF NOT EXISTS {schema}.meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _parse_ts(value: str | None) -> datetime | None:
    """Convert a memman ISO timestamp string into an aware datetime."""
    if not value:
        return None
    s = value
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_json_field(
        value: str | None, *, default_kind: str = 'array') -> object:
    """Decode a TEXT-JSON column into a Python object suitable for JSONB.

    `default_kind` controls what an empty/None value becomes: 'array'
    for entity lists, 'object' for metadata dicts. Malformed JSON
    falls back to the same default rather than raising -- the source
    SQLite store may have rows older than the current schema's
    invariants.
    """
    if not value:
        return [] if default_kind == 'array' else {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f'malformed JSON skipped: {value[:60]!r}')
        return [] if default_kind == 'array' else {}


def _decode_embedding(
        blob: bytes | None, dim: int) -> list[float] | None:
    """Decode a memman embedding BLOB into a float32 list for pgvector.

    `dim` is the source store's declared embedding dim (from
    `meta.embed_fingerprint`). A blob whose float64 element count does
    not match raises `ValueError` -- silently dropping a row would
    return a `None` vector that the destination column rejects only
    on insert, masking source corruption as a generic insert error.
    """
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float64)
    if arr.size != dim:
        raise ValueError(
            f'embedding dim {arr.size} does not match source'
            f' fingerprint dim {dim}')
    return arr.astype(np.float32).tolist()


def _ensure_schema(
        conn: psycopg.Connection, schema: str, dim: int) -> None:
    """Create the target schema and table DDL (idempotent).

    `dim` is interpolated into `vector(N)` for the embedding column.
    """
    if not schema.replace('_', '').isalnum():
        raise SystemExit(f'invalid schema name: {schema!r}')
    if dim <= 0:
        raise ValueError(f'invalid embedding dim: {dim}')
    with conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
        cur.execute(
            _SCHEMA_DDL.format(schema=schema, dim=dim))
        cur.execute(
            f'ALTER TABLE {schema}.oplog'
            f' ADD COLUMN IF NOT EXISTS legacy_id BIGINT')
        cur.execute(
            "select 1 from pg_constraint where conname = %s",
            (f'oplog_legacy_id_key_{schema}',))
        if cur.fetchone() is None:
            cur.execute(
                f'alter table {schema}.oplog'
                f' add constraint oplog_legacy_id_key_{schema}'
                f' unique (legacy_id)')


def _import_insights(
        sqlite_conn: sqlite3.Connection,
        pg_conn: psycopg.Connection,
        schema: str, dim: int) -> int:
    """Stream the insights table. Returns row count imported."""
    rows = sqlite_conn.execute(
        'SELECT id, content, category, importance, entities,'
        ' source, access_count, keywords, summary, semantic_facts,'
        ' last_accessed_at, embedding, effective_importance,'
        ' linked_at, enriched_at, created_at, updated_at,'
        ' deleted_at, prompt_version, model_id, embedding_model'
        ' FROM insights').fetchall()
    if not rows:
        return 0
    out = [(
            r[0], r[1], r[2], r[3],
            json.dumps(_parse_json_field(r[4], default_kind='array')),
            r[5], r[6],
            json.dumps(_parse_json_field(r[7], default_kind='array'))
                if r[7] else None,
            r[8],
            json.dumps(_parse_json_field(r[9], default_kind='array'))
                if r[9] else None,
            _parse_ts(r[10]),
            _decode_embedding(r[11], dim),
            r[12], _parse_ts(r[13]), _parse_ts(r[14]),
            _parse_ts(r[15]), _parse_ts(r[16]), _parse_ts(r[17]),
            r[18], r[19], r[20]) for r in rows]
    with pg_conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.insights ('
            ' id, content, category, importance, entities,'
            ' source, access_count, keywords, summary, semantic_facts,'
            ' last_accessed_at, embedding, effective_importance,'
            ' linked_at, enriched_at, created_at, updated_at,'
            ' deleted_at, prompt_version, model_id, embedding_model)'
            ' VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s,'
            ' %s::jsonb, %s, %s::jsonb,'
            ' %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
            ' ON CONFLICT (id) DO NOTHING',
            out)
    return len(out)


def _import_edges(
        sqlite_conn: sqlite3.Connection,
        pg_conn: psycopg.Connection,
        schema: str) -> int:
    """Stream the edges table. Returns row count imported."""
    rows = sqlite_conn.execute(
        'SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at FROM edges').fetchall()
    if not rows:
        return 0
    out = [(
            r[0], r[1], r[2], r[3],
            json.dumps(_parse_json_field(r[4], default_kind='object')),
            _parse_ts(r[5])) for r in rows]
    with pg_conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.edges'
            ' (source_id, target_id, edge_type, weight,'
            ' metadata, created_at)'
            ' VALUES (%s, %s, %s, %s, %s::jsonb, %s)'
            ' ON CONFLICT (source_id, target_id, edge_type)'
            ' DO NOTHING',
            out)
    return len(out)


def _import_oplog(
        sqlite_conn: sqlite3.Connection,
        pg_conn: psycopg.Connection,
        schema: str) -> int:
    """Stream the oplog table. Returns row count imported.

    The source SQLite `id` is copied into the destination's
    `legacy_id` column so that re-running migrate after a partial
    failure does not duplicate rows. The destination's own `id`
    (`bigserial`) remains independent.
    """
    rows = sqlite_conn.execute(
        'SELECT id, operation, insight_id, detail, created_at,'
        '       before, after'
        ' FROM oplog').fetchall()
    if not rows:
        return 0
    out = [
        (r[1], r[2], r[3], _parse_ts(r[4]), r[5], r[6], r[0])
        for r in rows
        ]
    with pg_conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.oplog'
            ' (operation, insight_id, detail, created_at,'
            '  before, after, legacy_id)'
            ' VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)'
            ' ON CONFLICT (legacy_id) DO NOTHING',
            out)
    return len(out)


def _read_source_dim(sqlite_conn: sqlite3.Connection) -> int:
    """Resolve the source store's embedding dim from `meta.embed_fingerprint`.

    Raises `SystemExit` when the source has no fingerprint -- the
    destination needs `vector(N)` baked in at schema creation, and
    guessing a default would silently corrupt the migration of any
    non-512-dim store.
    """
    row = sqlite_conn.execute(
        "select value from meta where key = 'embed_fingerprint'"
    ).fetchone()
    if row is None or not row[0]:
        raise SystemExit(
            "source has no meta.embed_fingerprint; cannot determine"
            " embedding dim. Run 'memman doctor' on the source store"
            " before migrating.")
    fp = json.loads(row[0])
    dim = int(fp['dim'])
    if dim <= 0:
        raise SystemExit(
            f'source meta.embed_fingerprint has invalid dim={dim}')
    return dim


def _import_meta(
        sqlite_conn: sqlite3.Connection,
        pg_conn: psycopg.Connection,
        schema: str) -> int:
    """Stream the meta table. Returns row count imported."""
    rows = sqlite_conn.execute(
        'SELECT key, value FROM meta').fetchall()
    if not rows:
        return 0
    with pg_conn.cursor() as cur:
        cur.executemany(
            f'INSERT INTO {schema}.meta (key, value) VALUES (%s, %s)'
            ' ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value',
            list(rows))
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
        0 on success, non-zero on validation failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Import a memman SQLite store into a Postgres schema.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--source', required=True,
        help='Path to the SQLite store directory'
             ' (contains memman.db).')
    parser.add_argument(
        '--target', required=True,
        help='Postgres connection URL (postgresql://...).')
    parser.add_argument(
        '--schema', required=True,
        help='Destination schema name (created if missing).')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable INFO logging.')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s')

    src_path = Path(args.source) / 'memman.db'
    if not src_path.exists():
        logger.error(f'source SQLite store not found: {src_path}')
        return 2

    logger.info(f'opening SQLite source: {src_path}')
    sqlite_conn = sqlite3.connect(str(src_path))
    try:
        dim = _read_source_dim(sqlite_conn)
        logger.info(f'opening Postgres target: {args.target}')
        with psycopg.connect(args.target, autocommit=False) as pg_conn:
            register_vector(pg_conn)
            _ensure_schema(pg_conn, args.schema, dim)
            counts = {
                'insights': _import_insights(
                    sqlite_conn, pg_conn, args.schema, dim),
                'edges': _import_edges(
                    sqlite_conn, pg_conn, args.schema),
                'oplog': _import_oplog(
                    sqlite_conn, pg_conn, args.schema),
                'meta': _import_meta(
                    sqlite_conn, pg_conn, args.schema),
                }
            pg_conn.commit()
    finally:
        sqlite_conn.close()
    for name, count in counts.items():
        print(f'{name}: imported {count} rows', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
