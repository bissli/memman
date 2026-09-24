"""Every insert path stores `content_hash`, the exact-duplicate key.

The hash is the sha256 hex of the content folded as the rung compares
it: whitespace runs collapsed to one space, ends stripped, lowercased.
The oracle in each test is that digest computed here by hand, never
the fold function under test.
"""

import hashlib
import shutil
from datetime import datetime, timezone

import pytest
from memman.store.db import open_db, set_meta, store_dir
from memman.store.model import Insight
from memman.store.node import insert_insight
from memman.store.postgres import PostgresMigrator, _store_schema
from memman.store.sqlite import SqliteBackend, SqliteMigrator
from tests.conftest import make_insight

RAW = '  Redis  Caches\tSession\nTokens '
FOLDED_DIGEST = hashlib.sha256(
    b'redis caches session tokens').hexdigest()
FINGERPRINT = '{"provider":"voyage","model":"voyage-3-lite","dim":512}'


def _stored_hash(backend, insight_id: str) -> str | None:
    """Read `content_hash` straight from the backend's table."""
    if isinstance(backend, SqliteBackend):
        row = backend._db._query(
            'select content_hash from insights where id = ?',
            (insight_id,)).fetchone()
    else:
        with backend._conn.cursor() as cur:
            cur.execute(
                f'select content_hash from {backend._schema}.insights'
                ' where id = %s', (insight_id,))
            row = cur.fetchone()
    return row[0] if row else None


def _seed_sqlite_store(data_dir: str, store: str) -> str:
    """Write one raw-spaced insight into a fresh SQLite store."""
    sdir = store_dir(data_dir, store)
    db = open_db(sdir)
    try:
        insert_insight(db, Insight(
            id='hash-migrate-1', content=RAW, category='fact',
            importance=3, entities=[], source='test', access_count=0,
            updated_at=datetime.now(timezone.utc)))
        set_meta(db, 'embed_fingerprint', FINGERPRINT)
    finally:
        db.close()
    return sdir


def test_insert_stores_the_folded_content_hash(backend):
    """Verify `nodes.insert` stores the digest of the folded content.

    Mutation: one backend's insert missing the column, or hashing the
        raw content unfolded.
    Oracle: the sha256 of 'redis caches session tokens', by hand.
    """
    backend.nodes.insert(make_insight(id='hash-1', content=RAW))
    assert _stored_hash(backend, 'hash-1') == FOLDED_DIGEST


def test_sqlite_migrate_stores_the_content_hash(tmp_path):
    """Verify a SQLite-to-SQLite migrate stores the hash on each row.

    Mutation: `SqliteMigrator.apply` missing the column.
    Oracle: the hand-computed digest on the destination row.
    """
    _seed_sqlite_store(str(tmp_path / 'src'), 'hashmig')
    payload = SqliteMigrator(str(tmp_path / 'src')).gather('hashmig')
    dest = tmp_path / 'dest'
    dest.mkdir()
    SqliteMigrator(str(dest)).apply('hashmig', payload)

    db = open_db(store_dir(str(dest), 'hashmig'))
    try:
        row = db._query(
            'select content_hash from insights where id = ?',
            ('hash-migrate-1',)).fetchone()
    finally:
        db.close()
    assert row[0] == FOLDED_DIGEST


@pytest.mark.postgres
def test_postgres_migrate_stores_the_content_hash(tmp_path, pg_dsn):
    """Verify a SQLite-to-Postgres migrate stores the hash on each row.

    Mutation: `PostgresMigrator.apply` missing the column.
    Oracle: the hand-computed digest on the Postgres row.
    """
    # psycopg is the optional postgres extra, so a sqlite-only install
    # still collects this file.
    import psycopg

    store = 'hash_migrate_pg'
    sdir = _seed_sqlite_store(str(tmp_path), store)
    schema = _store_schema(store)

    def _drop_schema() -> None:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')

    _drop_schema()
    try:
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(sdir)

        with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f'select content_hash from {schema}.insights'
                ' where id = %s', ('hash-migrate-1',))
            row = cur.fetchone()
        assert row[0] == FOLDED_DIGEST
    finally:
        _drop_schema()
