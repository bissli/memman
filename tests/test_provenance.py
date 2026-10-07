"""Provenance column `embedding_model`.

The baseline schema carries it, `insert_insight` persists it, and
`remember` stamps every new row.
"""

import json
import sqlite3
from pathlib import Path

from memman.cli import cli
from memman.queue import queue_db
from memman.store.db import open_db
from memman.store.model import Insight
from memman.store.node import insert_insight


def test_baseline_schema_has_provenance_columns(tmp_path):
    """A fresh open_db produces an insights table with the column.

    Mutation: dropping the `embedding_model` column line from
        `_BASELINE_SCHEMA`'s `insights` DDL.
    Oracle: the column names read back from `PRAGMA table_info`.
    """
    db = open_db(str(tmp_path))
    try:
        cols = db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        names = {row[1] for row in cols}
        assert 'embedding_model' in names
    finally:
        db.close()


def test_insert_insight_persists_provenance(tmp_path):
    """insert_insight writes embedding_model.

    Mutation: dropping `embedding_model` from `insert_insight`'s column
        or values list, or swapping it with another column.
    Oracle: the hand-supplied `'voyage-3-lite'`, read back by a
        `SELECT`.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(
            id='prov-1', content='provenance test',
            embedding_model='voyage-3-lite')
        insert_insight(db, ins)
        row = db._conn.execute(
            'select embedding_model'
            ' from insights where id = ?',
            (ins.id,)).fetchone()
        assert row == ('voyage-3-lite',)
    finally:
        db.close()


def test_insert_insight_tolerates_null_provenance(tmp_path):
    """Insight without a stamp (tests, fixtures) inserts with NULL.

    Mutation: `insert_insight` defaulting a `None` `embedding_model`
        to an empty string instead of passing it through as NULL.
    Oracle: the row read back as the tuple `(None,)`.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(id='null-prov', content='no stamp')
        insert_insight(db, ins)
        row = db._conn.execute(
            'select embedding_model'
            ' from insights where id = ?',
            (ins.id,)).fetchone()
        assert row == (None,)
    finally:
        db.close()


def test_remember_stamps_provenance(mm_runner):
    """`remember` stamps embedding_model on every row.

    Mutation: leaving `embedding_model` unset on a write, which the
        read path would then treat as never embedded.
    Oracle: the row read back by its queue_uuid, against the store's
        configured embed model.
    """
    r, data_dir = mm_runner

    result = r.invoke(cli, [
        '--data-dir', data_dir,
        'remember',
        'provenance stamping end-to-end'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    queue_id = data['queue_id']

    with queue_db(data_dir) as qconn:
        queue_uuid = qconn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()[0]
    store_path = Path(data_dir) / 'data' / 'default' / 'memman.db'
    conn = sqlite3.connect(str(store_path))
    try:
        embed_model, = conn.execute(
            'select embedding_model'
            ' from insights where queue_uuid = ?',
            (queue_uuid,)).fetchone()
    finally:
        conn.close()

    assert embed_model == 'voyageai/voyage-4-lite'
