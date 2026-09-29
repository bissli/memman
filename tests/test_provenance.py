"""Provenance columns `prompt_version` and `embedding_model`.

The baseline schema carries both, `insert_insight` persists them,
`compute_prompt_version` is stable across calls, and `remember` stamps
every new row.
"""

import json
import sqlite3
from pathlib import Path

from memman.cli import cli
from memman.pipeline import enrich
from memman.pipeline.remember import compute_prompt_version
from memman.queue import queue_db
from memman.store.db import open_db
from memman.store.model import Insight
from memman.store.node import insert_insight


def test_baseline_schema_has_provenance_columns(tmp_path):
    """A fresh open_db produces an insights table with the two columns.

    Mutation: dropping the `prompt_version` or `embedding_model`
        column line from `_BASELINE_SCHEMA`'s `insights` DDL.
    Oracle: the column names read back from `PRAGMA table_info`.
    """
    db = open_db(str(tmp_path))
    try:
        cols = db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        names = {row[1] for row in cols}
        assert 'prompt_version' in names
        assert 'embedding_model' in names
    finally:
        db.close()


def test_insert_insight_persists_provenance(tmp_path):
    """insert_insight writes prompt_version/embedding_model.

    Mutation: dropping `prompt_version`/`embedding_model` from
        `insert_insight`'s column or values list, or swapping their
        order against the placeholder list.
    Oracle: the hand-supplied `'pv_abc123'` /
        `'voyage-3-lite'` pair, read back by a `SELECT`.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(
            id='prov-1', content='provenance test',
            prompt_version='pv_abc123',
            embedding_model='voyage-3-lite')
        insert_insight(db, ins)
        row = db._conn.execute(
            'select prompt_version, embedding_model'
            ' from insights where id = ?',
            (ins.id,)).fetchone()
        assert row == ('pv_abc123', 'voyage-3-lite')
    finally:
        db.close()


def test_insert_insight_tolerates_null_provenance(tmp_path):
    """Insight without stamps (tests, fixtures) inserts with NULL columns.

    Mutation: `insert_insight` defaulting a `None` `prompt_version` or
        `embedding_model` to an empty string instead of passing it
        through as NULL.
    Oracle: the row read back as the tuple `(None, None)`.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(id='null-prov', content='no stamp')
        insert_insight(db, ins)
        row = db._conn.execute(
            'select prompt_version, embedding_model'
            ' from insights where id = ?',
            (ins.id,)).fetchone()
        assert row == (None, None)
    finally:
        db.close()


def test_compute_prompt_version_is_stable():
    """compute_prompt_version returns the same hash on repeated calls.

    Mutation: folding a per-process value (a clock reading, an object
    id, or a set's iteration order) into the hash, so every run
    reports every row stale.
    Oracle: two calls equal, and a 16-character lowercase hex string.
    """
    a = compute_prompt_version()
    b = compute_prompt_version()
    assert a == b
    assert len(a) == 16
    assert all(c in '0123456789abcdef' for c in a)


def test_compute_prompt_version_changes_with_prompt(monkeypatch):
    """Changing a REPLAYED prompt changes the hash.

    The key covers only what `enrich --stale-only` re-runs, so
    the enrichment prompt is the right lever here. A write-path-only
    prompt must NOT move it, which
    `tests/test_provenance_staleness.py` pins from the other side.

    Mutation: hashing a constant, or hashing the prompts by name
        rather than by value, so an edit to the enrichment prompt
        leaves the key unchanged and no row is ever reported stale.
    Oracle: the key recomputed with the enrichment prompt perturbed.
    """
    original = compute_prompt_version()
    compute_prompt_version.cache_clear()
    monkeypatch.setattr(
        enrich, 'ENRICHMENT_SYSTEM_PROMPT',
        enrich.ENRICHMENT_SYSTEM_PROMPT + '\n# mutated for test')
    mutated = compute_prompt_version()
    assert mutated != original
    compute_prompt_version.cache_clear()


def test_remember_stamps_provenance(mm_runner):
    """`remember` stamps prompt_version and embedding_model on every row.

    Mutation: leaving `prompt_version` or `embedding_model` unset on
        a write, which the read path would then treat as never
        enriched or embedded.
    Oracle: the row read back by its queue_uuid, against
        `compute_prompt_version()` and the store's configured embed
        model.
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
        prompt_v, embed_model = conn.execute(
            'select prompt_version, embedding_model'
            ' from insights where queue_uuid = ?',
            (queue_uuid,)).fetchone()
    finally:
        conn.close()

    assert prompt_v == compute_prompt_version()
    assert embed_model == 'voyage-3-lite'
