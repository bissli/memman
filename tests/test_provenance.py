"""Tests for B4 provenance columns (prompt_version, model_id, embedding_model).

Covers: the migration adds the three columns; insert_insight persists
them when the Insight dataclass carries them; compute_prompt_version()
hashes the write-path system prompts and is stable across calls; the
`remember` pipeline stamps every newly-inserted row.
"""

import json
from pathlib import Path
from unittest.mock import patch

from memman.cli import cli
from memman.pipeline.remember import compute_prompt_version
from memman.store.db import open_db
from memman.store.model import Insight
from memman.store.node import insert_insight


def test_baseline_schema_has_provenance_columns(tmp_path):
    """A fresh open_db produces an insights table with the three columns.
    """
    db = open_db(str(tmp_path))
    try:
        cols = db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        names = {row[1] for row in cols}
        assert 'prompt_version' in names
        assert 'model_id' in names
        assert 'embedding_model' in names
    finally:
        db.close()


def test_insert_insight_persists_provenance(tmp_path):
    """insert_insight writes prompt_version/model_id/embedding_model.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(
            id='prov-1', content='provenance test',
            prompt_version='pv_abc123',
            model_id='anthropic/claude-haiku-4.5',
            embedding_model='voyage-3-lite')
        insert_insight(db, ins)
        row = db._conn.execute(
            'SELECT prompt_version, model_id, embedding_model'
            ' FROM insights WHERE id = ?',
            (ins.id,)).fetchone()
        assert row == (
            'pv_abc123',
            'anthropic/claude-haiku-4.5',
            'voyage-3-lite')
    finally:
        db.close()


def test_insert_insight_tolerates_null_provenance(tmp_path):
    """Insight without stamps (tests, fixtures) inserts with NULL columns.
    """
    db = open_db(str(tmp_path))
    try:
        ins = Insight(id='null-prov', content='no stamp')
        insert_insight(db, ins)
        row = db._conn.execute(
            'SELECT prompt_version, model_id, embedding_model'
            ' FROM insights WHERE id = ?',
            (ins.id,)).fetchone()
        assert row == (None, None, None)
    finally:
        db.close()


def test_compute_prompt_version_is_stable():
    """compute_prompt_version returns the same hash on repeated calls.
    """
    a = compute_prompt_version()
    b = compute_prompt_version()
    assert a == b
    assert len(a) == 16
    assert all(c in '0123456789abcdef' for c in a)


def test_compute_prompt_version_changes_with_prompt(monkeypatch):
    """Changing a write-path prompt string changes the hash.
    """
    original = compute_prompt_version()
    compute_prompt_version.cache_clear()
    from memman.llm import extract as llm_extract
    monkeypatch.setattr(
        llm_extract, 'FACT_EXTRACTION_SYSTEM',
        llm_extract.FACT_EXTRACTION_SYSTEM + '\n# mutated for test')
    mutated = compute_prompt_version()
    assert mutated != original
    compute_prompt_version.cache_clear()


def test_remember_stamps_provenance(mm_runner):
    """`remember` stores rows with all three provenance columns set."""
    r, data_dir = mm_runner

    def _one_fact(client, content):
        return [{'text': content, 'category': 'fact',
                 'importance': 3, 'entities': []}]

    with patch('memman.llm.extract.extract_facts', _one_fact):
        result = r.invoke(cli, [
            '--data-dir', data_dir,
            'remember', '--no-reconcile',
            'provenance stamping end-to-end'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    queue_id = data['queue_id']

    import sqlite3
    store_path = Path(data_dir) / 'data' / 'default' / 'memman.db'
    conn = sqlite3.connect(str(store_path))
    try:
        prompt_v, model_id, embed_model = conn.execute(
            'SELECT prompt_version, model_id, embedding_model'
            ' FROM insights WHERE source = ?',
            (f'queue:{queue_id}',)).fetchone()
    finally:
        conn.close()

    assert prompt_v == compute_prompt_version()
    assert embed_model == 'voyage-3-lite'
    assert model_id is not None
