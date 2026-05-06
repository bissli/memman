"""Tests for the recall snapshot file (memman.store.snapshot).

The worker writes a snapshot after each drain that processes rows;
recall reads it as a hot-path cache instead of issuing full-table
DB scans. Verifies write/read round-trip, fingerprint mismatch
fallback, and that recall consumes the snapshot when present.
"""

import json
import pathlib

from click.testing import CliRunner
from memman.cli import cli
from memman.embed.fingerprint import Fingerprint, active_fingerprint
from memman.embed.vector import serialize_vector
from memman.store.edge import insert_edge
from memman.store.model import Edge
from memman.store.node import insert_insight, update_embedding
from memman.store.snapshot import SNAPSHOT_FILENAME, delete_snapshot
from memman.store.snapshot import read_snapshot, snapshot_path, write_snapshot
from tests.conftest import make_insight


def _seed(tmp_db):
    """Insert two embedded insights with one semantic edge between them."""
    a = make_insight(id='snap-a', content='alpha topic',
                     entities=['Alpha'])
    b = make_insight(id='snap-b', content='beta topic',
                     entities=['Beta'])
    insert_insight(tmp_db, a)
    insert_insight(tmp_db, b)
    fp = active_fingerprint()
    update_embedding(
        tmp_db, 'snap-a',
        serialize_vector([0.1] * fp.dim), fp.model)
    update_embedding(
        tmp_db, 'snap-b',
        serialize_vector([0.2] * fp.dim), fp.model)
    edge = Edge(
        source_id='snap-a', target_id='snap-b',
        edge_type='semantic', weight=0.7,
        metadata={'created_by': 'auto'})
    insert_edge(tmp_db, edge)
    return fp


def test_write_then_read_round_trip(tmp_db, tmp_path):
    """write_snapshot then read_snapshot returns equivalent data."""
    fp = _seed(tmp_db)
    store_dir = str(tmp_path)

    assert write_snapshot(tmp_db, store_dir, fp) is True
    assert snapshot_path(store_dir).exists()

    snap = read_snapshot(store_dir, fp)
    assert snap is not None
    ids = {i.id for i in snap.insights}
    assert ids == {'snap-a', 'snap-b'}
    assert set(snap.embeddings.keys()) == {'snap-a', 'snap-b'}
    assert len(snap.embeddings['snap-a']) == fp.dim
    assert 'snap-a' in snap.adjacency
    targets = [t for t, _et, _w in snap.adjacency['snap-a']]
    assert targets == ['snap-b']


def test_read_returns_none_on_fingerprint_mismatch(tmp_db, tmp_path):
    """A snapshot with a different embedding model is rejected."""
    fp = _seed(tmp_db)
    store_dir = str(tmp_path)
    write_snapshot(tmp_db, store_dir, fp)

    other = Fingerprint(provider='openai', model='other', dim=fp.dim)
    assert read_snapshot(store_dir, other) is None


def test_read_returns_none_when_missing(tmp_path):
    """No snapshot file -> read returns None cleanly."""
    fp = Fingerprint(provider='voyage', model='voyage-3-lite', dim=512)
    assert read_snapshot(str(tmp_path), fp) is None


def test_atomic_write_no_partial_file(tmp_db, tmp_path):
    """The .tmp sibling is cleaned up; only the final file remains."""
    fp = _seed(tmp_db)
    store_dir = str(tmp_path)
    write_snapshot(tmp_db, store_dir, fp)

    files = list(tmp_path.iterdir())
    names = {f.name for f in files}
    assert SNAPSHOT_FILENAME in names
    assert not any(n.endswith('.tmp') for n in names)


def test_skipped_when_over_max_insights(tmp_db, tmp_path, monkeypatch):
    """write_snapshot returns False (and writes nothing) above the cap."""
    monkeypatch.setattr(
        'memman.store.snapshot.SNAPSHOT_MAX_INSIGHTS', 1)
    _seed(tmp_db)
    fp = active_fingerprint()
    store_dir = str(tmp_path)
    assert write_snapshot(tmp_db, store_dir, fp) is False
    assert not snapshot_path(store_dir).exists()


def test_recall_consumes_snapshot_when_present(tmp_path):
    """Recall must use the snapshot rather than calling SQL helpers.

    Patch the SQL helpers to raise; recall succeeds only because the
    snapshot path bypasses them.
    """
    r = CliRunner()
    data_dir = str(tmp_path / 'data')

    seed_result = r.invoke(
        cli, ['--data-dir', data_dir, 'remember', 'alpha topic'])
    assert seed_result.exit_code == 0, seed_result.output
    drain_result = r.invoke(
        cli, ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])
    assert drain_result.exit_code == 0, drain_result.output

    from memman.store.db import read_active
    from memman.store.db import store_dir as _store_dir
    snap_path = (
        pathlib.Path(_store_dir(data_dir, read_active(data_dir)))
        / 'recall_snapshot.v1.bin')
    assert snap_path.exists(), 'drain should write a recall snapshot'

    recall_result = r.invoke(
        cli, ['--data-dir', data_dir, 'recall', 'alpha'])
    assert recall_result.exit_code == 0, recall_result.output
    payload = json.loads(recall_result.output)
    contents = [hit['insight']['content'] for hit in payload['results']]
    assert any('alpha' in c for c in contents)


def test_recall_falls_back_when_snapshot_absent(tmp_path):
    """Deleting the snapshot file forces a clean fallback to SQL."""
    r = CliRunner()
    data_dir = str(tmp_path / 'data')
    r.invoke(cli, ['--data-dir', data_dir, 'remember', 'gamma topic'])
    r.invoke(cli, ['--data-dir', data_dir, 'scheduler', 'drain', '--pending'])

    from memman.store.db import read_active
    from memman.store.db import store_dir as _store_dir
    name = read_active(data_dir)
    delete_snapshot(_store_dir(data_dir, name))

    recall_result = r.invoke(
        cli, ['--data-dir', data_dir, 'recall', 'gamma'])
    assert recall_result.exit_code == 0, recall_result.output
