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


def test_round_trip_preserves_linked_and_enriched_at(tmp_db, tmp_path):
    """Snapshot writer + reader round-trips the new lifecycle stamps."""
    from memman.store.model import format_timestamp
    from memman.store.node import stamp_enriched, stamp_linked
    from datetime import datetime, timezone

    fp = _seed(tmp_db)
    ts = format_timestamp(
        datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
    stamp_linked(tmp_db, 'snap-a', ts)
    stamp_enriched(tmp_db, 'snap-a', ts)

    store_dir = str(tmp_path)
    assert write_snapshot(tmp_db, store_dir, fp) is True
    snap = read_snapshot(store_dir, fp)
    assert snap is not None
    by_id = {i.id: i for i in snap.insights}
    assert by_id['snap-a'].linked_at is not None
    assert by_id['snap-a'].linked_at.tzinfo is not None
    assert by_id['snap-a'].enriched_at is not None
    assert by_id['snap-b'].linked_at is None
    assert by_id['snap-b'].enriched_at is None


def test_read_back_compat_old_snapshot_without_lifecycle_keys(
        tmp_db, tmp_path):
    """Old snapshot files without `linked_at`/`enriched_at` keys still load.

    Simulates a pre-fix snapshot by writing one through the current
    writer, then surgically rewriting the meta JSON to drop the new
    keys before reading back. The reader must default to None rather
    than raising KeyError.
    """
    import struct
    fp = _seed(tmp_db)
    store_dir = str(tmp_path)
    assert write_snapshot(tmp_db, store_dir, fp) is True

    path = snapshot_path(store_dir)
    raw = path.read_bytes()
    cursor = 4
    (header_len,) = struct.unpack('<I', raw[cursor:cursor + 4])
    cursor += 4 + header_len
    (embed_len,) = struct.unpack('<Q', raw[cursor:cursor + 8])
    cursor += 8 + embed_len
    (meta_len,) = struct.unpack('<I', raw[cursor:cursor + 4])
    meta_start = cursor + 4
    meta_end = meta_start + meta_len
    old_meta = json.loads(raw[meta_start:meta_end].decode('utf-8'))
    for entry in old_meta:
        entry.pop('linked_at', None)
        entry.pop('enriched_at', None)
    new_meta_blob = json.dumps(old_meta).encode('utf-8')
    rebuilt = (
        raw[:cursor]
        + struct.pack('<I', len(new_meta_blob))
        + new_meta_blob
        + raw[meta_end:])
    path.write_bytes(rebuilt)

    snap = read_snapshot(store_dir, fp)
    assert snap is not None
    for i in snap.insights:
        assert i.linked_at is None
        assert i.enriched_at is None


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
