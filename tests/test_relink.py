"""Auto-relink tests for constants hash change detection and edge management."""

import json
from datetime import datetime, timedelta, timezone

from click.testing import CliRunner
from mnemon.cli import cli
from mnemon.graph.engine import compute_constants_hash, relink_auto_edges
from mnemon.store.db import open_db
from mnemon.store.edge import get_all_edges, insert_edge
from mnemon.store.node import insert_insight
from tests.conftest import make_edge, make_insight


class TestOpenDbTriggersRelink:
    """open_db triggers relink when stored hash differs from current."""

    def test_triggers_relink_on_hash_mismatch(self, tmp_path):
        """Seed meta with wrong hash, open DB, verify hash updated."""
        db = open_db(str(tmp_path))
        db._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value)"
            " VALUES ('constants_hash', 'wrong_hash')")
        db.close()

        db = open_db(str(tmp_path))
        row = db._conn.execute(
            "SELECT value FROM meta WHERE key = 'constants_hash'"
            ).fetchone()
        assert row is not None
        assert row[0] == compute_constants_hash()
        db.close()


class TestOpenDbSkipsRelink:
    """open_db skips relink when stored hash matches current."""

    def test_skips_relink_when_hash_matches(self, tmp_path):
        """Seed meta with correct hash; open DB does not modify edges."""
        db = open_db(str(tmp_path))
        current_hash = compute_constants_hash()
        db._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value)"
            " VALUES ('constants_hash', ?)",
            (current_hash,))

        ins = make_insight(id='skip-1', content='test content')
        insert_insight(db, ins)
        auto_edge = make_edge(
            source_id='skip-1', target_id='skip-1',
            edge_type='semantic',
            metadata={'created_by': 'auto', 'cosine': '0.9'})
        insert_edge(db, auto_edge)
        db.close()

        db = open_db(str(tmp_path))
        edges = get_all_edges(db)
        semantic = [e for e in edges if e.edge_type == 'semantic']
        assert len(semantic) == 1
        db.close()


class TestRelinkPreservesManualSemanticEdges:
    """Manual semantic edges (created_by: claude) survive relink."""

    def test_preserves_manual_semantic(self, tmp_db):
        """Insert claude-created semantic edge, run relink, verify survival."""
        ins1 = make_insight(id='ms-1', content='first')
        ins2 = make_insight(id='ms-2', content='second')
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        manual_edge = make_edge(
            source_id='ms-1', target_id='ms-2',
            edge_type='semantic',
            metadata={'created_by': 'claude', 'cosine': '0.5'})
        insert_edge(tmp_db, manual_edge)

        relink_auto_edges(tmp_db)

        edges = get_all_edges(tmp_db)
        manual = [e for e in edges
                  if e.edge_type == 'semantic'
                  and e.metadata.get('created_by') == 'claude']
        assert len(manual) == 1


class TestRelinkPreservesManualEntityEdges:
    """Manual entity edges (created_by: claude) survive relink."""

    def test_preserves_manual_entity(self, tmp_db):
        """Insert claude-created entity edge, run relink, verify survival."""
        ins1 = make_insight(id='me-1', content='first',
                            entities=['Python'])
        ins2 = make_insight(id='me-2', content='second',
                            entities=['Python'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        manual_edge = make_edge(
            source_id='me-1', target_id='me-2',
            edge_type='entity',
            metadata={'entity': 'Python', 'created_by': 'claude'})
        insert_edge(tmp_db, manual_edge)

        relink_auto_edges(tmp_db)

        edges = get_all_edges(tmp_db)
        manual = [e for e in edges
                  if e.edge_type == 'entity'
                  and e.metadata.get('created_by') == 'claude']
        assert len(manual) == 1


class TestRelinkPreservesManualCreatedByEntityEdges:
    """Entity edges with created_by='manual' survive relink."""

    def test_preserves_manual_created_by_entity(self, tmp_db):
        """Insert manual-created entity edge, run relink, verify survival."""
        ins1 = make_insight(id='mm-1', content='first',
                            entities=['Go'])
        ins2 = make_insight(id='mm-2', content='second',
                            entities=['Go'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        manual_edge = make_edge(
            source_id='mm-1', target_id='mm-2',
            edge_type='entity',
            metadata={'entity': 'Go', 'created_by': 'manual'})
        insert_edge(tmp_db, manual_edge)

        relink_auto_edges(tmp_db)

        edges = get_all_edges(tmp_db)
        manual = [e for e in edges
                  if e.edge_type == 'entity'
                  and e.metadata.get('created_by') == 'manual']
        assert len(manual) == 1, (
            'relink_auto_edges deleted manual entity edges — '
            'filter should preserve both claude and manual created_by')


class TestRelinkDeletesAutoEntityEdges:
    """Auto entity edges (no created_by) are replaced during relink."""

    def test_deletes_auto_entity(self, tmp_db):
        """Insert auto entity edges, run relink, verify replaced."""
        ins1 = make_insight(id='ae-1', content='Go uses SQLite',
                            entities=['Go', 'SQLite'])
        ins2 = make_insight(id='ae-2', content='SQLite for storage',
                            entities=['SQLite'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        auto_edge = make_edge(
            source_id='ae-1', target_id='ae-2',
            edge_type='entity',
            metadata={'entity': 'SQLite'})
        insert_edge(tmp_db, auto_edge)

        relink_auto_edges(tmp_db)

        edges = get_all_edges(tmp_db)
        entity_edges = [e for e in edges if e.edge_type == 'entity']
        assert len(entity_edges) >= 1


class TestTemporalDeleteOnlySubThreshold:
    """Only low-weight proximity edges deleted during relink."""

    def test_removes_only_sub_threshold(self, tmp_db):
        """Mix of low/high weight proximity edges; only low deleted."""
        now = datetime.now(timezone.utc)
        ins1 = make_insight(id='tp-1', content='first',
                            created_at=now - timedelta(hours=1))
        ins2 = make_insight(id='tp-2', content='second',
                            created_at=now)
        ins3 = make_insight(id='tp-3', content='third',
                            created_at=now - timedelta(hours=20))
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)
        insert_insight(tmp_db, ins3)

        high_weight = make_edge(
            source_id='tp-2', target_id='tp-1',
            edge_type='temporal', weight=0.5,
            metadata={'sub_type': 'proximity', 'hours_diff': '1.0'})
        low_weight = make_edge(
            source_id='tp-1', target_id='tp-3',
            edge_type='temporal', weight=0.05,
            metadata={'sub_type': 'proximity', 'hours_diff': '19.0'})
        backbone = make_edge(
            source_id='tp-1', target_id='tp-2',
            edge_type='temporal', weight=1.0,
            metadata={'sub_type': 'backbone', 'direction': 'precedes'})
        insert_edge(tmp_db, high_weight)
        insert_edge(tmp_db, low_weight)
        insert_edge(tmp_db, backbone)

        relink_auto_edges(tmp_db)

        edges = get_all_edges(tmp_db)
        temporal = [e for e in edges if e.edge_type == 'temporal']
        low = [e for e in temporal
               if e.metadata.get('sub_type') == 'proximity'
               and e.weight < 0.10]
        assert len(low) == 0
        high = [e for e in temporal
                if e.metadata.get('sub_type') == 'proximity'
                and e.weight >= 0.10]
        assert len(high) >= 1
        backbones = [e for e in temporal
                     if e.metadata.get('sub_type') == 'backbone']
        assert len(backbones) >= 1


class TestRelinkDryRunCreationCounts:
    """Dry run returns non-zero creation counts without writing edges."""

    def test_dry_run_counts_entity_edges(self, tmp_db):
        """Seed two insights sharing an entity; dry run reports entity_created > 0."""
        ins1 = make_insight(
            id='drc-1', content='Python web framework',
            entities=['Python'])
        ins2 = make_insight(
            id='drc-2', content='Python data analysis',
            entities=['Python'])
        insert_insight(tmp_db, ins1)
        insert_insight(tmp_db, ins2)

        stats = relink_auto_edges(tmp_db, dry_run=True)

        assert stats['entity_created'] > 0
        edges = get_all_edges(tmp_db)
        entity_edges = [e for e in edges if e.edge_type == 'entity']
        assert len(entity_edges) == 0


class TestGraphRelinkCliDryRun:
    """CLI dry run shows stats without modifying DB."""

    def test_dry_run(self, tmp_path, monkeypatch):
        """Verify dry run outputs stats and makes no changes."""
        monkeypatch.delenv('MNEMON_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        db = open_db(str(store_path))
        ins1 = make_insight(id='dr-1', content='Python web app',
                            entities=['Python'])
        ins2 = make_insight(id='dr-2', content='Python data tool',
                            entities=['Python'])
        insert_insight(db, ins1)
        insert_insight(db, ins2)
        auto_edge = make_edge(
            source_id='dr-1', target_id='dr-1',
            edge_type='semantic',
            metadata={'created_by': 'auto', 'cosine': '0.9'})
        insert_edge(db, auto_edge)
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir,
            'graph', 'relink', '--dry-run',
            ])

        assert 'entity_created' in result.output
        assert '"entity_created": 0' not in result.output

        db = open_db(str(store_path))
        edges = get_all_edges(db)
        semantic = [e for e in edges if e.edge_type == 'semantic']
        assert len(semantic) == 1
        db.close()


class TestGraphRelinkCliLive:
    """CLI live relink modifies DB and logs to oplog."""

    def test_live_relink(self, tmp_path, monkeypatch):
        """Verify relink creates oplog entry with accurate edge counts."""
        monkeypatch.delenv('MNEMON_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        db = open_db(str(store_path))
        ins1 = make_insight(id='lr-1', content='Go uses SQLite',
                            entities=['Go', 'SQLite'])
        ins2 = make_insight(id='lr-2', content='SQLite storage layer',
                            entities=['SQLite'])
        insert_insight(db, ins1)
        insert_insight(db, ins2)
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir,
            'graph', 'relink',
            ])

        db = open_db(str(store_path))
        row = db._conn.execute(
            "SELECT COUNT(*) FROM oplog"
            " WHERE operation = 'relink'").fetchone()
        assert row[0] >= 1

        oplog_row = db._conn.execute(
            "SELECT detail FROM oplog WHERE operation = 'relink'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        logged_stats = json.loads(oplog_row[0])

        actual_entity = db._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'entity'"
            " AND (json_extract(metadata, '$.created_by') IS NULL"
            "      OR json_extract(metadata, '$.created_by') <> 'claude')"
            ).fetchone()[0]
        assert logged_stats['entity_created'] == actual_entity
        db.close()
