"""Tests for memman.doctor health-check module."""

import struct

from memman.store.edge import insert_edge
from memman.store.node import insert_insight, update_embedding
from memman.store.node import update_enrichment
from tests.conftest import make_edge, make_insight


def _fake_embedding(dim: int = 512) -> bytes:
    """Return a deterministic embedding blob of the given dimension."""
    return struct.pack(f'<{dim}d', *([0.1] * dim))


def _insert_healthy_insight(db, id: str, content: str = 'Healthy test insight with enough content') -> None:
    """Insert an insight with all enrichment fields populated."""
    ins = make_insight(id=id, content=content)
    insert_insight(db, ins)
    update_enrichment(db, id, ['kw1', 'kw2'], 'summary text', ['fact1'])
    update_embedding(db, id, _fake_embedding(), 'voyage-3-lite')


def _insert_edge_pair(db, id_a: str, id_b: str, edge_type: str = 'semantic') -> None:
    """Insert a bidirectional edge between two insights."""
    insert_edge(db, make_edge(source_id=id_a, target_id=id_b, edge_type=edge_type))
    insert_edge(db, make_edge(source_id=id_b, target_id=id_a, edge_type=edge_type))


class TestSqliteIntegrity:

    def test_pass_on_fresh_db(self, tmp_backend):
        """Fresh database passes integrity check."""
        from memman.doctor import check_integrity
        result = check_integrity(tmp_backend)
        assert result['name'] == 'integrity'
        assert result['status'] == 'pass'
        assert result['detail']['result'] == 'ok'


class TestEnrichmentCoverage:

    def test_full_pass(self, tmp_db, tmp_backend):
        """All enrichment fields populated returns pass."""
        from memman.doctor import check_enrichment_coverage
        _insert_healthy_insight(tmp_db, 'e-1')
        _insert_healthy_insight(tmp_db, 'e-2')
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['coverage_pct'] == 100.0

    def test_partial_warn(self, tmp_db, tmp_backend):
        """Some fields missing returns warn when coverage >= 90%."""
        from memman.doctor import check_enrichment_coverage
        for i in range(10):
            _insert_healthy_insight(tmp_db, f'e-{i}', f'Content for insight number {i}')
        ins = make_insight(id='e-bare', content='Bare insight without enrichment')
        insert_insight(tmp_db, ins)
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['missing_embedding'] == 1


class TestOrphanInsights:

    def test_none_pass(self, tmp_db, tmp_backend):
        """No orphans returns pass."""
        from memman.doctor import check_orphan_insights
        _insert_healthy_insight(tmp_db, 'o-1')
        _insert_healthy_insight(tmp_db, 'o-2')
        _insert_edge_pair(tmp_db, 'o-1', 'o-2')
        result = check_orphan_insights(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['orphan_count'] == 0

    def test_present_fail(self, tmp_db, tmp_backend):
        """All insights orphaned returns fail."""
        from memman.doctor import check_orphan_insights
        _insert_healthy_insight(tmp_db, 'o-1')
        _insert_healthy_insight(tmp_db, 'o-2')
        result = check_orphan_insights(tmp_backend)
        assert result['status'] == 'fail'
        assert result['detail']['orphan_count'] == 2


class TestDanglingEdges:

    def test_none_pass(self, tmp_db, tmp_backend):
        """Clean edges return pass."""
        from memman.doctor import check_dangling_edges
        _insert_healthy_insight(tmp_db, 'd-1')
        _insert_healthy_insight(tmp_db, 'd-2')
        _insert_edge_pair(tmp_db, 'd-1', 'd-2')
        result = check_dangling_edges(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['count'] == 0

    def test_present_fail(self, tmp_db, tmp_backend):
        """Edges pointing to soft-deleted insights return fail."""
        from memman.doctor import check_dangling_edges
        _insert_healthy_insight(tmp_db, 'd-1')
        _insert_healthy_insight(tmp_db, 'd-2')
        _insert_edge_pair(tmp_db, 'd-1', 'd-2')
        tmp_db._exec(
            "UPDATE insights SET deleted_at = '2026-01-01T00:00:00Z'"
            " WHERE id = 'd-2'")
        result = check_dangling_edges(tmp_backend)
        assert result['status'] == 'fail'
        assert result['detail']['count'] == 2


class TestEmbeddingConsistency:

    def test_consistent_pass(self, tmp_db, tmp_backend):
        """All embeddings same size returns pass."""
        from memman.doctor import check_embedding_consistency
        _insert_healthy_insight(tmp_db, 'emb-1')
        _insert_healthy_insight(tmp_db, 'emb-2')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'pass'

    def test_mixed_fail(self, tmp_db, tmp_backend):
        """Different embedding sizes returns fail."""
        from memman.doctor import check_embedding_consistency
        _insert_healthy_insight(tmp_db, 'emb-1')
        ins2 = make_insight(id='emb-2', content='Different dim embedding')
        insert_insight(tmp_db, ins2)
        update_embedding(tmp_db, 'emb-2', _fake_embedding(dim=256),
                         'voyage-3-lite')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'fail'
        assert len(result['detail']['sizes']) > 1


class TestProvenanceDrift:

    def test_no_rows_pass(self, tmp_db, tmp_backend):
        """Empty store: no stale rows."""
        from memman.doctor import check_provenance_drift
        result = check_provenance_drift(tmp_backend)
        assert result['name'] == 'provenance_drift'
        assert result['status'] == 'pass'
        assert result['detail']['stale_rows'] == 0

    def test_all_current_pass(self, tmp_db, tmp_backend):
        """All rows stamped with active prompt_version + slow model: pass."""
        from memman import config
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        active_model = config.require(config.LLM_MODEL_SLOW_CANONICAL)

        _insert_healthy_insight(tmp_db, 'p-1')
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?, model_id = ?'
            ' WHERE id = ?',
            (active_pv, active_model, 'p-1'))

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['stale_rows'] == 0

    def test_stale_prompt_version_warns(self, tmp_db, tmp_backend):
        """Rows with non-current prompt_version surface as a warning."""
        from memman import config
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        active_model = config.require(config.LLM_MODEL_SLOW_CANONICAL)

        _insert_healthy_insight(tmp_db, 'p-stale-1')
        _insert_healthy_insight(tmp_db, 'p-stale-2')
        _insert_healthy_insight(tmp_db, 'p-fresh')

        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?, model_id = ?'
            " WHERE id IN ('p-stale-1', 'p-stale-2')",
            ('deadbeefdeadbeef', active_model))
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?, model_id = ?'
            " WHERE id = 'p-fresh'",
            (active_pv, active_model))

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['stale_rows'] == 2
        assert 'remediation' in result['detail']

    def test_stale_model_id_warns(self, tmp_db, tmp_backend):
        """Rows with non-current model_id surface as a warning."""
        from memman import config
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        active_model = config.require(config.LLM_MODEL_SLOW_CANONICAL)

        _insert_healthy_insight(tmp_db, 'm-old')
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?, model_id = ?'
            " WHERE id = 'm-old'",
            (active_pv, 'anthropic/claude-haiku-1.0'))
        assert active_model != 'anthropic/claude-haiku-1.0'

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['stale_rows'] == 1


class TestEdgeDegree:

    def test_healthy_pass(self, tmp_db, tmp_backend):
        """Well-connected graph returns pass."""
        from memman.doctor import check_edge_degree
        ids = [f'deg-{i}' for i in range(6)]
        for id in ids:
            _insert_healthy_insight(tmp_db, id, f'Content for {id} with enough length')
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                _insert_edge_pair(tmp_db, id_a, id_b)
        result = check_edge_degree(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['median'] >= 5

    def test_sparse_fail(self, tmp_db, tmp_backend):
        """Insights with no edges returns fail."""
        from memman.doctor import check_edge_degree
        _insert_healthy_insight(tmp_db, 'deg-1')
        _insert_healthy_insight(tmp_db, 'deg-2')
        result = check_edge_degree(tmp_backend)
        assert result['status'] == 'fail'
        assert result['detail']['median'] == 0


class TestRunAllChecks:

    def test_structure(self, tmp_db, tmp_backend):
        """Verify output shape: status, checks list, total_active."""
        from memman.doctor import run_all_checks
        _insert_healthy_insight(tmp_db, 'all-1')
        result = run_all_checks(tmp_backend, tmp_db)
        assert 'status' in result
        assert 'checks' in result
        assert 'total_active' in result
        assert isinstance(result['checks'], list)
        assert result['status'] in {'pass', 'warn', 'fail'}

    def test_empty_db(self, tmp_db, tmp_backend):
        """Empty store returns status 'empty' with no checks."""
        from memman.doctor import run_all_checks
        result = run_all_checks(tmp_backend, tmp_db)
        assert result['status'] == 'empty'
        assert result['total_active'] == 0
        assert result['checks'] == []

    def test_healthy_db(self, tmp_db, tmp_backend):
        """Fully healthy DB returns status 'pass'."""
        from memman.doctor import run_all_checks
        ids = [f'h-{i}' for i in range(6)]
        for id in ids:
            _insert_healthy_insight(tmp_db, id, f'Healthy content for {id} insight')
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                _insert_edge_pair(tmp_db, id_a, id_b)
        result = run_all_checks(tmp_backend, tmp_db)
        assert result['status'] == 'pass', [
            (c['name'], c['status'], c.get('detail'))
            for c in result['checks'] if c['status'] != 'pass']
        assert all(c['status'] == 'pass' for c in result['checks'])
