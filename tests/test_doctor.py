"""Tests for memman.doctor health-check module."""

import json
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.doctor import check_drain_heartbeat, check_env_completeness
from memman.doctor import check_env_permissions, check_queue_schema
from memman.doctor import check_scheduler_heartbeat, check_scheduler_state
from memman.doctor import check_schema_columns
from memman.store.db import open_db
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

    @pytest.mark.parametrize(('stale_col', 'stale_value', 'stale_count'), [
        ('prompt_version', 'deadbeefdeadbeef', 2),
        ('model_id', 'anthropic/claude-haiku-1.0', 1),
    ])
    def test_drift_warns(
            self, tmp_db, tmp_backend,
            stale_col, stale_value, stale_count):
        """Rows with non-current prompt_version or model_id surface as warn."""
        from memman import config
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        active_model = config.require(config.LLM_MODEL_SLOW_CANONICAL)

        for i in range(stale_count):
            _insert_healthy_insight(tmp_db, f'p-stale-{i}')
        if stale_count == 2:
            _insert_healthy_insight(tmp_db, 'p-fresh')

        stale_ids = ', '.join(f"'p-stale-{i}'" for i in range(stale_count))
        if stale_col == 'prompt_version':
            tmp_db._exec(
                f'UPDATE insights SET prompt_version = ?, model_id = ?'
                f' WHERE id IN ({stale_ids})',
                (stale_value, active_model))
        else:
            tmp_db._exec(
                f'UPDATE insights SET prompt_version = ?, model_id = ?'
                f' WHERE id IN ({stale_ids})',
                (active_pv, stale_value))
        if stale_count == 2:
            tmp_db._exec(
                'UPDATE insights SET prompt_version = ?, model_id = ?'
                " WHERE id = 'p-fresh'",
                (active_pv, active_model))

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['stale_rows'] == stale_count
        if stale_col == 'prompt_version':
            assert 'remediation' in result['detail']


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
        result = run_all_checks(tmp_backend)
        assert 'status' in result
        assert 'checks' in result
        assert 'total_active' in result
        assert isinstance(result['checks'], list)
        assert result['status'] in {'pass', 'warn', 'fail'}

    def test_empty_db(self, tmp_db, tmp_backend):
        """Empty store returns status 'empty' with no checks."""
        from memman.doctor import run_all_checks
        result = run_all_checks(tmp_backend)
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
        result = run_all_checks(tmp_backend)
        assert result['status'] == 'pass', [
            (c['name'], c['status'], c.get('detail'))
            for c in result['checks'] if c['status'] != 'pass']
        assert all(c['status'] == 'pass' for c in result['checks'])


class TestEnvCompleteness:
    """check_env_completeness against INSTALLABLE_KEYS."""

    @pytest.fixture
    def write_env(self, tmp_path, monkeypatch):
        """Write a custom env file under a fresh data dir."""
        from memman import config
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))

        def _write(contents: str) -> None:
            (data_dir / config.ENV_FILENAME).write_text(contents)
            config.reset_file_cache()

        return _write

    def test_pass_when_all_present(self, write_env):
        """All INSTALLABLE_KEYS in the file -> status pass."""
        from memman import config
        lines = [f'{key}=value-for-{key}' for key in config.INSTALLABLE_KEYS]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'

    def test_warns_when_non_secret_missing(self, write_env):
        """Missing non-secret key -> warn with key in detail.missing."""
        from memman import config
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key != config.LLM_MODEL_FAST
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert config.LLM_MODEL_FAST in out['detail']['missing']
        assert 'memman install' in out['detail']['fix']

    def test_ignores_optional_secret(self, write_env):
        """Missing OPENAI_EMBED_API_KEY (optional secret) does not fail."""
        from memman import config
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key != config.OPENAI_EMBED_API_KEY
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'
        assert config.OPENAI_EMBED_API_KEY not in out.get('detail', {}).get(
            'missing', [])

    def test_legacy_backend_keys_are_optional(self, write_env):
        """`MEMMAN_BACKEND` / `MEMMAN_PG_DSN` are slated for removal in 2.6.

        Per-store keys carry the dispatch contract now; the legacy
        globals must not be flagged as missing on installs that have
        already migrated.
        """
        from memman import config
        legacy = {config.BACKEND, config.PG_DSN}
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key not in legacy
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'
        missing = out.get('detail', {}).get('missing', [])
        assert config.BACKEND not in missing
        assert config.PG_DSN not in missing


class TestCheckPerStoreKeys:
    """`check_per_store_keys` validates `MEMMAN_BACKEND_<store>` shape."""

    def test_pass_when_no_stores(self, tmp_path):
        """Empty data dir -> pass with empty stores list."""
        from memman.doctor import check_per_store_keys
        out = check_per_store_keys(str(tmp_path / 'memman'))
        assert out['name'] == 'per_store_keys'
        assert out['status'] == 'pass'
        assert out['detail']['stores'] == []

    def test_pass_when_per_store_key_resolves(self, tmp_path, env_file):
        """SQLite store with explicit per-store key -> pass."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'one').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'one', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('one'), 'sqlite')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        names = [s['store'] for s in out['detail']['stores']]
        assert 'one' in names

    def test_pass_when_falling_back_to_default(self, tmp_path, env_file):
        """No per-store key, default sqlite -> pass with fallback flag."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'fallback').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'fallback', 'memman.db').write_bytes(b'')
        env_file(config.DEFAULT_BACKEND, 'sqlite')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        match = next(s for s in out['detail']['stores']
                     if s['store'] == 'fallback')
        assert match['backend'] == 'sqlite'
        assert match['source'] == 'default'

    def test_fails_on_unknown_backend_value(self, tmp_path, env_file):
        """`MEMMAN_BACKEND_<store>=mongo` -> fail (unknown backend)."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'bad').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'bad', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('bad'), 'mongo')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'fail'
        bad = next(s for s in out['detail']['stores']
                   if s['store'] == 'bad')
        assert 'unknown backend' in bad.get('error', '').lower()

    def test_warns_when_postgres_dsn_missing(self, tmp_path, env_file):
        """`MEMMAN_BACKEND_<store>=postgres` without DSN -> fail."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_one').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_one', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_one'), 'postgres')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'fail'
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_one')
        assert 'dsn' in pg.get('error', '').lower()

    def test_postgres_default_dsn_satisfies(self, tmp_path, env_file):
        """`MEMMAN_DEFAULT_PG_DSN` covers a postgres store without a per-store DSN.
        """
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_two').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_two', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_two'), 'postgres')
        env_file(config.DEFAULT_PG_DSN, 'postgresql://x@y/z')

        out = check_per_store_keys(data_dir)
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_two')
        assert pg.get('error') is None
        assert pg['backend'] == 'postgres'


def _started_scheduler_status(interval=900):
    """Test helper: pretend the scheduler is installed + started."""
    return {
        'interval_seconds': interval,
        'state': 'started',
        'installed': True,
        }


class TestHardening:
    """B12 doctor checks: schema, env perms, scheduler, worker runs."""

    def test_schema_columns_passes_on_current_schema(self, tmp_path):
        """Fresh DB has all expected provenance columns."""
        db = open_db(str(tmp_path))
        try:
            from memman.store.sqlite import SqliteBackend
            result = check_schema_columns(SqliteBackend(db))
            assert result['status'] == 'pass'
            assert result['detail']['missing'] == []
        finally:
            db.close()

    def test_schema_columns_fails_when_column_missing(self, tmp_path):
        """A DB without provenance columns should fail the schema check.
        """
        db = open_db(str(tmp_path))
        try:
            db._conn.executescript(
                'CREATE TABLE insights_minimal (id TEXT PRIMARY KEY);'
                'DROP TABLE insights;'
                'ALTER TABLE insights_minimal RENAME TO insights;')
            from memman.store.sqlite import SqliteBackend
            result = check_schema_columns(SqliteBackend(db))
            assert result['status'] == 'fail'
            assert 'prompt_version' in result['detail']['missing']
            assert 'model_id' in result['detail']['missing']
            assert 'embedding_model' in result['detail']['missing']
        finally:
            db.close()

    def test_queue_schema_passes_with_worker_runs(self, tmp_path):
        """A fresh queue.db has the worker_runs table."""
        result = check_queue_schema(str(tmp_path))
        assert result['status'] == 'pass'
        assert result['detail']['missing'] == []

    @pytest.mark.parametrize(('mode', 'expected_status', 'assert_issue'), [
        (None, 'pass', False),
        (0o644, 'fail', True),
        (0o600, 'pass', False),
    ])
    def test_env_permissions(
            self, tmp_path, monkeypatch, mode, expected_status, assert_issue):
        """Permissions check passes for missing or 0600, fails for 0644."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        if mode is not None:
            mm = tmp_path / '.memman'
            mm.mkdir(mode=0o700)
            env = mm / 'env'
            env.write_text('OPENROUTER_API_KEY=fake\n')
            env.chmod(mode)
        result = check_env_permissions()
        assert result['status'] == expected_status
        if assert_issue:
            assert any('env file' in issue
                       for issue in result['detail']['issues'])

    def test_scheduler_state_warn_when_uninstalled(self, monkeypatch):
        """Scheduler-not-installed is a warn, not a fail."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': False, 'active': False, 'drift': False,
                     'state': 'off', 'interval_seconds': None})
        result = check_scheduler_state()
        assert result['status'] == 'warn'

    def test_scheduler_state_fail_on_drift(self, monkeypatch):
        """Drift between state file and OS truth is a fail."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': True, 'active': False, 'drift': True,
                     'state': 'active', 'interval_seconds': 900})
        result = check_scheduler_state()
        assert result['status'] == 'fail'

    def test_scheduler_heartbeat_fail_when_no_drains_and_started(self, tmp_path, monkeypatch):
        """Scheduler started + installed but no worker_runs row yet -> fail."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'
        assert 'no drains recorded' in result['detail']['reason']

    @pytest.mark.parametrize(('status', 'reason_snippet'), [
        ({'interval_seconds': 900, 'state': 'stopped',
          'installed': True}, "'stopped'"),
        ({'interval_seconds': None, 'state': 'stopped',
          'installed': False}, None),
    ])
    def test_scheduler_heartbeat_pass_when_inactive(
            self, tmp_path, monkeypatch, status, reason_snippet):
        """Scheduler stopped or uninstalled -> pass (no drain expected)."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(sch, 'status', lambda: status)
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass'
        if reason_snippet is not None:
            assert reason_snippet in result['detail']['reason']

    def test_scheduler_heartbeat_pass_on_recent_drain(self, tmp_path, monkeypatch):
        """A drain within the interval window passes."""
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass'

    def test_scheduler_heartbeat_threshold_floors_at_180s(self, tmp_path, monkeypatch):
        """At interval=0 (serve continuous), the threshold floors at 180s.

        Without the floor, `3 * 0 = 0` would fail every heartbeat check.
        With the floor (max(3*interval, 180s)), serve mode is robust to
        sub-minute intervals -- the rate-limited heartbeat writes 1/min so
        a 180s window allows two-miss tolerance.
        """
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'UPDATE worker_runs SET started_at = started_at - 90'
                ' WHERE id = ?', (run_id,))
            conn.commit()
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass', (
            f'90s old heartbeat at interval=0 should PASS under 180s floor;'
            f' got {result}')
        assert result['detail']['threshold_fail_seconds'] == 180

    def test_scheduler_heartbeat_fails_at_interval_zero_when_stale(
            self, tmp_path, monkeypatch):
        """At interval=0, a heartbeat older than 180s fails.

        Validates that the `interval and` truthiness guard is removed --
        interval=0 must reach the threshold comparison, not short-circuit
        to PASS.
        """
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'UPDATE worker_runs SET started_at = started_at - 200'
                ' WHERE id = ?', (run_id,))
            conn.commit()
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail', (
            f'200s old heartbeat at interval=0 should FAIL (180s floor);'
            f' got {result}')

    def test_scheduler_heartbeat_fail_on_recorded_error(self, tmp_path, monkeypatch):
        """A finished run with an error string flips the check to fail."""
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(
                conn, run_id, 1, 0, 1, error='RuntimeError: boom')
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'

    @pytest.fixture
    def runner(self, mm_runner):
        return mm_runner

    def test_doctor_text_mode_emits_colored_summary(self, runner):
        """`memman doctor --text` produces a human-readable report.

        Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor', '--text'])
        assert result.exit_code in {0, 1}, result.output
        assert 'memman doctor' in result.output
        assert ('sqlite_integrity' in result.output
                or 'env_permissions' in result.output)

    def test_doctor_json_default(self, runner):
        """`memman doctor` emits JSON by default.

        Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        assert result.exit_code in {0, 1}, result.output
        payload = json.loads(result.output)
        assert 'checks' in payload
        assert 'status' in payload

    def test_doctor_reports_llm_probe_failure(self, runner, monkeypatch):
        """`memman doctor` surfaces an LLM ConfigError and exits non-zero.

        Replaces the prior `keys test` surface; doctor's check_llm_probe
        is now the canonical key-validity gate.
        """
        from memman.exceptions import ConfigError

        r, data_dir = runner
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

        def _raise(role):
            raise ConfigError('OPENROUTER_API_KEY must be set')
        monkeypatch.setattr(
            'memman.llm.client.get_llm_client', _raise)

        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload['status'] == 'fail'
        llm_check = next(
            (c for c in payload['checks'] if c['name'] == 'llm_probe'),
            None)
        assert llm_check is not None
        assert llm_check['status'] == 'fail'
        assert 'OPENROUTER_API_KEY' in llm_check['detail']['error']

    def test_doctor_reports_probes_pass_under_mocks(self, runner):
        """With the autouse mocks both LLM and embed probes pass."""
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        payload = json.loads(result.output)
        llm_check = next(
            c for c in payload['checks'] if c['name'] == 'llm_probe')
        embed_check = next(
            c for c in payload['checks'] if c['name'] == 'embed_probe')
        assert llm_check['status'] == 'pass'
        assert embed_check['status'] == 'pass'


class TestDrainHeartbeat:
    """check_drain_heartbeat: postgres drain-heartbeat consumer."""

    pytestmark = pytest.mark.postgres

    def test_skips_on_sqlite(self):
        """SQLite mode: drain_heartbeat returns pass with skipped_reason."""
        result = check_drain_heartbeat()
        assert result['name'] == 'drain_heartbeat'
        assert result['status'] == 'pass'
        assert 'skipped_reason' in result['detail']

    def test_passes_when_no_in_progress_runs(self, env_file, pg_dsn):
        """Postgres mode with no in-progress runs: status pass."""
        env_file('MEMMAN_BACKEND', 'postgres')
        env_file('MEMMAN_PG_DSN', pg_dsn)
        from memman.store.postgres import PostgresCluster
        cluster = PostgresCluster(dsn=pg_dsn)
        try:
            cluster.drop_store(store='hb_doctor_setup', data_dir='')
        except Exception:
            pass
        backend = cluster.open(store='hb_doctor_setup', data_dir='')
        try:
            backend.close()
        finally:
            try:
                cluster.drop_store(store='hb_doctor_setup', data_dir='')
            except Exception:
                pass

        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE queue.worker_runs SET ended_at = now()'
                    ' WHERE ended_at IS NULL')

        result = check_drain_heartbeat()
        assert result['status'] == 'pass'
        assert result['detail']['in_progress'] == 0
        assert result['detail']['stale_runs'] == []

    def test_warns_no_drain_heartbeat_in_5m(self, env_file, pg_dsn):
        """Postgres mode: in-progress run with stale heartbeat -> warn."""
        env_file('MEMMAN_BACKEND', 'postgres')
        env_file('MEMMAN_PG_DSN', pg_dsn)

        stale = datetime.now(timezone.utc) - timedelta(minutes=10)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE queue.worker_runs SET ended_at = now()'
                    ' WHERE ended_at IS NULL')
                cur.execute(
                    'INSERT INTO queue.worker_runs'
                    ' (started_at, ended_at, last_heartbeat_at)'
                    ' VALUES (%s, NULL, %s) RETURNING id',
                    (stale, stale))
                stale_id = cur.fetchone()[0]
        try:
            result = check_drain_heartbeat()
            assert result['status'] == 'warn'
            stale_runs = result['detail']['stale_runs']
            assert any(s['run_id'] == stale_id for s in stale_runs)
            match = next(
                s for s in stale_runs if s['run_id'] == stale_id)
            assert match['age_seconds'] >= 5 * 60
        finally:
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE queue.worker_runs SET ended_at = now()'
                        ' WHERE id = %s',
                        (stale_id,))

    def test_no_warn_for_fresh_heartbeat(self, env_file, pg_dsn):
        """In-progress run with recent heartbeat does NOT warn."""
        env_file('MEMMAN_BACKEND', 'postgres')
        env_file('MEMMAN_PG_DSN', pg_dsn)

        fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE queue.worker_runs SET ended_at = now()'
                    ' WHERE ended_at IS NULL')
                cur.execute(
                    'INSERT INTO queue.worker_runs'
                    ' (started_at, ended_at, last_heartbeat_at)'
                    ' VALUES (%s, NULL, %s) RETURNING id',
                    (fresh, fresh))
                fresh_id = cur.fetchone()[0]
        try:
            result = check_drain_heartbeat()
            assert result['status'] == 'pass'
            assert result['detail']['stale_runs'] == []
            assert result['detail']['in_progress'] >= 1
        finally:
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE queue.worker_runs SET ended_at = now()'
                        ' WHERE id = %s',
                        (fresh_id,))


class TestDoctorBackendDispatch:
    """`memman doctor` runs against the active backend, not always SQLite."""

    pytestmark = pytest.mark.postgres

    def test_doctor_dispatches_to_postgres(
            self, tmp_path, env_file, pg_dsn, monkeypatch):
        """`db_path` reports the redacted DSN, not a filesystem path."""
        store = 'doctor_dispatch'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_PG_DSN_{store}', pg_dsn)
        monkeypatch.setenv('MEMMAN_STORE', store)

        from memman.store.postgres import PostgresCluster
        cluster = PostgresCluster(dsn=pg_dsn)
        try:
            cluster.drop_store(store=store, data_dir='')
        except Exception:
            pass
        b = cluster.open(store=store, data_dir='')
        b.close()

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ['--data-dir', str(tmp_path / 'memman'), 'doctor'])
            assert result.exit_code in {0, 1}, result.output
            data = json.loads(result.output)
            assert '#store_doctor_dispatch' in data['db_path']
        finally:
            try:
                cluster.drop_store(store=store, data_dir='')
            except Exception:
                pass
