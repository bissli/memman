"""Tests for B12 doctor checks: schema, env perms, scheduler, worker runs.

Covers the new health checks added in B12. The pre-existing
test_doctor.py still exercises the original checks; this file is
dedicated to the hardened surface.
"""

import json
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.doctor import check_env_permissions, check_last_worker_run
from memman.doctor import check_queue_schema, check_scheduler_state
from memman.doctor import check_schema_columns
from memman.store.db import open_db


def test_schema_columns_passes_on_current_schema(tmp_path):
    """Fresh DB has all expected provenance columns."""
    db = open_db(str(tmp_path))
    try:
        result = check_schema_columns(db)
        assert result['status'] == 'pass'
        assert result['detail']['missing'] == []
    finally:
        db.close()


def test_schema_columns_fails_when_column_missing(tmp_path):
    """A DB predating provenance columns should fail the schema check.
    """
    db = open_db(str(tmp_path))
    try:
        db._conn.executescript(
            'CREATE TABLE insights_legacy (id TEXT PRIMARY KEY);'
            'DROP TABLE insights;'
            'ALTER TABLE insights_legacy RENAME TO insights;')
        result = check_schema_columns(db)
        assert result['status'] == 'fail'
        assert 'prompt_version' in result['detail']['missing']
        assert 'model_id' in result['detail']['missing']
        assert 'embedding_model' in result['detail']['missing']
    finally:
        db.close()


def test_queue_schema_passes_with_worker_runs(tmp_path):
    """A fresh queue.db has the worker_runs table."""
    result = check_queue_schema(str(tmp_path))
    assert result['status'] == 'pass'
    assert result['detail']['missing'] == []


def test_env_permissions_pass_when_no_home_memman(tmp_path, monkeypatch):
    """Missing ~/.memman returns pass (nothing to secure)."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    result = check_env_permissions()
    assert result['status'] == 'pass'


def test_env_permissions_fail_on_world_readable_env(tmp_path, monkeypatch):
    """0644 env file trips the check."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    mm = tmp_path / '.memman'
    mm.mkdir(mode=0o700)
    env = mm / 'env'
    env.write_text('OPENROUTER_API_KEY=fake\n')
    env.chmod(0o644)
    result = check_env_permissions()
    assert result['status'] == 'fail'
    assert any('env file' in issue
               for issue in result['detail']['issues'])


def test_env_permissions_pass_on_0600_env(tmp_path, monkeypatch):
    """0600 env file + 0700 dir passes cleanly."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    mm = tmp_path / '.memman'
    mm.mkdir(mode=0o700)
    env = mm / 'env'
    env.write_text('OPENROUTER_API_KEY=fake\n')
    env.chmod(0o600)
    result = check_env_permissions()
    assert result['status'] == 'pass'
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_scheduler_state_warn_when_uninstalled(monkeypatch):
    """Scheduler-not-installed is a warn, not a fail."""
    from memman.setup import scheduler as sch
    monkeypatch.setattr(
        sch, 'status',
        lambda: {'installed': False, 'active': False, 'drift': False,
                 'state': 'off', 'interval_seconds': None})
    result = check_scheduler_state()
    assert result['status'] == 'warn'


def test_scheduler_state_fail_on_drift(monkeypatch):
    """Drift between state file and OS truth is a fail."""
    from memman.setup import scheduler as sch
    monkeypatch.setattr(
        sch, 'status',
        lambda: {'installed': True, 'active': False, 'drift': True,
                 'state': 'active', 'interval_seconds': 900})
    result = check_scheduler_state()
    assert result['status'] == 'fail'


def _started_scheduler_status(interval=900):
    """Test helper: pretend the scheduler is installed + started."""
    return {
        'interval_seconds': interval,
        'state': 'started',
        'installed': True,
        }


def test_last_worker_run_fail_when_no_drains_and_started(tmp_path, monkeypatch):
    """Scheduler started + installed but no worker_runs row yet -> fail."""
    from memman.setup import scheduler as sch
    monkeypatch.setattr(sch, 'status', _started_scheduler_status)
    result = check_last_worker_run(str(tmp_path))
    assert result['status'] == 'fail'
    assert 'no drains recorded' in result['detail']['reason']


def test_last_worker_run_pass_when_stopped(tmp_path, monkeypatch):
    """Scheduler stopped -> pass (no drain expected; recall-only mode)."""
    from memman.setup import scheduler as sch
    monkeypatch.setattr(
        sch, 'status', lambda: {
            'interval_seconds': 900, 'state': 'stopped',
            'installed': True})
    result = check_last_worker_run(str(tmp_path))
    assert result['status'] == 'pass'
    assert "'stopped'" in result['detail']['reason']


def test_last_worker_run_pass_when_uninstalled(tmp_path, monkeypatch):
    """Scheduler uninstalled -> pass (not relevant)."""
    from memman.setup import scheduler as sch
    monkeypatch.setattr(
        sch, 'status', lambda: {
            'interval_seconds': None, 'state': 'stopped',
            'installed': False})
    result = check_last_worker_run(str(tmp_path))
    assert result['status'] == 'pass'


def test_last_worker_run_pass_on_recent_drain(tmp_path, monkeypatch):
    """A drain within the interval window passes."""
    from memman.queue import finish_worker_run, open_queue_db, start_worker_run
    from memman.setup import scheduler as sch

    monkeypatch.setattr(sch, 'status', _started_scheduler_status)
    conn = open_queue_db(str(tmp_path))
    try:
        run_id = start_worker_run(conn, worker_pid=1)
        finish_worker_run(conn, run_id, 0, 0, 0)
    finally:
        conn.close()
    result = check_last_worker_run(str(tmp_path))
    assert result['status'] == 'pass'


def test_last_worker_run_fail_on_recorded_error(tmp_path, monkeypatch):
    """A finished run with an error string flips the check to fail."""
    from memman.queue import finish_worker_run, open_queue_db, start_worker_run
    from memman.setup import scheduler as sch

    monkeypatch.setattr(sch, 'status', _started_scheduler_status)
    conn = open_queue_db(str(tmp_path))
    try:
        run_id = start_worker_run(conn, worker_pid=1)
        finish_worker_run(
            conn, run_id, 1, 0, 1, error='RuntimeError: boom')
    finally:
        conn.close()
    result = check_last_worker_run(str(tmp_path))
    assert result['status'] == 'fail'


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to a fresh temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'mm')
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return r, data_dir


def test_doctor_text_mode_emits_colored_summary(runner):
    """`memman doctor --text` produces a human-readable report.

    Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
    """
    r, data_dir = runner
    result = r.invoke(cli, ['--data-dir', data_dir, 'doctor', '--text'])
    assert result.exit_code in {0, 1}, result.output
    assert 'memman doctor' in result.output
    assert 'sqlite_integrity' in result.output or 'env_permissions' in result.output


def test_doctor_json_default(runner):
    """`memman doctor` emits JSON by default.

    Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
    """
    r, data_dir = runner
    result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
    assert result.exit_code in {0, 1}, result.output
    payload = json.loads(result.output)
    assert 'checks' in payload
    assert 'status' in payload


def test_doctor_reports_llm_probe_failure(runner, monkeypatch):
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


def test_doctor_reports_probes_pass_under_mocks(runner):
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
