"""Tests for memman.doctor health-check module.
"""

import json
import os
import struct
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

try:
    import psycopg
except ImportError:
    psycopg = None

from click.testing import CliRunner
from memman import config
from memman import doctor as doctor_mod
from memman.cli import cli
from memman.doctor import check_claude_hooks, check_drain_heartbeat
from memman.doctor import check_embedding_consistency
from memman.doctor import check_enrichment_coverage, check_env_completeness
from memman.doctor import check_env_permissions, check_integrity
from memman.doctor import check_per_store_keys, check_scheduler_heartbeat
from memman.doctor import check_scheduler_state, run_all_checks
from memman.exceptions import ConfigError
from memman.queue import finish_worker_run, open_queue_db, start_worker_run
from memman.setup import scheduler as sch
from memman.store.db import DB
from memman.store.model import WorkerRun
from memman.store.node import insert_insight, update_embedding
from memman.store.node import update_enrichment
from tests.conftest import make_insight


def _fake_embedding(dim: int = 512) -> bytes:
    """Return a deterministic embedding blob of the given dimension.
    """
    return struct.pack(f'<{dim}d', *([0.1] * dim))


def _insert_healthy_insight(db: DB, id: str, content: str = 'Healthy test insight with enough content') -> None:
    """Insert an insight with all enrichment fields populated.
    """
    ins = make_insight(id=id, content=content)
    insert_insight(db, ins)
    update_enrichment(db, id, 'summary text', 'test-llm')
    update_embedding(db, id, _fake_embedding(), 'voyage-3-lite')


class TestSqliteIntegrity:

    def test_pass_on_fresh_db(self, tmp_backend):
        """Verify a fresh database passes the integrity check.

        Mutation: reading the wrong pragma column, or reporting a status other
            than pass for a healthy file.
        Oracle: SQLite's own `ok` result on a newly created database.
        """
        result = check_integrity(tmp_backend)
        assert result['name'] == 'integrity'
        assert result['status'] == 'pass'
        assert result['detail']['result'] == 'ok'


class TestEnrichmentCoverage:

    def test_full_pass(self, tmp_db, tmp_backend):
        """Verify rows with a summary and an embedding pass.

        Mutation: `enrichment_coverage` selecting a column the baseline
            no longer declares, such as `semantic_facts`, which raises
            rather than reports, or counting it as a missing field.
        Oracle: two rows carrying exactly the two graded fields.
        """
        _insert_healthy_insight(tmp_db, 'e-1')
        _insert_healthy_insight(tmp_db, 'e-2')
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['coverage_pct'] == 100.0

    def test_partial_warn(self, tmp_db, tmp_backend):
        """Verify one unembedded row in eleven warns instead of passing.

        Mutation: the check ignoring a missing embedding, or failing where
            coverage stays at or above 90%.
        Oracle: ten enriched rows plus one bare row, hand-counted.
        """
        for i in range(10):
            _insert_healthy_insight(tmp_db, f'e-{i}', f'Content for insight number {i}')
        ins = make_insight(id='e-bare', content='Bare insight without enrichment')
        insert_insight(tmp_db, ins)
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['missing_embedding'] == 1

    def test_stranded_row_warns_and_names_the_fix(self, backend):
        """Verify a stranded row warns by count and names its fix.

        Mutation: the coverage check grading only the `missing_*`
            counts, so a row whose enrichment call failed after a
            reset passes; or the stranded predicate dropping either
            timestamp term, which counts the pending or the enriched
            row too.
        Oracle: three rows that all carry a summary and a vector, so
            the `missing_*` counts are zero, and exactly one of them
            attempted and never enriched.
        """
        for rid in ('ok-1', 'strand-1', 'pend-1'):
            backend.nodes.insert(make_insight(
                id=rid, content=f'content for {rid} long enough'))
            backend.nodes.update_enrichment(
                rid, summary='summary text', summary_model='test-llm')
            backend.nodes.update_embedding(rid, [0.1] * 512, 'test-model')
        for rid in ('ok-1', 'strand-1'):
            backend.nodes.stamp_enrich_attempted(rid)
        backend.nodes.stamp_enriched('ok-1')
        result = check_enrichment_coverage(backend)
        assert result['status'] == 'warn'
        assert result['detail']['stranded'] == 1
        assert 'memman enrich --stranded-only' in result['detail']['remediation']


class TestEmbeddingConsistency:

    def test_consistent_pass(self, tmp_db, tmp_backend):
        """Verify embeddings of one size pass.

        Mutation: treating any pair of rows as inconsistent.
        Oracle: two rows built with the same vector dimension.
        """
        _insert_healthy_insight(tmp_db, 'emb-1')
        _insert_healthy_insight(tmp_db, 'emb-2')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'pass'

    def test_mixed_fail(self, tmp_db, tmp_backend):
        """Verify embeddings of two sizes fail and list both sizes.

        Mutation: comparing only the first row, or grouping by model name
            rather than by vector size.
        Oracle: one 512-dimension row and one 256-dimension row.
        """
        _insert_healthy_insight(tmp_db, 'emb-1')
        ins2 = make_insight(id='emb-2', content='Different dim embedding')
        insert_insight(tmp_db, ins2)
        update_embedding(tmp_db, 'emb-2', _fake_embedding(dim=256),
                         'voyage-3-lite')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'fail'
        assert len(result['detail']['sizes']) > 1


class TestRunAllChecks:

    def test_structure(self, tmp_db, tmp_backend):
        """Verify the report carries status, checks, and total_active.

        Mutation: dropping a top-level key, or returning checks as a dict.
        Oracle: the documented report keys and status vocabulary.
        """
        _insert_healthy_insight(tmp_db, 'all-1')
        result = run_all_checks(tmp_backend)
        assert 'status' in result
        assert 'checks' in result
        assert 'total_active' in result
        assert isinstance(result['checks'], list)
        assert result['status'] in {'pass', 'warn', 'fail'}

    def test_empty_db(self, tmp_db, tmp_backend):
        """Verify an empty store reports status empty with no checks.

        Mutation: running every check on an empty store, which reports
            failures for missing coverage.
        Oracle: hand-set `empty`, zero rows, and an empty check list.
        """
        result = run_all_checks(tmp_backend)
        assert result['status'] == 'empty'
        assert result['total_active'] == 0
        assert result['checks'] == []

    def test_healthy_db(self, tmp_db, tmp_backend):
        """Verify a fully enriched store passes every check.

        Mutation: any check flagging a healthy store, such as a coverage
            threshold applied to the wrong field.
        Oracle: six rows built with all enrichment fields set.
        """
        ids = [f'h-{i}' for i in range(6)]
        for id in ids:
            _insert_healthy_insight(tmp_db, id, f'Healthy content for {id} insight')
        result = run_all_checks(tmp_backend)
        assert result['status'] == 'pass', [
            (c['name'], c['status'], c.get('detail'))
            for c in result['checks'] if c['status'] != 'pass']
        assert all(c['status'] == 'pass' for c in result['checks'])


class TestEnvCompleteness:
    """check_env_completeness against INSTALLABLE_KEYS.
    """

    @pytest.fixture
    def write_env(self, tmp_path, monkeypatch):
        """Return a writer that replaces the env file under a fresh data dir.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))

        def _write(contents: str) -> None:
            (data_dir / config.ENV_FILENAME).write_text(contents)
            config.reset_file_cache()

        return _write

    def test_pass_when_all_present(self, write_env):
        """Verify an env file holding every installable key passes.

        Mutation: requiring a key outside INSTALLABLE_KEYS.
        Oracle: a file generated from INSTALLABLE_KEYS itself.
        """
        lines = [f'{key}=value-for-{key}' for key in config.INSTALLABLE_KEYS]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'

    def test_warns_when_non_secret_missing(self, write_env):
        """Verify a missing non-secret key warns and names the fix.

        Mutation: ignoring missing non-secret keys, or omitting the
            `memman install` remediation.
        Oracle: a file that lacks only MEMMAN_LLM_MODEL.
        """
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key != config.LLM_MODEL
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert config.LLM_MODEL in out['detail']['missing']
        assert 'memman install' in out['detail']['fix']

    def test_ignores_api_key_on_loopback_endpoint(self, write_env):
        """Verify a missing API key does not warn on a loopback endpoint.

        Mutation: treating MEMMAN_API_KEY as required on every endpoint,
            so a keyless local-server install warns forever.
        Oracle: a file that lacks only MEMMAN_API_KEY and points
            MEMMAN_ENDPOINT at localhost.
        """
        values = {
            key: 'v' for key in config.INSTALLABLE_KEYS
            if key != config.API_KEY}
        values[config.ENDPOINT] = 'http://localhost:11434/v1'
        write_env(''.join(f'{k}={v}\n' for k, v in values.items()))
        out = check_env_completeness()
        assert out['status'] == 'pass'

    def test_warns_when_api_key_missing_off_loopback(self, write_env):
        """Verify a missing API key warns on a remote endpoint.

        Mutation: treating MEMMAN_API_KEY as optional on every endpoint,
            so a keyless remote install passes the check.
        Oracle: a file that lacks only MEMMAN_API_KEY with a remote
            endpoint; `missing` is exactly that key.
        """
        values = {
            key: 'v' for key in config.INSTALLABLE_KEYS
            if key != config.API_KEY}
        values[config.ENDPOINT] = 'https://openrouter.ai/api/v1'
        write_env(''.join(f'{k}={v}\n' for k, v in values.items()))
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert out['detail']['missing'] == [config.API_KEY]

    def test_passes_on_fresh_install(self, write_env, monkeypatch):
        """The env file a fresh install writes passes the check.

        Mutation: requiring a key a fresh install never writes.
        Oracle: the file `collect_install_knobs` builds with only the
            API key exported.
        """
        for key in (*config.INSTALLABLE_KEYS,
                    config.OPENROUTER_NATIVE_API_KEY):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(config.API_KEY, 'api-key')
        write_env('')
        knobs = config.collect_install_knobs(os.environ[config.DATA_DIR])
        write_env(''.join(f'{k}={v}\n' for k, v in knobs.items()))
        out = check_env_completeness()
        assert out['status'] == 'pass', out['detail']

    def test_ignores_optional_backup_keys(self, write_env):
        """Verify absent backup keys do not warn.

        Mutation: requiring BACKUP_CRON, BACKUP_TARGET, or BACKUP_KEEP.
        Oracle: a file that lacks only the three opt-in backup keys.
        """
        optional_backup = {
            config.BACKUP_CRON, config.BACKUP_TARGET, config.BACKUP_KEEP}
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key not in optional_backup
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'
        missing = out.get('detail', {}).get('missing', [])
        assert config.BACKUP_CRON not in missing
        assert config.BACKUP_TARGET not in missing


class TestCheckPerStoreKeys:
    """`check_per_store_keys` validates `MEMMAN_BACKEND_<store>` shape.
    """

    def test_pass_when_no_stores(self, tmp_path):
        """Verify an empty data dir passes with an empty stores list.

        Mutation: raising on a missing data dir, or listing a phantom store.
        Oracle: a data dir path that does not exist.
        """
        out = check_per_store_keys(str(tmp_path / 'memman'))
        assert out['name'] == 'per_store_keys'
        assert out['status'] == 'pass'
        assert out['detail']['stores'] == []

    def test_pass_when_per_store_key_resolves(self, tmp_path, env_file):
        """Verify a sqlite store with its own backend key passes.

        Mutation: dropping stores that carry an explicit per-store key.
        Oracle: one store dir plus a `MEMMAN_BACKEND_one=sqlite` row.
        """

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'one').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'one', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('one'), 'sqlite')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        names = [s['store'] for s in out['detail']['stores']]
        assert 'one' in names

    def test_pass_when_falling_back_to_default(self, tmp_path, env_file):
        """Verify a store with no per-store key resolves via the default.

        Mutation: reporting the source as `store` for a default fallback, or
            failing a store that has no per-store key.
        Oracle: the default backend set to sqlite and no per-store row.
        """

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
        """Verify an unknown backend name fails the check.

        Mutation: accepting any string as a backend.
        Oracle: `MEMMAN_BACKEND_bad=mongo`, which no backend implements.
        """

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
        """Verify a postgres store with no DSN fails and names the DSN.

        Mutation: skipping the DSN lookup for a postgres store.
        Oracle: a postgres store with neither per-store nor default DSN.
        """

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
        """Verify the default DSN covers a postgres store with no own DSN.

        Mutation: requiring a per-store DSN whatever the default holds.
        Oracle: default DSN set and per-store DSN absent.
        """

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

    def test_no_warn_when_dsns_differ(self, tmp_path, env_file):
        """Verify a per-store DSN that differs from the default does not warn.

        Mutation: a warning restored for divergent DSNs, which flags the
            supported way to pin a store to its own cluster.
        Oracle: a per-store DSN and a distinct default DSN.
        """

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_pinned').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_pinned', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_pinned'), 'postgres')
        env_file(config.POSTGRES_DSN_FOR('pg_pinned'), 'postgresql://pinned@host/db')
        env_file(config.DEFAULT_PG_DSN, 'postgresql://default@host/db')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_pinned')
        assert pg.get('warning') is None
        assert pg.get('error') is None

    def test_check_per_store_keys_includes_declared_but_not_created_store(
            self, tmp_path, env_file):
        """Verify a declared store with no directory still appears.

        Mutation: enumerating only stores found on disk, which hides a
            declared store that was never created.
        Oracle: a `MEMMAN_BACKEND_declared_only` row and no store directory.
        """

        data_dir = str(tmp_path / 'memman')
        env_file(config.BACKEND_FOR('declared_only'), 'sqlite')

        out = check_per_store_keys(data_dir)
        names = [s['store'] for s in out['detail']['stores']
                 if s['store'] is not None]
        assert 'declared_only' in names


def _started_scheduler_status(interval: int = 900) -> dict:
    """Scheduler status of an installed, started unit at the given interval.
    """
    return {
        'interval_seconds': interval,
        'state': 'started',
        'installed': True,
        }


class TestHardening:
    """Doctor checks for env permissions, scheduler, and worker runs.
    """

    @pytest.mark.parametrize(('mode', 'expected_status', 'assert_issue'), [
        (None, 'pass', False),
        (0o644, 'fail', True),
        (0o600, 'pass', False),
    ])
    def test_env_permissions(
            self, tmp_path, monkeypatch, mode, expected_status, assert_issue):
        """Verify the check passes for a missing or 0600 env file and fails 0644.

        Mutation: comparing the mode with the wrong mask, or skipping a
            world-readable file.
        Oracle: files chmodded to known modes under a fake home.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        if mode is not None:
            mm = tmp_path / '.memman'
            mm.mkdir(mode=0o700)
            env = mm / 'env'
            env.write_text('MEMMAN_API_KEY=fake\n')
            env.chmod(mode)
        result = check_env_permissions()
        assert result['status'] == expected_status
        if assert_issue:
            assert any('env file' in issue
                       for issue in result['detail']['issues'])

    def test_scheduler_state_warn_when_uninstalled(self, monkeypatch):
        """Scheduler-not-installed is a warn, not a fail.

        Mutation: the `not installed` branch dropped or its status
            flipped to `pass` or `fail`.
        Oracle: the check's own status against a stubbed
            not-installed `status()`.
        """
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': False, 'active': False,
                     'state': 'stopped', 'interval_seconds': None})
        result = check_scheduler_state()
        assert result['status'] == 'warn'

    def test_scheduler_state_pass_when_installed(self, monkeypatch):
        """An installed, active scheduler passes.

        Mutation: the `installed` branch reporting `warn` or `fail`
            instead of `pass`.
        Oracle: the check's own status against a stubbed installed,
            active `status()`.
        """
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': True, 'active': True,
                     'state': 'started', 'interval_seconds': 900})
        result = check_scheduler_state()
        assert result['status'] == 'pass'

    def test_scheduler_heartbeat_fail_when_no_drains_and_started(self, tmp_path, monkeypatch):
        """Verify a started scheduler with no recorded drain fails.

        Mutation: passing when the worker_runs table is empty.
        Oracle: a stubbed started status and an empty queue database.
        """
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
        """Verify a stopped or uninstalled scheduler passes.

        Mutation: failing for a missing drain when none is expected.
        Oracle: stubbed stopped and uninstalled statuses, with no runs.
        """
        monkeypatch.setattr(sch, 'status', lambda: status)
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass'
        if reason_snippet is not None:
            assert reason_snippet in result['detail']['reason']

    def test_scheduler_heartbeat_pass_on_recent_drain(self, tmp_path, monkeypatch):
        """Verify a drain inside the interval window passes.

        Mutation: comparing the run age with the wrong bound so a fresh
            drain fails.
        Oracle: a run just finished against a 900-second interval.
        """

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
        """Verify interval 0 floors the fail threshold at 180 seconds.

        Mutation: dropping the floor, so `3 * 0 = 0` fails every heartbeat.
        Oracle: a 90-second-old run must pass and report a 180 threshold.
        """

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'update worker_runs set started_at = started_at - 90'
                ' where id = ?', (run_id,))
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
        """Verify interval 0 still fails a heartbeat older than 180 seconds.

        Mutation: a truthiness guard on the interval that returns pass
            before the threshold comparison runs.
        Oracle: a 200-second-old run just past the 180-second floor.
        """

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'update worker_runs set started_at = started_at - 200'
                ' where id = ?', (run_id,))
            conn.commit()
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail', (
            f'200s old heartbeat at interval=0 should FAIL (180s floor);'
            f' got {result}')

    def test_scheduler_heartbeat_fail_on_recorded_error(self, tmp_path, monkeypatch):
        """Verify a finished run with an error string fails the check.

        Mutation: reading only run timing and ignoring the error column.
        Oracle: a run finished with `RuntimeError: boom`.
        """

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
        """Verify `doctor --text` prints a readable report.

        Mutation: ignoring --text and emitting JSON, or omitting the check
            names.
        Oracle: the literal title and a known check name in the output.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor', '--text'])
        assert result.exit_code in {0, 1}, result.output
        assert 'memman doctor' in result.output
        assert ('sqlite_integrity' in result.output
                or 'env_permissions' in result.output)

    def test_doctor_json_default(self, runner):
        """Verify `doctor` emits JSON with checks and status by default.

        Mutation: defaulting to text output, or dropping a top-level key.
        Oracle: `json.loads` of the output and the two keys.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        assert result.exit_code in {0, 1}, result.output
        payload = json.loads(result.output)
        assert 'checks' in payload
        assert 'status' in payload

    def test_doctor_reports_llm_probe_failure(self, runner, monkeypatch):
        """Verify an LLM ConfigError fails the llm_probe check and exit code.

        Mutation: swallowing the probe error so doctor reports pass, or
            dropping the error text from the check detail.
        Oracle: a stub client factory that raises ConfigError naming the key.
        """

        r, data_dir = runner
        monkeypatch.delenv('MEMMAN_API_KEY', raising=False)

        def _raise() -> None:
            raise ConfigError('MEMMAN_API_KEY must be set')
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
        assert 'MEMMAN_API_KEY' in llm_check['detail']['error']

    def test_doctor_reports_probes_pass_under_mocks(self, runner):
        """Verify both probes pass under the autouse mocks.

        Mutation: a probe that fails despite a working client, or a missing
            embed_probe entry.
        Oracle: the mocked LLM and embed clients from conftest.
        """
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
    """check_drain_heartbeat: per-store drain-heartbeat consumer.
    """

    pytestmark = pytest.mark.postgres

    def test_skips_when_no_postgres_stores(self, tmp_path):
        """Verify the check passes with a skipped reason when no store is postgres.

        Mutation: failing or opening a backend when no postgres store exists.
        Oracle: an empty data dir.
        """
        result = check_drain_heartbeat(str(tmp_path))
        assert result['name'] == 'drain_heartbeat'
        assert result['status'] == 'pass'
        assert 'skipped_reason' in result['detail']

    def test_passes_when_no_in_progress_runs(self, env_file, pg_dsn):
        """Verify a postgres store with no in-progress run passes.

        Mutation: counting a finished run as in progress.
        Oracle: every open run closed by a direct SQL update.
        """

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_setup'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        drop_postgres_store(store, pg_dsn)
        backend = open_postgres_backend(store, pg_dsn, create=True)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'update {schema}.worker_runs set ended_at = now()'
                        f' where ended_at is null')

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'pass'
            assert result['detail']['in_progress'] == 0
            assert result['detail']['stale_runs'] == []
            assert store in result['detail']['stores_checked']
        finally:
            drop_postgres_store(store, pg_dsn)

    def test_warns_no_drain_heartbeat_in_5m(self, env_file, pg_dsn):
        """Verify an in-progress run silent for 10 minutes warns.

        Mutation: a threshold above 10 minutes, or a missing store name on
            the stale-run entry.
        Oracle: a row inserted with heartbeat and start 10 minutes old.
        """

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_stale'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        drop_postgres_store(store, pg_dsn)
        backend = open_postgres_backend(store, pg_dsn, create=True)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            stale = datetime.now(timezone.utc) - timedelta(minutes=10)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'insert into {schema}.worker_runs'
                        f' (started_at, ended_at, last_heartbeat_at)'
                        f' values (%s, null, %s) returning id',
                        (stale, stale))
                    stale_id = cur.fetchone()[0]

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'warn'
            stale_runs = result['detail']['stale_runs']
            assert any(s['run_id'] == stale_id for s in stale_runs)
            match = next(
                s for s in stale_runs if s['run_id'] == stale_id)
            assert match['age_seconds'] >= 5 * 60
            assert match['store'] == store
        finally:
            drop_postgres_store(store, pg_dsn)

    def test_no_warn_for_fresh_heartbeat(self, env_file, pg_dsn):
        """Verify an in-progress run with a 30-second heartbeat passes.

        Mutation: warning on any in-progress run whatever its heartbeat age.
        Oracle: a row inserted with heartbeat and start 30 seconds old.
        """

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_fresh'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        drop_postgres_store(store, pg_dsn)
        backend = open_postgres_backend(store, pg_dsn, create=True)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'insert into {schema}.worker_runs'
                        f' (started_at, ended_at, last_heartbeat_at)'
                        f' values (%s, null, %s) returning id',
                        (fresh, fresh))

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'pass'
            assert result['detail']['stale_runs'] == []
            assert result['detail']['in_progress'] >= 1
        finally:
            drop_postgres_store(store, pg_dsn)


class TestDrainHeartbeatSeverity:
    """check_drain_heartbeat severity ladder when failures and stale combine.
    """

    def test_failures_outrank_stale(self, tmp_path, monkeypatch):
        """Verify a store failure plus a stale run reports fail.

        Mutation: letting warn from a stale run override fail from an
            unreachable store, or dropping either list from the detail.
        Oracle: a stub backend set with one store that raises and one with a
            stale run.
        """

        @contextmanager
        def _fake_open_backend(store: str, data_dir: str, *, read_only: bool = False):
            if store == 'broken':
                raise RuntimeError('connection refused')
            yield _StaleRunsBackend()

        class _StaleRunsBackend:

            def recent_runs(self, *, limit: int) -> list[WorkerRun]:

                stale = datetime.now(timezone.utc) - timedelta(minutes=10)
                return [WorkerRun(
                    id=42, started_at=stale, ended_at=None,
                    last_heartbeat_at=stale)]

        monkeypatch.setattr(
            'memman.store.factory.list_stores',
            lambda data_dir: ['broken', 'has_stale'])
        monkeypatch.setattr(
            'memman.store.factory.resolve_store_backend',
            lambda store, data_dir: 'postgres')
        monkeypatch.setattr(
            'memman.store.factory.open_backend', _fake_open_backend)

        result = doctor_mod.check_drain_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'
        assert len(result['detail']['failures']) == 1
        assert result['detail']['failures'][0]['store'] == 'broken'
        assert len(result['detail']['stale_runs']) == 1
        assert result['detail']['stale_runs'][0]['store'] == 'has_stale'


class TestDoctorBackendDispatch:
    """`memman doctor` runs against the active backend, not always SQLite.
    """

    pytestmark = pytest.mark.postgres

    def test_doctor_dispatches_to_postgres(
            self, tmp_path, env_file, pg_dsn, monkeypatch):
        """Verify `doctor` reports the redacted DSN as db_path for a postgres store.

        Mutation: opening the sqlite file path regardless of the store's
            backend.
        Oracle: the `#store_doctor_dispatch` suffix of the postgres locator.
        """
        store = 'doctor_dispatch'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        monkeypatch.setenv('MEMMAN_STORE', store)

        from memman.store.postgres import drop_postgres_store
        from memman.store.postgres import open_postgres_backend
        drop_postgres_store(store, pg_dsn)
        b = open_postgres_backend(store, pg_dsn, create=True)
        b.close()

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ['--data-dir', str(tmp_path / 'memman'), 'doctor'])
            assert result.exit_code in {0, 1}, result.output
            data = json.loads(result.output)
            assert '#store_doctor_dispatch' in data['db_path']
        finally:
            drop_postgres_store(store, pg_dsn)


class TestClaudeHooksCheck:
    """`check_claude_hooks` compares live registrations to the installer.
    """

    def _install(self, home: Path, *, matcher: str = 'Agent|Task',
                 drop: str | None = None, extra_stop: bool = False,
                 dangle: bool = False) -> None:
        """Write a settings.json and hook scripts under a fake home.

        Parameters
        ----------
        home : Path
            Fake home directory.
        matcher : str
            Matcher of the task_recall registration.
        drop : str or None
            Event to remove from settings.
        extra_stop : bool
            Add a Stop registration that memman no longer writes.
        dangle : bool
            Leave out the task_recall.sh script file.
        """
        hooks_dir = home / '.claude' / 'hooks' / 'memman'
        hooks_dir.mkdir(parents=True)
        scripts = ['prime.sh', 'user_prompt.sh', 'compact.sh',
                   'task_recall.sh', 'exit_plan.sh']
        for name in scripts:
            if dangle and name == 'task_recall.sh':
                continue
            (hooks_dir / name).write_text('#!/bin/bash\n')

        def cmd(name: str) -> str:
            return f'~/.claude/hooks/memman/{name}'

        hooks = {
            'SessionStart': [{'hooks': [
                {'type': 'command', 'command': cmd('prime.sh')}]}],
            'UserPromptSubmit': [{'hooks': [
                {'type': 'command', 'command': cmd('user_prompt.sh')}]}],
            'PreCompact': [{'hooks': [
                {'type': 'command', 'command': cmd('compact.sh')}]}],
            'PreToolUse': [
                {'hooks': [{'type': 'command',
                            'command': cmd('task_recall.sh')}],
                 'matcher': matcher},
                {'hooks': [{'type': 'command',
                            'command': cmd('exit_plan.sh')}],
                 'matcher': 'ExitPlanMode'},
                ],
            }
        if extra_stop:
            hooks['Stop'] = [{'hooks': [
                {'type': 'command', 'command': cmd('stop.sh')}]}]
            (hooks_dir / 'stop.sh').write_text('#!/bin/bash\n')
        if drop:
            hooks.pop(drop)
        settings = home / '.claude' / 'settings.json'
        settings.write_text(json.dumps({'hooks': hooks}))

    def test_clean_install_passes(self, tmp_path, monkeypatch):
        """Verify a settings file the installer would write reports pass.

        Mutation: a check that compares the wrong shape and flags every
            healthy install, which would make the report unreadable.
        Oracle: a settings.json built to match what
            add_claude_hooks_selective emits.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path)
        assert check_claude_hooks()['status'] == 'pass'

    def test_no_claude_config_passes(self, tmp_path, monkeypatch):
        """Verify a machine with no Claude Code install is not a failure.

        Mutation: treating a missing settings.json as drift, which
            would fail doctor on a machine with no Claude Code install.
        Oracle: a home directory with no .claude at all.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        assert check_claude_hooks()['status'] == 'pass'

    def test_dangling_command_fails(self, tmp_path, monkeypatch):
        """Verify a registration whose script is gone reports fail.

        Mutation: a check that compares registrations but never probes
            the command path, so a pipx upgrade that removed a hook
            script leaves Claude Code running exit 127 unreported.
        Oracle: a settings entry naming a script absent from the hooks
            directory.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, dangle=True)
        result = check_claude_hooks()
        assert result['status'] == 'fail'
        assert any('task_recall.sh' in c
                   for c in result['detail']['dangling'])

    def test_retired_stop_entry_warns(self, tmp_path, monkeypatch):
        """Verify a hook event memman no longer registers is reported.

        Mutation: comparing only the events the installer writes, so a
            retired registration left by an older install stays
            invisible.
        Oracle: a Stop entry, which no current memman version writes.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, extra_stop=True)
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert any('Stop' in e for e in result['detail']['extra'])

    def test_stale_matcher_warns(self, tmp_path, monkeypatch):
        """Verify a matcher the installer no longer writes is reported.

        Mutation: comparing event and command but dropping the matcher,
            so a registration with a narrower matcher draws no warning.
        Oracle: the matcher `Task` against the installer's own
            current value.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, matcher='Task')
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert result['detail']['missing']
        assert result['detail']['extra']

    def test_missing_event_warns(self, tmp_path, monkeypatch):
        """Verify a hook the installer writes but settings lacks is seen.

        Mutation: comparing live against expected in one direction, so
            a registration dropped by hand is never noticed.
        Oracle: a settings.json with the PreCompact entry removed.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, drop='PreCompact')
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert any('compact.sh' in m for m in result['detail']['missing'])

    def test_foreign_hook_under_memman_checkout_passes(
            self, tmp_path, monkeypatch):
        """Verify a user hook whose path names memman is not drift.

        Mutation: judging ownership by 'memman' anywhere in the command,
            which reports a user's own hook under a memman source
            checkout as extra, a drift install never repairs.
        Oracle: a hook under ~/code/memman/scripts, outside the memman
            hooks directory.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path)
        settings = tmp_path / '.claude' / 'settings.json'
        data = json.loads(settings.read_text())
        data['hooks']['Stop'] = [{'hooks': [
            {'type': 'command', 'command': '~/code/memman/scripts/lint.sh'}]}]
        settings.write_text(json.dumps(data))
        assert check_claude_hooks()['status'] == 'pass'
