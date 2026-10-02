"""CLI tests for the `memman backup` group.
"""

import json
import tarfile
from datetime import datetime
from pathlib import Path

import memman.backup as backup_mod
from memman import config
from memman.cli import _maybe_fire_backup, cli, list_claude_permissions
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.setup import scheduler as sched
from memman.store.db import write_active
from memman.store.sqlite import open_sqlite_backend
from tests.conftest import invoke, make_insight


def _seed_store(data_dir: str, store: str = 'default') -> None:
    """Materialize a sqlite store with a fingerprint and one insight.
    """
    backend = open_sqlite_backend(store, data_dir)
    write_fingerprint(backend, Fingerprint('voyage-3-lite', 512))
    backend.nodes.insert(make_insight(id='k1', content='hi'))
    backend.close()
    write_active(data_dir, store)


class TestBackupRun:
    """`backup run` builds a bundle to a target dir.
    """

    def test_emits_bundle_json(self, mm_runner, tmp_path):
        """Verify `backup run TARGET` reports the created bundle path.

        Mutation: run printing no bundle path, or a path to a file never
            written.
        Oracle: the JSON action value and the bundle path's existence on disk.
        """
        _, data_dir = mm_runner
        _seed_store(data_dir)
        result = invoke(mm_runner, ['backup', 'run', str(tmp_path / 'arch')])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['action'] == 'backed_up'
        assert Path(out['bundle']).exists()

    def test_without_target_errors(self, mm_runner):
        """Verify `backup run` with no argument and no config exits with guidance.

        Mutation: run succeeding or crashing with a traceback when no target is
            set.
        Oracle: nonzero exit code and 'no target' in the output.
        """
        result = invoke(mm_runner, ['backup', 'run'])
        assert result.exit_code != 0
        assert 'no target' in result.output.lower()


class TestBackupList:
    """`backup list` reads sidecar manifests at the target.
    """

    def test_reads_sidecars(self, mm_runner, tmp_path):
        """Verify list shows a fresh bundle with its store names.

        Mutation: list ignoring sidecar manifests, or dropping the store names.
        Oracle: one seeded store 'default' backed up once.
        """
        _, data_dir = mm_runner
        _seed_store(data_dir)
        target = tmp_path / 'arch'
        invoke(mm_runner, ['backup', 'run', str(target)])
        result = invoke(mm_runner, ['backup', 'list', str(target)])
        out = json.loads(result.output)
        assert len(out['backups']) == 1
        assert out['backups'][0]['stores'] == ['default']


class TestBackupStatus:
    """`backup status` reports config + schedule shape.
    """

    def test_shape(self, mm_runner, monkeypatch):
        """Verify status returns every expected key and the configured scheduler.

        Mutation: omitting a status key, or reading the scheduler from the OS
            instead of MEMMAN_SCHEDULER_KIND.
        Oracle: the literal key tuple and 'serve' set through the env.
        """
        monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')
        result = invoke(mm_runner, ['backup', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        for key in ('cron', 'target', 'keep', 'last_fired',
                    'installed', 'next_run', 'scheduler'):
            assert key in out
        assert out['scheduler'] == 'serve'


class TestBackupSchedule:
    """`backup schedule` validates cron, writes env, installs the trigger.
    """

    def test_writes_env_and_creates_target(
            self, mm_runner, tmp_path, monkeypatch):
        """Verify schedule persists cron, target, and keep, and creates the target.

        Mutation: skipping an env key, a wrong default keep, or leaving a
            missing target directory uncreated.
        Oracle: parsed env file values; keep of 7 as the default.
        """
        _, data_dir = mm_runner
        monkeypatch.setattr(
            sched, 'install_backup',
            lambda dd, cron: {'platform': 'stub'})
        target = tmp_path / 'arch_sched'
        result = invoke(
            mm_runner, ['backup', 'schedule', '0 3 * * *', str(target)])
        assert result.exit_code == 0, result.output
        assert target.is_dir()
        env = config.parse_env_file(config.env_file_path(data_dir))
        assert env[config.BACKUP_CRON] == '0 3 * * *'
        assert env[config.BACKUP_TARGET] == str(target)
        assert env[config.BACKUP_KEEP] == '7'

    def test_rejects_bad_cron(self, mm_runner, tmp_path):
        """Verify an out-of-range cron field is rejected before install.

        Mutation: installing the trigger before validating the expression.
        Oracle: nonzero exit and 'invalid cron' in the output.
        """
        result = invoke(
            mm_runner,
            ['backup', 'schedule', '99 3 * * *', str(tmp_path / 'x')])
        assert result.exit_code != 0
        assert 'invalid cron' in result.output.lower()


class TestBackupRestore:
    """`backup restore` confirms before overwriting and runs under the lock.
    """

    def test_aborts_without_yes(self, mm_runner, tmp_path):
        """Verify answering 'n' at the confirm prompt aborts the restore.

        Mutation: restoring without honoring the confirmation answer.
        Oracle: nonzero exit code after input 'n'.
        """
        runner, data_dir = mm_runner
        _seed_store(data_dir)
        bundle = json.loads(invoke(
            mm_runner, ['backup', 'run', str(tmp_path / 'arch_r')]).output
            )['bundle']
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'backup', 'restore', bundle],
            input='n\n')
        assert result.exit_code != 0

    def test_incompatible_bundle_exits_clean(self, mm_runner, tmp_path):
        """Verify a bundle with an unsupported format_version exits with a clean error.

        Mutation: letting the RuntimeError from an unsupported format escape as
            a traceback.
        Oracle: nonzero exit, exception not a RuntimeError, and
            'format_version' in the output.
        """
        staging = tmp_path / 'st'
        staging.mkdir()
        (staging / 'manifest.json').write_text(json.dumps({
            'format_version': 999, 'stores': [], 'active_store': 'default'}))
        (staging / 'env.nonsecret').write_text('\n')
        bundle = tmp_path / 'bad.tar.gz'
        with tarfile.open(bundle, 'w:gz') as tar:
            tar.add(staging, arcname='.')
        result = invoke(mm_runner, ['backup', 'restore', str(bundle), '--yes'])
        assert result.exit_code != 0
        assert not isinstance(result.exception, RuntimeError)
        assert 'format_version' in result.output

    def test_runs_with_yes(self, mm_runner, tmp_path):
        """Verify `--yes` restores the store and reports the action.

        Mutation: restore skipping the store, or reporting a different action.
        Oracle: JSON action 'restored' with 'default' among the restored
            stores.
        """
        _, data_dir = mm_runner
        _seed_store(data_dir)
        bundle = json.loads(invoke(
            mm_runner, ['backup', 'run', str(tmp_path / 'arch_r2')]).output
            )['bundle']
        result = invoke(mm_runner, ['backup', 'restore', bundle, '--yes'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output[result.output.index('{'):])
        assert out['action'] == 'restored'
        assert 'default' in out['restored']


class TestBackupPermissions:
    """The worker is hidden and no backup subcommand is claude-callable.
    """

    def test_worker_hidden_from_help(self, mm_runner):
        """Verify `backup --help` does not advertise the hidden worker.

        Mutation: dropping hidden=True from the worker command.
        Oracle: the word 'worker' absent from the help text.
        """
        result = invoke(mm_runner, ['backup', '--help'])
        assert 'worker' not in result.output

    def test_backup_excluded_from_claude_permissions(self):
        """Verify no backup entry leaks into the Claude allow-list.

        Mutation: registering a backup subcommand as Claude-callable.
        Oracle: no 'backup' substring in list_claude_permissions().
        """
        assert not any('backup' in entry for entry in list_claude_permissions())


class TestMaybeFireBackup:
    """The serve-loop hook fires at most once per matching minute.
    """

    def test_fires_once_per_minute(
            self, mm_runner, fake_home, monkeypatch, env_file):
        """Verify a matching cron runs the backup once and a same-minute retry no-ops.

        Mutation: dropping the last-fired guard, so the serve loop backs up on
            every tick of the same minute.
        Oracle: a call counter after two invocations at one timestamp.
        """
        _, data_dir = mm_runner
        monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')
        env_file('MEMMAN_BACKUP_CRON', '* * * * *')
        calls: list = []
        monkeypatch.setattr(
            backup_mod, 'run_backup', calls.append)
        now = datetime(2026, 6, 27, 3, 0, 0)
        _maybe_fire_backup(data_dir, now)
        _maybe_fire_backup(data_dir, now)
        assert len(calls) == 1

    def test_settle_runs_before_backup(
            self, mm_runner, fake_home, monkeypatch, env_file):
        """Verify the settle hook runs before the backup when firing.

        Mutation: running the backup before the queue drain, or skipping
            settle.
        Oracle: recorded call order ['settle', 'backup'].
        """
        _, data_dir = mm_runner
        monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')
        env_file('MEMMAN_BACKUP_CRON', '* * * * *')
        order: list = []
        monkeypatch.setattr(
            backup_mod, 'run_backup', lambda dd: order.append('backup'))
        _maybe_fire_backup(
            data_dir, datetime(2026, 6, 27, 3, 0, 0),
            settle=lambda: order.append('settle'))
        assert order == ['settle', 'backup']

    def test_defers_to_native_timer(self, mm_runner, monkeypatch, env_file):
        """Verify the hook no-ops on a systemd host where the native timer owns backups.

        Mutation: firing from the serve loop while a native timer also fires.
        Oracle: an empty call list under a stubbed systemd scheduler.
        """
        _, data_dir = mm_runner
        monkeypatch.setattr(sched, 'detect_scheduler', lambda: 'systemd')
        env_file('MEMMAN_BACKUP_CRON', '* * * * *')
        calls: list = []
        monkeypatch.setattr(
            backup_mod, 'run_backup', calls.append)
        _maybe_fire_backup(data_dir, datetime(2026, 6, 27, 3, 0, 0))
        assert calls == []
