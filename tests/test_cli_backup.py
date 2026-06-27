"""CLI tests for the `memman backup` group."""

import json
from datetime import datetime
from pathlib import Path

from memman import config
from memman.cli import cli, list_claude_permissions
from tests.conftest import invoke, make_insight


def _seed_store(data_dir: str, store: str = 'default') -> None:
    """Materialize a sqlite store with a fingerprint and one insight."""
    from memman.embed.fingerprint import Fingerprint, write_fingerprint
    from memman.store.db import write_active
    from memman.store.sqlite import open_sqlite_backend

    backend = open_sqlite_backend(store, data_dir)
    write_fingerprint(backend, Fingerprint('voyage', 'voyage-3-lite', 512))
    backend.nodes.insert(make_insight(id='k1', content='hi'))
    backend.close()
    write_active(data_dir, store)


class TestBackupRun:
    """`backup run` builds a bundle to a target dir."""

    def test_emits_bundle_json(self, mm_runner, tmp_path):
        """`backup run TARGET` reports the created bundle path."""
        _, data_dir = mm_runner
        _seed_store(data_dir)
        result = invoke(mm_runner, ['backup', 'run', str(tmp_path / 'arch')])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['action'] == 'backed_up'
        assert Path(out['bundle']).exists()

    def test_without_target_errors(self, mm_runner):
        """`backup run` with neither arg nor config exits with guidance."""
        result = invoke(mm_runner, ['backup', 'run'])
        assert result.exit_code != 0
        assert 'no target' in result.output.lower()


class TestBackupList:
    """`backup list` reads sidecar manifests at the target."""

    def test_reads_sidecars(self, mm_runner, tmp_path):
        """A bundle then a list shows that bundle with its store names."""
        _, data_dir = mm_runner
        _seed_store(data_dir)
        target = tmp_path / 'arch'
        invoke(mm_runner, ['backup', 'run', str(target)])
        result = invoke(mm_runner, ['backup', 'list', str(target)])
        out = json.loads(result.output)
        assert len(out['backups']) == 1
        assert out['backups'][0]['stores'] == ['default']


class TestBackupStatus:
    """`backup status` reports config + schedule shape."""

    def test_shape(self, mm_runner, monkeypatch):
        """Status returns the expected keys without shelling to a scheduler."""
        monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')
        result = invoke(mm_runner, ['backup', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        for key in ('cron', 'target', 'keep', 'last_fired',
                    'installed', 'next_run', 'scheduler'):
            assert key in out
        assert out['scheduler'] == 'serve'


class TestBackupSchedule:
    """`backup schedule` validates cron, writes env, installs the trigger."""

    def test_writes_env_and_creates_target(
            self, mm_runner, tmp_path, monkeypatch):
        """schedule persists the 3 keys and creates a missing target dir."""
        from memman.setup import scheduler as sched

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
        """An out-of-range cron field is rejected before install."""
        result = invoke(
            mm_runner,
            ['backup', 'schedule', '99 3 * * *', str(tmp_path / 'x')])
        assert result.exit_code != 0
        assert 'invalid cron' in result.output.lower()


class TestBackupRestore:
    """`backup restore` confirms before overwriting and runs under the lock."""

    def test_aborts_without_yes(self, mm_runner, tmp_path):
        """A 'no' at the confirm prompt aborts the restore."""
        runner, data_dir = mm_runner
        _seed_store(data_dir)
        bundle = json.loads(invoke(
            mm_runner, ['backup', 'run', str(tmp_path / 'arch_r')]).output
            )['bundle']
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'backup', 'restore', bundle],
            input='n\n')
        assert result.exit_code != 0

    def test_runs_with_yes(self, mm_runner, tmp_path):
        """`--yes` restores the store and reports the action."""
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
    """The worker is hidden and no backup subcommand is claude-callable."""

    def test_worker_hidden_from_help(self, mm_runner):
        """`backup --help` does not advertise the hidden worker."""
        result = invoke(mm_runner, ['backup', '--help'])
        assert 'worker' not in result.output

    def test_backup_excluded_from_claude_permissions(self):
        """No `backup` entry leaks into the Claude allow-list."""
        assert not any('backup' in entry for entry in list_claude_permissions())


class TestMaybeFireBackup:
    """The serve-loop hook fires at most once per matching minute."""

    def test_fires_once_per_minute(
            self, mm_runner, fake_home, monkeypatch, env_file):
        """A matching cron spawns one worker; a same-minute retry no-ops."""
        import subprocess

        from memman.cli import _maybe_fire_backup

        _, data_dir = mm_runner
        env_file('MEMMAN_BACKUP_CRON', '* * * * *')
        calls: list = []
        monkeypatch.setattr(
            subprocess, 'Popen', lambda *a, **k: calls.append(a))
        now = datetime(2026, 6, 27, 3, 0, 0)
        _maybe_fire_backup('/fake/memman', data_dir, now)
        _maybe_fire_backup('/fake/memman', data_dir, now)
        assert len(calls) == 1
