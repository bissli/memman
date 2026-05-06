"""Prereq-check tests for run_install()."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.setup import claude as setup_claude
from tests.conftest import install_env_factory


def _write_keys(data_dir, openrouter=None, voyage=None):
    install_env_factory(data_dir, openrouter=openrouter, voyage=voyage)


@pytest.fixture
def all_prereqs_ok(monkeypatch):
    """Patch platform + binary checks to pass; API keys still need env setup."""
    monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(setup_claude, 'memman_binary_path',
                        lambda: '/fake/bin/memman')


class TestPrereqs:
    """Platform / binary / API-key checks at the run_install boundary."""

    def test_unsupported_platform_fails_loud(self, monkeypatch, tmp_path):
        """run_install raises when no scheduler platform is detected."""
        monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: '')
        monkeypatch.setattr(setup_claude, 'memman_binary_path',
                            lambda: '/fake/bin/memman')
        _write_keys(tmp_path, openrouter='x', voyage='y')
        with pytest.raises(click.ClickException, match='unsupported platform'):
            setup_claude.run_install(data_dir=str(tmp_path))

    def test_missing_memman_binary_fails_loud(self, monkeypatch, tmp_path):
        """run_install raises when the memman binary is not on PATH."""
        monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: 'systemd')

        def _not_found():
            raise RuntimeError('memman binary not on PATH')

        monkeypatch.setattr(setup_claude, 'memman_binary_path', _not_found)
        _write_keys(tmp_path, openrouter='x', voyage='y')
        with pytest.raises(click.ClickException, match='memman binary'):
            setup_claude.run_install(data_dir=str(tmp_path))

    @pytest.mark.parametrize(('present_key', 'missing_key'), [
        ('voyage', 'OPENROUTER_API_KEY'),
        ('openrouter', 'VOYAGE_API_KEY'),
    ])
    def test_missing_api_key_fails_loud(
            self, all_prereqs_ok, tmp_path, present_key, missing_key):
        """run_install raises when a mandatory API key is absent."""
        _write_keys(tmp_path, **{present_key: 'x'})
        with pytest.raises(click.ClickException, match=missing_key):
            setup_claude.run_install(data_dir=str(tmp_path))

    def test_uninstall_skips_prereq_checks(self, monkeypatch, tmp_path):
        """run_uninstall must not require API keys or platform detection."""
        monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: '')
        monkeypatch.setattr(setup_claude, 'detect_environments', list)
        monkeypatch.setattr(
            setup_claude, 'uninstall_scheduler',
            lambda data_dir=None: {'platform': 'unknown', 'actions': []})
        setup_claude.run_uninstall(data_dir=str(tmp_path))

    def test_invalid_target_fails_loud(self, monkeypatch, tmp_path):
        """run_install raises on an unknown --target value."""
        _write_keys(tmp_path, openrouter='x', voyage='y')
        with pytest.raises(click.ClickException, match='invalid target'):
            setup_claude.run_install(data_dir=str(tmp_path), target='bogus')

    def test_prereq_failure_writes_nothing_to_filesystem(
            self, monkeypatch, tmp_path):
        """A prereq failure must leave the filesystem untouched."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: 'systemd')
        monkeypatch.setattr(setup_claude, 'memman_binary_path',
                            lambda: '/fake/bin/memman')
        before = sorted(p for p in tmp_path.rglob('*'))

        with pytest.raises(click.ClickException):
            setup_claude.run_install(data_dir=str(tmp_path / 'data'))

        after = sorted(p for p in tmp_path.rglob('*'))
        assert before == after, (
            'filesystem changed during a prereq-failing install')


class TestCliCommands:
    """Top-level memman CLI surface for setup verbs."""

    def test_setup_command_removed(self):
        """The old `memman setup` command must no longer exist."""
        runner = CliRunner()
        result = runner.invoke(cli, ['setup', '--help'])
        assert result.exit_code != 0
        assert 'No such command' in result.output

    def test_install_command_exists(self):
        """`memman install --help` should work."""
        runner = CliRunner()
        result = runner.invoke(cli, ['install', '--help'])
        assert result.exit_code == 0
        assert '--target' in result.output

    def test_uninstall_command_exists(self):
        """`memman uninstall --help` should work."""
        runner = CliRunner()
        result = runner.invoke(cli, ['uninstall', '--help'])
        assert result.exit_code == 0
        assert '--target' in result.output
