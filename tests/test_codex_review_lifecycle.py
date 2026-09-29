"""Real integration files with isolated shared-service teardown boundaries."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli
from memman.setup import claude as setup
from memman.setup import detect, scheduler
from memman.setup.codex import install_codex
from memman.setup.deploy import symlink_asset


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    """Keep discovery/files real; replace all external service operations."""
    monkeypatch.setattr(detect.shutil, 'which', lambda name: None)
    codex_env = detect.detect_codex()
    install_codex(codex_env)
    data_dir = str(tmp_path / 'lifecycle-data')
    scheduler._write_env_keys({config.LLM_API_KEY: 'test-retained-secret'},
                              data_dir=data_dir)
    removed = []

    def remove_scheduler(**kwargs):
        assert kwargs['data_dir'] == data_dir
        removed.append('scheduler')
        scheduler._write_env_keys({}, removes={config.LLM_API_KEY},
                                  data_dir=data_dir)
        return {}

    monkeypatch.setattr(setup, 'uninstall_scheduler', remove_scheduler)
    monkeypatch.setattr(setup, 'uninstall_backup',
                        lambda: removed.append('backup') or {})
    return {
        'data_dir': data_dir,
        'claude_dir': Path.home() / '.claude',
        'codex_link': Path(codex_env['skills_dir']) / 'memman',
        'removed': removed,
        }


@pytest.mark.parametrize('part', ['skills', 'hooks'])
@pytest.mark.parametrize('kind', ['empty', 'foreign', 'foreign-symlink'])
def test_codex_uninstall_ignores_unrelated_claude_directories(
        lifecycle, tmp_path, part, kind):
    """Verify foreign or empty Claude memman dirs do not hold services.

    Mutation: _claude_integration_installed treats an existing skills/memman or
    hooks/memman path as owned, so the scheduler and secret stay.
    Oracle: Empty, foreign, and foreign-symlink directories built by hand;
    removed must equal ["scheduler", "backup"] and the key must be gone.
    """
    unrelated = lifecycle['claude_dir'] / part / 'memman'
    if kind == 'foreign-symlink':
        target = tmp_path / 'another-package'
        target.mkdir()
        (target / 'SKILL.md').write_text('An unrelated user skill.\n')
        unrelated.parent.mkdir(parents=True)
        unrelated.symlink_to(target, target_is_directory=True)
    else:
        unrelated.mkdir(parents=True)
        if kind == 'foreign':
            (unrelated / 'SKILL.md').write_text('An unrelated user skill.\n')

    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', '--codex'])

    assert result.exit_code == 0, result.output
    assert not lifecycle['codex_link'].is_symlink()
    assert unrelated.exists()
    if kind != 'empty':
        assert (unrelated / 'SKILL.md').read_text() == (
            'An unrelated user skill.\n')
    assert lifecycle['removed'] == ['scheduler', 'backup']
    assert config.get_scoped(config.LLM_API_KEY, lifecycle['data_dir']) is None


@pytest.mark.parametrize('stale', [False, True])
def test_codex_uninstall_retains_services_for_claude_hook_only(
        lifecycle, tmp_path, stale):
    """Verify a lone Claude hook, even a dangling one, keeps shared services.

    Mutation: _claude_integration_installed checks only skills/memman/SKILL.md,
    or follows the link and misses a hook into a removed environment.
    Oracle: A hook from symlink_asset and a hand-made dangling prime.sh link;
    removed must stay empty and the key must remain.
    """
    hook = lifecycle['claude_dir'] / 'hooks/memman/prime.sh'
    if stale:
        hook.parent.mkdir(parents=True)
        hook.symlink_to(tmp_path / 'old-env/memman/setup/assets/claude/prime.sh')
    else:
        symlink_asset('claude/prime.sh', hook)

    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', '--codex'])

    assert result.exit_code == 0, result.output
    assert not lifecycle['codex_link'].is_symlink()
    assert hook.is_symlink()
    assert lifecycle['removed'] == []
    assert config.get_scoped(config.LLM_API_KEY, lifecycle['data_dir']) == (
        'test-retained-secret')


@pytest.mark.parametrize('flags', [[], ['--claude-code', '--codex']])
def test_failed_claude_settings_cleanup_keeps_shared_services(
        lifecycle, flags):
    """Verify a Claude settings parse failure keeps shared services.

    Mutation: claude_uninstall drops the settings error from errs, or a later
    Codex success resets failed, so the scheduler is removed.
    Oracle: Invalid JSON written to settings.json; exit code 1, the message,
    the unchanged file, an empty removed list, and the retained key.
    """
    setup.claude_write_skill(str(lifecycle['claude_dir']))
    settings = lifecycle['claude_dir'] / 'settings.json'
    settings.write_text('{invalid JSON')

    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', *flags])

    assert result.exit_code == 1, result.output
    assert 'scheduler left in place' in result.output
    assert settings.read_text() == '{invalid JSON'
    assert not lifecycle['codex_link'].is_symlink()
    assert lifecycle['removed'] == []
    assert config.get_scoped(config.LLM_API_KEY, lifecycle['data_dir']) == (
        'test-retained-secret')


@pytest.mark.parametrize('part', ['skills', 'hooks'])
@pytest.mark.parametrize('flags', [[], ['--claude-code', '--codex']])
def test_claude_asset_removal_failure_keeps_shared_services(
        lifecycle, monkeypatch, part, flags):
    """Verify a Claude asset removal failure surfaces and keeps services.

    Mutation: _remove_claude_assets swallows the OSError or leaves it out of
    errs, so uninstall exits 0 and removes the scheduler.
    Oracle: An rmtree wrapper raising PermissionError for one path; exit code
    1, the message text, and the path still on disk.
    """
    setup.claude_write_skill(str(lifecycle['claude_dir']))
    setup.claude_write_hook(str(lifecycle['claude_dir']), 'prime.sh')
    failed_path = lifecycle['claude_dir'] / part / 'memman'
    original_rmtree = setup.shutil.rmtree

    def denied(path, *args, **kwargs):
        if Path(path) == failed_path:
            if kwargs.get('ignore_errors'):
                return
            raise PermissionError('asset directory is read-only')
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(setup.shutil, 'rmtree', denied)
    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', *flags])

    assert result.exit_code == 1, result.output
    assert 'asset directory is read-only' in result.output
    assert 'scheduler left in place' in result.output
    assert failed_path.exists()
    assert not lifecycle['codex_link'].is_symlink()
    assert lifecycle['removed'] == []
    assert config.get_scoped(config.LLM_API_KEY, lifecycle['data_dir']) == (
        'test-retained-secret')


@pytest.mark.parametrize(('contents', 'retained'), [
    (json.dumps({'hooks': {'SessionStart': [{'hooks': [{
        'type': 'command', 'command': 'memman prime'}]}]}}), True),
    (json.dumps({'hooks': {'Stop': [{'hooks': [{
        'type': 'command', 'command': 'memman remember "session summary"'}]}]}}),
     True),
    ('{invalid JSON', True),
    ('[]', True),
    (json.dumps({'hooks': {'SessionStart': [{'hooks': [{
        'type': 'command', 'command': 'echo ready'}]}]}}), False),
    (json.dumps({'hooks': {'Stop': [{'hooks': [{
        'type': 'command', 'command': 'echo done'}]}]}}), False),
    (json.dumps({'hooks': {}}), False),
])
def test_codex_uninstall_respects_unselected_claude_settings(
        lifecycle, contents, retained):
    """Verify memman hooks in Claude settings keep shared services.

    Mutation: _contains_memman is skipped or limited to SessionStart, or a bad-
    JSON or non-dict settings file counts as unused.
    Oracle: Hand-written settings JSON per row with a hand-written retained
    flag; the file must stay byte-identical.
    """
    settings = lifecycle['claude_dir'] / 'settings.json'
    settings.parent.mkdir()
    settings.write_text(contents)

    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', '--codex'])

    assert result.exit_code == 0, result.output
    assert settings.read_text() == contents
    assert not lifecycle['codex_link'].is_symlink()
    assert lifecycle['removed'] == ([] if retained else ['scheduler', 'backup'])


@pytest.mark.parametrize(('flags', 'exit_code'), [
    (['--codex'], 0),
    (['--claude-code', '--codex'], 1),
])
def test_unreadable_claude_settings_are_preserved(
        lifecycle, monkeypatch, flags, exit_code):
    """Verify unreadable Claude settings are never rewritten or judged unused.

    Mutation: _claude_integration_installed lets the OSError escape or returns
    False on it, or claude_uninstall rewrites the file.
    Oracle: A read_text stub raising PermissionError for that path; contents
    read with the saved original method and exit codes per row.
    """
    settings = lifecycle['claude_dir'] / 'settings.json'
    settings.parent.mkdir()
    original_contents = '{"user-setting": true}'
    settings.write_text(original_contents)
    original_read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == settings:
            raise PermissionError('settings are unreadable')
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', denied)
    result = CliRunner().invoke(cli, [
        '--data-dir', lifecycle['data_dir'], 'uninstall', *flags])

    assert result.exit_code == exit_code, result.output
    assert original_read_text(settings) == original_contents
    assert not lifecycle['codex_link'].is_symlink()
    assert lifecycle['removed'] == []
    assert config.get_scoped(config.LLM_API_KEY, lifecycle['data_dir']) == (
        'test-retained-secret')
