"""Codex discovery, skill lifecycle, and CLI integration selection."""

import subprocess
from importlib.resources import files
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli, list_claude_permissions
from memman.setup import claude as setup
from memman.setup import detect
from memman.setup.codex import install_codex, uninstall_codex
from memman.setup.deploy import symlink_asset


@pytest.fixture
def codex_env(monkeypatch):
    monkeypatch.setattr(detect.shutil, 'which', lambda name: None)
    return detect.detect_codex()


def test_detect_codex_sources(monkeypatch, tmp_path, codex_env):
    """Verify detect_codex reads CODEX_HOME, ~/.codex, and the codex binary.

    Mutation: detect_codex ignores CODEX_HOME, derives skills_dir from it, or
    lets a TimeoutExpired from the version probe escape or clear detected.
    Oracle: An empty tmp CODEX_HOME dir, a created ~/.codex, and stubbed which
    and check_output returning "codex-cli 1.2.3".
    """
    assert not codex_env['detected']
    assert codex_env['skills_dir'] == str(Path.home() / '.agents/skills')

    custom_home = tmp_path / 'custom-codex'
    monkeypatch.setenv('CODEX_HOME', str(custom_home))
    assert not detect.detect_codex()['detected']
    custom_home.mkdir()
    assert detect.detect_codex()['config_dir'] == str(custom_home)
    assert detect.detect_codex()['detected']
    assert detect.detect_codex()['skills_dir'] == codex_env['skills_dir']
    custom_home.rmdir()

    default_home = Path.home() / '.codex'
    monkeypatch.delenv('CODEX_HOME')
    default_home.mkdir()
    assert detect.detect_codex()['detected']
    default_home.rmdir()

    monkeypatch.setattr(detect.shutil, 'which', lambda name: '/bin/codex')
    monkeypatch.setattr(detect.subprocess, 'check_output',
                        lambda *args, **kwargs: b'codex-cli 1.2.3\n')
    assert detect.detect_codex()['version'] == 'codex-cli 1.2.3'
    assert detect.detect_codex()['bin_path'] == '/bin/codex'

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('codex', 5)

    monkeypatch.setattr(detect.subprocess, 'check_output', timeout)
    assert detect.detect_codex()['detected']
    assert detect.detect_codex()['version'] == ''


def test_skill_lifecycle_preserves_other_files(codex_env):
    """Verify install and uninstall are repeatable and touch only the link.

    Mutation: uninstall_codex removes the whole skills or rules dir,
    install_codex rewrites config.toml or another rules file, or a repeated
    call raises because unlink lost missing_ok.
    Oracle: A config.toml, a sibling rules file, and a sibling skill written
    before the act, and the packaged SKILL.md read from the asset dir.
    """
    config = Path(codex_env['config_dir'])
    config.mkdir()
    settings = config / 'config.toml'
    settings.write_text('# user configuration\n')
    user_rules = config / 'rules' / 'default.rules'
    user_rules.parent.mkdir()
    user_rules.write_text('prefix_rule(pattern=["git"], decision="allow")\n')
    skills = Path(codex_env['skills_dir'])
    other = skills / 'other' / 'SKILL.md'
    other.parent.mkdir(parents=True)
    other.write_text('another skill')
    asset = Path(str(files('memman.setup.assets').joinpath('codex')))

    install_codex(codex_env)
    install_codex(codex_env)
    link = skills / 'memman'
    assert link.is_symlink()
    assert (link / 'SKILL.md').read_text() == (asset / 'SKILL.md').read_text()
    assert settings.read_text() == '# user configuration\n'
    assert user_rules.read_text() \
        == 'prefix_rule(pattern=["git"], decision="allow")\n'

    uninstall_codex(codex_env)
    uninstall_codex(codex_env)
    assert not link.is_symlink()
    assert (asset / 'SKILL.md').is_file()
    assert other.read_text() == 'another skill'
    assert settings.read_text() == '# user configuration\n'
    assert user_rules.read_text() \
        == 'prefix_rule(pattern=["git"], decision="allow")\n'


def test_install_allows_agent_verbs_and_uninstall_revokes_them(codex_env):
    """Verify install allows each agent verb in Codex; uninstall revokes it.

    Mutation: install_codex writes no rules file, a rule drops a verb or
    the allow decision, or uninstall_codex leaves the file behind.
    Oracle: the Bash(memman ...:*) entries Claude Code pre-approves,
    rewritten by hand as Codex prefix rules.
    """
    rules = Path(codex_env['config_dir']) / 'rules' / 'memman.rules'
    expected = [
        'prefix_rule(pattern=["memman", '
        + ', '.join(f'"{token}"' for token in
                    entry.removeprefix('Bash(memman ').removesuffix(':*)')
                    .split())
        + '], decision="allow")'
        for entry in list_claude_permissions()]

    install_codex(codex_env, no_wizard=True)
    assert rules.read_text().splitlines() == expected

    uninstall_codex(codex_env)
    assert not rules.exists()


@pytest.mark.parametrize('confirmed', [True, False])
def test_interactive_install_asks_before_writing_rules(
        monkeypatch, codex_env, confirmed):
    """Verify an interactive install writes the Codex rules only on consent.

    Mutation: the confirm prompt is skipped, or its answer is ignored.
    Oracle: the rules file exists exactly when the stubbed prompt says yes.
    """
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    monkeypatch.setattr(click, 'confirm', lambda *a, **kw: confirmed)
    install_codex(codex_env, no_wizard=False)
    rules = Path(codex_env['config_dir']) / 'rules' / 'memman.rules'
    assert rules.exists() == confirmed


@pytest.mark.parametrize('operation', [install_codex, uninstall_codex])
@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling', 'cyclic'])
def test_preserves_foreign_memman_skill(codex_env, tmp_path, operation, kind):
    """Verify install and uninstall refuse a memman skill they do not own.

    Mutation: _check_owned returns early for any existing path, or drops the
    is_symlink() half of its test so a dangling or cyclic link counts as free.
    Oracle: A foreign directory, symlink, dangling link, and cyclic link built
    by hand; contents and readlink target are compared after the act.
    """
    link = Path(codex_env['skills_dir']) / 'memman'
    link.parent.mkdir(parents=True)
    target = tmp_path / 'user-skill'
    if kind == 'directory':
        link.mkdir()
        (link / 'SKILL.md').write_text('custom skill')
    else:
        if kind == 'cyclic':
            target = link
        if kind == 'symlink':
            target.mkdir()
        link.symlink_to(target)
    with pytest.raises(click.ClickException, match='already exists'):
        operation(codex_env)
    if kind == 'directory':
        assert (link / 'SKILL.md').read_text() == 'custom skill'
    else:
        assert link.is_symlink()
        assert link.readlink() == target


def test_detect_refresh_and_remove_stale_packaged_link(codex_env, tmp_path):
    """Verify a dangling link to an old packaged asset is detected and reused.

    Mutation: is_asset_link follows the link, or detect_codex checks
    link.exists(), so a link into a removed environment counts as foreign.
    Oracle: A hand-made link to a missing old-env assets/codex path; SKILL.md
    resolves after install and detected is False after uninstall.
    """
    link = Path(codex_env['skills_dir']) / 'memman'
    link.parent.mkdir(parents=True)
    stale = tmp_path / 'old-env/lib/memman/setup/assets/codex'
    link.symlink_to(stale)
    assert detect.detect_codex()['detected']
    install_codex(codex_env)
    assert (link / 'SKILL.md').is_file()
    link.unlink()
    link.symlink_to(stale)
    uninstall_codex(codex_env)
    assert not link.is_symlink()
    assert not detect.detect_codex()['detected']


@pytest.mark.parametrize('command', ['install', 'uninstall'])
@pytest.mark.parametrize(('flags', 'expected'), [
    ([], (False, False)),
    (['--codex'], (False, True)),
    (['--claude-code', '--codex'], (True, True)),
])
def test_cli_passes_agent_flags(monkeypatch, command, flags, expected):
    """Verify install and uninstall forward --claude-code and --codex.

    Mutation: The install or uninstall command drops the codex option, swaps
    the two flags, or forces one of them to True.
    Oracle: A spy on run_install and run_uninstall recording the (claude_code,
    codex) kwargs, compared to hand-written tuples.
    """
    calls = []
    monkeypatch.setattr(setup, f'run_{command}',
                        lambda *args, **kwargs: calls.append(
                            (kwargs['claude_code'], kwargs['codex'])))
    result = CliRunner().invoke(cli, [command, *flags])
    assert result.exit_code == 0, result.output
    assert calls == [expected]


@pytest.mark.parametrize('command', ['install', 'uninstall'])
@pytest.mark.parametrize(('detected', 'claude_flag', 'codex_flag', 'expected'), [
    (True, False, False, ['claude', 'codex', 'scheduler']),
    (True, True, False, ['claude', 'scheduler']),
    (True, False, True, ['codex', 'scheduler']),
    (False, False, True, ['codex', 'scheduler']),
    (False, True, True, ['claude', 'codex', 'scheduler']),
    (False, False, False, ['scheduler']),
])
def test_flows_select_integrations(monkeypatch, tmp_path, codex_env,
                                   command, detected, claude_flag,
                                   codex_flag, expected):
    """Verify the flows choose integrations from flags first, then detection.

    Mutation: _run_install_flow or _run_uninstall_flow lets detection override
    an explicit flag, or skips the scheduler, or _init_default_store is skipped
    for a Codex-only install.
    Oracle: Spies appending the name of each installer that ran, compared to a
    hand-written list per row; a spy list of initialized data dirs.
    """
    calls = []
    claude_env = dict(codex_env, display='Claude Code', detected=detected)
    codex_env['detected'] = detected
    monkeypatch.setattr(setup, '_install_claude_code',
                        lambda *a, **kw: calls.append('claude'))
    monkeypatch.setattr(setup, '_uninstall_env',
                        lambda *a: calls.append('claude') or False)
    monkeypatch.setattr(setup, f'{command}_codex',
                        lambda *a, **kw: calls.append('codex'))
    monkeypatch.setattr(setup, f'{command}_scheduler',
                        lambda *a, **kw: calls.append('scheduler') or {})
    initialized = []
    monkeypatch.setattr(setup, '_init_default_store', initialized.append)
    monkeypatch.setattr(setup.openrouter_models, 'refresh_model_state',
                        lambda *a, **kw: None)
    kwargs = {'knobs': {}} if command == 'install' else {}
    getattr(setup, f'_run_{command}_flow')(
        claude_env, claude_code=claude_flag, codex=codex_flag,
        codex_env=codex_env, data_dir=str(tmp_path), **kwargs)
    assert calls == expected
    if command == 'install' and expected == ['codex', 'scheduler']:
        assert initialized == [str(tmp_path)]


def test_codex_cleanup_failure_keeps_scheduler(monkeypatch, tmp_path, codex_env):
    """Verify a failed Codex cleanup raises and leaves the scheduler in place.

    Mutation: _run_uninstall_flow swallows the uninstall_codex error without
    setting failed, so uninstall_scheduler runs.
    Oracle: An uninstall_codex stub raising PermissionError, and a spy on
    uninstall_scheduler that must record no call.
    """
    def fail(env):
        raise PermissionError('skill is read-only')

    monkeypatch.setattr(setup, 'uninstall_codex', fail)
    calls = []
    monkeypatch.setattr(setup, 'uninstall_scheduler',
                        lambda **kw: calls.append('removed'))
    with pytest.raises(click.ClickException, match='scheduler left in place'):
        setup._run_uninstall_flow(
            codex_env, claude_code=False, codex=True,
            codex_env=codex_env, data_dir=str(tmp_path))
    assert calls == []


def test_install_and_uninstall_commands_deploy_skill(monkeypatch, codex_env):
    """Verify install --codex writes skill and rules; uninstall removes skill.

    Mutation: install_codex writes no link or rules, or run_uninstall finds
    Codex only through the binary or config dir, so a link-only install stays
    behind.
    Oracle: Real detection and filesystem writes under the isolated home; only
    Claude detection, the wizard, and the scheduler are stubbed.
    """
    monkeypatch.setattr(setup, 'detect_claude_code',
                        lambda: dict(codex_env, detected=False))
    monkeypatch.setattr(setup.wizard, 'run_wizard', lambda *a, **kw: {})
    monkeypatch.setattr(setup, 'check_prereqs', lambda *a: {})
    monkeypatch.setattr(setup, 'install_scheduler', lambda *a: {})
    monkeypatch.setattr(setup, 'uninstall_scheduler', lambda **kw: {})
    monkeypatch.setattr(setup, 'uninstall_backup', dict)
    monkeypatch.setattr(setup, '_init_default_store', lambda *a: None)
    monkeypatch.setattr(setup.openrouter_models, 'refresh_model_state',
                        lambda *a, **kw: None)
    runner = CliRunner()
    result = runner.invoke(cli, ['install', '--codex', '--no-wizard'])
    assert result.exit_code == 0, result.output
    skill = Path(codex_env['skills_dir']) / 'memman'
    assert (skill / 'SKILL.md').is_file()
    assert (Path(codex_env['config_dir']) / 'rules/memman.rules').is_file()

    # With the CLI and config directory absent, the installed skill is
    # enough for automatic discovery on a subsequent uninstall.
    result = runner.invoke(cli, ['uninstall'])
    assert result.exit_code == 0, result.output
    assert not skill.is_symlink()


def test_foreign_skill_does_not_detect_codex(codex_env):
    """Verify a foreign memman skill directory does not mark Codex detected.

    Mutation: detect_codex sets detected for any skills_dir/memman path,
    whoever owns it.
    Oracle: A real directory holding a hand-written foreign SKILL.md; the
    detected flag must be False.
    """
    skill = Path(codex_env['skills_dir']) / 'memman'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('A skill installed by another agent.')
    assert not detect.detect_codex()['detected']


@pytest.mark.parametrize('command', ['install', 'uninstall'])
def test_skill_conflict_precedes_other_setup_mutations(
        monkeypatch, codex_env, command):
    """Verify a foreign skill aborts install and uninstall before changes.

    Mutation: run_install or run_uninstall drops its check_codex_skill call, so
    the wizard, the Claude change, or the backup removal runs first.
    Oracle: Spies appending to a mutations list that must stay empty, plus exit
    code 1 and "already exists" in the output.
    """
    skill = Path(codex_env['skills_dir']) / 'memman'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('User-owned skill')
    monkeypatch.setattr(setup, 'detect_claude_code',
                        lambda: dict(codex_env, detected=True))
    mutations = []
    monkeypatch.setattr(setup.wizard, 'run_wizard',
                        lambda *a, **kw: mutations.append('wizard') or {})
    monkeypatch.setattr(setup, 'check_prereqs', lambda *a: {})
    monkeypatch.setattr(setup, '_install_claude_code',
                        lambda *a, **kw: mutations.append('claude'))
    monkeypatch.setattr(setup, '_uninstall_env',
                        lambda *a: mutations.append('claude') or False)
    monkeypatch.setattr(setup, 'uninstall_backup',
                        lambda: mutations.append('backup') or {})
    result = CliRunner().invoke(cli, [command, '--codex', '--claude-code'])
    assert result.exit_code == 1
    assert 'already exists' in result.output
    assert mutations == []


def test_failed_codex_cleanup_preserves_backup(monkeypatch, codex_env):
    """Verify a failed Codex cleanup exits 1 and skips backup removal.

    Mutation: _run_uninstall_flow drops its failed check and returns True, so
    run_uninstall removes the backup.
    Oracle: An uninstall_codex stub raising PermissionError, and a spy on
    uninstall_backup that must record no call.
    """
    monkeypatch.setattr(setup, 'detect_claude_code',
                        lambda: dict(codex_env, detected=False))
    mutations = []
    monkeypatch.setattr(setup, 'uninstall_backup',
                        lambda: mutations.append('backup') or {})

    def fail(env):
        raise PermissionError('skill directory is read-only')

    monkeypatch.setattr(setup, 'uninstall_codex', fail)
    result = CliRunner().invoke(cli, ['uninstall', '--codex'])
    assert result.exit_code == 1
    assert 'scheduler left in place' in result.output
    assert mutations == []


@pytest.mark.parametrize('failure', ['symlink_to', 'replace'])
def test_failed_skill_refresh_preserves_existing_link(
        monkeypatch, tmp_path, failure):
    """Verify a failed link refresh keeps the old link and leaves no litter.

    Mutation: symlink_asset unlinks dest before creating the new link, or
    leaves its .memman- staging directory behind.
    Oracle: The original readlink target read before the act; PermissionError
    injected at Path.symlink_to and Path.replace; the dir listing must equal
    [link].
    """
    directory = tmp_path / 'skills'
    link = directory / 'memman'
    symlink_asset('codex', link)
    original = link.readlink()

    def fail(*a, **kw):
        raise PermissionError('symlink creation denied')

    monkeypatch.setattr(Path, failure, fail)
    with pytest.raises(PermissionError):
        symlink_asset('codex', link)
    assert link.is_symlink()
    assert link.readlink() == original
    assert list(directory.iterdir()) == [link]


@pytest.mark.no_default_env
@pytest.mark.parametrize('flags', [['--codex'], ['--codex', '--claude-code']])
def test_fresh_install_initializes_from_persisted_config(
        monkeypatch, tmp_path, codex_env, flags):
    """Verify a fresh install seeds the default store from persisted config.

    Mutation: _run_install_flow initializes the default store before
    install_scheduler persists the provider defaults, so the fresh env file
    fails.
    Oracle: The real config writer, skill deployment, and store init run;
    assertions read the persisted embed provider, memman.db, and SKILL.md.
    """
    monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'test-openrouter-key')
    monkeypatch.setenv(config.VOYAGE_API_KEY, 'test-voyage-key')
    monkeypatch.setattr(setup.openrouter_models, 'refresh_model_state',
                        lambda *a, **kw: None)
    data_dir = tmp_path / 'fresh-install'
    result = CliRunner().invoke(
        cli, ['--data-dir', str(data_dir), 'install', '--no-wizard', *flags])
    assert result.exit_code == 0, result.output
    assert config.get_scoped(config.EMBED_PROVIDER, str(data_dir)) == 'voyage'
    assert (data_dir / 'data/default/memman.db').is_file()
    assert (Path(codex_env['skills_dir']) / 'memman/SKILL.md').is_file()


@pytest.mark.parametrize('selected', ['codex', 'claude-code'])
def test_selective_uninstall_keeps_shared_services_for_other_agent(
        monkeypatch, codex_env, selected):
    """Verify removing one agent keeps the worker and key the other needs.

    Mutation: _run_uninstall_flow drops its other_claude or other_codex check,
    so the scheduler, backup, or LLM key goes, or the final uninstall skips
    teardown.
    Oracle: A remove_scheduler spy that deletes the key from the env file; the
    removed list and stored key are read after each uninstall.
    """
    from memman.setup import scheduler

    claude_dir = Path.home() / '.claude'
    setup.claude_write_skill(str(claude_dir))
    install_codex(codex_env)
    # Both detections use their installed files, not host CLI binaries.
    data_dir = str(Path.home() / '.memman')
    scheduler._write_env_keys({config.LLM_API_KEY: 'retained-key'},
                              data_dir=data_dir)
    removed = []

    def remove_scheduler(**kwargs):
        removed.append('scheduler')
        scheduler._write_env_keys({}, removes={config.LLM_API_KEY},
                                  data_dir=data_dir)
        return {}

    monkeypatch.setattr(setup, 'uninstall_scheduler', remove_scheduler)
    monkeypatch.setattr(setup, 'uninstall_backup',
                        lambda: removed.append('backup') or {})
    runner = CliRunner()
    result = runner.invoke(cli, ['--data-dir', data_dir,
                                 'uninstall', f'--{selected}'])
    assert result.exit_code == 0, result.output
    assert removed == []
    assert config.get_scoped(config.LLM_API_KEY, data_dir) == 'retained-key'
    assert (claude_dir / 'skills/memman/SKILL.md').exists() == (selected == 'codex')
    assert (Path(codex_env['skills_dir']) / 'memman').is_symlink() == (
        selected == 'claude-code')

    # Removing the remaining integration still performs the full teardown.
    result = runner.invoke(cli, ['--data-dir', data_dir, 'uninstall'])
    assert result.exit_code == 0, result.output
    assert removed == ['scheduler', 'backup']
    assert config.get_scoped(config.LLM_API_KEY, data_dir) is None
