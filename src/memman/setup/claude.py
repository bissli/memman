"""Claude Code integration: install, eject, and setup orchestration."""

import os
import platform
import shutil
from importlib.resources import files as pkg_files
from pathlib import Path

import click
from memman.setup.detect import detect_environments, home_dir
from memman.setup.markdown import eject_memory_block
from memman.setup.prompt import detection_line, status_error, status_ok
from memman.setup.prompt import status_updated
from memman.setup.scheduler import detect_scheduler
from memman.setup.scheduler import install as install_scheduler
from memman.setup.scheduler import memman_binary_path
from memman.setup.scheduler import uninstall as uninstall_scheduler
from memman.setup.settings import add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, write_json_file
from memman.setup.settings import write_or_remove_json_file


def check_prereqs() -> tuple[str, str]:
    """Validate install prerequisites; raise ClickException on failure.

    Returns (openrouter_api_key, voyage_api_key) once all checks pass.
    """
    if not detect_scheduler():
        raise click.ClickException(
            f'unsupported platform {platform.system()!r}: expected'
            ' Linux+systemd or macOS+launchd')
    try:
        memman_binary_path()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    openrouter_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not openrouter_key:
        raise click.ClickException(
            'OPENROUTER_API_KEY is required for the background enrichment'
            ' worker; export it and re-run')

    voyage_key = os.environ.get('VOYAGE_API_KEY', '').strip()
    if not voyage_key:
        raise click.ClickException(
            'VOYAGE_API_KEY is required for memory embeddings;'
            ' export it and re-run')

    return openrouter_key, voyage_key


def _asset_bytes(rel_path: str) -> bytes:
    """Read an embedded asset file."""
    return (pkg_files('memman.setup.assets')
            .joinpath(rel_path).read_bytes())


def write_prompt_files() -> str:
    """Write guide.md and skill.md to ~/.memman/prompt/."""
    prompt_dir = os.path.join(home_dir(), '.memman', 'prompt')
    Path(prompt_dir).mkdir(mode=0o755, exist_ok=True, parents=True)

    guide_path = os.path.join(prompt_dir, 'guide.md')
    Path(guide_path).write_bytes(_asset_bytes('claude/guide.md'))
    Path(guide_path).chmod(0o644)

    skill_path = os.path.join(prompt_dir, 'skill.md')
    Path(skill_path).write_bytes(_asset_bytes('claude/SKILL.md'))
    Path(skill_path).chmod(0o644)

    return prompt_dir


def claude_write_skill(config_dir: str) -> str:
    """Write the memman skill to the config dir."""
    skill_dir = os.path.join(config_dir, 'skills', 'memman')
    Path(skill_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    skill_path = os.path.join(skill_dir, 'SKILL.md')
    Path(skill_path).write_bytes(_asset_bytes('claude/SKILL.md'))
    Path(skill_path).chmod(0o644)
    return skill_path


def claude_write_hook(config_dir: str, filename: str, content: bytes) -> str:
    """Write a hook script to the hooks dir."""
    hooks_dir = os.path.join(config_dir, 'hooks', 'mm')
    Path(hooks_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    hook_path = os.path.join(hooks_dir, filename)
    Path(hook_path).write_bytes(content)
    Path(hook_path).chmod(0o755)
    return hook_path


def claude_register_hooks(config_dir: str,
                          remind: bool = True, nudge: bool = True,
                          compact: bool = True,
                          task_recall: bool = True,
                          exit_plan: bool = True) -> str:
    """Register hooks in settings.json."""
    hooks_dir = os.path.join(config_dir, 'hooks', 'mm')
    settings_path = os.path.join(config_dir, 'settings.json')
    data = read_json_file(settings_path)
    add_claude_hooks_selective(
        data, hooks_dir,
        remind=remind, nudge=nudge,
        compact=compact, task_recall=task_recall,
        exit_plan=exit_plan)
    write_json_file(settings_path, data)
    return settings_path


def claude_eject(config_dir: str) -> list[Exception]:
    """Remove memman integration from the given Claude Code config dir."""
    errs: list[Exception] = []

    print(f'\nRemoving Claude Code integration ({config_dir})...')

    hooks_dir = os.path.join(config_dir, 'hooks', 'mm')
    shutil.rmtree(hooks_dir, ignore_errors=True)
    status_ok('Hooks', hooks_dir + ' removed')
    remove_if_empty(os.path.join(config_dir, 'hooks'))

    settings_path = os.path.join(config_dir, 'settings.json')
    try:
        data = read_json_file(settings_path)
        remove_claude_hooks(data)
        remove_memman_permission(data)
        write_or_remove_json_file(settings_path, data)
        status_ok('Settings', settings_path + ' cleaned')
    except Exception as e:
        status_error('Settings', e)
        errs.append(e)

    skill_dir = os.path.join(config_dir, 'skills', 'memman')
    try:
        shutil.rmtree(skill_dir, ignore_errors=True)
        status_ok('Skill', skill_dir + ' removed')
    except Exception as e:
        status_error('Skill', e)
        errs.append(e)
    remove_if_empty(os.path.join(config_dir, 'skills'))

    remove_if_empty(config_dir)
    return errs


def _init_default_store(data_dir: str) -> None:
    """Ensure the default store exists."""
    from memman.store.db import open_db, store_dir, store_exists

    if not store_exists(data_dir, 'default'):
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        db.close()
        print(f'  Initialized default store at {sdir}')


def _install_claude_code(env: dict, data_dir: str) -> None:
    """Install memman into Claude Code (~/.claude/)."""
    config_dir = env['config_dir']

    print(f'\nSetting up Claude Code ({config_dir})...')

    print('\n[1/3] Skill')
    path = claude_write_skill(config_dir)
    status_ok('Skill', path)

    print('\n[2/3] Prompts')
    path = write_prompt_files()
    status_ok('Prompts', path)

    print('\n[3/3] Hooks')
    hook_assets = [
        ('prime.sh', 'prime'),
        ('user_prompt.sh', 'remind'),
        ('stop.sh', 'nudge'),
        ('compact.sh', 'compact'),
        ('task_recall.sh', 'recall'),
        ('exit_plan.sh', 'exit_plan'),
        ]
    for filename, label in hook_assets:
        path = claude_write_hook(
            config_dir, filename, _asset_bytes(f'claude/{filename}'))
        status_ok(f'Hook: {label}', path)

    path = claude_register_hooks(config_dir)
    status_updated('Settings', path)

    settings_path = os.path.join(config_dir, 'settings.json')
    data = read_json_file(settings_path)
    add_memman_permission(data)
    write_json_file(settings_path, data)
    status_ok('Permission',
              'Bash(memman:*) added to settings.json')

    print()
    print('Setup complete!')
    print('  Hooks   prime, remind, nudge, compact, recall, exit_plan')
    print('  Prompts ~/.memman/prompt/ (guide.md, skill.md)')
    print()
    print('Start a new Claude Code session to activate.')
    print('Edit ~/.memman/prompt/guide.md to customize behavior.')
    print("Run 'memman setup --eject' to remove.")

    _init_default_store(data_dir)


def _eject_markdown(file_path: str) -> None:
    """Remove memory guidance block from a markdown file if present."""
    if eject_memory_block(file_path):
        print(f'  Memory guidance removed from {file_path}')


def _eject_env(env: dict) -> bool:
    """Eject memman from a single environment."""
    if env['name'] == 'claude-code':
        errs = claude_eject(env['config_dir'])
        _eject_markdown('CLAUDE.md')
        return len(errs) > 0

    if env['name'] == 'openclaw':
        from memman.setup.openclaw import openclaw_eject
        errs = openclaw_eject(env['config_dir'])
        _eject_markdown('AGENTS.md')
        return len(errs) > 0

    return False


def run_setup(data_dir: str, target: str = '',
              eject: bool = False) -> None:
    """Main setup orchestrator called by cli.py."""
    if target and target not in {'claude-code', 'openclaw'}:
        raise click.ClickException(
            f'invalid target {target!r}'
            ' (must be claude-code or openclaw)')

    envs = detect_environments()

    if eject:
        _run_eject_flow(envs, target=target)
        return

    openrouter_key, voyage_key = check_prereqs()
    _run_install_flow(envs, target=target, data_dir=data_dir,
                      openrouter_key=openrouter_key,
                      voyage_key=voyage_key)


def _run_install_flow(envs: list[dict], target: str,
                      data_dir: str,
                      openrouter_key: str,
                      voyage_key: str) -> None:
    """Install CLI integrations and the scheduler unit."""
    if target:
        matched = next((e for e in envs if e['name'] == target), None)
        if matched is None:
            raise click.ClickException(f'unknown target {target!r}')
        _install_env(matched, data_dir=data_dir)
    else:
        print('Detecting LLM CLI environments...')
        print()

        detected = []
        for env in envs:
            detection_line(
                env['detected'], env['display'],
                env['version'], env['config_dir'])
            if env['detected']:
                detected.append(env)

        if not detected:
            print('\nNo CLI integration installed'
                  ' (no Claude Code or OpenClaw detected).')
            print('Installing scheduler only; manual'
                  ' `memman remember` calls will still work.')
        else:
            err_count = 0
            for env in detected:
                try:
                    _install_env(env, data_dir=data_dir)
                except Exception:
                    err_count += 1
            if err_count > 0:
                raise click.ClickException(
                    f'{err_count} error(s) during CLI integration install')

    print('\n[scheduler]')
    result = install_scheduler(
        data_dir,
        openrouter_api_key=openrouter_key,
        voyage_api_key=voyage_key)
    for action in result.get('env_actions', []) + result.get('actions', []):
        status_ok(result['platform'], action)


def _install_env(env: dict, data_dir: str) -> None:
    """Install memman into a single environment."""
    if env['name'] == 'claude-code':
        _install_claude_code(env, data_dir=data_dir)
    elif env['name'] == 'openclaw':
        from memman.setup.openclaw import install_openclaw
        install_openclaw(env, data_dir=data_dir)


def _run_eject_flow(envs: list[dict], target: str) -> None:
    """Eject CLI integrations and uninstall the scheduler unit."""
    if target:
        matched = next((e for e in envs if e['name'] == target), None)
        if matched is None:
            raise click.ClickException(f'unknown target {target!r}')
        _eject_env(matched)
    else:
        print('Detecting LLM CLI environments...')
        print()

        installed = []
        for env in envs:
            detection_line(
                env['detected'], env['display'],
                env['version'], env['config_dir'])
            if env['detected']:
                installed.append(env)

        if not installed:
            print('\nNo CLI integration detected.')
        else:
            err_count = 0
            for env in installed:
                if _eject_env(env):
                    err_count += 1
            if err_count > 0:
                raise click.ClickException(
                    f'{err_count} error(s) during CLI integration eject')

    print('\n[scheduler]')
    result = uninstall_scheduler()
    for action in result.get('actions', []):
        status_ok(result['platform'], action)

    print()
    print('Done! All detected integrations removed.')
