"""Claude Code integration: install and uninstall orchestration."""

import os
import platform
import shutil
from pathlib import Path

import click
from memman.setup.deploy import symlink_asset
from memman.setup.detect import detect_environments
from memman.setup.markdown import remove_memory_block
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


def claude_write_skill(config_dir: str) -> str:
    """Symlink the memman skill into the config dir."""
    link = Path(config_dir) / 'skills' / 'memman' / 'SKILL.md'
    symlink_asset('claude/SKILL.md', link)
    return str(link)


def claude_write_hook(config_dir: str, filename: str) -> str:
    """Symlink a hook script into the config dir."""
    link = Path(config_dir) / 'hooks' / 'mm' / filename
    symlink_asset(f'claude/{filename}', link)
    return str(link)


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


def claude_uninstall(config_dir: str) -> list[Exception]:
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

    logs_dir = Path.home() / '.memman' / 'logs'
    logs_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    print('\n[1/2] Skill')
    path = claude_write_skill(config_dir)
    status_ok('Skill', path)

    print('\n[2/2] Hooks')
    hook_filenames = [
        ('prime.sh', 'prime'),
        ('user_prompt.sh', 'remind'),
        ('stop.sh', 'nudge'),
        ('compact.sh', 'compact'),
        ('task_recall.sh', 'recall'),
        ('exit_plan.sh', 'exit_plan'),
        ]
    for filename, label in hook_filenames:
        path = claude_write_hook(config_dir, filename)
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
    print()
    print('Start a new Claude Code session to activate.')

    _init_default_store(data_dir)


def _uninstall_markdown(file_path: str) -> None:
    """Remove memory guidance block from a markdown file if present."""
    if remove_memory_block(file_path):
        print(f'  Memory guidance removed from {file_path}')


def _uninstall_env(env: dict) -> bool:
    """Uninstall memman from a single environment."""
    if env['name'] == 'claude-code':
        errs = claude_uninstall(env['config_dir'])
        _uninstall_markdown('CLAUDE.md')
        return len(errs) > 0

    if env['name'] == 'openclaw':
        from memman.setup.openclaw import openclaw_uninstall
        errs = openclaw_uninstall(env['config_dir'])
        _uninstall_markdown('AGENTS.md')
        return len(errs) > 0

    if env['name'] == 'nanoclaw':
        from memman.setup.nanoclaw import uninstall_nanoclaw
        errs = uninstall_nanoclaw()
        return len(errs) > 0

    return False


def _validate_target(target: str) -> None:
    """Raise if target is set and not one of the known environments."""
    if target and target not in {'claude-code', 'openclaw', 'nanoclaw'}:
        raise click.ClickException(
            f'invalid target {target!r}'
            ' (must be claude-code, openclaw, or nanoclaw)')


def run_install(data_dir: str, target: str = '') -> None:
    """Install memman integration. Called by the `memman install` command."""
    _validate_target(target)
    envs = detect_environments()
    openrouter_key, voyage_key = check_prereqs()
    _run_install_flow(envs, target=target, data_dir=data_dir,
                      openrouter_key=openrouter_key,
                      voyage_key=voyage_key)


def run_uninstall(data_dir: str, target: str = '') -> None:
    """Remove memman integration. Called by the `memman uninstall` command."""
    _validate_target(target)
    envs = detect_environments()
    _run_uninstall_flow(envs, target=target)


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
    elif env['name'] == 'nanoclaw':
        from memman.setup.nanoclaw import install_nanoclaw
        install_nanoclaw()


def _run_uninstall_flow(envs: list[dict], target: str) -> None:
    """Uninstall CLI integrations and remove the scheduler unit."""
    if target:
        matched = next((e for e in envs if e['name'] == target), None)
        if matched is None:
            raise click.ClickException(f'unknown target {target!r}')
        _uninstall_env(matched)
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
                if _uninstall_env(env):
                    err_count += 1
            if err_count > 0:
                raise click.ClickException(
                    f'{err_count} error(s) during CLI integration uninstall')

    print('\n[scheduler]')
    result = uninstall_scheduler()
    for action in result.get('actions', []):
        status_ok(result['platform'], action)

    print()
    print('Done! All detected integrations removed.')
