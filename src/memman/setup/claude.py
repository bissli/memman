"""Claude Code integration: install and uninstall orchestration.
"""

import os
import shutil
import sys
from pathlib import Path

import click
import httpx
from memman import config
from memman.cli import list_claude_permissions
from memman.embed import get_client
from memman.embed.fingerprint import seed_if_fresh
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.llm import openrouter_models
from memman.setup import wizard
from memman.setup.deploy import symlink_asset
from memman.setup.detect import detect_claude_code
from memman.setup.prompt import detection_line, status_error, status_ok
from memman.setup.prompt import status_updated
from memman.setup.scheduler import _write_env_keys, detect_scheduler
from memman.setup.scheduler import install as install_scheduler
from memman.setup.scheduler import memman_binary_path
from memman.setup.scheduler import uninstall as uninstall_scheduler
from memman.setup.scheduler import uninstall_backup
from memman.setup.settings import add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, write_json_file
from memman.setup.settings import write_or_remove_json_file
from memman.store.db import store_dir, store_exists
from memman.store.factory import open_backend, resolve_store_backend


def check_prereqs(data_dir: str) -> dict[str, str]:
    """Validate install prerequisites; raise ClickException on failure.

    Parameters
    ----------
    data_dir : str
        Data directory holding the env file.

    Returns
    -------
    dict[str, str]
        Install-time knobs (env-or-default for every `INSTALLABLE_KEYS`
        entry). `collect_install_knobs` validates the mandatory keys,
        and a model for a non-OpenRouter endpoint.

    Raises
    ------
    RuntimeError
        The host has no scheduler (from `detect_scheduler`).
    click.ClickException
        The memman binary is missing, or the knobs fail validation.
    """
    detect_scheduler()
    try:
        memman_binary_path()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        return config.collect_install_knobs(data_dir)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def claude_write_skill(config_dir: str) -> str:
    """Symlink the memman skill into the config dir.

    Parameters
    ----------
    config_dir : str
        Claude Code config directory (`~/.claude`).

    Returns
    -------
    str
        Path of the symlink, `<config_dir>/skills/memman/SKILL.md`.
    """
    link = Path(config_dir) / 'skills' / 'memman' / 'SKILL.md'
    symlink_asset('claude/SKILL.md', link)
    return str(link)


def claude_write_hook(config_dir: str, filename: str) -> str:
    """Symlink a hook script into the config dir.

    Parameters
    ----------
    config_dir : str
        Claude Code config directory (`~/.claude`).
    filename : str
        Shipped hook script under `assets/claude/`, e.g. `prime.sh`.

    Returns
    -------
    str
        Path of the symlink, `<config_dir>/hooks/memman/<filename>`.
    """
    link = Path(config_dir) / 'hooks' / 'memman' / filename
    symlink_asset(f'claude/{filename}', link)
    return str(link)


def claude_uninstall(config_dir: str) -> list[Exception]:
    """Remove memman integration from the given Claude Code config dir.

    Parameters
    ----------
    config_dir : str
        Claude Code config directory (`~/.claude`).

    Returns
    -------
    list[Exception]
        Errors raised while cleaning the settings file. Empty on full
        success.
    """
    errs: list[Exception] = []

    print(f'\nRemoving Claude Code integration ({config_dir})...')

    hooks_dir = os.path.join(config_dir, 'hooks', 'memman')
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
    shutil.rmtree(skill_dir, ignore_errors=True)
    status_ok('Skill', skill_dir + ' removed')
    remove_if_empty(os.path.join(config_dir, 'skills'))

    remove_if_empty(config_dir)
    return errs


def _init_default_store(data_dir: str) -> None:
    """Ensure the default store exists with a seeded embed fingerprint.

    Delegates to `seed_if_fresh` so the install-time and lazy
    first-open paths share a single seed implementation, including
    the unavailable-client and dim>0 validation.
    """
    backend_kind = resolve_store_backend('default', data_dir)
    if backend_kind == 'sqlite' and not store_exists(data_dir, 'default'):
        with open_backend('default', data_dir) as backend:
            try:
                seed_if_fresh(backend, get_client())
            except (EmbedFingerprintError, ConfigError) as exc:
                raise click.ClickException(str(exc)) from exc
        print(f'  Initialized default store at {store_dir(data_dir, "default")}')


def _install_claude_code(env: dict, data_dir: str,
                         no_wizard: bool = False) -> None:
    """Install memman into Claude Code (~/.claude/).
    """
    config_dir = env['config_dir']

    print(f'\nSetting up Claude Code ({config_dir})...')

    logs_dir = Path.home() / '.memman' / 'logs'
    # Notes:
    # - 0700, matching the rest of ~/.memman: worker logs carry
    #   memory content, and doctor's `env_permissions` check warns on
    #   any group or other bit under that directory.
    # - chmod as well as mkdir: mkdir's mode is masked by the umask,
    #   so it alone cannot state a mode, and it does nothing at all
    #   for a directory an earlier install already created.
    logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    logs_dir.chmod(0o700)

    print('\n[1/2] Skill')
    path = claude_write_skill(config_dir)
    status_ok('Skill', path)

    print('\n[2/2] Hooks')
    hook_filenames = [
        ('prime.sh', 'prime'),
        ('user_prompt.sh', 'remind'),
        ('compact.sh', 'compact'),
        ('task_recall.sh', 'recall'),
        ('exit_plan.sh', 'exit_plan'),
        ]
    for filename, label in hook_filenames:
        path = claude_write_hook(config_dir, filename)
        status_ok(f'Hook: {label}', path)

    settings_path = os.path.join(config_dir, 'settings.json')
    hooks_dir = os.path.join(config_dir, 'hooks', 'memman')
    data = read_json_file(settings_path)
    add_claude_hooks_selective(
        data, hooks_dir,
        remind=True, compact=True,
        task_recall=True, exit_plan=True)

    permissions = list_claude_permissions()

    if sys.stdin.isatty() and not no_wizard:
        print()
        print('memman would like to add these entries to'
              f' {settings_path} permissions.allow:')
        for entry in permissions:
            print(f'  {entry}')
        if click.confirm('Add them?', default=True):
            add_memman_permission(data, permissions)
            permission_msg = f'{len(permissions)} memman verbs added'
        else:
            permission_msg = (
                'skipped (use "Allow always" on first call, or re-run'
                ' memman install)')
    else:
        add_memman_permission(data, permissions)
        permission_msg = f'{len(permissions)} memman verbs added'

    write_json_file(settings_path, data)
    status_updated('Settings', settings_path)
    status_ok('Permissions', permission_msg)

    print()
    print('Setup complete!')
    print('  Hooks   prime, remind, compact, recall, exit_plan')
    print()
    print('Start a new Claude Code session to activate.')

    _init_default_store(data_dir)


def _uninstall_env(env: dict) -> bool:
    """Remove memman from Claude Code; True when cleanup reported an error.
    """
    errs = claude_uninstall(env['config_dir'])
    return len(errs) > 0


def run_install(data_dir: str, claude_code: bool = False,
                backend: str | None = None,
                pg_dsn: str | None = None,
                llm_endpoint: str | None = None,
                embed_provider: str | None = None,
                no_wizard: bool = False) -> None:
    """Install memman integration. Called by the `memman install` command.

    Parameters
    ----------
    data_dir : str
        Holds the env file, the stores, and the model state.
    claude_code : bool, default False
        Install into `~/.claude` even when the `claude` binary is not
        on `PATH`.
    backend : str or None, default None
        `sqlite` or `postgres`; None leaves the choice to the wizard or
        the env file.
    pg_dsn : str or None, default None
        Postgres DSN for `backend='postgres'`.
    llm_endpoint : str or None, default None
        OpenAI-compatible LLM endpoint URL.
    embed_provider : str or None, default None
        Embed provider name.
    no_wizard : bool, default False
        Take flags, the env file, and defaults only; never prompt.

    Raises
    ------
    click.ClickException
        A flag disagrees with a value already in the env file, or a
        prerequisite is missing.
    """
    _reject_flag_file_conflicts(
        data_dir=data_dir, backend=backend, pg_dsn=pg_dsn,
        llm_endpoint=llm_endpoint, embed_provider=embed_provider)
    env = detect_claude_code()
    wizard_out = wizard.run_wizard(
        data_dir, backend=backend, pg_dsn=pg_dsn,
        llm_endpoint=llm_endpoint, embed_provider=embed_provider,
        no_wizard=no_wizard)
    if wizard_out:
        # A secret the wizard collected counts toward the prereq check.
        _write_env_keys(wizard_out, data_dir=data_dir)
    knobs = check_prereqs(data_dir)
    _run_install_flow(env, claude_code=claude_code, data_dir=data_dir,
                      knobs=knobs, no_wizard=no_wizard)


def _reject_flag_file_conflicts(
        *,
        data_dir: str,
        backend: str | None,
        pg_dsn: str | None,
        llm_endpoint: str | None,
        embed_provider: str | None) -> None:
    """Exit 1 when a flag value conflicts with the env file's current value.

    The env-file canonical model means install flags are sticky-seed:
    they fill blanks but never override an existing value. Silently
    swallowing a flag is a footgun, so any conflict surfaces here with
    the exact `memman config set ...` command the user should run.
    """
    file_values = config.parse_env_file(config.env_file_path(data_dir))
    pairs: list[tuple[str, str | None]] = [
        (config.DEFAULT_BACKEND, backend),
        (config.DEFAULT_PG_DSN, pg_dsn),
        (config.LLM_ENDPOINT, llm_endpoint),
        (config.EMBED_PROVIDER, embed_provider),
        ]
    for key, flag_value in pairs:
        if flag_value is None:
            continue
        existing = file_values.get(key, '').strip()
        if existing and existing != flag_value:
            env_path = config.env_file_path(data_dir)
            raise click.ClickException(
                f'{key} is already set to {existing!r} in {env_path};'
                f' refusing to silently override with flag value'
                f' {flag_value!r}.\nRun: memman config set {key} {flag_value}')


def run_uninstall(data_dir: str, claude_code: bool = False) -> None:
    """Remove memman integration. Called by the `memman uninstall` command.

    Parameters
    ----------
    data_dir : str
        Holds the env file and the stores. The stores stay on disk;
        the env file loses its secret keys and keeps the rest.
    claude_code : bool, default False
        Remove from `~/.claude` even when the `claude` binary is not
        on `PATH`.

    Raises
    ------
    click.ClickException
        The Claude Code cleanup reported an error; the scheduler unit
        is left in place.
    """
    env = detect_claude_code()
    print('\n[backup]')
    try:
        backup_result = uninstall_backup()
        for action in backup_result.get('actions', []):
            status_ok(backup_result['platform'], action)
    except RuntimeError:
        pass
    _run_uninstall_flow(env, claude_code=claude_code, data_dir=data_dir)


def _run_install_flow(env: dict, claude_code: bool,
                      data_dir: str,
                      knobs: dict[str, str],
                      no_wizard: bool = False) -> None:
    """Install Claude Code integration and the scheduler, then check the model.

    Parameters
    ----------
    env : dict
        `detect_claude_code` output.
    claude_code : bool
        Force the Claude Code install even when not detected.
    data_dir : str
        Holds the env file and the model state.
    knobs : dict[str, str]
        Install values from `check_prereqs`, handed to the scheduler.
    no_wizard : bool, default False
        Passed through to the Claude Code install.
    """
    if claude_code:
        _install_claude_code(env, data_dir=data_dir, no_wizard=no_wizard)
    else:
        print('Detecting LLM CLI environments...')
        print()
        detection_line(
            env['detected'], env['display'],
            env['version'], env['config_dir'])
        if env['detected']:
            _install_claude_code(env, data_dir=data_dir, no_wizard=no_wizard)
        else:
            print('\nNo CLI integration installed'
                  ' (no Claude Code detected).')
            print('Installing scheduler only; manual'
                  ' `memman remember` calls will still work.')

    print('\n[scheduler]')
    result = install_scheduler(data_dir, knobs)
    for action in result.get('env_actions', []) + result.get('actions', []):
        status_ok(result['platform'], action)

    # Runs once the env file is final. A catalog outage prints an error
    # and the install still finishes.
    try:
        notice = openrouter_models.refresh_model_state(data_dir, force=True)
    except (httpx.HTTPError, RuntimeError) as exc:
        print('\n[model]')
        status_error(
            'openrouter', f'cannot read the OpenRouter catalogs: {exc}')
        return
    if notice is None:
        return
    print('\n[model]')
    if notice:
        status_error('openrouter', notice)
    else:
        model = config.get_scoped(config.LLM_MODEL, data_dir)
        status_ok('openrouter', f'{model} routes under the provider pin')


def _run_uninstall_flow(env: dict, claude_code: bool,
                        data_dir: str) -> None:
    """Uninstall Claude Code integration and remove the scheduler unit.
    """
    failed = False
    if claude_code:
        failed = _uninstall_env(env)
    else:
        print('Detecting LLM CLI environments...')
        print()
        detection_line(
            env['detected'], env['display'],
            env['version'], env['config_dir'])
        if env['detected']:
            failed = _uninstall_env(env)
        else:
            print('\nNo CLI integration detected.')
    if failed:
        raise click.ClickException(
            'error during Claude Code integration uninstall;'
            ' scheduler left in place')

    print('\n[scheduler]')
    result = uninstall_scheduler(data_dir=data_dir)
    for action in result.get('actions', []):
        status_ok(result['platform'], action)

    print()
    print('Done! All detected integrations removed.')
