"""Agent integration orchestration and Claude Code setup.
"""

import json
import os
import shutil
import sys
from itertools import starmap
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
from memman.setup.codex import check_codex_skill, install_codex
from memman.setup.codex import uninstall_codex
from memman.setup.deploy import is_asset_link, symlink_asset
from memman.setup.detect import detect_claude_code, detect_codex
from memman.setup.prompt import detection_line, status_error, status_ok
from memman.setup.prompt import status_updated
from memman.setup.scheduler import _write_env_keys, detect_scheduler
from memman.setup.scheduler import install as install_scheduler
from memman.setup.scheduler import memman_binary_path
from memman.setup.scheduler import uninstall as uninstall_scheduler
from memman.setup.scheduler import uninstall_backup
from memman.setup.settings import _contains_memman, add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, strip_json5
from memman.setup.settings import write_json_file, write_or_remove_json_file
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
        Errors raised while removing assets or cleaning settings.
        Empty on full success.
    """
    errs: list[Exception] = []

    print(f'\nRemoving Claude Code integration ({config_dir})...')

    hooks_dir = os.path.join(config_dir, 'hooks', 'memman')
    _remove_claude_assets(hooks_dir, 'Hooks', errs)
    remove_if_empty(os.path.join(config_dir, 'hooks'))

    settings_path = os.path.join(config_dir, 'settings.json')
    try:
        data = _read_claude_settings(Path(settings_path))
        remove_claude_hooks(data)
        remove_memman_permission(data)
        write_or_remove_json_file(settings_path, data)
        status_ok('Settings', settings_path + ' cleaned')
    except Exception as e:
        status_error('Settings', e)
        errs.append(e)

    skill_dir = os.path.join(config_dir, 'skills', 'memman')
    _remove_claude_assets(skill_dir, 'Skill', errs)
    remove_if_empty(os.path.join(config_dir, 'skills'))

    remove_if_empty(config_dir)
    return errs


def _remove_claude_assets(path: str, label: str,
                          errors: list[Exception]) -> None:
    """Report failed asset removal without following directory symlinks."""
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError) or os.path.lexists(path):
            status_error(label, exc)
            errors.append(exc)
            return
    status_ok(label, path + ' removed')


def _read_claude_settings(path: Path) -> dict:
    """A missing file is empty; unreadable settings must remain untouched."""
    try:
        contents = path.read_text()
    except FileNotFoundError:
        return {}
    return json.loads(strip_json5(contents)) if contents else {}


def _claude_integration_installed(config_dir: str) -> bool:
    """Recognize owned assets or hook registrations before shared teardown."""
    base = Path(config_dir)
    assets = [('claude/SKILL.md', base / 'skills/memman/SKILL.md')]
    assets.extend(
        (f'claude/{name}', base / 'hooks/memman' / name)
        for name in ('prime.sh', 'user_prompt.sh', 'compact.sh',
                     'task_recall.sh', 'exit_plan.sh'))
    if any(starmap(is_asset_link, assets)):
        return True
    try:
        data = _read_claude_settings(base / 'settings.json')
        if not isinstance(data, dict):
            return True
        # Dependency checks include legacy/custom events beyond the events
        # this installer manages (for example, a Stop hook calling memman).
        return _contains_memman(data.get('hooks', {}))
    except (OSError, ValueError):
        # An unselected agent's settings are outside this uninstall's scope.
        # If they cannot be inspected, retain its possible dependencies.
        return True


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


def _uninstall_env(env: dict) -> bool:
    """Remove memman from Claude Code; True when cleanup reported an error.
    """
    errs = claude_uninstall(env['config_dir'])
    return len(errs) > 0


def run_install(data_dir: str, claude_code: bool = False,
                backend: str | None = None,
                pg_dsn: str | None = None,
                endpoint: str | None = None,
                no_wizard: bool = False, codex: bool = False) -> None:
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
    endpoint : str or None, default None
        OpenAI-compatible endpoint URL for the LLM, embed, and rerank
        paths.
    no_wizard : bool, default False
        Take flags, the env file, and defaults only; never prompt.
    codex : bool, default False
        Install the Codex skill even when Codex is not detected.
        Explicit agent flags select only the named integrations.

    Raises
    ------
    click.ClickException
        A flag disagrees with a value already in the env file, or a
        prerequisite is missing.
    """
    _reject_flag_file_conflicts(
        data_dir=data_dir, backend=backend, pg_dsn=pg_dsn,
        endpoint=endpoint)
    env = detect_claude_code()
    codex_env = detect_codex()
    if codex or (not claude_code and codex_env['detected']):
        check_codex_skill(codex_env)
    wizard_out = wizard.run_wizard(
        data_dir, backend=backend, pg_dsn=pg_dsn,
        endpoint=endpoint, no_wizard=no_wizard)
    if wizard_out:
        # A secret the wizard collected counts toward the prereq check.
        _write_env_keys(wizard_out, data_dir=data_dir)
    knobs = check_prereqs(data_dir)
    _run_install_flow(env, claude_code=claude_code, data_dir=data_dir,
                      knobs=knobs, no_wizard=no_wizard,
                      codex_env=codex_env, codex=codex)


def _reject_flag_file_conflicts(
        *,
        data_dir: str,
        backend: str | None,
        pg_dsn: str | None,
        endpoint: str | None) -> None:
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
        (config.ENDPOINT, endpoint),
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


def run_uninstall(data_dir: str, claude_code: bool = False,
                  codex: bool = False) -> None:
    """Remove memman integration. Called by the `memman uninstall` command.

    Parameters
    ----------
    data_dir : str
        Holds the env file and the stores. The stores stay on disk;
        secrets are removed only when shared services are removed.
    claude_code : bool, default False
        Remove from `~/.claude` even when the `claude` binary is not
        on `PATH`.
    codex : bool, default False
        Remove the Codex skill even when Codex is not detected.

    Raises
    ------
    click.ClickException
        An integration cleanup reported an error; the scheduler unit
        is left in place.
    """
    env = detect_claude_code()
    codex_env = detect_codex()
    if codex or (not claude_code and codex_env['detected']):
        check_codex_skill(codex_env)
    removed_shared = _run_uninstall_flow(
        env, claude_code=claude_code, data_dir=data_dir,
        codex_env=codex_env, codex=codex)
    if removed_shared:
        print('\n[backup]')
        try:
            backup_result = uninstall_backup()
            for action in backup_result.get('actions', []):
                status_ok(backup_result['platform'], action)
        except RuntimeError:
            pass
    print('\nDone! Selected integrations removed.')


def _run_install_flow(env: dict, claude_code: bool,
                      data_dir: str,
                      knobs: dict[str, str],
                      no_wizard: bool = False,
                      codex_env: dict | None = None,
                      codex: bool = False) -> None:
    """Install selected integrations and the scheduler, then check the model.

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
        Passed through to the Claude Code and Codex installs.
    codex_env : dict or None, default None
        `detect_codex` output, when Codex discovery was requested.
    codex : bool, default False
        Force the Codex install; explicit flags select integrations.
    """
    explicit = claude_code or codex
    if not explicit:
        print('Detecting LLM CLI environments...\n')
        for detected_env in (env, codex_env):
            if detected_env is not None:
                detection_line(
                    detected_env['detected'], detected_env['display'],
                    detected_env['version'], detected_env['config_dir'])
    use_claude = claude_code or (not explicit and env['detected'])
    use_codex = codex or (not explicit and codex_env is not None
                          and codex_env['detected'])
    if use_claude:
        _install_claude_code(env, data_dir=data_dir, no_wizard=no_wizard)
    if use_codex:
        install_codex(codex_env if codex_env is not None else detect_codex(),
                      no_wizard=no_wizard)
    if not use_claude and not use_codex:
        print('\nNo CLI integration installed (no supported agent detected).')
        print('Installing scheduler only; manual'
              ' `memman remember` calls will still work.')

    print('\n[scheduler]')
    result = install_scheduler(data_dir, knobs)
    for action in result.get('env_actions', []) + result.get('actions', []):
        status_ok(result['platform'], action)
    if use_claude or use_codex:
        # The scheduler install persists the provider defaults first.
        # Initializing in an agent installer fails on a fresh env file.
        _init_default_store(data_dir)

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
                        data_dir: str, codex_env: dict | None = None,
                        codex: bool = False) -> bool:
    """Uninstall integrations; return whether shared services were removed.

    An explicit selection leaves shared services in place while another
    memman integration remains installed. Detection of an agent CLI alone
    is insufficient: check the actual memman assets.
    """
    failed = False
    explicit = claude_code or codex
    if claude_code or (not explicit and env['detected']):
        failed = _uninstall_env(env)
    if codex or (not explicit and codex_env is not None
                 and codex_env['detected']):
        try:
            uninstall_codex(
                codex_env if codex_env is not None else detect_codex())
        except (OSError, click.ClickException) as exc:
            status_error('Codex', exc)
            failed = True
    if failed:
        raise click.ClickException(
            'error during agent integration uninstall;'
            ' scheduler left in place')

    other_claude = not claude_code and _claude_integration_installed(
        env['config_dir'])
    other_codex = (not codex and codex_env is not None and is_asset_link(
        'codex', Path(codex_env['skills_dir']) / 'memman'))
    if explicit and (other_claude or other_codex):
        print('\nAnother memman integration remains installed; keeping'
              ' the shared scheduler, backups, and provider settings.')
        return False

    print('\n[scheduler]')
    result = uninstall_scheduler(data_dir=data_dir)
    for action in result.get('actions', []):
        status_ok(result['platform'], action)
    return True
