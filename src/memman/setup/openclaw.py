"""OpenClaw integration: install, eject."""

import json
import os
import shutil
from importlib.resources import files as pkg_files
from pathlib import Path

import memman
from memman.setup.prompt import status_error, status_ok, status_updated
from memman.setup.settings import remove_if_empty


def _asset_bytes(rel_path: str) -> bytes:
    """Read an embedded asset file."""
    return (pkg_files('memman.setup.assets')
            .joinpath(rel_path).read_bytes())


def openclaw_write_skill(config_dir: str) -> str:
    """Write the SKILL.md to the OpenClaw skills directory."""
    skill_dir = os.path.join(config_dir, 'skills', 'memman')
    Path(skill_dir).mkdir(mode=0o755, exist_ok=True, parents=True)
    skill_path = os.path.join(skill_dir, 'SKILL.md')
    Path(skill_path).write_bytes(
        _asset_bytes('openclaw/SKILL.md'))
    Path(skill_path).chmod(0o644)
    return skill_path


def openclaw_write_hook(config_dir: str) -> str:
    """Write the memman-prime internal hook."""
    hook_dir = os.path.join(
        config_dir, 'hooks', 'memman-prime')
    Path(hook_dir).mkdir(mode=0o755, exist_ok=True, parents=True)

    hook_md_path = os.path.join(hook_dir, 'HOOK.md')
    Path(hook_md_path).write_bytes(
        _asset_bytes('openclaw/hooks/memman-prime/HOOK.md'))
    Path(hook_md_path).chmod(0o644)

    handler_path = os.path.join(hook_dir, 'handler.js')
    Path(handler_path).write_bytes(
        _asset_bytes('openclaw/hooks/memman-prime/handler.js'))
    Path(handler_path).chmod(0o644)

    return hook_dir


def openclaw_write_plugin(config_dir: str, ver: str) -> str:
    """Write the memman plugin to the OpenClaw extensions directory."""
    plugin_dir = os.path.join(
        config_dir, 'extensions', 'memman')
    Path(plugin_dir).mkdir(mode=0o755, exist_ok=True, parents=True)

    manifest = _asset_bytes(
        'openclaw/plugin/openclaw.plugin.json')
    if ver and ver != 'dev':
        try:
            m = json.loads(manifest)
            m['version'] = ver
            manifest = (
                json.dumps(m, indent=2) + '\n').encode()
        except Exception:
            pass

    file_list = [
        ('package.json',
         _asset_bytes('openclaw/plugin/package.json')),
        ('openclaw.plugin.json', manifest),
        ('index.js',
         _asset_bytes('openclaw/plugin/index.js')),
        ]
    for name, data in file_list:
        fpath = os.path.join(plugin_dir, name)
        Path(fpath).write_bytes(data)
        Path(fpath).chmod(0o644)

    return plugin_dir


def openclaw_register_plugin(config_dir: str,
                             remind: bool = True,
                             nudge: bool = True) -> str:
    """Add the memman plugin entry to openclaw.json."""
    cfg_path = os.path.join(config_dir, 'openclaw.json')

    try:
        data = Path(cfg_path).read_text()
        cfg = json.loads(data)
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        cfg = {}

    plugins = cfg.get('plugins')
    if not isinstance(plugins, dict):
        plugins = {}
    entries = plugins.get('entries')
    if not isinstance(entries, dict):
        entries = {}

    entries['memman'] = {
        'enabled': True,
        'config': {
            'remind': remind,
            'nudge': nudge,
            },
        }
    plugins['entries'] = entries
    cfg['plugins'] = plugins

    content = json.dumps(cfg, indent=2) + '\n'
    Path(cfg_path).write_text(content)
    Path(cfg_path).chmod(0o600)

    return cfg_path


def openclaw_eject(config_dir: str) -> list[Exception]:
    """Remove memman skill, hook, and plugin from OpenClaw."""
    errs: list[Exception] = []

    print(f'\nRemoving OpenClaw integration ({config_dir})...')

    targets = [
        ('Skill',
         os.path.join(config_dir, 'skills', 'memman')),
        ('Hook',
         os.path.join(config_dir, 'hooks', 'memman-prime')),
        ('Plugin',
         os.path.join(config_dir, 'extensions', 'memman')),
        ]

    for (label, path) in targets:
        try:
            shutil.rmtree(path, ignore_errors=True)
            status_ok(label,
                      path + ' removed')
        except Exception as e:
            status_error(label, e)
            errs.append(e)

    remove_if_empty(os.path.join(config_dir, 'skills'))
    remove_if_empty(os.path.join(config_dir, 'hooks'))
    remove_if_empty(os.path.join(config_dir, 'extensions'))

    cfg_path = os.path.join(config_dir, 'openclaw.json')
    try:
        data = Path(cfg_path).read_text()
        cfg = json.loads(data)
        plugins = cfg.get('plugins', {})
        entries = plugins.get('entries', {})
        if isinstance(entries, dict):
            entries.pop('memman', None)
            plugins['entries'] = entries
            cfg['plugins'] = plugins
            content = json.dumps(cfg, indent=2) + '\n'
            Path(cfg_path).write_text(content)
            Path(cfg_path).chmod(0o600)
    except Exception:
        pass

    remove_if_empty(config_dir)
    return errs


def install_openclaw(env: dict, data_dir: str) -> None:
    """Install memman into OpenClaw (~/.openclaw/)."""
    from memman.setup.claude import _init_default_store, write_prompt_files

    config_dir = env['config_dir']

    print(f'\nSetting up OpenClaw ({config_dir})...')

    print('\n[1/4] Skill')
    path = openclaw_write_skill(config_dir)
    status_ok('Skill', path)

    print('\n[2/4] Prompts')
    path = write_prompt_files()
    status_ok('Prompts', path)

    print('\n[3/4] Hook')
    path = openclaw_write_hook(config_dir)
    status_ok('Hook: prime', path)

    print('\n[4/4] Plugin')
    ver = memman.__version__
    path = openclaw_write_plugin(config_dir, ver)
    status_ok('Plugin', path)

    path = openclaw_register_plugin(config_dir)
    status_updated('Config', path)

    print()
    print('Setup complete!')
    print(f'  Skill   {config_dir}/skills/memman/SKILL.md')
    print(f'  Hook    {config_dir}/hooks/memman-prime/'
          ' (agent:bootstrap)')
    print(f'  Plugin  {config_dir}/extensions/memman/'
          ' (hooks: remind, nudge)')
    print('  Prompts ~/.memman/prompt/ (guide.md, skill.md)')
    print()
    print('Restart the OpenClaw gateway to activate.')
    print('Edit ~/.memman/prompt/guide.md to customize behavior.')
    print("Run 'memman setup --eject' to remove.")

    _init_default_store(data_dir)
