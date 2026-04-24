"""OpenClaw integration: install, uninstall."""

import json
import os
import shutil
from pathlib import Path

from memman.setup.deploy import symlink_asset
from memman.setup.prompt import status_error, status_ok, status_updated
from memman.setup.settings import remove_if_empty


def openclaw_write_skill(config_dir: str) -> str:
    """Symlink the SKILL.md into the OpenClaw skills directory."""
    link = Path(config_dir) / 'skills' / 'memman' / 'SKILL.md'
    symlink_asset('openclaw/SKILL.md', link)
    return str(link)


def openclaw_write_hook(config_dir: str) -> str:
    """Symlink the memman-prime internal hook files."""
    hook_dir = Path(config_dir) / 'hooks' / 'memman-prime'
    symlink_asset(
        'openclaw/hooks/memman-prime/HOOK.md',
        hook_dir / 'HOOK.md')
    symlink_asset(
        'openclaw/hooks/memman-prime/handler.js',
        hook_dir / 'handler.js')
    return str(hook_dir)


def openclaw_write_plugin(config_dir: str) -> str:
    """Symlink the memman plugin into the OpenClaw extensions directory."""
    plugin_dir = Path(config_dir) / 'extensions' / 'memman'
    symlink_asset('openclaw/plugin/package.json',
                  plugin_dir / 'package.json')
    symlink_asset('openclaw/plugin/openclaw.plugin.json',
                  plugin_dir / 'openclaw.plugin.json')
    symlink_asset('openclaw/plugin/index.js',
                  plugin_dir / 'index.js')
    return str(plugin_dir)


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


def openclaw_uninstall(config_dir: str) -> list[Exception]:
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
    from memman.setup.claude import _init_default_store

    config_dir = env['config_dir']

    print(f'\nSetting up OpenClaw ({config_dir})...')

    print('\n[1/3] Skill')
    path = openclaw_write_skill(config_dir)
    status_ok('Skill', path)

    print('\n[2/3] Hook')
    path = openclaw_write_hook(config_dir)
    status_ok('Hook: prime', path)

    print('\n[3/3] Plugin')
    path = openclaw_write_plugin(config_dir)
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
    print()
    print('Restart the OpenClaw gateway to activate.')

    _init_default_store(data_dir)
