"""Environment detection for LLM CLI integrations."""

import os
import shutil
import subprocess
from pathlib import Path


def home_dir() -> str:
    """Return the user's home directory."""
    return str(Path.home())


def clean_version(v: str) -> str:
    """Strip parenthesized suffixes like '(Claude Code)' from version strings."""
    idx = v.find(' (')
    if idx > 0:
        return v[:idx]
    return v


def _detect_claude() -> dict:
    """Detect Claude Code CLI environment."""
    home = home_dir()
    config_dir = os.path.join(home, '.claude')

    env = {
        'name': 'claude-code',
        'display': 'Claude Code',
        'detected': False,
        'bin_path': '',
        'installed': False,
        'version': '',
        'config_dir': config_dir,
        }

    bin_path = shutil.which('claude')
    if bin_path:
        env['detected'] = True
        env['bin_path'] = bin_path
    if Path(config_dir).exists():
        env['detected'] = True

    skill_path = os.path.join(config_dir, 'skills', 'memman', 'SKILL.md')
    if Path(skill_path).exists():
        env['installed'] = True

    if env['bin_path']:
        try:
            out = subprocess.check_output(
                [env['bin_path'], '--version'],
                timeout=5, stderr=subprocess.DEVNULL)
            env['version'] = clean_version(out.decode().strip())
        except Exception:
            pass

    return env


def _detect_openclaw() -> dict:
    """Detect OpenClaw CLI environment."""
    home = home_dir()
    config_dir = os.path.join(home, '.openclaw')

    env = {
        'name': 'openclaw',
        'display': 'OpenClaw',
        'detected': False,
        'bin_path': '',
        'installed': False,
        'version': '',
        'config_dir': config_dir,
        }

    bin_path = shutil.which('openclaw')
    if bin_path:
        env['detected'] = True
        env['bin_path'] = bin_path
    if Path(config_dir).exists():
        env['detected'] = True

    skill_path = os.path.join(config_dir, 'skills', 'memman', 'SKILL.md')
    if Path(skill_path).exists():
        env['installed'] = True

    if env['bin_path']:
        try:
            out = subprocess.check_output(
                [env['bin_path'], '--version'],
                timeout=5, stderr=subprocess.DEVNULL)
            env['version'] = clean_version(out.decode().strip())
        except Exception:
            pass

    return env


def detect_environments() -> list[dict]:
    """Probe for all supported LLM CLI environments."""
    return [
        _detect_claude(),
        _detect_openclaw(),
        ]
