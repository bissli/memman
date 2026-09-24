"""Environment detection for LLM CLI integrations."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger('memman')


def home_dir() -> str:
    """Return the user's home directory."""
    return str(Path.home())


def clean_version(v: str) -> str:
    """Strip parenthesized suffixes like '(Claude Code)' from version strings."""
    idx = v.find(' (')
    if idx > 0:
        return v[:idx]
    return v


def detect_claude_code() -> dict:
    """Probe for the Claude Code CLI environment.

    Returns
    -------
    dict
        Environment dict with keys name, display, detected, bin_path,
        installed, version, config_dir.
    """
    config_dir = os.path.join(home_dir(), '.claude')
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
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug(
                f'version probe failed for {env["bin_path"]}: {exc}')

    return env
