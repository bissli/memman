"""Environment detection for LLM CLI integrations.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from memman.setup.deploy import is_asset_link

logger = logging.getLogger('memman')


def home_dir() -> str:
    """Return the user's home directory.
    """
    return str(Path.home())


def clean_version(v: str) -> str:
    """Strip parenthesized suffixes like '(Claude Code)' from version strings.
    """
    idx = v.find(' (')
    if idx > 0:
        return v[:idx]
    return v


def detect_claude_code() -> dict:
    """Probe for the Claude Code CLI environment.

    Returns
    -------
    dict
        Environment dict with keys display, detected, bin_path,
        version, config_dir.
    """
    return _detect_cli('claude', 'Claude Code',
                       os.path.join(home_dir(), '.claude'))


def detect_codex() -> dict:
    """Detect Codex via its CLI, config directory, or installed skill.

    CODEX_HOME changes the config location; user skills live under
    ~/.agents/skills independently of that setting.
    """
    config_dir = str(Path(os.environ.get('CODEX_HOME') or
                          os.path.join(home_dir(), '.codex')).expanduser())
    env = _detect_cli('codex', 'Codex', config_dir)
    env['skills_dir'] = os.path.join(home_dir(), '.agents', 'skills')
    if is_asset_link('codex', Path(env['skills_dir']) / 'memman'):
        env['detected'] = True
    return env


def _detect_cli(binary: str, display: str, config_dir: str) -> dict:
    """Probe a CLI without requiring its version command to succeed."""
    env = {
        'display': display,
        'detected': False,
        'bin_path': '',
        'version': '',
        'config_dir': config_dir,
        }

    bin_path = shutil.which(binary)
    if bin_path:
        env['detected'] = True
        env['bin_path'] = bin_path
    if Path(config_dir).exists():
        env['detected'] = True

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
