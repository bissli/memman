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


def _probe_cli_environment(name: str, display: str, bin_name: str,
                           config_dir: str) -> dict:
    """Probe for a CLI integration with a `.<name>/skills/memman/SKILL.md` layout.

    Claude Code and OpenClaw share an identical detection shape:
    look for the binary on PATH, look for the config dir, probe
    `--version` if the binary exists, and check for an installed
    skill marker. Returns an environment dict with the same fields
    in both cases.
    """
    env = {
        'name': name,
        'display': display,
        'detected': False,
        'bin_path': '',
        'installed': False,
        'version': '',
        'config_dir': config_dir,
        }

    bin_path = shutil.which(bin_name)
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


def detect_environments() -> list[dict]:
    """Probe for all supported LLM CLI environments.

    NanoClaw is a containerised agent platform; detection requires
    (a) Docker available on the host and (b) a top-level `nanoclaw/`
    directory in the current working directory. `.claude/` alone is
    not a NanoClaw signal -- every Claude Code project has one. Pass
    `--target nanoclaw` explicitly to install ahead of time when no
    project marker is present.
    """
    home = home_dir()
    claude = _probe_cli_environment(
        'claude-code', 'Claude Code', 'claude',
        os.path.join(home, '.claude'))
    openclaw = _probe_cli_environment(
        'openclaw', 'OpenClaw', 'openclaw',
        os.path.join(home, '.openclaw'))

    nanoclaw_config_dir = str(
        Path.home() / '.claude' / 'skills' / 'add-memman')
    nanoclaw = {
        'name': 'nanoclaw',
        'display': 'NanoClaw',
        'detected': False,
        'bin_path': '',
        'installed': False,
        'version': '',
        'config_dir': nanoclaw_config_dir,
        }
    docker_path = shutil.which('docker')
    if docker_path and Path('nanoclaw').exists():
        nanoclaw['detected'] = True
        nanoclaw['bin_path'] = docker_path
    if (Path(nanoclaw_config_dir) / 'SKILL.md').exists():
        nanoclaw['installed'] = True

    return [claude, openclaw, nanoclaw]
