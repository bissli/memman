"""Background-scheduler setup for memman enrich worker.

Detects platform (systemd on Linux, launchd on macOS) and writes the
appropriate user-scope unit / plist that runs `memman enrich --pending`
on a recurring interval. Units handle sleep/power-off catch-up natively.

The scheduler path always routes through OpenRouter with ZDR enforced.
The OpenRouter API key is written to `~/.memman/env` at mode 600 and
referenced by EnvironmentFile (systemd) or sourced via a wrapper
(launchd).
"""

import platform
import shutil
import subprocess
from pathlib import Path

SYSTEMD_TIMER_NAME = 'memman-enrich.timer'
SYSTEMD_SERVICE_NAME = 'memman-enrich.service'
LAUNCHD_LABEL = 'com.memman.enrich'
ENV_FILENAME = 'env'


def detect_scheduler() -> str:
    """Return 'systemd' on Linux with systemd, 'launchd' on macOS, else ''."""
    system = platform.system()
    if system == 'Darwin':
        return 'launchd'
    if system == 'Linux':
        if shutil.which('systemctl') and Path('/run/systemd/system').exists():
            return 'systemd'
    return ''


def memman_binary_path() -> str:
    """Return the absolute path to the memman binary."""
    path = shutil.which('memman')
    if not path:
        raise RuntimeError(
            'memman binary not on PATH; install with pipx or make install')
    return path


def install(data_dir: str, interval_seconds: int = 900,
            openrouter_api_key: str | None = None,
            dry_run: bool = False) -> dict:
    """Install the scheduler unit for the current platform.

    If openrouter_api_key is given, it is written to ~/.memman/env at
    mode 600. If None, the existing env file (if any) is left unchanged
    and the scheduler relies on whatever env the user has configured.
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError(
            f'no supported scheduler detected on platform {platform.system()!r};'
            ' expected systemd (Linux) or launchd (macOS)')
    binary = memman_binary_path()

    env_actions = []
    if openrouter_api_key and not dry_run:
        env_actions = _write_env_file(openrouter_api_key)

    if kind == 'systemd':
        result = _install_systemd(binary, data_dir, interval_seconds, dry_run)
    else:
        result = _install_launchd(binary, data_dir, interval_seconds, dry_run)

    result['env_actions'] = env_actions
    return result


def _env_file_path() -> Path:
    return Path.home() / '.memman' / ENV_FILENAME


def _write_env_file(openrouter_api_key: str) -> list[str]:
    """Write ~/.memman/env at mode 600 with OpenRouter provider vars."""
    path = _env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            existing[k] = v

    existing['MEMMAN_LLM_PROVIDER'] = 'openrouter'
    existing['OPENROUTER_API_KEY'] = openrouter_api_key

    contents = '\n'.join(f'{k}={v}' for k, v in existing.items()) + '\n'
    path.write_text(contents)
    Path(path).chmod(0o600)
    return [f'wrote {path} (mode 600)']


def uninstall(dry_run: bool = False) -> dict:
    """Remove the scheduler unit for the current platform."""
    kind = detect_scheduler()
    if not kind:
        return {'platform': 'unknown', 'action': 'noop'}
    if kind == 'systemd':
        return _uninstall_systemd(dry_run)
    return _uninstall_launchd(dry_run)


def _systemd_unit_dir() -> Path:
    return Path.home() / '.config' / 'systemd' / 'user'


def _launchd_agent_dir() -> Path:
    return Path.home() / 'Library' / 'LaunchAgents'


def _install_systemd(binary: str, data_dir: str,
                     interval_seconds: int,
                     dry_run: bool) -> dict:
    """Write systemd timer+service units and enable the timer."""
    unit_dir = _systemd_unit_dir()
    timer_path = unit_dir / SYSTEMD_TIMER_NAME
    service_path = unit_dir / SYSTEMD_SERVICE_NAME
    exec_timeout = max(60, interval_seconds - 20)

    timer_contents = (
        '[Unit]\n'
        'Description=MemMan background enrichment timer\n\n'
        '[Timer]\n'
        'OnBootSec=2min\n'
        f'OnUnitActiveSec={interval_seconds}s\n'
        'Persistent=true\n\n'
        '[Install]\n'
        'WantedBy=timers.target\n')

    env_file = Path.home() / '.memman' / 'env'
    service_contents = (
        '[Unit]\n'
        'Description=MemMan enrichment worker\n\n'
        '[Service]\n'
        'Type=oneshot\n'
        f'Environment=MEMMAN_DATA_DIR={data_dir}\n'
        f'EnvironmentFile=-{env_file}\n'
        f'ExecStart={binary} enrich --pending --timeout {exec_timeout}\n'
        'StandardOutput=journal\n'
        'StandardError=journal\n')

    actions = []
    if not dry_run:
        unit_dir.mkdir(parents=True, exist_ok=True)
        timer_path.write_text(timer_contents)
        service_path.write_text(service_contents)
        actions.extend((f'wrote {timer_path}', f'wrote {service_path}'))
        subprocess.run(
            ['systemctl', '--user', 'daemon-reload'], check=False)
        subprocess.run(
            ['systemctl', '--user', 'enable', '--now',
             SYSTEMD_TIMER_NAME], check=False)
        actions.append('systemctl --user enable --now memman-enrich.timer')

    return {
        'platform': 'systemd',
        'timer_path': str(timer_path),
        'service_path': str(service_path),
        'interval_seconds': interval_seconds,
        'dry_run': dry_run,
        'actions': actions,
        }


def _uninstall_systemd(dry_run: bool) -> dict:
    """Disable the timer and remove unit files."""
    unit_dir = _systemd_unit_dir()
    timer_path = unit_dir / SYSTEMD_TIMER_NAME
    service_path = unit_dir / SYSTEMD_SERVICE_NAME
    actions = []
    if not dry_run:
        subprocess.run(
            ['systemctl', '--user', 'disable', '--now',
             SYSTEMD_TIMER_NAME], check=False)
        actions.append('systemctl --user disable --now memman-enrich.timer')
        for p in (timer_path, service_path):
            if p.exists():
                p.unlink()
                actions.append(f'removed {p}')
        subprocess.run(
            ['systemctl', '--user', 'daemon-reload'], check=False)
    return {'platform': 'systemd', 'dry_run': dry_run, 'actions': actions}


def _install_launchd(binary: str, data_dir: str,
                     interval_seconds: int,
                     dry_run: bool) -> dict:
    """Write launchd plist and load it."""
    agent_dir = _launchd_agent_dir()
    plist_path = agent_dir / f'{LAUNCHD_LABEL}.plist'
    wrapper_path = Path.home() / '.memman' / 'bin' / 'memman-enrich-wrapper.sh'
    exec_timeout = max(60, interval_seconds - 20)

    wrapper_contents = (
        '#!/bin/sh\n'
        '# Source env file (if present) so OPENROUTER_API_KEY etc. flow in\n'
        f'[ -f "{Path.home()}/.memman/env" ] && . "{Path.home()}/.memman/env"\n'
        f'export MEMMAN_DATA_DIR="{data_dir}"\n'
        f'exec "{binary}" enrich --pending --timeout {exec_timeout}\n')

    plist_contents = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
        ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f'  <key>Label</key><string>{LAUNCHD_LABEL}</string>\n'
        '  <key>ProgramArguments</key>\n'
        '  <array>\n'
        f'    <string>{wrapper_path}</string>\n'
        '  </array>\n'
        f'  <key>StartInterval</key><integer>{interval_seconds}</integer>\n'
        '  <key>RunAtLoad</key><true/>\n'
        '  <key>StandardOutPath</key>'
        '<string>/tmp/memman-enrich.log</string>\n'
        '  <key>StandardErrorPath</key>'
        '<string>/tmp/memman-enrich.err</string>\n'
        '</dict></plist>\n')

    actions = []
    if not dry_run:
        agent_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(wrapper_contents)
        Path(wrapper_path).chmod(0o755)
        plist_path.write_text(plist_contents)
        actions.extend((f'wrote {wrapper_path} (mode 755)', f'wrote {plist_path}'))
        subprocess.run(
            ['launchctl', 'unload', str(plist_path)], check=False)
        subprocess.run(
            ['launchctl', 'load', '-w', str(plist_path)], check=False)
        actions.append(f'launchctl load -w {plist_path}')

    return {
        'platform': 'launchd',
        'plist_path': str(plist_path),
        'wrapper_path': str(wrapper_path),
        'interval_seconds': interval_seconds,
        'dry_run': dry_run,
        'actions': actions,
        }


def _uninstall_launchd(dry_run: bool) -> dict:
    """Unload plist and remove files."""
    agent_dir = _launchd_agent_dir()
    plist_path = agent_dir / f'{LAUNCHD_LABEL}.plist'
    wrapper_path = Path.home() / '.memman' / 'bin' / 'memman-enrich-wrapper.sh'
    actions = []
    if not dry_run:
        if plist_path.exists():
            subprocess.run(
                ['launchctl', 'unload', str(plist_path)], check=False)
            plist_path.unlink()
            actions.append(f'removed {plist_path}')
        if wrapper_path.exists():
            wrapper_path.unlink()
            actions.append(f'removed {wrapper_path}')
    return {'platform': 'launchd', 'dry_run': dry_run, 'actions': actions}
