"""Background-scheduler setup for memman enrich worker.

Detects platform (systemd on Linux, launchd on macOS) and writes the
appropriate user-scope unit / plist that runs `memman enrich --pending`
every 15 min. Units handle sleep/power-off catch-up natively.

The scheduler path always routes through OpenRouter with ZDR enforced.
Both OPENROUTER_API_KEY and VOYAGE_API_KEY are written to `~/.memman/env`
at mode 600 and referenced by EnvironmentFile (systemd) or sourced via
a wrapper script (launchd).
"""

import platform
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYSTEMD_TIMER_NAME = 'memman-enrich.timer'
SYSTEMD_SERVICE_NAME = 'memman-enrich.service'
LAUNCHD_LABEL = 'com.memman.enrich'
ENV_FILENAME = 'env'
DEFAULT_INTERVAL_SECONDS = 900


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
            'memman binary not on PATH; install with pipx')
    return path


def install(data_dir: str,
            openrouter_api_key: str,
            voyage_api_key: str,
            interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> dict:
    """Install the scheduler unit for the current platform.

    Writes both API keys to ~/.memman/env at mode 600 (merging with any
    existing keys) and installs the timer/plist that runs
    `memman enrich --pending` at the given interval.
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError(
            f'no supported scheduler detected on platform {platform.system()!r};'
            ' expected systemd (Linux) or launchd (macOS)')
    binary = memman_binary_path()

    env_actions = _write_env_file(openrouter_api_key, voyage_api_key)

    if kind == 'systemd':
        result = _install_systemd(binary, data_dir, interval_seconds)
    else:
        result = _install_launchd(binary, data_dir, interval_seconds)

    result['env_actions'] = env_actions
    return result


def _env_file_path() -> Path:
    return Path.home() / '.memman' / ENV_FILENAME


def _write_env_file(openrouter_api_key: str,
                    voyage_api_key: str) -> list[str]:
    """Write ~/.memman/env at mode 600 with provider + embedding keys.

    Atomic: writes to a .tmp sibling at mode 600 then os.replace() to the
    final path so a concurrent reader never sees a mode-644 or partial file.
    """
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
    existing['VOYAGE_API_KEY'] = voyage_api_key

    contents = '\n'.join(f'{k}={v}' for k, v in existing.items()) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(contents)
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)
    return [f'wrote {path} (mode 600, atomic)']


def uninstall() -> dict:
    """Remove the scheduler unit for the current platform."""
    kind = detect_scheduler()
    if not kind:
        return {'platform': 'unknown', 'actions': []}
    if kind == 'systemd':
        return _uninstall_systemd()
    return _uninstall_launchd()


def start() -> dict:
    """Resume the installed scheduler unit (no filesystem changes).

    Runs `systemctl --user enable --now` or `launchctl load -w` on the
    already-installed unit. Raises FileNotFoundError if the unit isn't
    installed yet (user should run `memman install` first). Raises
    RuntimeError if the scheduler fails to become active within a short
    poll window — catches silent-no-op environments (WSL2 without
    linger, containers, CI).
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError(
            'no supported scheduler on this platform')
    if kind == 'systemd':
        timer_path = _systemd_unit_dir() / SYSTEMD_TIMER_NAME
        if not timer_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {timer_path};'
                " run 'memman install' first")
        subprocess.run(
            ['systemctl', '--user', 'enable', '--now',
             SYSTEMD_TIMER_NAME], check=False)
        _verify_systemd_active()
        return {
            'platform': 'systemd',
            'actions': [
                'systemctl --user enable --now memman-enrich.timer'],
            }
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    if not plist_path.exists():
        raise FileNotFoundError(
            f'scheduler unit not installed at {plist_path};'
            " run 'memman install' first")
    subprocess.run(
        ['launchctl', 'load', '-w', str(plist_path)], check=False)
    _verify_launchd_loaded()
    return {
        'platform': 'launchd',
        'actions': [f'launchctl load -w {plist_path}'],
        }


def stop() -> dict:
    """Pause the scheduler unit without removing its files."""
    kind = detect_scheduler()
    if not kind:
        return {'platform': 'unknown', 'actions': []}
    if kind == 'systemd':
        timer_path = _systemd_unit_dir() / SYSTEMD_TIMER_NAME
        if not timer_path.exists():
            return {'platform': 'systemd', 'actions': [],
                    'note': 'not installed'}
        subprocess.run(
            ['systemctl', '--user', 'disable', '--now',
             SYSTEMD_TIMER_NAME], check=False)
        return {
            'platform': 'systemd',
            'actions': [
                'systemctl --user disable --now memman-enrich.timer'],
            }
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    if not plist_path.exists():
        return {'platform': 'launchd', 'actions': [],
                'note': 'not installed'}
    subprocess.run(
        ['launchctl', 'unload', str(plist_path)], check=False)
    return {
        'platform': 'launchd',
        'actions': [f'launchctl unload {plist_path}'],
        }


def trigger() -> dict:
    """Run the installed scheduler unit once, immediately.

    Kicks the installed service so the drain runs under the scheduler's
    environment and logs land where scheduled runs land. Does not
    disturb the timer's next-fire schedule. Raises FileNotFoundError if
    the unit file is absent (run `memman setup` first).
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError('no supported scheduler on this platform')
    if kind == 'systemd':
        service_path = _systemd_unit_dir() / SYSTEMD_SERVICE_NAME
        if not service_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {service_path};'
                " run 'memman setup' first")
        cmd = ['systemctl', '--user', 'start', '--no-block',
               SYSTEMD_SERVICE_NAME]
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False)
        if out.returncode != 0:
            stderr = (out.stderr or '').lower()
            if 'already' in stderr:
                return {
                    'platform': 'systemd',
                    'actions': [' '.join(cmd)],
                    'note': 'a scheduled run is already in progress;'
                            ' see `journalctl --user -u memman-enrich`',
                    }
            raise RuntimeError(
                f'systemctl start failed (rc={out.returncode}):'
                f' {out.stderr.strip() or out.stdout.strip()}')
        return {
            'platform': 'systemd',
            'actions': [' '.join(cmd)],
            'note': 'dispatched; see `journalctl --user -u memman-enrich`',
            }
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    if not plist_path.exists():
        raise FileNotFoundError(
            f'scheduler unit not installed at {plist_path};'
            " run 'memman setup' first")
    cmd = ['launchctl', 'start', LAUNCHD_LABEL]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        'platform': 'launchd',
        'actions': [' '.join(cmd)],
        'note': 'dispatched; see ~/.memman/logs/enrich.log',
        }


def _verify_systemd_active() -> None:
    """Poll systemctl is-active; raise if the timer isn't active."""
    try:
        out = subprocess.run(
            ['systemctl', '--user', 'is-active', SYSTEMD_TIMER_NAME],
            capture_output=True, text=True, check=False, timeout=5)
        state = out.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError(
            f'could not verify systemd timer state: {exc}') from exc
    if state != 'active':
        raise RuntimeError(
            f'systemd timer is {state!r} after enable;'
            ' check `journalctl --user -u memman-enrich` and confirm'
            ' `loginctl enable-linger` if this is a headless session')


def _verify_launchd_loaded() -> None:
    """Check launchctl list; raise if the job isn't loaded."""
    try:
        out = subprocess.run(
            ['launchctl', 'list', LAUNCHD_LABEL],
            capture_output=True, text=True, check=False, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError(
            f'could not verify launchd state: {exc}') from exc
    if out.returncode != 0:
        raise RuntimeError(
            f'launchd job {LAUNCHD_LABEL} is not loaded after'
            ' launchctl load; check ~/.memman/logs/enrich.err')


def _systemd_unit_dir() -> Path:
    return Path.home() / '.config' / 'systemd' / 'user'


def _launchd_agent_dir() -> Path:
    return Path.home() / 'Library' / 'LaunchAgents'


def _install_systemd(binary: str, data_dir: str,
                     interval_seconds: int) -> dict:
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

    env_file = _env_file_path()
    service_contents = (
        '[Unit]\n'
        'Description=MemMan enrichment worker\n\n'
        '[Service]\n'
        'Type=oneshot\n'
        f'Environment=MEMMAN_DATA_DIR={data_dir}\n'
        f'EnvironmentFile={env_file}\n'
        f'ExecStartPre=/bin/mkdir -p {Path.home()}/.memman/logs\n'
        f'ExecStart={binary} enrich --pending --timeout {exec_timeout}\n'
        'StandardOutput=append:%h/.memman/logs/enrich.log\n'
        'StandardError=append:%h/.memman/logs/enrich.err\n')

    unit_dir.mkdir(parents=True, exist_ok=True)
    timer_path.write_text(timer_contents)
    service_path.write_text(service_contents)
    actions = [f'wrote {timer_path}', f'wrote {service_path}']
    subprocess.run(
        ['systemctl', '--user', 'daemon-reload'], check=False)
    subprocess.run(
        ['systemctl', '--user', 'enable', '--now',
         SYSTEMD_TIMER_NAME], check=False)
    actions.append('systemctl --user enable --now memman-enrich.timer')
    _verify_systemd_active()

    return {
        'platform': 'systemd',
        'timer_path': str(timer_path),
        'service_path': str(service_path),
        'interval_seconds': interval_seconds,
        'actions': actions,
        }


def _uninstall_systemd() -> dict:
    """Disable the timer and remove unit files."""
    unit_dir = _systemd_unit_dir()
    timer_path = unit_dir / SYSTEMD_TIMER_NAME
    service_path = unit_dir / SYSTEMD_SERVICE_NAME
    actions = []
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
    return {'platform': 'systemd', 'actions': actions}


def _install_launchd(binary: str, data_dir: str,
                     interval_seconds: int) -> dict:
    """Write launchd plist and load it."""
    agent_dir = _launchd_agent_dir()
    plist_path = agent_dir / f'{LAUNCHD_LABEL}.plist'
    wrapper_path = Path.home() / '.memman' / 'bin' / 'memman-enrich-wrapper.sh'
    exec_timeout = max(60, interval_seconds - 20)

    env_file_q = shlex.quote(str(_env_file_path()))
    data_dir_q = shlex.quote(data_dir)
    binary_q = shlex.quote(binary)
    logs_dir = Path.home() / '.memman' / 'logs'
    logs_dir_q = shlex.quote(str(logs_dir))
    wrapper_contents = (
        '#!/bin/sh\n'
        f'mkdir -p {logs_dir_q}\n'
        f'[ -f {env_file_q} ] && . {env_file_q}\n'
        f'export MEMMAN_DATA_DIR={data_dir_q}\n'
        f'exec {binary_q} enrich --pending --timeout {exec_timeout}\n')

    log_path = logs_dir / 'enrich.log'
    err_path = logs_dir / 'enrich.err'
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
        f'  <key>StandardOutPath</key><string>{log_path}</string>\n'
        f'  <key>StandardErrorPath</key><string>{err_path}</string>\n'
        '</dict></plist>\n')

    agent_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(wrapper_contents)
    Path(wrapper_path).chmod(0o755)
    plist_path.write_text(plist_contents)
    actions = [
        f'wrote {wrapper_path} (mode 755)', f'wrote {plist_path}']
    subprocess.run(
        ['launchctl', 'unload', str(plist_path)], check=False)
    subprocess.run(
        ['launchctl', 'load', '-w', str(plist_path)], check=False)
    actions.append(f'launchctl load -w {plist_path}')
    _verify_launchd_loaded()

    return {
        'platform': 'launchd',
        'plist_path': str(plist_path),
        'wrapper_path': str(wrapper_path),
        'interval_seconds': interval_seconds,
        'actions': actions,
        }


def _uninstall_launchd() -> dict:
    """Unload plist and remove files."""
    agent_dir = _launchd_agent_dir()
    plist_path = agent_dir / f'{LAUNCHD_LABEL}.plist'
    wrapper_path = Path.home() / '.memman' / 'bin' / 'memman-enrich-wrapper.sh'
    actions = []
    if plist_path.exists():
        subprocess.run(
            ['launchctl', 'unload', str(plist_path)], check=False)
        plist_path.unlink()
        actions.append(f'removed {plist_path}')
    if wrapper_path.exists():
        wrapper_path.unlink()
        actions.append(f'removed {wrapper_path}')
    return {'platform': 'launchd', 'actions': actions}


def change_interval(data_dir: str, new_seconds: int) -> dict:
    """Rewrite the scheduler unit with a new interval.

    Does not touch ~/.memman/env. Requires the scheduler to already be
    installed (an env file with the API keys should exist).
    """
    if new_seconds < 60:
        raise RuntimeError(
            f'interval {new_seconds}s is too short; minimum is 60s')
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError(
            'no supported scheduler on this platform')
    binary = memman_binary_path()
    if kind == 'systemd':
        return _install_systemd(binary, data_dir, new_seconds)
    return _install_launchd(binary, data_dir, new_seconds)


def _parse_interval_from_systemd_timer(path: Path) -> int | None:
    """Extract OnUnitActiveSec from the systemd timer file."""
    if not path.exists():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith('OnUnitActiveSec='):
            value = line.split('=', 1)[1].strip()
            value = value.removesuffix('s')
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _parse_interval_from_launchd_plist(path: Path) -> int | None:
    """Extract StartInterval from the launchd plist file."""
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(
        r'<key>StartInterval</key>\s*<integer>(\d+)</integer>', text)
    if m:
        return int(m.group(1))
    return None


def _systemd_status() -> dict:
    """Collect systemd timer status."""
    unit_dir = _systemd_unit_dir()
    timer_path = unit_dir / SYSTEMD_TIMER_NAME
    service_path = unit_dir / SYSTEMD_SERVICE_NAME
    result = {
        'platform': 'systemd',
        'timer_path': str(timer_path),
        'service_path': str(service_path),
        'installed': timer_path.exists() and service_path.exists(),
        'enabled': False,
        'active': False,
        'next_run': None,
        'interval_seconds': _parse_interval_from_systemd_timer(timer_path),
        }
    if not result['installed']:
        return result

    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, check=False, timeout=5)
            return out.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ''

    enabled = _run(
        ['systemctl', '--user', 'is-enabled', SYSTEMD_TIMER_NAME])
    result['enabled'] = (enabled == 'enabled')
    active = _run(
        ['systemctl', '--user', 'is-active', SYSTEMD_TIMER_NAME])
    result['active'] = (active == 'active')
    last = _run([
        'systemctl', '--user', 'show',
        '--property=LastTriggerUSec', '--value',
        SYSTEMD_TIMER_NAME])
    if last and last != 'n/a' and result['interval_seconds']:
        last_dt = _parse_systemd_timestamp(last)
        if last_dt is not None:
            next_dt = last_dt + timedelta(
                seconds=result['interval_seconds'])
            result['next_run'] = next_dt.astimezone(
                timezone.utc).isoformat()
    return result


def _parse_systemd_timestamp(raw: str) -> datetime | None:
    """Parse a systemd wall-clock string like `Fri 2026-04-24 14:18:58 EDT`.

    Returns a timezone-aware datetime (system local tz) or None.
    """
    match = re.match(
        r'\S+\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\S+',
        raw.strip())
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        naive = datetime.fromisoformat(f'{date_str}T{time_str}')
    except ValueError:
        return None
    return naive.astimezone()


def _launchd_status() -> dict:
    """Collect launchd agent status."""
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    result = {
        'platform': 'launchd',
        'plist_path': str(plist_path),
        'installed': plist_path.exists(),
        'enabled': False,
        'active': False,
        'next_run': None,
        'interval_seconds': _parse_interval_from_launchd_plist(plist_path),
        }
    if not result['installed']:
        return result
    try:
        out = subprocess.run(
            ['launchctl', 'list', LAUNCHD_LABEL],
            capture_output=True, text=True, check=False, timeout=5)
        result['enabled'] = (out.returncode == 0)
        result['active'] = (out.returncode == 0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    log_path = Path.home() / '.memman' / 'logs' / 'enrich.log'
    if (result['active']
            and result['interval_seconds']
            and log_path.exists()):
        try:
            log_mtime = log_path.stat().st_mtime
            next_dt = datetime.fromtimestamp(
                log_mtime + result['interval_seconds'], tz=timezone.utc)
            result['next_run'] = next_dt.isoformat()
        except OSError:
            pass
    return result


def status() -> dict:
    """Return the scheduler's current status."""
    kind = detect_scheduler()
    if kind == 'systemd':
        return _systemd_status()
    if kind == 'launchd':
        return _launchd_status()
    return {
        'platform': 'unknown',
        'installed': False,
        'enabled': False,
        'active': False,
        'next_run': None,
        'interval_seconds': None,
        }
