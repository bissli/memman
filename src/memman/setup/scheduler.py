"""Background-scheduler setup for the memman drain worker.

Detects platform (systemd on Linux, launchd on macOS) and writes the
appropriate user-scope unit / plist that runs `memman scheduler drain
--pending` every 60 s. Units handle sleep/power-off catch-up natively.

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
STATE_FILENAME = 'scheduler.state'
DEFAULT_INTERVAL_SECONDS = 60

STATE_ACTIVE = 'active'
STATE_PAUSED = 'paused'
STATE_OFF = 'off'
VALID_STATES = (STATE_ACTIVE, STATE_PAUSED, STATE_OFF)

DEBUG_STATE_FILENAME = 'debug.state'
DEBUG_ON = 'on'
DEBUG_OFF = 'off'
VALID_DEBUG_STATES = (DEBUG_ON, DEBUG_OFF)


def _state_file_path() -> Path:
    """Return ~/.memman/scheduler.state. Per-host; never synced."""
    return Path.home() / '.memman' / STATE_FILENAME


def _enforce_data_dir_perms(data_dir: str) -> None:
    """Tighten ~/.memman, ~/.memman/logs, ~/.memman/data to 0700.

    Idempotent — safe to call from install or any setup path. The env
    file inside ~/.memman is written separately at 0600 by
    _write_env_file. SQLite databases under ~/.memman/data/ are owned
    by the user; the dir mode is what enforces "no other user can see
    insight content".
    """
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    base.chmod(0o700)
    for sub in ('logs', 'data'):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)


def read_state() -> str:
    """Read the scheduler intent state. Missing file -> 'active'."""
    path = _state_file_path()
    try:
        value = path.read_text().strip()
    except (OSError, FileNotFoundError):
        return STATE_ACTIVE
    return value if value in VALID_STATES else STATE_ACTIVE


def write_state(state: str) -> None:
    """Atomically persist the scheduler intent state."""
    if state not in VALID_STATES:
        raise ValueError(f'invalid scheduler state {state!r}')
    path = _state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(state + '\n')
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)


def clear_state() -> None:
    """Remove the state file if present (used on uninstall)."""
    path = _state_file_path()
    if path.exists():
        path.unlink()


def _debug_state_file_path() -> Path:
    """Return ~/.memman/debug.state. Per-host; never synced."""
    return Path.home() / '.memman' / DEBUG_STATE_FILENAME


def read_debug_state() -> str:
    """Read the debug trace intent state. Missing file -> 'off'."""
    path = _debug_state_file_path()
    try:
        value = path.read_text().strip()
    except (OSError, FileNotFoundError):
        return DEBUG_OFF
    return value if value in VALID_DEBUG_STATES else DEBUG_OFF


def write_debug_state(state: str) -> None:
    """Atomically persist the debug trace intent state."""
    if state not in VALID_DEBUG_STATES:
        raise ValueError(f'invalid debug state {state!r}')
    path = _debug_state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(state + '\n')
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)


def clear_debug_state() -> None:
    """Remove the debug state file if present (used on uninstall)."""
    path = _debug_state_file_path()
    if path.exists():
        path.unlink()


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
    `memman scheduler drain --pending` at the given interval.

    Tightens permissions on ~/.memman, ~/.memman/logs, ~/.memman/data
    to 0700 — the env file holds API keys (mode 0600 by atomic write),
    the data dir holds insight content, the logs dir holds worker
    output that may include LLM request payloads. Owner-only is the
    correct default for all three.
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError(
            f'no supported scheduler detected on platform {platform.system()!r};'
            ' expected systemd (Linux) or launchd (macOS)')
    binary = memman_binary_path()

    _enforce_data_dir_perms(data_dir)
    env_actions = _write_env_file(openrouter_api_key, voyage_api_key)

    if kind == 'systemd':
        result = _install_systemd(binary, data_dir, interval_seconds)
    else:
        result = _install_launchd(binary, data_dir, interval_seconds)

    write_state(STATE_ACTIVE)
    result['state'] = STATE_ACTIVE
    result['env_actions'] = env_actions
    return result


def _env_file_path() -> Path:
    return Path.home() / '.memman' / ENV_FILENAME


def _read_env_file() -> dict[str, str]:
    """Parse ~/.memman/env into a dict. Missing file -> empty dict."""
    path = _env_file_path()
    existing: dict[str, str] = {}
    if not path.exists():
        return existing
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        existing[k] = v
    return existing


def _write_env_keys(updates: dict[str, str],
                    removes: set[str] | None = None) -> list[str]:
    """Merge updates into ~/.memman/env, atomically, at mode 600.

    Preserves any keys already in the file that are not in updates or
    removes. Atomic: writes to a .tmp sibling at mode 600 then
    os.replace() so a concurrent reader never sees a partial file.
    """
    path = _env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_env_file()
    for k in (removes or ()):
        existing.pop(k, None)
    existing.update(updates)
    contents = '\n'.join(f'{k}={v}' for k, v in existing.items()) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(contents)
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)
    return [f'wrote {path} (mode 600, atomic)']


def _write_env_file(openrouter_api_key: str,
                    voyage_api_key: str) -> list[str]:
    """Write ~/.memman/env at mode 600 with provider + embedding keys."""
    return _write_env_keys({
        'MEMMAN_LLM_PROVIDER': 'openrouter',
        'OPENROUTER_API_KEY': openrouter_api_key,
        'VOYAGE_API_KEY': voyage_api_key,
        })


def set_debug(on: bool) -> list[str]:
    """Toggle debug trace state in ~/.memman/debug.state atomically."""
    value = DEBUG_ON if on else DEBUG_OFF
    write_debug_state(value)
    return [f'wrote {_debug_state_file_path()} = {value} (mode 600, atomic)']


def get_debug() -> bool:
    """Return True if ~/.memman/debug.state says 'on'."""
    return read_debug_state() == DEBUG_ON


def uninstall() -> dict:
    """Remove the scheduler unit for the current platform.

    Also clears both state files. This is the full teardown path used
    by `memman uninstall`; for a scheduler-only teardown that keeps the
    rest of memman intact, see `off()`.
    """
    clear_state()
    clear_debug_state()
    kind = detect_scheduler()
    if not kind:
        return {'platform': 'unknown', 'actions': []}
    if kind == 'systemd':
        return _uninstall_systemd()
    return _uninstall_launchd()


def resume() -> dict:
    """Transition to `active`: enable the installed scheduler unit.

    Raises FileNotFoundError if the unit isn't installed (user should
    run `memman install` first). Raises RuntimeError if the scheduler
    fails to become active within a short poll window — catches
    silent-no-op environments (WSL2 without linger, containers, CI).
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
        write_state(STATE_ACTIVE)
        return {
            'platform': 'systemd',
            'state': STATE_ACTIVE,
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
    write_state(STATE_ACTIVE)
    return {
        'platform': 'launchd',
        'state': STATE_ACTIVE,
        'actions': [f'launchctl load -w {plist_path}'],
        }


def pause() -> dict:
    """Transition to `paused`: stop timer firing, keep unit files.

    Raises FileNotFoundError if the unit isn't installed — `pause` is
    meaningless without units. Use `off` to represent the no-units
    state explicitly.
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError('no supported scheduler on this platform')
    if kind == 'systemd':
        timer_path = _systemd_unit_dir() / SYSTEMD_TIMER_NAME
        if not timer_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {timer_path};'
                " run 'memman install' first")
        subprocess.run(
            ['systemctl', '--user', 'stop', SYSTEMD_TIMER_NAME],
            check=False, capture_output=True)
        subprocess.run(
            ['systemctl', '--user', 'disable', SYSTEMD_TIMER_NAME],
            check=False, capture_output=True)
        write_state(STATE_PAUSED)
        return {
            'platform': 'systemd',
            'state': STATE_PAUSED,
            'actions': [
                'systemctl --user stop memman-enrich.timer',
                'systemctl --user disable memman-enrich.timer'],
            }
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    if not plist_path.exists():
        raise FileNotFoundError(
            f'scheduler unit not installed at {plist_path};'
            " run 'memman install' first")
    subprocess.run(
        ['launchctl', 'unload', '-w', str(plist_path)], check=False)
    write_state(STATE_PAUSED)
    return {
        'platform': 'launchd',
        'state': STATE_PAUSED,
        'actions': [f'launchctl unload -w {plist_path}'],
        }


def off() -> dict:
    """Transition to `off`: remove scheduler unit files; keep integrations.

    Queue protection (pending rows refused unless force/drain) is
    enforced by the caller — CLI handles the `--force` / `--drain-first`
    flags and calls this after the guard.
    """
    kind = detect_scheduler()
    if not kind:
        write_state(STATE_OFF)
        return {'platform': 'unknown', 'state': STATE_OFF, 'actions': []}
    if kind == 'systemd':
        result = _uninstall_systemd()
    else:
        result = _uninstall_launchd()
    result['state'] = STATE_OFF
    write_state(STATE_OFF)
    return result


def reconcile() -> dict:
    """Detect actual OS scheduler state and rewrite the state file."""
    s = status()
    if not s.get('installed'):
        detected = STATE_OFF
    elif s.get('enabled') or s.get('active'):
        detected = STATE_ACTIVE
    else:
        detected = STATE_PAUSED
    write_state(detected)
    return {
        'platform': s.get('platform'),
        'state': detected,
        'actions': [f'wrote {_state_file_path()} = {detected}'],
        }


def trigger() -> dict:
    """Run the installed scheduler unit once, immediately.

    Kicks the installed service so the drain runs under the scheduler's
    environment and logs land where scheduled runs land. Does not
    disturb the timer's next-fire schedule. Raises FileNotFoundError if
    the unit file is absent (run `memman install` first).
    """
    kind = detect_scheduler()
    if not kind:
        raise RuntimeError('no supported scheduler on this platform')
    if kind == 'systemd':
        service_path = _systemd_unit_dir() / SYSTEMD_SERVICE_NAME
        if not service_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {service_path};'
                " run 'memman install' first")
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
                            ' see `memman scheduler logs`',
                    }
            raise RuntimeError(
                f'systemctl start failed (rc={out.returncode}):'
                f' {out.stderr.strip() or out.stdout.strip()}')
        return {
            'platform': 'systemd',
            'actions': [' '.join(cmd)],
            'note': 'dispatched; see `memman scheduler logs`',
            }
    plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
    if not plist_path.exists():
        raise FileNotFoundError(
            f'scheduler unit not installed at {plist_path};'
            " run 'memman install' first")
    cmd = ['launchctl', 'start', LAUNCHD_LABEL]
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(
            f'launchctl start failed (rc={out.returncode}):'
            f' {out.stderr.strip() or out.stdout.strip()}')
    return {
        'platform': 'launchd',
        'actions': [' '.join(cmd)],
        'note': 'dispatched; see `memman scheduler logs`',
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
        f'OnActiveSec={interval_seconds}s\n'
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
        f'Environment="MEMMAN_DATA_DIR={data_dir}"\n'
        'Environment=MEMMAN_WORKER=1\n'
        f'EnvironmentFile={env_file}\n'
        f'ExecStartPre=/bin/mkdir -p {Path.home()}/.memman/logs\n'
        f'ExecStart={binary} scheduler drain --pending --timeout {exec_timeout}\n'
        'StandardOutput=append:%h/.memman/logs/enrich.log\n'
        'StandardError=append:%h/.memman/logs/enrich.err\n')

    unit_dir.mkdir(parents=True, exist_ok=True)
    timer_path.write_text(timer_contents)
    service_path.write_text(service_contents)
    actions = [f'wrote {timer_path}', f'wrote {service_path}']
    subprocess.run(
        ['systemctl', '--user', 'daemon-reload'], check=False)
    subprocess.run(
        ['systemctl', '--user', 'enable', SYSTEMD_TIMER_NAME],
        check=False, capture_output=True)
    actions.append('systemctl --user enable memman-enrich.timer')
    # restart (not just `enable --now`) ensures a freshly-rearmed timer
    # when re-installing over an already-active unit; `enable --now` is
    # a no-op on a running timer, leaving the schedule based on the
    # pre-reload state.
    subprocess.run(
        ['systemctl', '--user', 'restart', SYSTEMD_TIMER_NAME],
        check=False, capture_output=True)
    actions.append('systemctl --user restart memman-enrich.timer')
    _verify_systemd_active()

    return {
        'platform': 'systemd',
        'timer_path': str(timer_path),
        'service_path': str(service_path),
        'interval_seconds': interval_seconds,
        'actions': actions,
        }


def _uninstall_systemd() -> dict:
    """Stop the timer, disable the unit, and remove unit files.

    Splits stop+disable into separate calls instead of `disable --now`
    because some systemd versions surface a benign-but-noisy
    `DisableUnitFilesWithFlagsAndInstallInfo` dbus error from the
    combined path; the split path is universally compatible. Stderr is
    captured so transient dbus chatter never reaches the user.
    """
    unit_dir = _systemd_unit_dir()
    timer_path = unit_dir / SYSTEMD_TIMER_NAME
    service_path = unit_dir / SYSTEMD_SERVICE_NAME
    actions = []
    subprocess.run(
        ['systemctl', '--user', 'stop', SYSTEMD_TIMER_NAME],
        check=False, capture_output=True)
    actions.append('systemctl --user stop memman-enrich.timer')
    subprocess.run(
        ['systemctl', '--user', 'disable', SYSTEMD_TIMER_NAME],
        check=False, capture_output=True)
    actions.append('systemctl --user disable memman-enrich.timer')
    for p in (timer_path, service_path):
        if p.exists():
            p.unlink()
            actions.append(f'removed {p}')
    subprocess.run(
        ['systemctl', '--user', 'daemon-reload'],
        check=False, capture_output=True)
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
        'export MEMMAN_WORKER=1\n'
        f'exec {binary_q} scheduler drain --pending --timeout {exec_timeout}\n')

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
    installed (an env file with the API keys should exist). Preserves
    the prior enabled/disabled state: if the scheduler was disabled
    before the call, it is re-disabled after the unit is rewritten.
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
        was_enabled = _systemd_is_enabled()
        result = _install_systemd(binary, data_dir, new_seconds)
        if not was_enabled:
            subprocess.run(
                ['systemctl', '--user', 'stop', SYSTEMD_TIMER_NAME],
                check=False, capture_output=True)
            subprocess.run(
                ['systemctl', '--user', 'disable', SYSTEMD_TIMER_NAME],
                check=False, capture_output=True)
            result['actions'].append(
                'systemctl --user stop+disable memman-enrich.timer'
                ' (restored prior disabled state)')
        return result
    was_loaded = _launchd_is_loaded()
    result = _install_launchd(binary, data_dir, new_seconds)
    if not was_loaded:
        plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
        subprocess.run(
            ['launchctl', 'unload', '-w', str(plist_path)], check=False)
        result['actions'].append(
            f'launchctl unload -w {plist_path}'
            ' (restored prior unloaded state)')
    return result


def _systemd_is_enabled() -> bool:
    """True if the systemd timer is currently enabled."""
    try:
        out = subprocess.run(
            ['systemctl', '--user', 'is-enabled', SYSTEMD_TIMER_NAME],
            capture_output=True, text=True, check=False, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return out.returncode == 0 and out.stdout.strip() == 'enabled'


def _launchd_is_loaded() -> bool:
    """True if the launchd agent is currently loaded."""
    try:
        out = subprocess.run(
            ['launchctl', 'list', LAUNCHD_LABEL],
            capture_output=True, text=True, check=False, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return out.returncode == 0


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
    """Return the scheduler's current status, including tri-state.

    Fields:
      - platform / installed / enabled / active / next_run /
        interval_seconds — OS truth.
      - state — persisted user intent (active | paused | off).
      - drift — True when state disagrees with OS truth.
    """
    kind = detect_scheduler()
    if kind == 'systemd':
        result = _systemd_status()
    elif kind == 'launchd':
        result = _launchd_status()
    else:
        result = {
            'platform': 'unknown',
            'installed': False,
            'enabled': False,
            'active': False,
            'next_run': None,
            'interval_seconds': None,
            }
    intent = read_state()
    result['state'] = intent
    if intent == STATE_OFF:
        expected_installed = False
        expected_enabled = False
    elif intent == STATE_PAUSED:
        expected_installed = True
        expected_enabled = False
    else:
        expected_installed = True
        expected_enabled = True
    result['drift'] = (
        bool(result.get('installed')) != expected_installed
        or bool(result.get('enabled')) != expected_enabled)
    return result
