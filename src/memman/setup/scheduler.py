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
SERVE_INTERVAL_FILENAME = 'scheduler.serve_interval'
SCHEDULER_KIND_ENV = 'MEMMAN_SCHEDULER_KIND'
SCHEDULER_KIND_SERVE = 'serve'
DEBUG_STATE_FILENAME = 'debug.state'
DEFAULT_INTERVAL_SECONDS = 60

STATE_STARTED = 'started'
STATE_STOPPED = 'stopped'

DEBUG_ON = 'on'
DEBUG_OFF = 'off'
VALID_DEBUG_STATES = (DEBUG_ON, DEBUG_OFF)


def _state_file_path() -> Path:
    """Return ~/.memman/scheduler.state. Per-host; never synced."""
    return Path.home() / '.memman' / STATE_FILENAME


def _enforce_data_dir_perms(data_dir: str) -> None:
    """Tighten ~/.memman, ~/.memman/logs, ~/.memman/data to 0700.
    """
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    base.chmod(0o700)
    for sub in ('logs', 'data'):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)


def read_state() -> str:
    """Return STATE_STARTED iff state file says 'started', else STATE_STOPPED.
    """
    path = _state_file_path()
    try:
        value = path.read_text().strip()
    except (OSError, FileNotFoundError):
        return STATE_STOPPED
    return STATE_STARTED if value == STATE_STARTED else STATE_STOPPED


def write_state(state: str) -> None:
    """Atomically persist the scheduler intent state."""
    if state not in {STATE_STARTED, STATE_STOPPED}:
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


def _serve_interval_path() -> Path:
    """Return ~/.memman/scheduler.serve_interval. Per-host; never synced."""
    return Path.home() / '.memman' / SERVE_INTERVAL_FILENAME


def write_serve_interval(seconds: int) -> None:
    """Persist the active serve loop's interval for status/doctor reads."""
    path = _serve_interval_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(f'{int(seconds)}\n')
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)


def read_serve_interval() -> int | None:
    """Read the persisted serve interval, or None if absent/unreadable."""
    path = _serve_interval_path()
    try:
        raw = path.read_text().strip()
    except (OSError, FileNotFoundError):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def clear_serve_interval() -> None:
    """Remove the serve interval file if present."""
    path = _serve_interval_path()
    if path.exists():
        path.unlink()


def _debug_state_file_path() -> Path:
    """Return ~/.memman/debug.state. Per-host; never synced."""
    return Path.home() / '.memman' / DEBUG_STATE_FILENAME


def read_debug_state() -> str:
    """Read the persistent debug-trace state. Missing file -> 'off'."""
    path = _debug_state_file_path()
    try:
        value = path.read_text().strip()
    except (OSError, FileNotFoundError):
        return DEBUG_OFF
    return value if value in VALID_DEBUG_STATES else DEBUG_OFF


def write_debug_state(state: str) -> None:
    """Atomically persist the debug-trace state."""
    if state not in VALID_DEBUG_STATES:
        raise ValueError(f'invalid debug state {state!r}')
    path = _debug_state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(state + '\n')
    Path(tmp).chmod(0o600)
    Path(tmp).replace(path)


def clear_debug_state() -> None:
    """Remove the debug-state file (used on uninstall)."""
    path = _debug_state_file_path()
    if path.exists():
        path.unlink()


def set_debug(on: bool) -> list[str]:
    """Toggle the persistent debug-trace flag in ~/.memman/debug.state."""
    value = DEBUG_ON if on else DEBUG_OFF
    write_debug_state(value)
    return [f'wrote {_debug_state_file_path()} = {value} (mode 600, atomic)']


def get_debug() -> bool:
    """Return True if ~/.memman/debug.state says 'on'."""
    return read_debug_state() == DEBUG_ON


def detect_scheduler() -> str:
    """Return 'systemd', 'launchd', or 'serve' for this host.

    Resolution order:
    1. `MEMMAN_SCHEDULER_KIND=serve` -> 'serve' (explicit opt-in for
       containers and any host that runs `memman scheduler serve` itself).
    2. macOS -> 'launchd'.
    3. Linux with systemctl + /run/systemd/system -> 'systemd'.

    Raises `RuntimeError` on hosts that match none of the above. Set
    `MEMMAN_SCHEDULER_KIND=serve` and run `memman scheduler serve`, or
    install systemd/launchd integration.
    """
    import os as _os
    if _os.environ.get(SCHEDULER_KIND_ENV) == SCHEDULER_KIND_SERVE:
        return SCHEDULER_KIND_SERVE
    system = platform.system()
    if system == 'Darwin':
        return 'launchd'
    if system == 'Linux':
        if shutil.which('systemctl') and Path('/run/systemd/system').exists():
            return 'systemd'
    raise RuntimeError(
        'no scheduler available on this host'
        f' (platform={system!r});'
        f' set {SCHEDULER_KIND_ENV}=serve and run'
        ' `memman scheduler serve`, or install systemd / launchd')


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
    """Install the scheduler trigger for the current environment.

    Writes both API keys to ~/.memman/env at mode 600 (merging with any
    existing keys) and installs the trigger that runs
    `memman scheduler drain --pending` at the given interval.

    Trigger by environment:
      - systemd (Linux host) -> user timer + service
      - launchd (Mac host) -> launch agent plist
      - serve (`MEMMAN_SCHEDULER_KIND=serve`) -> records the configured
        interval and expects the operator to run `memman scheduler serve`
        themselves (typically as PID 1 in a container).

    Always ends with state file = STATE_STARTED so the install path
    never leaves the user with installed-but-stopped state.
    """
    binary = memman_binary_path()

    _enforce_data_dir_perms(data_dir)
    env_actions = _write_env_file(openrouter_api_key, voyage_api_key)

    kind = detect_scheduler()
    if kind == 'systemd':
        result = _install_systemd(binary, data_dir, interval_seconds)
    elif kind == 'launchd':
        result = _install_launchd(binary, data_dir, interval_seconds)
    elif kind == SCHEDULER_KIND_SERVE:
        result = _install_serve(interval_seconds)

    write_state(STATE_STARTED)
    result['state'] = STATE_STARTED
    result['env_actions'] = env_actions
    return result


def _install_serve(interval_seconds: int) -> dict:
    """Record the configured interval for serve-mode hosts.

    Serve mode has no installable artifact — the user runs
    `memman scheduler serve` themselves (typically as PID 1 of a
    container). Install just records the interval so `status` and
    `doctor` can report it.
    """
    write_serve_interval(interval_seconds)
    return {
        'platform': SCHEDULER_KIND_SERVE,
        'interval_seconds': interval_seconds,
        'actions': [
            f'wrote {_serve_interval_path()} (mode 600)',
            ('start `memman scheduler serve` to run the drain loop'
             ' (typically as PID 1 in a container)'),
            ],
        }


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


def uninstall() -> dict:
    """Remove the scheduler trigger and clear the state files.

    Removes systemd units, launchd plist, or inline marker — whichever
    is present. Clears scheduler.state and debug.state last.
    """
    clear_state()
    clear_debug_state()
    kind = detect_scheduler()
    if kind == 'systemd':
        return _uninstall_systemd()
    if kind == 'launchd':
        return _uninstall_launchd()
    clear_serve_interval()
    return {
        'platform': SCHEDULER_KIND_SERVE,
        'actions': [f'removed {_serve_interval_path()}'],
        }


def start() -> dict:
    """Activate the scheduler trigger. Idempotent.

    systemd: `systemctl --user enable --now`. launchd: `launchctl load -w`.
    inline: no-op beyond writing the state file.
    Raises FileNotFoundError if the trigger isn't installed (run
    `memman install` first).
    """
    kind = detect_scheduler()
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
        write_state(STATE_STARTED)
        return {
            'platform': 'systemd',
            'state': STATE_STARTED,
            'actions': [
                'systemctl --user enable --now memman-enrich.timer'],
            }
    if kind == 'launchd':
        plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
        if not plist_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {plist_path};'
                " run 'memman install' first")
        subprocess.run(
            ['launchctl', 'load', '-w', str(plist_path)], check=False)
        _verify_launchd_loaded()
        write_state(STATE_STARTED)
        return {
            'platform': 'launchd',
            'state': STATE_STARTED,
            'actions': [f'launchctl load -w {plist_path}'],
            }
    write_state(STATE_STARTED)
    return {
        'platform': SCHEDULER_KIND_SERVE,
        'state': STATE_STARTED,
        'actions': [
            f'wrote {_state_file_path()} = {STATE_STARTED}',
            'start `memman scheduler serve` if it is not already running',
            ],
        }


def stop() -> dict:
    """Deactivate the scheduler trigger. Trigger files stay on disk.

    systemd: stop + disable timer (unit file kept). launchd: unload
    plist (plist kept). inline: no-op beyond writing the state file.
    Use `uninstall` to remove trigger files.
    """
    kind = detect_scheduler()
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
        write_state(STATE_STOPPED)
        return {
            'platform': 'systemd',
            'state': STATE_STOPPED,
            'actions': [
                'systemctl --user stop memman-enrich.timer',
                'systemctl --user disable memman-enrich.timer'],
            }
    if kind == 'launchd':
        plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
        if not plist_path.exists():
            raise FileNotFoundError(
                f'scheduler unit not installed at {plist_path};'
                " run 'memman install' first")
        subprocess.run(
            ['launchctl', 'unload', '-w', str(plist_path)], check=False)
        write_state(STATE_STOPPED)
        return {
            'platform': 'launchd',
            'state': STATE_STOPPED,
            'actions': [f'launchctl unload -w {plist_path}'],
            }
    write_state(STATE_STOPPED)
    return {
        'platform': SCHEDULER_KIND_SERVE,
        'state': STATE_STOPPED,
        'actions': [
            f'wrote {_state_file_path()} = {STATE_STOPPED}',
            ('the running serve loop polls this state and will exit'
             ' on its next iteration (within ~`--interval` seconds)'),
            ],
        }


def trigger() -> dict:
    """Run the scheduler worker once, immediately.

    systemd: `systemctl start --no-block` the service. launchd:
    `launchctl start` the agent. inline: callers (the CLI) drain
    in-process via _drain_queue; this function rejects on inline since
    the trigger is implicit, not on-demand.
    """
    kind = detect_scheduler()
    if kind == SCHEDULER_KIND_SERVE:
        raise RuntimeError(
            'scheduler trigger is not applicable in serve mode;'
            ' the serve loop drains on its own cadence.'
            ' Use `memman scheduler serve --once` to run a single drain.')
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
    """Rewrite the scheduler trigger with a new interval. Preserves state.

    systemd/launchd: rewrite unit file with new interval (min 60s).
    serve: write the requested interval to the serve interval file
    (any value >= 0; 0 means continuous mode). The running serve loop
    reads its interval from the CLI flag, so the file write is
    advisory until the next `memman scheduler serve` restart.
    """
    if new_seconds < 0:
        raise RuntimeError(
            f'interval {new_seconds}s is negative; must be >= 0')
    prior_state = read_state()
    kind = detect_scheduler()
    if kind in {'systemd', 'launchd'} and new_seconds < 60:
        raise RuntimeError(
            f'interval {new_seconds}s is too short for {kind};'
            ' minimum is 60s. To use sub-minute intervals, set'
            ' MEMMAN_SCHEDULER_KIND=serve and run'
            ' `memman scheduler serve --interval N`.')
    if kind == SCHEDULER_KIND_SERVE and (
            _systemd_is_enabled() or _launchd_is_loaded()):
        import logging as _logging
        _logging.getLogger('memman').warning(
            'MEMMAN_SCHEDULER_KIND=serve is set but a systemd/launchd'
            ' unit is still active. Drains may run from both. Run'
            ' `memman uninstall` to remove the OS timer if you intend'
            ' to use serve mode exclusively.')
    if kind == 'systemd':
        binary = memman_binary_path()
        result = _install_systemd(binary, data_dir, new_seconds)
        if prior_state == STATE_STOPPED:
            subprocess.run(
                ['systemctl', '--user', 'stop', SYSTEMD_TIMER_NAME],
                check=False, capture_output=True)
            subprocess.run(
                ['systemctl', '--user', 'disable', SYSTEMD_TIMER_NAME],
                check=False, capture_output=True)
            result['actions'].append(
                'systemctl --user stop+disable memman-enrich.timer'
                ' (restored prior stopped state)')
            write_state(STATE_STOPPED)
        else:
            write_state(STATE_STARTED)
        return result
    if kind == 'launchd':
        binary = memman_binary_path()
        result = _install_launchd(binary, data_dir, new_seconds)
        if prior_state == STATE_STOPPED:
            plist_path = _launchd_agent_dir() / f'{LAUNCHD_LABEL}.plist'
            subprocess.run(
                ['launchctl', 'unload', '-w', str(plist_path)], check=False)
            result['actions'].append(
                f'launchctl unload -w {plist_path}'
                ' (restored prior stopped state)')
            write_state(STATE_STOPPED)
        else:
            write_state(STATE_STARTED)
        return result
    write_serve_interval(new_seconds)
    return {
        'platform': SCHEDULER_KIND_SERVE,
        'interval_seconds': new_seconds,
        'actions': [
            f'wrote {_serve_interval_path()} = {new_seconds}',
            ('the running serve loop reads its interval from the CLI'
             ' flag, not this file; restart `memman scheduler serve`'
             f' with --interval {new_seconds} to apply'),
            ],
        }


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
    """Return the scheduler's current status.

    Fields:
      - platform — 'systemd' | 'launchd' | 'serve'
      - installed — True iff the trigger file/marker exists
      - active — True iff the trigger is currently active (timer
        running on systemd/launchd; STATE_STARTED for serve)
      - next_run — best-effort next-fire timestamp (systemd/launchd only)
      - interval_seconds — configured interval (read from
        ~/.memman/scheduler.serve_interval for serve mode)
      - state — persisted user intent ('started' | 'stopped')

    `state` is the pause/resume gate: `memman scheduler stop` flips it
    to STOPPED; the serve loop polls it every iteration and exits when
    stopped, while `_require_started` rejects writes in the same state.
    `memman scheduler start` resumes both.
    """
    kind = detect_scheduler()
    if kind == 'systemd':
        result = _systemd_status()
    elif kind == 'launchd':
        result = _launchd_status()
    else:
        interval = read_serve_interval()
        result = {
            'platform': SCHEDULER_KIND_SERVE,
            'installed': interval is not None,
            'active': read_state() == STATE_STARTED,
            'next_run': None,
            'interval_seconds': interval,
            }
    result['state'] = read_state()
    return result
