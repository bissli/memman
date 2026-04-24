"""Unit tests for memman.setup.scheduler."""

import os
from pathlib import Path

import pytest
from memman.setup import scheduler as sch


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect HOME and scheduler dirs to a tmp_path."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_binary(monkeypatch):
    """Pretend memman is installed at a known path."""
    monkeypatch.setattr(sch, 'memman_binary_path',
                        lambda: '/fake/bin/memman')


def _no_subprocess(monkeypatch, active: bool = True):
    """Suppress real subprocess calls for systemctl/launchctl tests."""
    class _FakeResult:
        returncode = 0 if active else 3
        stdout = 'active' if active else 'inactive'
        stderr = ''

    def _fake_run(*args, **kwargs):
        return _FakeResult()

    fake = type('S', (), {
        'run': staticmethod(_fake_run),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(sch, 'subprocess', fake)


def test_install_systemd_writes_timer_and_service(
        fake_home, fake_binary, monkeypatch):
    """Systemd install creates timer + service files with correct content.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)

    result = sch.install(
        data_dir=str(fake_home / '.memman'),
        openrouter_api_key='sk-or-x',
        voyage_api_key='vk-y',
        interval_seconds=600)

    assert result['platform'] == 'systemd'
    assert result['interval_seconds'] == 600
    timer = Path(result['timer_path']).read_text()
    service = Path(result['service_path']).read_text()
    assert 'OnUnitActiveSec=600s' in timer
    assert 'Persistent=true' in timer
    assert '/fake/bin/memman enrich --pending' in service
    assert 'MEMMAN_DATA_DIR=' in service
    assert 'EnvironmentFile=' in service


def test_install_launchd_writes_plist_and_wrapper(
        fake_home, fake_binary, monkeypatch):
    """Launchd install creates plist + wrapper script with correct content.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
    _no_subprocess(monkeypatch)

    result = sch.install(
        data_dir=str(fake_home / '.memman'),
        openrouter_api_key='sk-or-x',
        voyage_api_key='vk-y',
        interval_seconds=1800)

    assert result['platform'] == 'launchd'
    plist = Path(result['plist_path']).read_text()
    wrapper = Path(result['wrapper_path']).read_text()
    assert '<key>StartInterval</key><integer>1800</integer>' in plist
    assert '/fake/bin/memman' in wrapper
    assert 'enrich --pending' in wrapper
    assert os.access(result['wrapper_path'], os.X_OK)


def test_install_unknown_platform_raises(monkeypatch):
    """Install raises when no supported scheduler is detected.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: '')
    with pytest.raises(RuntimeError, match='no supported scheduler'):
        sch.install(data_dir='/tmp',
                    openrouter_api_key='x',
                    voyage_api_key='y')


def test_install_writes_both_keys_to_env_file(
        fake_home, fake_binary, monkeypatch):
    """Both OPENROUTER_API_KEY and VOYAGE_API_KEY are written at mode 600.
    """
    import stat
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)

    sch.install(data_dir=str(fake_home),
                openrouter_api_key='sk-or-fake',
                voyage_api_key='vk-fake')
    env_path = fake_home / '.memman' / 'env'
    assert env_path.exists()
    contents = env_path.read_text()
    assert 'OPENROUTER_API_KEY=sk-or-fake' in contents
    assert 'VOYAGE_API_KEY=vk-fake' in contents
    assert 'MEMMAN_LLM_PROVIDER=openrouter' in contents
    mode = stat.S_IMODE(os.stat(env_path).st_mode)
    assert mode == 0o600


def test_install_merges_existing_env_file(
        fake_home, fake_binary, monkeypatch):
    """Existing env keys are preserved across installs.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    env_path = fake_home / '.memman' / 'env'
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        'SOMETHING_ELSE=keep\nMEMMAN_LLM_PROVIDER=anthropic\n')

    sch.install(data_dir=str(fake_home),
                openrouter_api_key='sk-or-new',
                voyage_api_key='vk-new')
    contents = env_path.read_text()
    assert 'SOMETHING_ELSE=keep' in contents
    assert 'MEMMAN_LLM_PROVIDER=openrouter' in contents
    assert 'OPENROUTER_API_KEY=sk-or-new' in contents
    assert 'VOYAGE_API_KEY=vk-new' in contents


def test_change_interval_rewrites_unit_without_touching_env(
        fake_home, fake_binary, monkeypatch):
    """change_interval updates the unit file but leaves ~/.memman/env alone.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)

    sch.install(data_dir=str(fake_home),
                openrouter_api_key='sk-or-1',
                voyage_api_key='vk-1',
                interval_seconds=900)
    env_before = (fake_home / '.memman' / 'env').read_text()

    sch.change_interval(str(fake_home), 300)
    timer = (fake_home / '.config' / 'systemd' / 'user'
             / 'memman-enrich.timer').read_text()
    assert 'OnUnitActiveSec=300s' in timer
    env_after = (fake_home / '.memman' / 'env').read_text()
    assert env_before == env_after


def test_change_interval_rejects_too_short(
        fake_home, fake_binary, monkeypatch):
    """change_interval refuses values below the 60s floor.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    with pytest.raises(RuntimeError, match='too short'):
        sch.change_interval(str(fake_home), 30)


def test_status_not_installed(fake_home, monkeypatch):
    """status() reports installed=False when no unit file exists.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    result = sch.status()
    assert result['platform'] == 'systemd'
    assert result['installed'] is False
    assert result['interval_seconds'] is None


def test_status_installed_parses_interval(
        fake_home, fake_binary, monkeypatch):
    """status() parses OnUnitActiveSec from the installed timer.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x',
                voyage_api_key='y',
                interval_seconds=1800)
    result = sch.status()
    assert result['installed'] is True
    assert result['interval_seconds'] == 1800


def test_status_launchd_parses_interval(
        fake_home, fake_binary, monkeypatch):
    """Launchd status parses StartInterval from the plist.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
    _no_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x',
                voyage_api_key='y',
                interval_seconds=1200)
    result = sch.status()
    assert result['platform'] == 'launchd'
    assert result['installed'] is True
    assert result['interval_seconds'] == 1200


def test_parse_interval_non_s_unit(fake_home):
    """Parser returns None on OnUnitActiveSec values without an 's' suffix.
    """
    timer_path = fake_home / 'memman-enrich.timer'
    timer_path.write_text(
        '[Timer]\nOnUnitActiveSec=15min\nPersistent=true\n')
    assert sch._parse_interval_from_systemd_timer(timer_path) is None


def test_change_interval_launchd(fake_home, fake_binary, monkeypatch):
    """change_interval rewrites the launchd plist with a new StartInterval.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
    _no_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x',
                voyage_api_key='y',
                interval_seconds=900)
    sch.change_interval(str(fake_home), 3600)
    plist = (fake_home / 'Library' / 'LaunchAgents'
             / 'com.memman.enrich.plist').read_text()
    assert '<key>StartInterval</key><integer>3600</integer>' in plist


def test_uninstall_systemd_removes_unit_files(
        fake_home, fake_binary, monkeypatch):
    """uninstall() removes systemd timer and service files.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x',
                voyage_api_key='y')
    timer_path = (fake_home / '.config' / 'systemd' / 'user'
                  / sch.SYSTEMD_TIMER_NAME)
    service_path = (fake_home / '.config' / 'systemd' / 'user'
                    / sch.SYSTEMD_SERVICE_NAME)
    assert timer_path.exists()
    assert service_path.exists()
    sch.uninstall()
    assert not timer_path.exists()
    assert not service_path.exists()


def test_start_raises_when_not_installed(fake_home, monkeypatch):
    """start() raises FileNotFoundError when unit files are absent.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    with pytest.raises(FileNotFoundError, match='not installed'):
        sch.start()


def test_stop_when_not_installed_is_noop(fake_home, monkeypatch):
    """stop() returns cleanly when unit files are absent.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    result = sch.stop()
    assert result.get('note') == 'not installed'


def _record_subprocess(monkeypatch, *, returncode: int = 0,
                       stderr: str = '', stdout: str = 'active',
                       responses: dict | None = None):
    """Stub subprocess.run, record argvs, and (optionally) route by argv.

    `responses` maps an argv-tuple prefix to a stdout string. When a
    call's argv starts with a key, that response is returned; otherwise
    the default `stdout` is used.
    """
    calls: list = []

    class _FakeResult:
        def __init__(self, rc: int, out: str, err: str) -> None:
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def _fake_run(cmd, *args, **kwargs):
        argv = tuple(cmd)
        calls.append(list(cmd))
        out = stdout
        if responses:
            if argv in responses:
                out = responses[argv]
            else:
                for key, value in responses.items():
                    if argv[:len(key)] == key:
                        out = value
                        break
        return _FakeResult(returncode, out, stderr)

    fake = type('S', (), {
        'run': staticmethod(_fake_run),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(sch, 'subprocess', fake)
    return calls


def test_trigger_systemd_uses_no_block(
        fake_home, fake_binary, monkeypatch):
    """trigger() on systemd runs `systemctl --user start --no-block`.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    calls = _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y')
    calls.clear()

    result = sch.trigger()
    assert result['platform'] == 'systemd'
    assert calls == [[
        'systemctl', '--user', 'start', '--no-block',
        'memman-enrich.service',
        ]]


def test_trigger_systemd_handles_already_running(
        fake_home, fake_binary, monkeypatch):
    """trigger() returns informational note when a run is already active.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y')

    _record_subprocess(
        monkeypatch, returncode=1,
        stderr='Job for memman-enrich.service already running')
    result = sch.trigger()
    assert result['platform'] == 'systemd'
    assert 'already' in result.get('note', '').lower()


def test_trigger_launchd_runs_job(
        fake_home, fake_binary, monkeypatch):
    """trigger() on launchd runs `launchctl start com.memman.enrich`.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
    calls = _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y')
    calls.clear()

    result = sch.trigger()
    assert result['platform'] == 'launchd'
    assert calls == [['launchctl', 'start', 'com.memman.enrich']]


def test_trigger_raises_when_not_installed(fake_home, monkeypatch):
    """trigger() raises FileNotFoundError when the unit file is absent.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _no_subprocess(monkeypatch)
    with pytest.raises(FileNotFoundError, match='not installed'):
        sch.trigger()


def test_systemd_status_computes_next_run(
        fake_home, fake_binary, monkeypatch):
    """status() reports next_run = LastTriggerUSec + interval.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y',
                interval_seconds=900)

    _record_subprocess(monkeypatch, responses={
        ('systemctl', '--user', 'is-enabled'): 'enabled',
        ('systemctl', '--user', 'is-active'): 'active',
        ('systemctl', '--user', 'show',
         '--property=LastTriggerUSec', '--value'):
            'Fri 2026-04-24 14:18:58 EDT',
        })
    result = sch.status()
    assert result['next_run'] is not None
    assert result['next_run'].startswith('2026-04-24T18:33:58')


def test_systemd_status_next_run_when_never_fired(
        fake_home, fake_binary, monkeypatch):
    """status() returns next_run=None when the timer has never fired.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y')

    _record_subprocess(monkeypatch, responses={
        ('systemctl', '--user', 'show',
         '--property=LastTriggerUSec', '--value'): 'n/a',
        })
    result = sch.status()
    assert result['next_run'] is None


def test_systemd_status_next_run_when_malformed(
        fake_home, fake_binary, monkeypatch):
    """status() returns next_run=None on an unparseable timestamp.
    """
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
    _record_subprocess(monkeypatch)
    sch.install(data_dir=str(fake_home),
                openrouter_api_key='x', voyage_api_key='y')

    _record_subprocess(monkeypatch, responses={
        ('systemctl', '--user', 'show',
         '--property=LastTriggerUSec', '--value'):
            'not a real timestamp',
        })
    result = sch.status()
    assert result['next_run'] is None
