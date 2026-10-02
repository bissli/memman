"""Unit tests for memman.setup.scheduler.
"""

import logging
import os
import platform
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli
from memman.setup import scheduler as sch
from memman.setup.claude import _install_claude_code
from tests.conftest import fake_subprocess, install_env_factory


@pytest.fixture
def uninstall_home(fake_home, monkeypatch):
    """`fake_home` plus a `MEMMAN_DATA_DIR` pin under it.

    Used by `TestUninstall` for tests that reach into a fake env file
    and assert what `sch.uninstall` strips. Standalone scheduler tests
    keep using `fake_home` directly.
    """
    data_dir = fake_home / 'memman'
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
    config.reset_file_cache()
    return fake_home, data_dir


def _knobs(api_key: str = 'sk-or-test',
           **extra: str) -> dict[str, str]:
    """Build the install-time knobs dict used by `sch.install`.
    """
    base = {
        'MEMMAN_ENDPOINT': 'https://openrouter.ai/api/v1',
        'MEMMAN_API_KEY': api_key,
        }
    base.update(extra)
    return base


@pytest.fixture
def fake_binary(monkeypatch):
    """Pretend memman is installed at a known path.
    """
    monkeypatch.setattr(sch, 'memman_binary_path',
                        lambda: '/fake/bin/memman')


def _no_subprocess(monkeypatch, active: bool = True):
    """Thin wrapper for shared `fake_subprocess` keyed on `sch`.
    """
    fake_subprocess(monkeypatch, sch, active=active)


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


class TestInstall:
    """systemd / launchd install: file generation, env file merge.
    """

    def test_install_systemd_writes_timer_and_service(
            self, fake_home, fake_binary, monkeypatch):
        """Verify a systemd install writes a correct timer and service.

        Mutation: the interval is not rendered into OnUnitActiveSec, Persistent
            is dropped, or the service runs the wrong binary or command.
        Oracle: hand-written expected unit-file substrings.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)

        result = sch.install(
            data_dir=str(fake_home / '.memman'),
            knobs=_knobs(api_key='sk-or-x'),
            interval_seconds=600)

        assert result['platform'] == 'systemd'
        assert result['interval_seconds'] == 600
        timer = Path(result['timer_path']).read_text()
        service = Path(result['service_path']).read_text()
        assert 'OnUnitActiveSec=600s' in timer
        assert 'Persistent=true' in timer
        assert '/fake/bin/memman scheduler drain' in service
        assert 'MEMMAN_DATA_DIR=' in service
        assert 'EnvironmentFile=' in service

    def test_install_launchd_writes_plist_and_wrapper(
            self, fake_home, fake_binary, monkeypatch):
        """Verify a launchd install writes a plist and an executable wrapper.

        Mutation: StartInterval or the drain command is not written, or the
            wrapper loses its execute bit.
        Oracle: hand-written plist and wrapper substrings, and os.access.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _no_subprocess(monkeypatch)

        result = sch.install(
            data_dir=str(fake_home / '.memman'),
            knobs=_knobs(api_key='sk-or-x'),
            interval_seconds=1800)

        assert result['platform'] == 'launchd'
        plist = Path(result['plist_path']).read_text()
        wrapper = Path(result['wrapper_path']).read_text()
        assert '<key>StartInterval</key><integer>1800</integer>' in plist
        assert '/fake/bin/memman' in wrapper
        assert 'scheduler drain' in wrapper
        assert os.access(result['wrapper_path'], os.X_OK)

    def test_install_writes_both_keys_to_env_file(
            self, fake_home, fake_binary, monkeypatch):
        """Verify install writes the API key and the endpoint at mode 600.

        Mutation: a key is omitted from the env file, or the file is left
            readable by group or other.
        Oracle: literal env lines and the mode 0o600.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)

        sch.install(data_dir=str(fake_home / '.memman'),
                    knobs=_knobs(api_key='sk-or-fake'))
        env_path = fake_home / '.memman' / 'env'
        assert env_path.exists()
        contents = env_path.read_text()
        assert 'MEMMAN_API_KEY=sk-or-fake' in contents
        assert 'MEMMAN_ENDPOINT=https://openrouter.ai/api/v1' in contents
        mode = stat.S_IMODE(os.stat(env_path).st_mode)
        assert mode == 0o600

    def test_install_merges_existing_env_file(
            self, fake_home, fake_binary, monkeypatch):
        """Verify install keeps unrelated env keys and refreshes knobs.

        Mutation: install overwrites the env file, or keeps a stale knob value.
        Oracle: a seeded file holding one foreign key and an old endpoint.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        env_path = fake_home / '.memman' / 'env'
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            'SOMETHING_ELSE=keep\n'
            'MEMMAN_ENDPOINT=https://api.openai.com/v1\n')

        sch.install(data_dir=str(fake_home / '.memman'),
                    knobs=_knobs(api_key='sk-or-new'))
        contents = env_path.read_text()
        assert 'SOMETHING_ELSE=keep' in contents
        assert 'MEMMAN_ENDPOINT=https://openrouter.ai/api/v1' in contents
        assert 'MEMMAN_API_KEY=sk-or-new' in contents

    def test_install_without_interval_uses_60s_default(
            self, fake_home, fake_binary, monkeypatch):
        """Verify install with no interval writes a 60s timer.

        Mutation: the default interval is dropped or differs from 60 seconds.
        Oracle: literal 60 in the result and 'OnUnitActiveSec=60s' in the
            timer.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        result = sch.install(data_dir=str(fake_home),
                             knobs=_knobs(api_key='x'))
        assert result['interval_seconds'] == 60
        timer = Path(result['timer_path']).read_text()
        assert 'OnUnitActiveSec=60s' in timer

    def test_default_interval_is_60_seconds(self):
        """Verify DEFAULT_INTERVAL_SECONDS is 60.

        Mutation: the constant is raised to a slower poll.
        Oracle: the literal 60.
        """
        assert sch.DEFAULT_INTERVAL_SECONDS == 60


class TestDebugState:
    """Debug-state file (`~/.memman/debug.state`) round-trip.
    """

    def test_set_debug_on_writes_state_file_mode_600(self, fake_home):
        """Verify set_debug(True) writes 'on' to debug.state at mode 600.

        Mutation: the state lands at another path or with group/other bits set.
        Oracle: the literal path, DEBUG_ON, and mode 0o600.
        """
        sch.set_debug(True)
        p = fake_home / '.memman' / 'debug.state'
        assert p.exists()
        assert p.read_text().strip() == sch.DEBUG_ON
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600

    def test_set_debug_off_writes_off_to_state_file(self, fake_home):
        """Verify set_debug(False) turns a stored 'on' back to 'off'.

        Mutation: set_debug(False) leaves the file unchanged.
        Oracle: DEBUG_OFF read from the state file after on then off.
        """
        sch.set_debug(True)
        sch.set_debug(False)
        p = fake_home / '.memman' / 'debug.state'
        assert p.read_text().strip() == sch.DEBUG_OFF

    def test_get_debug_round_trips_state_file(self, fake_home):
        """Verify get_debug follows the last state written.

        Mutation: get_debug caches its first answer or inverts the flag.
        Oracle: the state values written in sequence, off by default.
        """
        assert sch.get_debug() is False
        sch.write_debug_state(sch.DEBUG_ON)
        assert sch.get_debug() is True
        sch.write_debug_state(sch.DEBUG_OFF)
        assert sch.get_debug() is False

    def test_set_debug_does_not_touch_env_file(self, fake_home):
        """Verify toggling debug leaves the env file byte-identical.

        Mutation: set_debug rewrites the env file, e.g. to store the flag
            there.
        Oracle: the seeded env text.
        """
        env_path = fake_home / '.memman' / 'env'
        env_path.parent.mkdir(parents=True, exist_ok=True)
        original = (
            'MEMMAN_ENDPOINT=https://openrouter.ai/api/v1\n'
            'MEMMAN_API_KEY=sk-x\n')
        env_path.write_text(original)
        sch.set_debug(True)
        sch.set_debug(False)
        assert env_path.read_text() == original


class TestChangeInterval:
    """`sch.change_interval` validation and unit rewrite.
    """

    def test_change_interval_rewrites_unit_without_touching_env(
            self, fake_home, fake_binary, monkeypatch):
        """Verify change_interval rewrites the timer, not the env file.

        Mutation: the timer keeps the old interval, or the env file is
            rewritten.
        Oracle: 'OnUnitActiveSec=300s' and the env text before and after.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)

        sch.install(data_dir=str(fake_home / '.memman'),
                    knobs=_knobs(api_key='sk-or-1'),
                    interval_seconds=900)
        env_before = (fake_home / '.memman' / 'env').read_text()

        sch.change_interval(str(fake_home / '.memman'), 300)
        timer = (fake_home / '.config' / 'systemd' / 'user'
                 / 'memman-enrich.timer').read_text()
        assert 'OnUnitActiveSec=300s' in timer
        env_after = (fake_home / '.memman' / 'env').read_text()
        assert env_before == env_after

    def test_change_interval_rejects_too_short(
            self, fake_home, fake_binary, monkeypatch):
        """Verify systemd refuses an interval below the 60s floor.

        Mutation: the floor check is dropped or skipped for systemd.
        Oracle: RuntimeError 'too short for systemd' for 30 seconds.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        with pytest.raises(RuntimeError, match='too short for systemd'):
            sch.change_interval(str(fake_home), 30)

    def test_change_interval_rejects_below_60_for_launchd(
            self, fake_home, fake_binary, monkeypatch):
        """Verify launchd refuses an interval below the 60s floor.

        Mutation: the floor check is dropped or skipped for launchd.
        Oracle: RuntimeError 'too short for launchd' for 30 seconds.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _no_subprocess(monkeypatch)
        with pytest.raises(RuntimeError, match='too short for launchd'):
            sch.change_interval(str(fake_home), 30)

    def test_change_interval_accepts_zero_for_serve(
            self, fake_home, fake_binary, monkeypatch):
        """Verify serve mode accepts 0 as its continuous-loop interval.

        Mutation: the 60s floor is applied to serve mode as well.
        Oracle: result and stored serve interval both 0.
        """
        monkeypatch.setattr(sch, 'detect_scheduler',
                            lambda: sch.SCHEDULER_KIND_SERVE)
        monkeypatch.setattr(sch, '_systemd_is_enabled', lambda: False)
        monkeypatch.setattr(sch, '_launchd_is_loaded', lambda: False)
        _no_subprocess(monkeypatch)
        result = sch.change_interval(str(fake_home), 0)
        assert result['interval_seconds'] == 0
        assert sch.read_serve_interval() == 0

    def test_change_interval_rejects_negative(
            self, fake_home, fake_binary, monkeypatch):
        """Verify a negative interval is refused in every mode.

        Mutation: the negative check is dropped for serve mode.
        Oracle: RuntimeError matching 'negative' for -1.
        """
        monkeypatch.setattr(sch, 'detect_scheduler',
                            lambda: sch.SCHEDULER_KIND_SERVE)
        _no_subprocess(monkeypatch)
        with pytest.raises(RuntimeError, match='negative'):
            sch.change_interval(str(fake_home), -1)

    def test_change_interval_warns_on_mixed_mode(
            self, fake_home, fake_binary, monkeypatch, caplog):
        """Verify serve mode alongside enabled systemd warns.

        Mutation: the mixed-mode warning is dropped.
        Oracle: a WARNING record naming both 'serve' and 'systemd'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler',
                            lambda: sch.SCHEDULER_KIND_SERVE)
        monkeypatch.setattr(sch, '_systemd_is_enabled', lambda: True)
        monkeypatch.setattr(sch, '_launchd_is_loaded', lambda: False)
        _no_subprocess(monkeypatch)
        with caplog.at_level(logging.WARNING, logger='memman'):
            sch.change_interval(str(fake_home), 30)
        assert any('serve' in rec.message and 'systemd' in rec.message
                   for rec in caplog.records), (
            f'expected mixed-mode warning; got: '
            f'{[r.message for r in caplog.records]}')

    def test_change_interval_launchd(self, fake_home, fake_binary, monkeypatch):
        """Verify change_interval rewrites the launchd plist interval.

        Mutation: the plist keeps the old StartInterval.
        Oracle: '<key>StartInterval</key><integer>3600</integer>' in the plist.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _no_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'),
                    interval_seconds=900)
        sch.change_interval(str(fake_home), 3600)
        plist = (fake_home / 'Library' / 'LaunchAgents'
                 / 'com.memman.enrich.plist').read_text()
        assert '<key>StartInterval</key><integer>3600</integer>' in plist


class TestStatusParsing:
    """`sch.status` parses systemd / launchd output.
    """

    def test_status_not_installed(self, fake_home, monkeypatch):
        """Verify status reports not installed when no unit file exists.

        Mutation: status reports installed, or an interval, with no unit file.
        Oracle: installed False and interval None.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        result = sch.status()
        assert result['platform'] == 'systemd'
        assert result['installed'] is False
        assert result['interval_seconds'] is None

    def test_status_installed_parses_interval(
            self, fake_home, fake_binary, monkeypatch):
        """Verify status reads the interval back from the installed timer.

        Mutation: the OnUnitActiveSec value is misparsed or dropped.
        Oracle: the 1800 seconds passed to install.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'),
                    interval_seconds=1800)
        result = sch.status()
        assert result['installed'] is True
        assert result['interval_seconds'] == 1800

    def test_status_launchd_parses_interval(
            self, fake_home, fake_binary, monkeypatch):
        """Verify status reads StartInterval back from the launchd plist.

        Mutation: the StartInterval value is misparsed or dropped.
        Oracle: the 1200 seconds passed to install.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _no_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'),
                    interval_seconds=1200)
        result = sch.status()
        assert result['platform'] == 'launchd'
        assert result['installed'] is True
        assert result['interval_seconds'] == 1200

    def test_parse_interval_non_s_unit(self, fake_home):
        """Verify the timer parser returns None without an 's' suffix.

        Mutation: '15min' is read as 15 seconds.
        Oracle: a hand-written timer file with OnUnitActiveSec=15min.
        """
        timer_path = fake_home / 'memman-enrich.timer'
        timer_path.write_text(
            '[Timer]\nOnUnitActiveSec=15min\nPersistent=true\n')
        assert sch._parse_interval_from_systemd_timer(timer_path) is None

    def test_systemd_status_computes_next_run(
            self, fake_home, fake_binary, monkeypatch):
        """Verify next_run is LastTriggerUSec plus the interval, in UTC.

        Mutation: the interval is not added, or the EDT stamp is read as UTC.
        Oracle: hand-computed 14:18:58 EDT + 900s = 2026-04-24T18:33:58 UTC.
        """
        prev_tz = os.environ.get('TZ')
        os.environ['TZ'] = 'EST5EDT,M3.2.0,M11.1.0'
        time.tzset()
        try:
            monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
            _record_subprocess(monkeypatch)
            sch.install(data_dir=str(fake_home),
                        knobs=_knobs(api_key='x'),
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
        finally:
            if prev_tz is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = prev_tz
            time.tzset()

    def test_systemd_status_next_run_when_never_fired(
            self, fake_home, fake_binary, monkeypatch):
        """Verify next_run is None when the timer has never fired.

        Mutation: the 'n/a' stamp is parsed into a date or raises.
        Oracle: LastTriggerUSec stubbed to 'n/a'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))

        _record_subprocess(monkeypatch, responses={
            ('systemctl', '--user', 'show',
             '--property=LastTriggerUSec', '--value'): 'n/a',
            })
        result = sch.status()
        assert result['next_run'] is None

    def test_launchd_status_computes_next_run(
            self, fake_home, fake_binary, monkeypatch):
        """Verify launchd next_run is the log mtime plus the interval.

        Mutation: the interval is not added, or the mtime is ignored.
        Oracle: a log file with a fixed mtime, plus 900s.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'),
                    interval_seconds=900)

        log_path = fake_home / '.memman' / 'logs' / 'enrich.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        fixed_mtime = 1_800_000_000.0
        os.utime(log_path, (fixed_mtime, fixed_mtime))

        _record_subprocess(monkeypatch)
        result = sch.status()
        assert result['platform'] == 'launchd'
        assert result['active'] is True
        expected = datetime.fromtimestamp(
            fixed_mtime + 900, tz=timezone.utc).isoformat()
        assert result['next_run'] == expected

    def test_launchd_status_next_run_without_log(
            self, fake_home, fake_binary, monkeypatch):
        """Verify launchd next_run is None before any enrich.log exists.

        Mutation: a missing log raises or yields a made-up time.
        Oracle: no log file on disk.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))

        _record_subprocess(monkeypatch)
        result = sch.status()
        assert result['platform'] == 'launchd'
        assert result['next_run'] is None

    def test_systemd_status_next_run_when_malformed(
            self, fake_home, fake_binary, monkeypatch):
        """Verify next_run is None when LastTriggerUSec cannot be parsed.

        Mutation: an unparseable stamp raises or yields a bogus date.
        Oracle: LastTriggerUSec stubbed to a non-date string.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))

        _record_subprocess(monkeypatch, responses={
            ('systemctl', '--user', 'show',
             '--property=LastTriggerUSec', '--value'):
            'not a real timestamp',
            })
        result = sch.status()
        assert result['next_run'] is None


class TestActivation:
    """`uninstall`, `start`, `stop`, `trigger` semantics.
    """

    def test_uninstall_systemd_removes_unit_files(
            self, fake_home, fake_binary, monkeypatch):
        """Verify uninstall deletes the systemd timer and service files.

        Mutation: uninstall leaves one of the unit files on disk.
        Oracle: both paths exist after install and neither after uninstall.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))
        timer_path = (fake_home / '.config' / 'systemd' / 'user'
                      / sch.SYSTEMD_TIMER_NAME)
        service_path = (fake_home / '.config' / 'systemd' / 'user'
                        / sch.SYSTEMD_SERVICE_NAME)
        assert timer_path.exists()
        assert service_path.exists()
        sch.uninstall()
        assert not timer_path.exists()
        assert not service_path.exists()

    def test_start_raises_when_not_installed(self, fake_home, monkeypatch):
        """Verify start() without unit files raises FileNotFoundError.

        Mutation: start() calls systemctl and reports success with nothing
            installed.
        Oracle: FileNotFoundError matching 'not installed'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        with pytest.raises(FileNotFoundError, match='not installed'):
            sch.start()

    def test_stop_raises_when_not_installed(self, fake_home, monkeypatch):
        """Verify stop() without unit files raises FileNotFoundError.

        Mutation: stop() calls systemctl and reports success with nothing
            installed.
        Oracle: FileNotFoundError matching 'not installed'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        with pytest.raises(FileNotFoundError, match='not installed'):
            sch.stop()

    def test_trigger_systemd_uses_no_block(
            self, fake_home, fake_binary, monkeypatch):
        """Verify trigger() starts the systemd service without blocking.

        Mutation: --no-block is dropped, or the timer is started in place of
            the service.
        Oracle: the exact recorded systemctl argv.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        calls = _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))
        calls.clear()

        result = sch.trigger()
        assert result['platform'] == 'systemd'
        assert calls == [[
            'systemctl', '--user', 'start', '--no-block',
            'memman-enrich.service',
            ]]

    def test_trigger_systemd_handles_already_running(
            self, fake_home, fake_binary, monkeypatch):
        """Verify trigger() returns a note when a run is already active.

        Mutation: the nonzero systemctl exit raises instead of returning a
            note.
        Oracle: a stub returning exit 1 with 'already running' on stderr.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))

        _record_subprocess(
            monkeypatch, returncode=1,
            stderr='Job for memman-enrich.service already running')
        result = sch.trigger()
        assert result['platform'] == 'systemd'
        assert 'already' in result.get('note', '').lower()

    def test_trigger_launchd_runs_job(
            self, fake_home, fake_binary, monkeypatch):
        """Verify trigger() on launchd starts the enrich job by label.

        Mutation: the wrong launchctl subcommand or label is used.
        Oracle: the exact recorded launchctl argv.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        calls = _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home),
                    knobs=_knobs(api_key='x'))
        calls.clear()

        result = sch.trigger()
        assert result['platform'] == 'launchd'
        assert calls == [['launchctl', 'start', 'com.memman.enrich']]

    def test_trigger_raises_when_not_installed(self, fake_home, monkeypatch):
        """Verify trigger() without unit files raises FileNotFoundError.

        Mutation: trigger() reports success with nothing installed.
        Oracle: FileNotFoundError matching 'not installed'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch)
        with pytest.raises(FileNotFoundError, match='not installed'):
            sch.trigger()


class TestStateFile:
    """Scheduler state-file read/write/clear.
    """

    def test_state_file_round_trip(self, fake_home):
        """Verify read_state returns the state write_state stored.

        Mutation: write_state and read_state use different files.
        Oracle: STATE_STARTED, which differs from the read default.
        """
        sch.write_state(sch.STATE_STARTED)
        assert sch.read_state() == sch.STATE_STARTED

    @pytest.mark.parametrize('file_content', [None, 'garbage\n'])
    def test_read_state_defaults_to_stopped(self, fake_home, file_content):
        """Verify a missing or invalid state file reads as stopped.

        Mutation: a missing or garbage file reads as started.
        Oracle: no file, and a file holding 'garbage', both giving
            STATE_STOPPED.
        """
        if file_content is not None:
            sch._state_file_path().parent.mkdir(parents=True, exist_ok=True)
            sch._state_file_path().write_text(file_content)
        assert sch.read_state() == sch.STATE_STOPPED

    def test_write_state_rejects_bad_value(self, fake_home):
        """Verify write_state refuses a value outside the known states.

        Mutation: any string is accepted and persisted.
        Oracle: ValueError for 'banana'.
        """
        with pytest.raises(ValueError):
            sch.write_state('banana')

    def test_clear_state_removes_file(self, fake_home):
        """Verify clear_state deletes the state file.

        Mutation: clear_state leaves the file in place.
        Oracle: the path exists after write_state and is gone after
            clear_state.
        """
        sch.write_state(sch.STATE_STARTED)
        assert sch._state_file_path().exists()
        sch.clear_state()
        assert not sch._state_file_path().exists()


class TestServe:
    """Serve mode: continuous-loop scheduler.
    """

    def test_serve_install_records_interval(self, fake_home, monkeypatch):
        """Verify a serve-mode install records interval and STATE_STARTED.

        Mutation: the interval or the started state is not persisted.
        Oracle: DEFAULT_INTERVAL_SECONDS and STATE_STARTED read back from disk.
        """
        monkeypatch.setattr(sch, 'detect_scheduler',
                            lambda: sch.SCHEDULER_KIND_SERVE)
        monkeypatch.setattr(sch, 'memman_binary_path', lambda: '/usr/local/bin/memman')
        monkeypatch.setattr(sch, '_write_env_keys', lambda *a, **k: ['noop'])
        result = sch.install(
            str(fake_home / 'data'),
            knobs=_knobs(api_key='or-key'))
        assert result['platform'] == sch.SCHEDULER_KIND_SERVE
        assert result['state'] == sch.STATE_STARTED
        assert sch.read_serve_interval() == sch.DEFAULT_INTERVAL_SECONDS
        assert sch.read_state() == sch.STATE_STARTED


class TestDetectScheduler:
    """Platform detection.
    """

    def test_detect_raises_when_no_scheduler(self, monkeypatch):
        """Verify detect_scheduler raises when no scheduler kind is available.

        Mutation: detection falls back to a kind that is not installed.
        Oracle: Linux with no systemctl on PATH and no override in the env.
        """
        monkeypatch.delenv(config.SCHEDULER_KIND, raising=False)
        monkeypatch.setattr(platform, 'system', lambda: 'Linux')
        monkeypatch.setattr(sch.shutil, 'which', lambda _: None)
        with pytest.raises(RuntimeError, match='no scheduler available'):
            sch.detect_scheduler()


class TestMemmanBinaryPath:
    """`memman_binary_path` self-identifies via sys.executable's sibling.
    """

    def test_returns_sibling_of_sys_executable(self, tmp_path, monkeypatch):
        """Verify memman_binary_path prefers the script beside sys.executable.

        Mutation: shutil.which is preferred over the interpreter's sibling, so
            a side-by-side install (pipx plus a dev venv) writes the wrong path
            into the systemd unit.
        Oracle: a sibling script and a PATH decoy at known paths.
        """
        venv_bin = tmp_path / 'venv' / 'bin'
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / 'python'
        fake_python.touch()
        fake_memman = venv_bin / 'memman'
        fake_memman.write_text('#!/bin/sh\necho fake\n')
        Path(fake_memman).chmod(0o755)

        other_bin = tmp_path / 'other'
        other_bin.mkdir()
        decoy = other_bin / 'memman'
        decoy.write_text('#!/bin/sh\necho decoy\n')
        Path(decoy).chmod(0o755)

        monkeypatch.setattr(sch.sys, 'executable', str(fake_python))
        monkeypatch.setattr(sch.shutil, 'which', lambda _: str(decoy))

        assert sch.memman_binary_path() == str(fake_memman)

    def test_falls_back_to_path_when_sibling_missing(
            self, tmp_path, monkeypatch):
        """Verify memman_binary_path falls back to PATH when no sibling exists.

        Mutation: a missing sibling raises instead of consulting shutil.which.
        Oracle: a script at a known PATH location.
        """
        venv_bin = tmp_path / 'venv' / 'bin'
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / 'python'
        fake_python.touch()

        path_bin = tmp_path / 'pathfound'
        path_bin.mkdir()
        path_memman = path_bin / 'memman'
        path_memman.write_text('#!/bin/sh\necho path\n')
        Path(path_memman).chmod(0o755)

        monkeypatch.setattr(sch.sys, 'executable', str(fake_python))
        monkeypatch.setattr(
            sch.shutil, 'which', lambda _: str(path_memman))

        assert sch.memman_binary_path() == str(path_memman)

    def test_raises_when_neither_resolves(self, tmp_path, monkeypatch):
        """Verify memman_binary_path raises with an install hint.

        Mutation: an unresolved name is returned instead of raising.
        Oracle: RuntimeError matching 'install with pipx'.
        """
        venv_bin = tmp_path / 'venv' / 'bin'
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / 'python'
        fake_python.touch()

        monkeypatch.setattr(sch.sys, 'executable', str(fake_python))
        monkeypatch.setattr(sch.shutil, 'which', lambda _: None)

        with pytest.raises(RuntimeError, match='install with pipx'):
            sch.memman_binary_path()

    def test_sibling_must_be_executable(self, tmp_path, monkeypatch):
        """Verify a non-executable file beside python is skipped.

        Mutation: the sibling is accepted on existence alone.
        Oracle: a mode 0644 sibling and an executable script on PATH.
        """
        venv_bin = tmp_path / 'venv' / 'bin'
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / 'python'
        fake_python.touch()
        non_exec = venv_bin / 'memman'
        non_exec.write_text('not a script')
        Path(non_exec).chmod(0o644)

        path_bin = tmp_path / 'pathfound'
        path_bin.mkdir()
        path_memman = path_bin / 'memman'
        path_memman.write_text('#!/bin/sh\necho path\n')
        Path(path_memman).chmod(0o755)

        monkeypatch.setattr(sch.sys, 'executable', str(fake_python))
        monkeypatch.setattr(
            sch.shutil, 'which', lambda _: str(path_memman))

        assert sch.memman_binary_path() == str(path_memman)


class TestSchedulerLogs:
    """`memman log worker` and the install-time logs/ directory.
    """

    def test_install_creates_logs_directory(self, tmp_path, monkeypatch):
        """_install_claude_code leaves ~/.memman/logs/ owner-only.

        Worker logs carry memory content, and doctor's
        `env_permissions` check warns on any group or other bit under
        ~/.memman.

        Mutation: dropping the explicit chmod, or asking mkdir for a
            mode with group/other bits. `mkdir(mode=...)` is masked by
            the umask and does nothing at all for an existing
            directory, so only the chmod can tighten one an earlier
            install left at 0755 - which is why the fixture pre-creates
            it that way.
        Oracle: the directory's stat mode after install, compared
            against 0o700 exactly.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        # The case mkdir cannot fix: a directory that already exists,
        # group- and world-readable, from an earlier install.
        preexisting = tmp_path / '.memman' / 'logs'
        preexisting.mkdir(parents=True)
        preexisting.chmod(0o755)
        monkeypatch.setattr(
            'memman.setup.claude._init_default_store', lambda dd: None)
        env = {'config_dir': str(tmp_path / 'claude')}
        _install_claude_code(env, data_dir=str(tmp_path / 'data'))
        logs_dir = tmp_path / '.memman' / 'logs'
        assert logs_dir.is_dir()
        assert (logs_dir.stat().st_mode & 0o777) == 0o700

    def test_log_worker_reads_log_file(self, tmp_path, monkeypatch):
        """Verify `log worker --lines 2` prints the last two enrich.log lines.

        Mutation: --lines is ignored, so the whole file prints.
        Oracle: a four-line file with the marker on the last line.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        logs_dir = tmp_path / '.memman' / 'logs'
        logs_dir.mkdir(parents=True)
        (logs_dir / 'enrich.log').write_text(
            'line1\nline2\nline3\nLOG-MARKER\n')
        runner = CliRunner()
        result = runner.invoke(cli, ['log', 'worker', '--lines', '2'])
        assert result.exit_code == 0
        assert 'LOG-MARKER' in result.output
        assert 'line1' not in result.output

    def test_log_worker_errors_flag(self, tmp_path, monkeypatch):
        """Verify `log worker --errors` reads enrich.err, not enrich.log.

        Mutation: --errors is ignored and enrich.log prints.
        Oracle: distinct markers in the two files.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        logs_dir = tmp_path / '.memman' / 'logs'
        logs_dir.mkdir(parents=True)
        (logs_dir / 'enrich.err').write_text('ERR-MARKER\n')
        (logs_dir / 'enrich.log').write_text('LOG-NOT-THIS\n')
        runner = CliRunner()
        result = runner.invoke(cli, ['log', 'worker', '--errors'])
        assert result.exit_code == 0
        assert 'ERR-MARKER' in result.output
        assert 'LOG-NOT-THIS' not in result.output

    def test_log_worker_missing_file(self, tmp_path, monkeypatch):
        """Verify a missing log file gives a friendly message and exit 0.

        Mutation: a missing file raises or exits non-zero.
        Oracle: no logs directory, and the 'no log file yet' text.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['log', 'worker'])
        assert result.exit_code == 0
        assert 'no log file yet' in result.output.lower() \
            or 'no log file yet' in (result.stderr or '').lower()

    def test_log_worker_stack_reads_the_data_dir_rotated_log(
            self, tmp_path, monkeypatch):
        """`log worker --stack` reads <data-dir>/logs/memman.log.

        Mutation: resolving --stack under ~/.memman/logs like the two
            enrich targets, which is not where the rotating handler
            writes it once --data-dir moves.
        Oracle: a decoy memman.log seeded under the fake home; only
            the data-dir file's marker may surface.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        data_dir = tmp_path / 'elsewhere'
        (data_dir / 'logs').mkdir(parents=True)
        (data_dir / 'logs' / 'memman.log').write_text('STACK-MARKER\n')
        home_logs = tmp_path / '.memman' / 'logs'
        home_logs.mkdir(parents=True)
        (home_logs / 'memman.log').write_text('HOME-DECOY\n')
        result = CliRunner().invoke(
            cli, ['--data-dir', str(data_dir), 'log', 'worker', '--stack'])
        assert result.exit_code == 0, result.output
        assert 'STACK-MARKER' in result.output
        assert 'HOME-DECOY' not in result.output

    def test_log_worker_enrich_targets_ignore_the_data_dir(
            self, tmp_path, monkeypatch):
        """`log worker` reads ~/.memman/logs under a custom data dir.

        Mutation: resolving enrich.log or enrich.err from --data-dir as
            well, which would break either - the systemd unit pins
            those redirects to %h/.memman/logs and the launchd plist
            bakes in the absolute home, so neither follows the data
            dir. Both targets are exercised, so narrowing the mutation
            to one file does not escape.
        Oracle: a decoy of each name seeded under the data dir; only
            the home files' markers may surface.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        data_dir = tmp_path / 'elsewhere'
        (data_dir / 'logs').mkdir(parents=True)
        home_logs = tmp_path / '.memman' / 'logs'
        home_logs.mkdir(parents=True)
        for name in ('enrich.log', 'enrich.err'):
            (data_dir / 'logs' / name).write_text(f'DATADIR-DECOY-{name}\n')
            (home_logs / name).write_text(f'HOME-MARKER-{name}\n')
        for name, flags in (('enrich.log', []), ('enrich.err', ['--errors'])):
            result = CliRunner().invoke(
                cli,
                ['--data-dir', str(data_dir), 'log', 'worker'] + flags)
            assert result.exit_code == 0, result.output
            assert f'HOME-MARKER-{name}' in result.output
            assert f'DATADIR-DECOY-{name}' not in result.output

    def test_log_worker_stack_spans_the_rotation_backups(
            self, tmp_path, monkeypatch):
        """`log worker --stack` reads memman.log.N, not just the live file.

        Mutation: reading only the live `memman.log`. Rotation keeps
            three backups and BOTH the enrichment and backup workers
            write the file, so the traceback a CLI error pointed at is
            often already in `memman.log.1`; reading the live file
            alone prints a tail with no traceback in it while the stack
            is still on disk.
        Oracle: the traceback marker exists ONLY in `memman.log.1`,
            and ordering - the backup is older, so it must print
            before the live line.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        data_dir = tmp_path / 'elsewhere'
        (data_dir / 'logs').mkdir(parents=True)
        (data_dir / 'logs' / 'memman.log.1').write_text(
            'ROTATED-TRACEBACK\n')
        (data_dir / 'logs' / 'memman.log').write_text('LIVE-LINE\n')
        result = CliRunner().invoke(
            cli, ['--data-dir', str(data_dir), 'log', 'worker', '--stack'])
        assert result.exit_code == 0, result.output
        assert 'ROTATED-TRACEBACK' in result.output
        assert 'LIVE-LINE' in result.output
        assert (result.output.index('ROTATED-TRACEBACK')
                < result.output.index('LIVE-LINE'))

    def test_log_worker_rejects_both_target_flags(
            self, tmp_path, monkeypatch):
        """`log worker --errors --stack` is a usage error.

        Mutation: dropping the guard so one flag silently wins and the
            operator reads a different file than the one they asked
            for.
        Oracle: Click's usage exit code 2, and both flag names in the
            message.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        home_logs = tmp_path / '.memman' / 'logs'
        home_logs.mkdir(parents=True)
        (home_logs / 'enrich.err').write_text('ERR-MARKER\n')
        result = CliRunner().invoke(
            cli, ['log', 'worker', '--errors', '--stack'])
        assert result.exit_code == 2, result.output
        assert '--errors' in result.output
        assert '--stack' in result.output
        assert 'ERR-MARKER' not in result.output


def _install_env_full(data_dir):
    """Seed an env file with a representative mix of keys.

    Used by `TestUninstall` to verify the strip-secrets path.
    """
    install_env_factory(
        data_dir,
        **{
            config.ENDPOINT: 'https://openrouter.ai/api/v1',
            config.API_KEY: 'sk-installed',
            config.LLM_MODEL: 'anthropic/claude-sonnet-4.6',
            config.EMBED_MODEL: 'voyageai/voyage-4-lite',
            config.DEFAULT_BACKEND: 'postgres',
            config.DEFAULT_PG_DSN: 'postgresql://user:pw@host/db',
            })


class TestUninstall:
    """`sch.uninstall` strips secrets and is a no-op on empty data dirs.
    """

    def test_uninstall_strips_secrets_keeps_settings(
            self, uninstall_home, monkeypatch):
        """Verify uninstall removes secrets and keeps non-secret settings.

        Mutation: a secret key survives, or a setting is stripped with them.
        Oracle: a seeded env file with a known mix of both kinds.
        """
        _, data_dir = uninstall_home
        _install_env_full(data_dir)

        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch, active=False)

        sch.uninstall(data_dir=str(data_dir))

        contents = (data_dir / config.ENV_FILENAME).read_text()
        assert config.API_KEY not in contents
        assert config.DEFAULT_PG_DSN not in contents
        assert (f'{config.ENDPOINT}=https://openrouter.ai/api/v1'
                in contents)
        assert config.LLM_MODEL in contents
        assert config.EMBED_MODEL in contents
        assert f'{config.DEFAULT_BACKEND}=postgres' in contents

    @pytest.mark.no_default_env
    def test_uninstall_no_op_when_no_env_file(
            self, uninstall_home, monkeypatch):
        """Verify uninstall with no env file reports no env actions.

        Mutation: uninstall raises, or reports an action, when the file is
            absent.
        Oracle: an empty data dir and an empty env_actions list.
        """
        _, data_dir = uninstall_home
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _no_subprocess(monkeypatch, active=False)

        result = sch.uninstall(data_dir=str(data_dir))
        assert result['env_actions'] == []


class TestBackupScheduler:
    """`install_backup` / `uninstall_backup` and the backup-state helpers.
    """

    def test_install_backup_systemd_writes_calendar_timer(
            self, fake_home, fake_binary, monkeypatch):
        """Verify install_backup writes an OnCalendar timer and service.

        Mutation: the cron expression converts to the wrong OnCalendar, or the
            enable and restart calls are skipped.
        Oracle: '0 3 * * *' giving '*-*-* 03:00:00', and the recorded systemctl
            argvs.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        calls = _record_subprocess(monkeypatch)
        result = sch.install_backup(str(fake_home / '.memman'), '0 3 * * *')
        assert result['platform'] == 'systemd'
        timer = Path(result['timer_path']).read_text()
        service = Path(result['service_path']).read_text()
        assert 'OnCalendar=*-*-* 03:00:00' in timer
        assert 'Persistent=true' in timer
        assert '/fake/bin/memman backup worker' in service
        argvs = [' '.join(c) for c in calls]
        assert any('daemon-reload' in a for a in argvs)
        assert any('enable memman-backup.timer' in a for a in argvs)
        assert any('restart memman-backup.timer' in a for a in argvs)

    def test_install_backup_systemd_carries_install_path(
            self, fake_home, fake_binary, monkeypatch):
        """Verify the systemd backup service runs with the install-time PATH.

        Mutation: the service inheriting systemd's minimal PATH, so a
            pg_dump in a user directory is never found and every Postgres
            store's backup fails.
        Oracle: the PATH value set before install, written verbatim.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        monkeypatch.setenv('PATH', '/opt/pg/bin:/usr/bin')
        _record_subprocess(monkeypatch)
        result = sch.install_backup(str(fake_home / '.memman'), '0 3 * * *')
        service = Path(result['service_path']).read_text()
        assert 'Environment="PATH=/opt/pg/bin:/usr/bin"\n' in service

    def test_install_backup_launchd_carries_install_path(
            self, fake_home, fake_binary, monkeypatch):
        """Verify the launchd backup wrapper exports the install-time PATH.

        Mutation: the wrapper inheriting launchd's minimal PATH, so a
            pg_dump in a user directory is never found.
        Oracle: the PATH value set before install, written verbatim.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        monkeypatch.setenv('PATH', '/opt/pg/bin:/usr/bin')
        _record_subprocess(monkeypatch)
        result = sch.install_backup(str(fake_home / '.memman'), '0 3 * * *')
        wrapper = Path(result['wrapper_path']).read_text()
        assert 'export PATH=/opt/pg/bin:/usr/bin\n' in wrapper

    def test_install_backup_launchd_writes_calendar_plist(
            self, fake_home, fake_binary, monkeypatch):
        """Verify a multi-value cron gives a launchd interval array.

        Mutation: the plist omits the array, or sets RunAtLoad true.
        Oracle: hand-written plist substrings for '*/15 * * * *'.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _record_subprocess(monkeypatch)
        result = sch.install_backup(
            str(fake_home / '.memman'), '*/15 * * * *')
        plist = Path(result['plist_path']).read_text()
        wrapper = Path(result['wrapper_path']).read_text()
        assert '<key>StartCalendarInterval</key>' in plist
        assert '<array>' in plist
        assert '<key>RunAtLoad</key><false/>' in plist
        assert 'backup worker' in wrapper
        assert os.access(result['wrapper_path'], os.X_OK)

    def test_install_backup_launchd_single_value_is_dict(
            self, fake_home, fake_binary, monkeypatch):
        """Verify a single-value cron gives one calendar dict, not an array.

        Mutation: a single value is wrapped in a one-element array.
        Oracle: Minute 0 and Hour 3 in a dict, with no array element.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'launchd')
        _record_subprocess(monkeypatch)
        result = sch.install_backup(str(fake_home / '.memman'), '0 3 * * *')
        plist = Path(result['plist_path']).read_text()
        assert '<key>Minute</key><integer>0</integer>' in plist
        assert '<key>Hour</key><integer>3</integer>' in plist
        assert '<key>StartCalendarInterval</key>\n  <dict>' in plist
        assert '<key>StartCalendarInterval</key>\n  <array>' not in plist

    def test_uninstall_backup_leaves_enrich_units(
            self, fake_home, fake_binary, monkeypatch):
        """Verify uninstall_backup removes backup units, keeps enrich ones.

        Mutation: uninstall_backup removes the enrich timer or keeps backup
            state.
        Oracle: unit file existence, and read_backup_state returning None.
        """
        monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')
        _record_subprocess(monkeypatch)
        sch.install(data_dir=str(fake_home / '.memman'),
                    knobs=_knobs(api_key='x'))
        sch.install_backup(str(fake_home / '.memman'), '0 3 * * *')
        sch.write_backup_state('2026-06-27T03:00')
        unit_dir = fake_home / '.config' / 'systemd' / 'user'
        assert (unit_dir / sch.SYSTEMD_BACKUP_TIMER_NAME).exists()

        sch.uninstall_backup()

        assert not (unit_dir / sch.SYSTEMD_BACKUP_TIMER_NAME).exists()
        assert not (unit_dir / sch.SYSTEMD_BACKUP_SERVICE_NAME).exists()
        assert (unit_dir / sch.SYSTEMD_TIMER_NAME).exists()
        assert sch.read_backup_state() is None

    def test_backup_state_round_trip(self, fake_home):
        """Verify the backup state stamp writes, reads, and clears.

        Mutation: read_backup_state disagrees with write, or clear leaves the
            stamp.
        Oracle: the literal stamp '2026-06-27T03:00'.
        """
        sch.write_backup_state('2026-06-27T03:00')
        assert sch.read_backup_state() == '2026-06-27T03:00'
        sch.clear_backup_state()
        assert sch.read_backup_state() is None
