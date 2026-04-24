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


def _no_subprocess(monkeypatch):
    """Stop scheduler functions from invoking systemctl/launchctl."""
    monkeypatch.setattr(sch, 'subprocess',
                        type('S', (), {'run': staticmethod(
                            lambda *a, **kw: None)})())


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
