"""End-to-end test: `memman uninstall` strips secrets from env file."""

import pytest
from memman import config
from memman.setup import scheduler as sch


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect HOME and pin MEMMAN_DATA_DIR."""
    from pathlib import Path
    data_dir = tmp_path / 'memman'
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
    config.reset_file_cache()
    return tmp_path, data_dir


def _install_env(data_dir):
    """Write an env file with a representative mix of keys."""
    data_dir.mkdir(parents=True, exist_ok=True)
    contents = '\n'.join([
        f'{config.LLM_PROVIDER}=openrouter',
        f'{config.LLM_MODEL_FAST}=anthropic/claude-haiku-4.5',
        f'{config.LLM_MODEL_SLOW_CANONICAL}=anthropic/claude-sonnet-4.6',
        f'{config.EMBED_PROVIDER}=voyage',
        f'{config.OPENROUTER_API_KEY}=sk-or-installed',
        f'{config.VOYAGE_API_KEY}=pa-installed',
        f'{config.OPENAI_EMBED_API_KEY}=sk-oa-installed',
        f'{config.BACKEND}=postgres',
        f'{config.PG_DSN}=postgresql://user:pw@host/db',
        ]) + '\n'
    (data_dir / config.ENV_FILENAME).write_text(contents)
    (data_dir / config.ENV_FILENAME).chmod(0o600)
    config.reset_file_cache()


def test_uninstall_strips_secrets_keeps_settings(fake_home, monkeypatch):
    """Secrets removed; non-secret settings preserved."""
    _, data_dir = fake_home
    _install_env(data_dir)

    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')

    class _FakeResult:
        returncode = 0
        stdout = 'inactive'
        stderr = ''

    fake_subprocess = type('S', (), {
        'run': staticmethod(lambda *a, **k: _FakeResult()),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(sch, 'subprocess', fake_subprocess)

    sch.uninstall(data_dir=str(data_dir))

    contents = (data_dir / config.ENV_FILENAME).read_text()
    assert config.OPENROUTER_API_KEY not in contents
    assert config.VOYAGE_API_KEY not in contents
    assert config.OPENAI_EMBED_API_KEY not in contents
    assert config.PG_DSN not in contents
    assert f'{config.LLM_PROVIDER}=openrouter' in contents
    assert config.LLM_MODEL_FAST in contents
    assert config.EMBED_PROVIDER in contents
    assert f'{config.BACKEND}=postgres' in contents


@pytest.mark.no_default_env
def test_uninstall_no_op_when_no_env_file(fake_home, monkeypatch):
    """Uninstall against an empty data dir doesn't error."""
    _, data_dir = fake_home
    monkeypatch.setattr(sch, 'detect_scheduler', lambda: 'systemd')

    class _FakeResult:
        returncode = 0
        stdout = ''
        stderr = ''

    fake_subprocess = type('S', (), {
        'run': staticmethod(lambda *a, **k: _FakeResult()),
        'TimeoutExpired': TimeoutError,
        })()
    monkeypatch.setattr(sch, 'subprocess', fake_subprocess)

    result = sch.uninstall(data_dir=str(data_dir))
    assert result['env_actions'] == []
