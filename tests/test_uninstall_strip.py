"""End-to-end test: `memman uninstall` strips secrets from env file."""

import pytest
from memman import config
from memman.setup import scheduler as sch
from tests.conftest import install_env_factory


@pytest.fixture
def uninstall_home(fake_home, monkeypatch):
    """fake_home plus a `MEMMAN_DATA_DIR` pin under it."""
    data_dir = fake_home / 'memman'
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
    config.reset_file_cache()
    return fake_home, data_dir


def _install_env(data_dir):
    install_env_factory(
        data_dir,
        **{
            config.LLM_PROVIDER: 'openrouter',
            config.LLM_MODEL_FAST: 'anthropic/claude-haiku-4.5',
            config.LLM_MODEL_SLOW_CANONICAL: 'anthropic/claude-sonnet-4.6',
            config.EMBED_PROVIDER: 'voyage',
            config.OPENROUTER_API_KEY: 'sk-or-installed',
            config.VOYAGE_API_KEY: 'pa-installed',
            config.OPENAI_EMBED_API_KEY: 'sk-oa-installed',
            config.BACKEND: 'postgres',
            config.PG_DSN: 'postgresql://user:pw@host/db',
            })


def test_uninstall_strips_secrets_keeps_settings(uninstall_home, monkeypatch):
    """Secrets removed; non-secret settings preserved."""
    _, data_dir = uninstall_home
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
def test_uninstall_no_op_when_no_env_file(uninstall_home, monkeypatch):
    """Uninstall against an empty data dir doesn't error."""
    _, data_dir = uninstall_home
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
