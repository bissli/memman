"""Tests for `memman.doctor.check_env_completeness`."""

import pytest
from memman import config
from memman.doctor import check_env_completeness


@pytest.fixture
def write_env(tmp_path, monkeypatch):
    """Write a custom env file under a fresh data dir."""
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(config.DATA_DIR, str(data_dir))

    def _write(contents: str) -> None:
        (data_dir / config.ENV_FILENAME).write_text(contents)
        config.reset_file_cache()

    return _write


def test_env_completeness_pass_when_all_present(write_env):
    """All INSTALLABLE_KEYS in the file -> status pass."""
    lines = []
    for key in config.INSTALLABLE_KEYS:
        lines.append(f'{key}=value-for-{key}')
    write_env('\n'.join(lines) + '\n')
    out = check_env_completeness()
    assert out['status'] == 'pass'


def test_env_completeness_warns_when_non_secret_missing(write_env):
    """Missing non-secret key -> warn with key in detail.missing."""
    lines = [
        f'{key}=v' for key in config.INSTALLABLE_KEYS
        if key != config.LLM_MODEL_FAST
        ]
    write_env('\n'.join(lines) + '\n')
    out = check_env_completeness()
    assert out['status'] == 'warn'
    assert config.LLM_MODEL_FAST in out['detail']['missing']
    assert 'memman install' in out['detail']['fix']


def test_env_completeness_ignores_optional_secret(write_env):
    """Missing OPENAI_EMBED_API_KEY (optional secret) does not fail."""
    lines = [
        f'{key}=v' for key in config.INSTALLABLE_KEYS
        if key != config.OPENAI_EMBED_API_KEY
        ]
    write_env('\n'.join(lines) + '\n')
    out = check_env_completeness()
    assert out['status'] == 'pass'
    assert config.OPENAI_EMBED_API_KEY not in out.get('detail', {}).get(
        'missing', [])
