"""Tests for `memman.config.collect_install_knobs`."""

import pytest
from memman import config
from memman.exceptions import ConfigError


@pytest.fixture
def stub_resolver(monkeypatch):
    """Stub resolve_latest_in_family so tests don't hit the network."""
    calls = []

    def fake(api_key, endpoint, family):
        calls.append((api_key, endpoint, family))
        return f'anthropic/claude-{family}-4.5'

    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_in_family', fake)
    return calls


def test_env_value_wins_over_file_and_default(
        tmp_path, monkeypatch, stub_resolver):
    """os.environ takes precedence over the env file and INSTALL_DEFAULTS."""
    data_dir = str(tmp_path / 'memman')
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'env-or-key')
    monkeypatch.setenv(config.VOYAGE_API_KEY, 'env-vy-key')
    monkeypatch.setenv(config.LLM_MODEL_FAST, 'env/haiku-pin')
    config.reset_file_cache()
    knobs = config.collect_install_knobs(data_dir)
    assert knobs[config.LLM_MODEL_FAST] == 'env/haiku-pin'
    assert knobs[config.OPENROUTER_API_KEY] == 'env-or-key'


@pytest.mark.no_default_env
def test_file_value_wins_over_resolver_and_default(
        tmp_path, monkeypatch, stub_resolver):
    """Existing env-file value is preserved across re-installs."""
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.LLM_MODEL_FAST}=file/haiku-pinned\n'
        f'{config.LLM_MODEL_SLOW_CANONICAL}=file/sonnet-pinned\n'
        f'{config.LLM_MODEL_SLOW_METADATA}=file/sonnet-pinned\n'
        f'{config.OPENROUTER_API_KEY}=file-or-key\n'
        f'{config.VOYAGE_API_KEY}=file-vy-key\n'
        f'{config.LLM_PROVIDER}=openrouter\n')
    monkeypatch.setenv(config.DATA_DIR, str(data_dir))
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    monkeypatch.delenv(config.LLM_MODEL_SLOW_CANONICAL, raising=False)
    monkeypatch.delenv(config.LLM_MODEL_SLOW_METADATA, raising=False)
    monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
    monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
    config.reset_file_cache()
    knobs = config.collect_install_knobs(str(data_dir))
    assert knobs[config.LLM_MODEL_FAST] == 'file/haiku-pinned'
    assert knobs[config.LLM_MODEL_SLOW_CANONICAL] == 'file/sonnet-pinned'
    assert knobs[config.LLM_MODEL_SLOW_METADATA] == 'file/sonnet-pinned'
    assert knobs[config.OPENROUTER_API_KEY] == 'file-or-key'
    assert stub_resolver == [], 'resolver should NOT fire when file has value'


@pytest.mark.no_default_env
def test_resolver_fires_when_neither_env_nor_file_set(
        tmp_path, monkeypatch, stub_resolver):
    """Resolver runs only when both env and file are empty."""
    data_dir = str(tmp_path / 'memman')
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'or-key')
    monkeypatch.setenv(config.VOYAGE_API_KEY, 'vy-key')
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    monkeypatch.delenv(config.LLM_MODEL_SLOW_CANONICAL, raising=False)
    config.reset_file_cache()
    knobs = config.collect_install_knobs(data_dir)
    families = [c[2] for c in stub_resolver]
    assert 'haiku' in families
    assert 'sonnet' in families
    assert knobs[config.LLM_MODEL_FAST] == 'anthropic/claude-haiku-4.5'
    assert knobs[config.LLM_MODEL_SLOW_CANONICAL] == 'anthropic/claude-sonnet-4.5'


@pytest.mark.no_default_env
def test_resolver_none_falls_back_to_install_defaults(
        tmp_path, monkeypatch):
    """When the resolver returns None, INSTALL_DEFAULTS is used."""
    data_dir = str(tmp_path / 'memman')
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'or-key')
    monkeypatch.setenv(config.VOYAGE_API_KEY, 'vy-key')
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_in_family',
        lambda *a, **k: None)
    config.reset_file_cache()
    knobs = config.collect_install_knobs(data_dir)
    assert knobs[config.LLM_MODEL_FAST] == \
        config.INSTALL_DEFAULTS[config.LLM_MODEL_FAST]


@pytest.mark.no_default_env
def test_missing_mandatory_secret_raises(tmp_path, monkeypatch):
    """ConfigError is raised when a mandatory secret has no value."""
    data_dir = str(tmp_path / 'memman')
    monkeypatch.setenv(config.DATA_DIR, data_dir)
    monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
    monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
    config.reset_file_cache()
    with pytest.raises(ConfigError, match='OPENROUTER_API_KEY'):
        config.collect_install_knobs(data_dir)
