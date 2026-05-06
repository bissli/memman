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


@pytest.mark.no_default_env
class TestCollectInstallKnobs:
    """`config.collect_install_knobs` precedence and resolver behavior.

    All tests share `stub_resolver` and the `no_default_env` mark. The
    autouse `_isolate_env` fixture skips its env seeding so each test
    can craft the file/shell state it asserts on.
    """

    def test_file_value_wins_over_shell_env(
            self, tmp_path, monkeypatch, stub_resolver):
        """File values are sticky; shell env never overrides them on reinstall."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=file-or-key\n'
            f'{config.VOYAGE_API_KEY}=file-vy-key\n'
            f'{config.LLM_MODEL_FAST}=file/haiku-pin\n'
            f'{config.LLM_PROVIDER}=openrouter\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv(config.LLM_MODEL_FAST, 'env/haiku-OVERRIDE')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'env-or-OVERRIDE')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL_FAST] == 'file/haiku-pin'
        assert knobs[config.OPENROUTER_API_KEY] == 'file-or-key'

    def test_shell_env_seeds_file_when_key_missing(
            self, tmp_path, monkeypatch, stub_resolver):
        """Shell env values fill blanks in the file at install time."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        monkeypatch.setenv(config.LLM_MODEL_FAST, 'shell/haiku-seed')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENROUTER_API_KEY] == 'shell-or-key'
        assert knobs[config.VOYAGE_API_KEY] == 'shell-vy-key'
        assert knobs[config.LLM_MODEL_FAST] == 'shell/haiku-seed'

    def test_file_value_wins_over_resolver_and_default(
            self, tmp_path, monkeypatch, stub_resolver):
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

    def test_resolver_fires_when_file_lacks_model_keys(
            self, tmp_path, monkeypatch, stub_resolver):
        """Resolver runs only when the env file has no value for a model key."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n'
            f'{config.LLM_PROVIDER}=openrouter\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        families = [c[2] for c in stub_resolver]
        assert 'haiku' in families
        assert 'sonnet' in families
        assert knobs[config.LLM_MODEL_FAST] == 'anthropic/claude-haiku-4.5'
        assert knobs[config.LLM_MODEL_SLOW_CANONICAL] == 'anthropic/claude-sonnet-4.5'

    def test_resolver_none_falls_back_to_install_defaults(
            self, tmp_path, monkeypatch):
        """When the resolver returns None, INSTALL_DEFAULTS is used."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setattr(
            'memman.llm.openrouter_models.resolve_latest_in_family',
            lambda *a, **k: None)
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL_FAST] == \
            config.INSTALL_DEFAULTS[config.LLM_MODEL_FAST]

    def test_missing_mandatory_secret_raises(self, tmp_path, monkeypatch):
        """ConfigError when a mandatory secret is in neither file nor env."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
        monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match='OPENROUTER_API_KEY'):
            config.collect_install_knobs(data_dir)

    def test_backend_default_is_sqlite(
            self, tmp_path, monkeypatch, stub_resolver):
        """`MEMMAN_BACKEND` resolves to 'sqlite' from INSTALL_DEFAULTS."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.BACKEND] == 'sqlite'

    def test_backend_value_round_trips_from_file(
            self, tmp_path, monkeypatch, stub_resolver):
        """Existing `MEMMAN_BACKEND=postgres` in the file is preserved."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n'
            f'{config.BACKEND}=postgres\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.BACKEND] == 'postgres'
