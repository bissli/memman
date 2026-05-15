"""Tests for `memman.config.collect_install_knobs`."""

import pytest
from memman import config
from memman.exceptions import ConfigError


@pytest.fixture
def stub_resolver(monkeypatch):
    """Stub resolve_latest_for_role so tests don't hit the network."""
    calls = []

    def fake(role, endpoint='https://openrouter.ai/api/v1'):
        calls.append((role, endpoint))
        family = 'haiku' if role == 'fast' else 'sonnet'
        return f'anthropic/claude-{family}-4.5'

    monkeypatch.setattr(
        'memman.llm.openrouter_models.resolve_latest_for_role', fake)
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
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
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
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
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
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        roles = [c[0] for c in stub_resolver]
        assert 'fast' in roles
        assert 'slow' in roles
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
            'memman.llm.openrouter_models.resolve_latest_for_role',
            lambda *a, **k: None)
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL_FAST] == \
            config.INSTALL_DEFAULTS[config.LLM_MODEL_FAST]

    def test_missing_mandatory_secret_raises(self, tmp_path, monkeypatch):
        """ConfigError when the embed provider's mandatory secret is missing."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
        monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match='MEMMAN_VOYAGE_API_KEY'):
            config.collect_install_knobs(data_dir)

    def test_backend_default_is_sqlite(
            self, tmp_path, monkeypatch, stub_resolver):
        """`MEMMAN_DEFAULT_BACKEND` resolves to 'sqlite' from INSTALL_DEFAULTS."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.DEFAULT_BACKEND] == 'sqlite'

    def test_native_voyage_key_seeds_memman_voyage_key(
            self, tmp_path, monkeypatch, stub_resolver):
        """Native `VOYAGE_API_KEY` (no MEMMAN- prefix) seeds the memman key."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('VOYAGE_API_KEY', 'native-vy-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_API_KEY] == 'native-vy-key'

    def test_native_openrouter_key_cascades_into_llm_api_key(
            self, tmp_path, monkeypatch, stub_resolver):
        """Native `OPENROUTER_API_KEY` seeds OR key AND cascades to LLM key."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENROUTER_API_KEY] == 'native-or-key'
        assert knobs[config.LLM_API_KEY] == 'native-or-key'

    def test_native_openai_key_seeds_memman_openai_embed_key(
            self, tmp_path, monkeypatch, stub_resolver):
        """Native `OPENAI_API_KEY` seeds MEMMAN_OPENAI_EMBED_API_KEY."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('OPENAI_API_KEY', 'native-oai-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENAI_EMBED_API_KEY] == 'native-oai-key'

    def test_memman_prefixed_wins_over_native(
            self, tmp_path, monkeypatch, stub_resolver):
        """When both MEMMAN- and native are exported, MEMMAN- wins."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'memman-vy-key')
        monkeypatch.setenv('VOYAGE_API_KEY', 'native-vy-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'memman-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_API_KEY] == 'memman-vy-key'

    def test_file_wins_over_native_shell(
            self, tmp_path, monkeypatch, stub_resolver):
        """File value still wins over native shell fallback (sticky seed)."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.VOYAGE_API_KEY}=file-vy-key\n'
            f'{config.OPENROUTER_API_KEY}=file-or-key\n'
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv('VOYAGE_API_KEY', 'native-vy-key')
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.VOYAGE_API_KEY] == 'file-vy-key'
        assert knobs[config.OPENROUTER_API_KEY] == 'file-or-key'

    def test_voyage_embed_model_default_is_written(
            self, tmp_path, monkeypatch, stub_resolver):
        """`MEMMAN_VOYAGE_EMBED_MODEL=voyage-3-lite` lands from INSTALL_DEFAULTS."""
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_EMBED_MODEL] == 'voyage-3-lite'

    def test_backend_value_round_trips_from_file(
            self, tmp_path, monkeypatch, stub_resolver):
        """Existing `MEMMAN_DEFAULT_BACKEND=postgres` in the file is preserved."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n'
            f'{config.DEFAULT_BACKEND}=postgres\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.DEFAULT_BACKEND] == 'postgres'
