"""Tests for `memman.config.collect_install_knobs`.
"""

import httpx
import pytest
from memman import config
from memman.exceptions import ConfigError


@pytest.mark.no_default_env
class TestCollectInstallKnobs:
    """`config.collect_install_knobs` precedence and refusals.

    All tests share the `no_default_env` mark. The autouse
    `_isolate_env` fixture skips its env seeding so each test can craft
    the file/shell state it asserts on.
    """

    def test_file_value_wins_over_shell_env(
            self, tmp_path, monkeypatch):
        """Verify an env-file value beats a conflicting shell export.

        Mutation: reading os.environ before the env file, so a later shell
            export overrides a pinned value on reinstall.
        Oracle: the file values 'file/sonnet-pin' and 'file-or-key' against
            different shell values.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=file-or-key\n'
            f'{config.LLM_MODEL}=file/sonnet-pin\n'
            f'{config.ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv(config.LLM_MODEL, 'env/sonnet-OVERRIDE')
        monkeypatch.setenv(config.API_KEY, 'env-or-OVERRIDE')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL] == 'file/sonnet-pin'
        assert knobs[config.API_KEY] == 'file-or-key'

    def test_shell_env_seeds_file_when_key_missing(
            self, tmp_path, monkeypatch):
        """Verify a shell export fills a key the env file lacks.

        Mutation: ignoring os.environ for keys absent from the file, so install
            fails or writes the default.
        Oracle: the two shell values read back from the knobs.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.API_KEY, 'shell-or-key')
        monkeypatch.setenv(config.LLM_MODEL, 'shell/sonnet-seed')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.API_KEY] == 'shell-or-key'
        assert knobs[config.LLM_MODEL] == 'shell/sonnet-seed'

    def test_file_value_wins_over_default(
            self, tmp_path, monkeypatch):
        """Existing env-file value is preserved across re-installs.

        Mutation: reading `INSTALL_DEFAULTS` ahead of an env-file value
            already on disk, which would overwrite a pinned model slug
            on every reinstall.
        Oracle: the pinned file values, read back from
            `collect_install_knobs`.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.LLM_MODEL}=file/sonnet-pinned\n'
            f'{config.API_KEY}=file-or-key\n'
            f'{config.ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        monkeypatch.delenv(config.API_KEY, raising=False)
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL] == 'file/sonnet-pinned'
        assert knobs[config.API_KEY] == 'file-or-key'

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        """Verify install raises ConfigError for a missing shared API key.

        Mutation: dropping the key check, so install writes a file with no
            key for a remote endpoint.
        Oracle: ConfigError naming MEMMAN_API_KEY, with the key absent
            from file and shell on the default OpenRouter endpoint.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.delenv(config.API_KEY, raising=False)
        monkeypatch.delenv(config.OPENROUTER_NATIVE_API_KEY, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match='MEMMAN_API_KEY'):
            config.collect_install_knobs(data_dir)

    def test_blank_api_key_allowed_on_loopback(self, tmp_path, monkeypatch):
        """Verify a loopback endpoint installs without an API key.

        Mutation: requiring the key on every endpoint, which blocks a
            keyless local server install.
        Oracle: no ConfigError for a localhost endpoint with all three
            models set and no key.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.ENDPOINT}=http://localhost:11434/v1\n'
            f'{config.LLM_MODEL}=m\n'
            f'{config.EMBED_MODEL}=e\n'
            f'{config.RERANK_MODEL}=r\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert config.API_KEY not in knobs

    def test_backend_default_is_sqlite(
            self, tmp_path, monkeypatch):
        """Verify DEFAULT_BACKEND resolves to 'sqlite' from INSTALL_DEFAULTS.

        Mutation: INSTALL_DEFAULTS omitting DEFAULT_BACKEND or naming another
            backend.
        Oracle: the literal 'sqlite'.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=or-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.DEFAULT_BACKEND] == 'sqlite'

    def test_backup_keep_default_present_cron_target_absent(
            self, tmp_path, monkeypatch):
        """Verify BACKUP_KEEP defaults to '7' and BACKUP_CRON/TARGET stay unset.

        Mutation: seeding a default for BACKUP_CRON or BACKUP_TARGET, raising
            for them, or dropping the BACKUP_KEEP default.
        Oracle: the literal '7' and the absence of both keys.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=or-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.BACKUP_KEEP] == '7'
        assert config.BACKUP_CRON not in knobs
        assert config.BACKUP_TARGET not in knobs

    def test_native_openrouter_key_seeds_shared_key(
            self, tmp_path, monkeypatch):
        """Verify the native OPENROUTER_API_KEY seeds MEMMAN_API_KEY.

        Mutation: dropping the native-name seed, so install fails for a
            user who exported only the vendor variable.
        Oracle: the native value read back under MEMMAN_API_KEY on the
            default OpenRouter endpoint.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.API_KEY] == 'native-or-key'

    def test_native_openrouter_key_ignored_off_openrouter(
            self, tmp_path, monkeypatch):
        """Verify the native OPENROUTER_API_KEY never seeds another endpoint.

        Mutation: seeding the shared key from the OpenRouter export on any
            endpoint, which sends the OpenRouter secret to a third party.
        Oracle: ConfigError naming MEMMAN_API_KEY for a remote
            non-OpenRouter endpoint with all three models set.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.ENDPOINT}=https://api.example.com/v1\n'
            f'{config.LLM_MODEL}=m\n'
            f'{config.EMBED_MODEL}=e\n'
            f'{config.RERANK_MODEL}=r\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        config.reset_file_cache()
        with pytest.raises(ConfigError, match='MEMMAN_API_KEY'):
            config.collect_install_knobs(str(data_dir))

    def test_memman_prefixed_wins_over_native(
            self, tmp_path, monkeypatch):
        """Verify the MEMMAN- name beats the vendor-native name.

        Mutation: checking the native name first.
        Oracle: 'memman-key' read back with both names exported.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.API_KEY, 'memman-key')
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.API_KEY] == 'memman-key'

    def test_file_wins_over_native_shell(
            self, tmp_path, monkeypatch):
        """Verify an env-file value beats a vendor-native shell export.

        Mutation: letting the native-name fallback override a file value.
        Oracle: the file value read back with a different native export set.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=file-or-key\n'
            f'{config.ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.API_KEY] == 'file-or-key'

    def test_embed_and_rerank_model_defaults_are_written(
            self, tmp_path, monkeypatch):
        """Verify the embed and rerank models default from INSTALL_DEFAULTS.

        Mutation: dropping EMBED_MODEL or RERANK_MODEL from
            INSTALL_DEFAULTS, or changing a value.
        Oracle: the literals 'voyageai/voyage-4-lite' and
            'voyageai/rerank-3-lite'.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.API_KEY, 'key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.EMBED_MODEL] == 'voyageai/voyage-4-lite'
        assert knobs[config.RERANK_MODEL] == 'voyageai/rerank-3-lite'

    def test_backend_value_round_trips_from_file(
            self, tmp_path, monkeypatch):
        """Verify a DEFAULT_BACKEND already in the env file survives install.

        Mutation: overwriting the file value with the 'sqlite' default.
        Oracle: the literal 'postgres' read back.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=or-key\n'
            f'{config.DEFAULT_BACKEND}=postgres\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.DEFAULT_BACKEND] == 'postgres'

    def test_install_never_writes_a_catalog_picked_model(
            self, tmp_path, monkeypatch):
        """An OpenRouter install with no model seeds the shipped default.

        Mutation: install consults the OpenRouter catalog and writes the
            model it picks, switching the model with no operator choice.
        Oracle: `INSTALL_DEFAULTS`, against a stubbed catalog whose
            newest snapshot of the default's line differs from it.
        """
        newer = {'data': [{'id': 'qwen/qwen3-235b-a22b-2607'}]}
        monkeypatch.setattr(
            httpx.Client, 'get',
            lambda self, url, **kwargs: httpx.Response(
                200, json=newer, request=httpx.Request('GET', url)))
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.API_KEY}=or-key\n'
            f'{config.ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL] == \
            config.INSTALL_DEFAULTS[config.LLM_MODEL]

    def test_non_openrouter_install_with_no_model_refuses(
            self, tmp_path, monkeypatch):
        """A non-OpenRouter install with no model names the missing key.

        Mutation: the install falls back to the OpenRouter qwen default,
            which the endpoint rejects on the first enrichment call.
        Oracle: an Anthropic endpoint with the model absent from both
            the env file and the shell.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.ENDPOINT}=https://api.anthropic.com/v1\n'
            f'{config.API_KEY}=sk-ant-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match=config.LLM_MODEL):
            config.collect_install_knobs(str(data_dir))
