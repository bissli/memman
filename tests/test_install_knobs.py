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
            f'{config.OPENROUTER_API_KEY}=file-or-key\n'
            f'{config.VOYAGE_API_KEY}=file-vy-key\n'
            f'{config.LLM_MODEL}=file/sonnet-pin\n'
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.setenv(config.LLM_MODEL, 'env/sonnet-OVERRIDE')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'env-or-OVERRIDE')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL] == 'file/sonnet-pin'
        assert knobs[config.OPENROUTER_API_KEY] == 'file-or-key'

    def test_shell_env_seeds_file_when_key_missing(
            self, tmp_path, monkeypatch):
        """Verify a shell export fills a key the env file lacks.

        Mutation: ignoring os.environ for keys absent from the file, so install
            fails or writes the default.
        Oracle: the three shell values read back from the knobs.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        monkeypatch.setenv(config.LLM_MODEL, 'shell/sonnet-seed')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENROUTER_API_KEY] == 'shell-or-key'
        assert knobs[config.VOYAGE_API_KEY] == 'shell-vy-key'
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
            f'{config.OPENROUTER_API_KEY}=file-or-key\n'
            f'{config.VOYAGE_API_KEY}=file-vy-key\n'
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
        monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.LLM_MODEL] == 'file/sonnet-pinned'
        assert knobs[config.OPENROUTER_API_KEY] == 'file-or-key'

    def test_missing_mandatory_secret_raises(self, tmp_path, monkeypatch):
        """Verify install raises ConfigError for a missing embed-provider secret.

        Mutation: dropping the required_install_keys check, so install writes a
            file with no embed key.
        Oracle: ConfigError naming MEMMAN_VOYAGE_API_KEY, with the key absent
            from file and shell.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
        monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match='MEMMAN_VOYAGE_API_KEY'):
            config.collect_install_knobs(data_dir)

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
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
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
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        config.reset_file_cache()
        knobs = config.collect_install_knobs(str(data_dir))
        assert knobs[config.BACKUP_KEEP] == '7'
        assert config.BACKUP_CRON not in knobs
        assert config.BACKUP_TARGET not in knobs

    def test_native_voyage_key_seeds_memman_voyage_key(
            self, tmp_path, monkeypatch):
        """Verify the vendor-native VOYAGE_API_KEY seeds the memman key.

        Mutation: dropping the native-name fallback, so install fails for a
            user who exported only the vendor variable.
        Oracle: the native value read back under MEMMAN_VOYAGE_API_KEY.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('VOYAGE_API_KEY', 'native-vy-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_API_KEY] == 'native-vy-key'

    def test_native_openrouter_key_cascades_into_llm_api_key(
            self, tmp_path, monkeypatch):
        """Verify the native OPENROUTER_API_KEY seeds the OR key and the LLM key.

        Mutation: seeding the OpenRouter key without copying it to
            MEMMAN_LLM_API_KEY, so the LLM client has no key.
        Oracle: both knobs read back as 'native-or-key'.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENROUTER_API_KEY] == 'native-or-key'
        assert knobs[config.LLM_API_KEY] == 'native-or-key'

    def test_native_openai_key_seeds_memman_openai_embed_key(
            self, tmp_path, monkeypatch):
        """Verify the native OPENAI_API_KEY seeds MEMMAN_OPENAI_EMBED_API_KEY.

        Mutation: omitting OPENAI_API_KEY from the native fallback table.
        Oracle: the native value read back under MEMMAN_OPENAI_EMBED_API_KEY.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv('OPENAI_API_KEY', 'native-oai-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.OPENAI_EMBED_API_KEY] == 'native-oai-key'

    def test_memman_prefixed_wins_over_native(
            self, tmp_path, monkeypatch):
        """Verify the MEMMAN- name beats the vendor-native name.

        Mutation: checking the native name first.
        Oracle: 'memman-vy-key' read back with both names exported.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'memman-vy-key')
        monkeypatch.setenv('VOYAGE_API_KEY', 'native-vy-key')
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'memman-or-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_API_KEY] == 'memman-vy-key'

    def test_file_wins_over_native_shell(
            self, tmp_path, monkeypatch):
        """Verify an env-file value beats a vendor-native shell export.

        Mutation: letting the native-name fallback override a file value.
        Oracle: the file values read back with different native exports set.
        """
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
            self, tmp_path, monkeypatch):
        """Verify the Voyage embed model defaults from INSTALL_DEFAULTS.

        Mutation: dropping VOYAGE_EMBED_MODEL from INSTALL_DEFAULTS, or
            changing its value.
        Oracle: the literal 'voyage-3-lite'.
        """
        data_dir = str(tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, data_dir)
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'or-key')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'vy-key')
        config.reset_file_cache()
        knobs = config.collect_install_knobs(data_dir)
        assert knobs[config.VOYAGE_EMBED_MODEL] == 'voyage-3-lite'

    def test_backend_value_round_trips_from_file(
            self, tmp_path, monkeypatch):
        """Verify a DEFAULT_BACKEND already in the env file survives install.

        Mutation: overwriting the file value with the 'sqlite' default.
        Oracle: the literal 'postgres' read back.
        """
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
            f'{config.OPENROUTER_API_KEY}=or-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n'
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n')
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
            f'{config.LLM_ENDPOINT}=https://api.anthropic.com/v1\n'
            f'{config.LLM_API_KEY}=sk-ant-key\n'
            f'{config.VOYAGE_API_KEY}=vy-key\n')
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        config.reset_file_cache()
        with pytest.raises(ConfigError, match=config.LLM_MODEL):
            config.collect_install_knobs(str(data_dir))
