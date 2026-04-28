"""Tests for the env-var resolver in `memman.config`."""

from pathlib import Path

import pytest
from memman import config


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Pin MEMMAN_DATA_DIR to tmp and return the env-file path."""
    monkeypatch.setenv(config.DATA_DIR, str(tmp_path))
    config.reset_file_cache()
    yield tmp_path / config.ENV_FILENAME
    config.reset_file_cache()


def write_env(path: Path, contents: str) -> None:
    """Write contents to the env file and reset the cache."""
    path.write_text(contents)
    config.reset_file_cache()


def test_get_returns_env_value_when_set(env_file, monkeypatch):
    monkeypatch.setenv(config.LLM_MODEL_FAST, 'env-model')
    write_env(env_file, f'{config.LLM_MODEL_FAST}=file-model\n')
    assert config.get(config.LLM_MODEL_FAST) == 'env-model'


def test_get_returns_file_value_when_env_unset(env_file, monkeypatch):
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    write_env(env_file, f'{config.LLM_MODEL_FAST}=file-model\n')
    assert config.get(config.LLM_MODEL_FAST) == 'file-model'


def test_get_returns_none_when_both_unset(env_file, monkeypatch):
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    assert config.get(config.LLM_MODEL_FAST) is None


def test_get_treats_empty_env_as_unset(env_file, monkeypatch):
    monkeypatch.setenv(config.LLM_MODEL_FAST, '')
    write_env(env_file, f'{config.LLM_MODEL_FAST}=file-model\n')
    assert config.get(config.LLM_MODEL_FAST) == 'file-model'


def test_parser_skips_blank_lines_and_comments(env_file):
    contents = '\n'.join([
        '# This is a comment',
        '',
        f'{config.LLM_MODEL_FAST}=fast',
        '   ',
        '# Another comment',
        f'{config.LLM_MODEL_SLOW_CANONICAL}=slow',
        ])
    write_env(env_file, contents + '\n')
    assert config.get(config.LLM_MODEL_FAST) == 'fast'
    assert config.get(config.LLM_MODEL_SLOW_CANONICAL) == 'slow'


def test_parser_strips_quoted_values(env_file):
    contents = '\n'.join([
        f'{config.LLM_MODEL_FAST}="quoted-fast"',
        f"{config.LLM_MODEL_SLOW_CANONICAL}='quoted-slow'",
        ])
    write_env(env_file, contents + '\n')
    assert config.get(config.LLM_MODEL_FAST) == 'quoted-fast'
    assert config.get(config.LLM_MODEL_SLOW_CANONICAL) == 'quoted-slow'


def test_parser_does_not_expand_variables(env_file):
    contents = f'{config.LLM_MODEL_FAST}=${{HOME}}/models\n'
    write_env(env_file, contents)
    assert config.get(config.LLM_MODEL_FAST) == '${HOME}/models'


def test_missing_file_returns_none(env_file, monkeypatch):
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    assert not env_file.exists()
    assert config.get(config.LLM_MODEL_FAST) is None


def test_data_dir_change_invalidates_cache(tmp_path, monkeypatch):
    dir_a = tmp_path / 'a'
    dir_b = tmp_path / 'b'
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / config.ENV_FILENAME).write_text(
        f'{config.LLM_MODEL_FAST}=from-a\n')
    (dir_b / config.ENV_FILENAME).write_text(
        f'{config.LLM_MODEL_FAST}=from-b\n')

    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    monkeypatch.setenv(config.DATA_DIR, str(dir_a))
    config.reset_file_cache()
    assert config.get(config.LLM_MODEL_FAST) == 'from-a'

    monkeypatch.setenv(config.DATA_DIR, str(dir_b))
    assert config.get(config.LLM_MODEL_FAST) == 'from-b'


def test_get_bool_resolves_through_chain(env_file, monkeypatch):
    monkeypatch.delenv(config.DEBUG, raising=False)
    write_env(env_file, f'{config.DEBUG}=on\n')
    assert config.get_bool(config.DEBUG) is True


def test_get_bool_env_overrides_file(env_file, monkeypatch):
    monkeypatch.setenv(config.DEBUG, 'off')
    write_env(env_file, f'{config.DEBUG}=on\n')
    assert config.get_bool(config.DEBUG) is False


def test_effective_source_reports_layer(env_file, monkeypatch):
    monkeypatch.setenv(config.LLM_MODEL_FAST, 'env-val')
    assert config.effective_source(config.LLM_MODEL_FAST) == 'env'

    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    write_env(env_file, f'{config.LLM_MODEL_FAST}=file-val\n')
    assert config.effective_source(config.LLM_MODEL_FAST) == 'file'

    write_env(env_file, '')
    assert config.effective_source(config.LLM_MODEL_FAST) == 'unset'


def test_enumerate_effective_config_redacts_secrets(env_file, monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'super-secret')
    out = config.enumerate_effective_config(redact=True)
    assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'

    out_unredacted = config.enumerate_effective_config(redact=False)
    assert out_unredacted[config.OPENROUTER_API_KEY] == 'super-secret'


def test_enumerate_resolves_through_file(env_file, monkeypatch):
    monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
    write_env(env_file, f'{config.LLM_MODEL_FAST}=file-only\n')
    out = config.enumerate_effective_config(redact=False)
    assert out[config.LLM_MODEL_FAST] == 'file-only'


def test_process_control_vars_bypass_file(env_file, monkeypatch):
    monkeypatch.delenv(config.WORKER, raising=False)
    write_env(env_file, f'{config.WORKER}=1\n')
    out = config.enumerate_effective_config()
    assert out[config.WORKER] is None


def test_installable_keys_excludes_process_control():
    process_control = {
        config.DATA_DIR,
        config.STORE,
        config.WORKER,
        config.DEBUG,
        }
    for var in process_control:
        assert var not in config.INSTALLABLE_KEYS


def test_installable_keys_covers_secrets():
    for secret in config.SECRET_VARS:
        assert secret in config.INSTALLABLE_KEYS
