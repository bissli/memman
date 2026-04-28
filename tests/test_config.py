"""Tests for memman.config -- central env-var configuration module.

Verifies the canonical env var names, typed helpers (`get_bool`,
`is_worker`), and effective-config enumeration with secret redaction.
"""

import pytest
from memman import config

ALL_EXPECTED_NAMES = {
    'MEMMAN_DATA_DIR',
    'MEMMAN_STORE',
    'MEMMAN_LLM_PROVIDER',
    'MEMMAN_LLM_MODEL_FAST',
    'MEMMAN_LLM_MODEL_SLOW_CANONICAL',
    'MEMMAN_LLM_MODEL_SLOW_METADATA',
    'MEMMAN_EMBED_PROVIDER',
    'MEMMAN_OPENROUTER_ENDPOINT',
    'MEMMAN_DEBUG',
    'MEMMAN_WORKER',
    'MEMMAN_LOG_LEVEL',
    'OPENROUTER_API_KEY',
    'VOYAGE_API_KEY',
    'MEMMAN_OPENAI_EMBED_API_KEY',
    'MEMMAN_OPENAI_EMBED_ENDPOINT',
    'MEMMAN_OPENAI_EMBED_MODEL',
    'MEMMAN_OLLAMA_HOST',
    'MEMMAN_OLLAMA_EMBED_MODEL',
    'MEMMAN_OPENROUTER_EMBED_MODEL',
    }


def test_mandatory_keys_subset_of_installable():
    """MANDATORY_INSTALL_KEYS must be a subset of INSTALLABLE_KEYS."""
    assert set(config.MANDATORY_INSTALL_KEYS) <= set(config.INSTALLABLE_KEYS)


def test_secret_vars_subset_of_installable():
    """Every secret must be in INSTALLABLE_KEYS so install can write it."""
    assert config.SECRET_VARS <= set(config.INSTALLABLE_KEYS)


def test_install_defaults_keys_subset_of_installable():
    """INSTALL_DEFAULTS must not contain ghost keys outside INSTALLABLE_KEYS."""
    assert set(config.INSTALL_DEFAULTS) <= set(config.INSTALLABLE_KEYS)


def test_all_vars_covers_installable_plus_process_control():
    """_ALL_VARS = INSTALLABLE_KEYS + process-control vars."""
    expected = set(config.INSTALLABLE_KEYS) | {
        config.DATA_DIR, config.STORE, config.WORKER, config.DEBUG,
        }
    assert set(config._ALL_VARS) == expected


def test_log_level_bootstrap_literal_matches_install_default():
    """The cli.py bootstrap fall-through literal must equal INSTALL_DEFAULTS.

    `cli._configure_logging` uses `or 'WARNING'` as a pre-install
    bootstrap default. If `INSTALL_DEFAULTS[LOG_LEVEL]` ever changes,
    the literal must be updated alongside; this test wires them together.
    """
    assert config.INSTALL_DEFAULTS[config.LOG_LEVEL] == 'WARNING'


def test_constants_match_expected_names():
    """Every env var the rest of the codebase uses has a constant here.
    """
    actual = {
        config.DATA_DIR, config.STORE, config.LLM_PROVIDER,
        config.LLM_MODEL_FAST,
        config.LLM_MODEL_SLOW_CANONICAL,
        config.LLM_MODEL_SLOW_METADATA,
        config.EMBED_PROVIDER,
        config.OPENROUTER_ENDPOINT,
        config.DEBUG, config.WORKER, config.LOG_LEVEL,
        config.OPENROUTER_API_KEY,
        config.VOYAGE_API_KEY,
        config.OPENAI_EMBED_API_KEY,
        config.OPENAI_EMBED_ENDPOINT,
        config.OPENAI_EMBED_MODEL,
        config.OLLAMA_HOST,
        config.OLLAMA_EMBED_MODEL,
        config.OPENROUTER_EMBED_MODEL,
        }
    assert actual == ALL_EXPECTED_NAMES


def test_get_bool_truthy_values(monkeypatch):
    """get_bool() treats '1', 'true', 'yes', 'on' (any case) as True.
    """
    for val in ['1', 'true', 'TRUE', 'yes', 'ON', 'On']:
        monkeypatch.setenv(config.DEBUG, val)
        assert config.get_bool(config.DEBUG) is True


def test_get_bool_falsy_values(monkeypatch):
    """get_bool() returns False for '0', 'false', '', and unset vars.
    """
    monkeypatch.delenv(config.DEBUG, raising=False)
    assert config.get_bool(config.DEBUG) is False
    for val in ['0', 'false', 'no', 'off', '', 'garbage']:
        monkeypatch.setenv(config.DEBUG, val)
        assert config.get_bool(config.DEBUG) is False


def test_is_worker_detects_worker_env(monkeypatch):
    """is_worker() returns True only when MEMMAN_WORKER=1 exactly.
    """
    monkeypatch.delenv(config.WORKER, raising=False)
    assert config.is_worker() is False
    monkeypatch.setenv(config.WORKER, '1')
    assert config.is_worker() is True
    monkeypatch.setenv(config.WORKER, '0')
    assert config.is_worker() is False
    monkeypatch.setenv(config.WORKER, 'true')
    assert config.is_worker() is False


@pytest.mark.no_default_env
def test_enumerate_returns_all_known_vars(monkeypatch):
    """enumerate_effective_config() includes every known env var name.

    Marked `no_default_env` so the conftest fixture skips seeding the
    INSTALL_DEFAULTS file; with both env and file empty, every var
    resolves to None.
    """
    for name in ALL_EXPECTED_NAMES:
        if name == config.DATA_DIR:
            continue
        monkeypatch.delenv(name, raising=False)
    config.reset_file_cache()
    out = config.enumerate_effective_config()
    assert set(out.keys()) == ALL_EXPECTED_NAMES
    for name, value in out.items():
        if name == config.DATA_DIR:
            continue
        assert value is None, f'{name}={value!r}'


def test_enumerate_reflects_current_env(monkeypatch):
    """enumerate_effective_config() returns live values for set vars.
    """
    monkeypatch.setenv(config.LLM_PROVIDER, 'openrouter')
    monkeypatch.setenv(config.LLM_MODEL_FAST, 'anthropic/claude-haiku-4.5')
    monkeypatch.setenv(
        config.LLM_MODEL_SLOW_CANONICAL, 'anthropic/claude-sonnet-4.6')
    out = config.enumerate_effective_config()
    assert out[config.LLM_PROVIDER] == 'openrouter'
    assert out[config.LLM_MODEL_FAST] == 'anthropic/claude-haiku-4.5'
    assert out[config.LLM_MODEL_SLOW_CANONICAL] == 'anthropic/claude-sonnet-4.6'


def test_enumerate_redacts_secrets_by_default(monkeypatch):
    """API keys are replaced with ***REDACTED*** in the default output.
    """
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'sk-or-secret-value')
    monkeypatch.setenv(config.VOYAGE_API_KEY, 'pa-secret')
    out = config.enumerate_effective_config()
    assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'
    assert out[config.VOYAGE_API_KEY] == '***REDACTED***'


def test_enumerate_redact_false_exposes_secrets(monkeypatch):
    """redact=False returns the raw secret values (diagnostic override).
    """
    monkeypatch.setenv(config.OPENROUTER_API_KEY, 'sk-or-plaintext')
    out = config.enumerate_effective_config(redact=False)
    assert out[config.OPENROUTER_API_KEY] == 'sk-or-plaintext'


@pytest.mark.no_default_env
def test_enumerate_empty_string_is_unset(monkeypatch):
    """Empty-string env vars map to None (not the empty string).
    """
    monkeypatch.setenv(config.LLM_MODEL_FAST, '')
    out = config.enumerate_effective_config()
    assert out[config.LLM_MODEL_FAST] is None
