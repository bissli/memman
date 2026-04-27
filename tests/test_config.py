"""Tests for memman.config -- central env-var configuration module.

Verifies the canonical env var names, typed helpers (`get_bool`,
`is_worker`), and effective-config enumeration with secret redaction.
"""

from memman import config

ALL_EXPECTED_NAMES = {
    'MEMMAN_DATA_DIR',
    'MEMMAN_STORE',
    'MEMMAN_LLM_PROVIDER',
    'MEMMAN_LLM_MODEL_FAST',
    'MEMMAN_LLM_MODEL_SLOW',
    'MEMMAN_EMBED_PROVIDER',
    'MEMMAN_OPENROUTER_ENDPOINT',
    'MEMMAN_CACHE_DIR',
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
    }


def test_constants_match_expected_names():
    """Every env var the rest of the codebase uses has a constant here.
    """
    actual = {
        config.DATA_DIR, config.STORE, config.LLM_PROVIDER,
        config.LLM_MODEL_FAST, config.LLM_MODEL_SLOW,
        config.EMBED_PROVIDER,
        config.OPENROUTER_ENDPOINT, config.CACHE_DIR,
        config.DEBUG, config.WORKER, config.LOG_LEVEL,
        config.OPENROUTER_API_KEY,
        config.VOYAGE_API_KEY,
        config.OPENAI_EMBED_API_KEY,
        config.OPENAI_EMBED_ENDPOINT,
        config.OPENAI_EMBED_MODEL,
        config.OLLAMA_HOST,
        config.OLLAMA_EMBED_MODEL,
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


def test_enumerate_returns_all_known_vars(monkeypatch):
    """enumerate_effective_config() includes every known env var name.
    """
    for name in ALL_EXPECTED_NAMES:
        monkeypatch.delenv(name, raising=False)
    out = config.enumerate_effective_config()
    assert set(out.keys()) == ALL_EXPECTED_NAMES
    assert all(v is None for v in out.values())


def test_enumerate_reflects_current_env(monkeypatch):
    """enumerate_effective_config() returns live values for set vars.
    """
    monkeypatch.setenv(config.LLM_PROVIDER, 'openrouter')
    monkeypatch.setenv(config.LLM_MODEL_FAST, 'anthropic/claude-haiku-4.5')
    monkeypatch.setenv(config.LLM_MODEL_SLOW, 'anthropic/claude-sonnet-4.6')
    out = config.enumerate_effective_config()
    assert out[config.LLM_PROVIDER] == 'openrouter'
    assert out[config.LLM_MODEL_FAST] == 'anthropic/claude-haiku-4.5'
    assert out[config.LLM_MODEL_SLOW] == 'anthropic/claude-sonnet-4.6'


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


def test_enumerate_empty_string_is_unset(monkeypatch):
    """Empty-string env vars map to None (not the empty string).
    """
    monkeypatch.setenv(config.LLM_MODEL_FAST, '')
    out = config.enumerate_effective_config()
    assert out[config.LLM_MODEL_FAST] is None
