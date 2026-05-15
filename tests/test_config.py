"""Tests for memman.config -- variables, set command, and env-var resolver."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli

ALL_EXPECTED_NAMES = {
    'MEMMAN_DATA_DIR',
    'MEMMAN_STORE',
    'MEMMAN_LLM_ENDPOINT',
    'MEMMAN_LLM_API_KEY',
    'MEMMAN_LLM_MODEL_FAST',
    'MEMMAN_LLM_MODEL_SLOW_CANONICAL',
    'MEMMAN_LLM_MODEL_SLOW_METADATA',
    'MEMMAN_EMBED_PROVIDER',
    'MEMMAN_OPENROUTER_ENDPOINT',
    'MEMMAN_RERANK_PROVIDER',
    'MEMMAN_RERANK_ENABLED',
    'MEMMAN_VOYAGE_RERANK_MODEL',
    'MEMMAN_DEBUG',
    'MEMMAN_WORKER',
    'MEMMAN_LOG_LEVEL',
    'MEMMAN_DEFAULT_BACKEND',
    'MEMMAN_DEFAULT_POSTGRES_DSN',
    'MEMMAN_OPENROUTER_API_KEY',
    'MEMMAN_VOYAGE_API_KEY',
    'MEMMAN_OPENAI_EMBED_API_KEY',
    'MEMMAN_OPENAI_EMBED_ENDPOINT',
    'MEMMAN_OPENAI_EMBED_MODEL',
    'MEMMAN_OLLAMA_HOST',
    'MEMMAN_OLLAMA_EMBED_MODEL',
    'MEMMAN_OLLAMA_MAX_INPUT_CHARS',
    'MEMMAN_OPENROUTER_EMBED_MODEL',
    'MEMMAN_INTERVAL',
    'MEMMAN_SCHEDULER_KIND',
    'MEMMAN_EMBED_SWAP_BATCH_SIZE',
    'MEMMAN_EMBED_SWAP_INDEX_TIMEOUT',
    'MEMMAN_REINDEX_TIMEOUT',
    }


def test_required_install_keys_returns_subset_of_installable():
    """required_install_keys output stays within INSTALLABLE_KEYS for every
    embed provider.
    """
    for embed in ('voyage', 'openai', 'openrouter'):
        keys = config.required_install_keys(embed)
        assert keys <= set(config.INSTALLABLE_KEYS)


def test_required_install_keys_picks_curated_secrets():
    """Each curated embed provider maps to the API key install must verify."""
    assert config.required_install_keys('voyage') == {config.VOYAGE_API_KEY}
    assert config.required_install_keys('openai') == {
        config.OPENAI_EMBED_API_KEY}
    assert config.required_install_keys('openrouter') == {
        config.OPENROUTER_API_KEY}


def test_required_install_keys_experimental_returns_empty():
    """Experimental embed providers return an empty set."""
    assert config.required_install_keys('cohere') == set()


class TestIsOpenrouterEndpoint:
    """`config.is_openrouter_endpoint` matches OR hosts robustly."""

    def test_canonical_url(self):
        assert config.is_openrouter_endpoint(
            'https://openrouter.ai/api/v1') is True

    def test_trailing_slash(self):
        assert config.is_openrouter_endpoint(
            'https://openrouter.ai/api/v1/') is True

    def test_http_scheme(self):
        assert config.is_openrouter_endpoint(
            'http://openrouter.ai/api/v1') is True

    def test_regional_subdomain(self):
        assert config.is_openrouter_endpoint(
            'https://eu.openrouter.ai/api/v1') is True

    def test_www_prefix(self):
        assert config.is_openrouter_endpoint(
            'https://www.openrouter.ai/api/v1') is True

    def test_other_endpoints_are_not_or(self):
        assert config.is_openrouter_endpoint(
            'https://api.openai.com/v1') is False
        assert config.is_openrouter_endpoint(
            'https://api.anthropic.com/v1') is False
        assert config.is_openrouter_endpoint(
            'http://localhost:11434/v1') is False


class TestIsLoopbackEndpoint:
    """`config.is_loopback_endpoint` matches the standard loopback hosts."""

    def test_localhost(self):
        assert config.is_loopback_endpoint(
            'http://localhost:11434/v1') is True

    def test_loopback_ipv4(self):
        assert config.is_loopback_endpoint('http://127.0.0.1:1234') is True

    def test_loopback_ipv6(self):
        assert config.is_loopback_endpoint('http://[::1]:11434/v1') is True

    def test_dotted_localhost_subdomain(self):
        assert config.is_loopback_endpoint(
            'http://api.localhost:1234') is True

    def test_remote_endpoint_is_not_loopback(self):
        assert config.is_loopback_endpoint(
            'https://api.openai.com/v1') is False
        assert config.is_loopback_endpoint(
            'https://openrouter.ai/api/v1') is False


def test_secret_vars_subset_of_installable():
    """Every secret must be in INSTALLABLE_KEYS so install can write it."""
    assert config.SECRET_VARS <= set(config.INSTALLABLE_KEYS)


def test_install_defaults_keys_subset_of_installable():
    """INSTALL_DEFAULTS must not contain ghost keys outside INSTALLABLE_KEYS."""
    assert set(config.INSTALL_DEFAULTS) <= set(config.INSTALLABLE_KEYS)


def test_all_vars_covers_installable_plus_direct_env_vars():
    """_ALL_VARS = INSTALLABLE_KEYS + process-control vars + tuning vars."""
    expected = set(config.INSTALLABLE_KEYS) | {
        config.DATA_DIR, config.STORE, config.WORKER, config.DEBUG,
        config.SCHEDULER_KIND,
        config.EMBED_SWAP_BATCH_SIZE,
        config.EMBED_SWAP_INDEX_TIMEOUT,
        config.REINDEX_TIMEOUT,
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
        config.DATA_DIR, config.STORE,
        config.LLM_ENDPOINT, config.LLM_API_KEY,
        config.LLM_MODEL_FAST,
        config.LLM_MODEL_SLOW_CANONICAL,
        config.LLM_MODEL_SLOW_METADATA,
        config.EMBED_PROVIDER,
        config.OPENROUTER_ENDPOINT,
        config.RERANK_PROVIDER,
        config.RERANK_ENABLED,
        config.VOYAGE_RERANK_MODEL,
        config.DEBUG, config.WORKER, config.LOG_LEVEL,
        config.OPENROUTER_API_KEY,
        config.VOYAGE_API_KEY,
        config.OPENAI_EMBED_API_KEY,
        config.OPENAI_EMBED_ENDPOINT,
        config.OPENAI_EMBED_MODEL,
        config.OLLAMA_HOST,
        config.OLLAMA_EMBED_MODEL,
        config.OLLAMA_MAX_INPUT_CHARS,
        config.OPENROUTER_EMBED_MODEL,
        config.DEFAULT_BACKEND,
        config.DEFAULT_PG_DSN,
        config.INTERVAL,
        config.SCHEDULER_KIND,
        config.EMBED_SWAP_BATCH_SIZE,
        config.EMBED_SWAP_INDEX_TIMEOUT,
        config.REINDEX_TIMEOUT,
        }
    assert actual == ALL_EXPECTED_NAMES


def test_get_bool_truthy_values(env_file):
    """get_bool() treats '1', 'true', 'yes', 'on' (any case) as True.
    """
    for val in ['1', 'true', 'TRUE', 'yes', 'ON', 'On']:
        env_file(config.LOG_LEVEL, val)
        assert config.get_bool(config.LOG_LEVEL) is True


def test_get_bool_falsy_values(env_file):
    """get_bool() returns False for '0', 'false', '', and unset vars.
    """
    env_file(config.LOG_LEVEL, None)
    assert config.get_bool(config.LOG_LEVEL) is False
    for val in ['0', 'false', 'no', 'off', '', 'garbage']:
        env_file(config.LOG_LEVEL, val)
        assert config.get_bool(config.LOG_LEVEL) is False


def test_auto_threshold_for_returns_per_store_key():
    """`AUTO_THRESHOLD_FOR(store)` produces the canonical env-key name.
    """
    assert config.AUTO_THRESHOLD_FOR('mystore') == \
        'MEMMAN_AUTO_SEMANTIC_THRESHOLD_mystore'


def test_get_store_auto_threshold_unset_returns_none(env_file):
    """No per-store key set -> None (defer to calibrated / fallback).
    """
    env_file(config.AUTO_THRESHOLD_FOR('s1'), None)
    assert config.get_store_auto_threshold('s1') is None


def test_get_store_auto_threshold_numeric(env_file):
    """A numeric value is parsed to float.
    """
    env_file(config.AUTO_THRESHOLD_FOR('s2'), '0.72')
    assert config.get_store_auto_threshold('s2') == 0.72


def test_get_store_auto_threshold_skip_sentinel(env_file):
    """`skip` is a sentinel meaning 'do not emit semantic edges'.
    """
    env_file(config.AUTO_THRESHOLD_FOR('s3'), 'skip')
    assert config.get_store_auto_threshold('s3') == 'skip'


def test_get_store_auto_threshold_none_sentinel(env_file):
    """`none` is an alias for `skip`.
    """
    env_file(config.AUTO_THRESHOLD_FOR('s4'), 'none')
    assert config.get_store_auto_threshold('s4') == 'skip'


def test_get_store_auto_threshold_invalid_raises(env_file):
    """Non-numeric, non-sentinel values raise ValueError.
    """
    env_file(config.AUTO_THRESHOLD_FOR('s5'), 'not-a-number')
    with pytest.raises(ValueError):
        config.get_store_auto_threshold('s5')


def test_get_store_auto_threshold_out_of_range_raises(env_file):
    """Values outside (0.0, 1.0) raise ValueError -- cosine cutoffs only.
    """
    env_file(config.AUTO_THRESHOLD_FOR('s6'), '1.5')
    with pytest.raises(ValueError):
        config.get_store_auto_threshold('s6')


def test_get_store_auto_threshold_boundary_zero_raises(env_file):
    """Exactly 0.0 raises -- the bound is strictly exclusive.
    """
    env_file(config.AUTO_THRESHOLD_FOR('zerobound'), '0.0')
    with pytest.raises(ValueError):
        config.get_store_auto_threshold('zerobound')


def test_get_store_auto_threshold_boundary_one_raises(env_file):
    """Exactly 1.0 raises -- the bound is strictly exclusive.
    """
    env_file(config.AUTO_THRESHOLD_FOR('onebound'), '1.0')
    with pytest.raises(ValueError):
        config.get_store_auto_threshold('onebound')


def test_get_store_auto_threshold_whitespace_returns_none(env_file):
    """Whitespace-only value is treated as unset, not as malformed.
    """
    env_file(config.AUTO_THRESHOLD_FOR('blank'), '   ')
    assert config.get_store_auto_threshold('blank') is None


def test_get_store_auto_threshold_scientific_notation(env_file):
    """Scientific notation (`5e-1`) parses to a float in range.
    """
    env_file(config.AUTO_THRESHOLD_FOR('sci'), '5e-1')
    assert config.get_store_auto_threshold('sci') == 0.5


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


def test_enumerate_reflects_current_env(env_file):
    """enumerate_effective_config() returns live values for set vars.
    """
    env_file(config.LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(config.LLM_MODEL_FAST, 'anthropic/claude-haiku-4.5')
    env_file(config.LLM_MODEL_SLOW_CANONICAL, 'anthropic/claude-sonnet-4.6')
    out = config.enumerate_effective_config()
    assert out[config.LLM_ENDPOINT] == 'https://openrouter.ai/api/v1'
    assert out[config.LLM_MODEL_FAST] == 'anthropic/claude-haiku-4.5'
    assert out[config.LLM_MODEL_SLOW_CANONICAL] == 'anthropic/claude-sonnet-4.6'


def test_enumerate_redacts_secrets_by_default(env_file):
    """API keys are replaced with ***REDACTED*** in the default output.
    """
    env_file(config.OPENROUTER_API_KEY, 'sk-or-secret-value')
    env_file(config.VOYAGE_API_KEY, 'pa-secret')
    out = config.enumerate_effective_config()
    assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'
    assert out[config.VOYAGE_API_KEY] == '***REDACTED***'


def test_enumerate_redact_false_exposes_secrets(env_file):
    """redact=False returns the raw secret values (diagnostic override).
    """
    env_file(config.OPENROUTER_API_KEY, 'sk-or-plaintext')
    out = config.enumerate_effective_config(redact=False)
    assert out[config.OPENROUTER_API_KEY] == 'sk-or-plaintext'


@pytest.mark.no_default_env
def test_enumerate_empty_string_is_unset(env_file):
    """Empty-string env vars map to None (not the empty string).
    """
    env_file(config.LLM_MODEL_FAST, '')
    out = config.enumerate_effective_config()
    assert out[config.LLM_MODEL_FAST] is None


class TestConfigSet:
    """`memman config set` writes and validates env-file entries."""

    def test_writes_env_file(self, tmp_path):
        """`config set` writes the key into the env file at mode 0600."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set',
                  config.DEFAULT_BACKEND, 'postgres'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[config.DEFAULT_BACKEND] == 'postgres'

    def test_rejects_unknown_key(self, tmp_path):
        """`config set` exits non-zero with the new shape-list hint
        when KEY is not in INSTALLABLE_KEYS or a per-store form.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set',
                  'MEMMAN_BOGUS_KEY', 'value'])
        assert result.exit_code != 0
        assert 'not a recognized config key' in result.output

    def test_overrides_existing_value(self, tmp_path):
        """`config set` overrides an existing env-file value."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.DEFAULT_BACKEND}=sqlite\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'set',
                  config.DEFAULT_BACKEND, 'postgres'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(str(data_dir)))
        assert parsed[config.DEFAULT_BACKEND] == 'postgres'

    def test_preserves_other_rows(self, tmp_path):
        """`config set` merges -- other env-file rows are preserved verbatim."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.LLM_ENDPOINT}=https://openrouter.ai/api/v1\n'
            f'{config.OPENROUTER_API_KEY}=keep-me\n'
            f'{config.DEFAULT_BACKEND}=sqlite\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'set',
                  config.DEFAULT_BACKEND, 'postgres'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(str(data_dir)))
        assert parsed[config.DEFAULT_BACKEND] == 'postgres'
        assert parsed[config.LLM_ENDPOINT] == 'https://openrouter.ai/api/v1'
        assert parsed[config.OPENROUTER_API_KEY] == 'keep-me'

    def test_writes_per_store_auto_threshold_numeric(self, tmp_path):
        """`MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>=0.55` writes to env file.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        key = config.AUTO_THRESHOLD_FOR('mystore')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set', key, '0.55'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[key] == '0.55'

    def test_writes_per_store_auto_threshold_skip_sentinel(self, tmp_path):
        """`MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>=skip` writes to env file.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        key = config.AUTO_THRESHOLD_FOR('noedges')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set', key, 'skip'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[key] == 'skip'

    def test_rejects_per_store_auto_threshold_garbage(self, tmp_path):
        """A non-numeric, non-sentinel value is rejected by `config set`."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        key = config.AUTO_THRESHOLD_FOR('badval')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set', key, 'garbage'])
        assert result.exit_code != 0
        assert 'must be a float' in result.output

    def test_rejects_per_store_auto_threshold_out_of_range(self, tmp_path):
        """A float outside (0.0, 1.0) is rejected by `config set`."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        key = config.AUTO_THRESHOLD_FOR('oor')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set', key, '1.5'])
        assert result.exit_code != 0
        assert 'out of range' in result.output

    def test_rejects_invalid_surface(self, tmp_path):
        """`MEMMAN_SURFACE_<store>=legal` is rejected (closed set)."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        key = config.SURFACE_FOR('wrong')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set', key, 'legal'])
        assert result.exit_code != 0
        assert 'must be one of' in result.output

    def test_get_store_surface_silently_falls_back_on_invalid(self, tmp_path):
        """An invalid surface value in the env file returns 'code'
        instead of raising. Validation happens in `memman doctor`;
        the read-side helper must stay total so recall does not blow
        up on a misconfigured store.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.SURFACE_FOR("bogus")}=legal\n')
        assert config.get_store_surface(
            'bogus', data_dir=str(data_dir)) == 'code'


class TestConfigGet:
    """`memman config get KEY` prints env-file values, exits 1 on unset."""

    def test_get_returns_value(self, tmp_path):
        """`config get` echoes the stored value for a set key."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.DEFAULT_BACKEND}=sqlite\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'get',
                  config.DEFAULT_BACKEND])
        assert result.exit_code == 0, result.output
        assert 'sqlite' in result.output

    def test_get_exits_nonzero_for_unset_key(self, tmp_path):
        """`config get` exits 1 with a message when the key is unset."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text('')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'get',
                  config.DEFAULT_BACKEND])
        assert result.exit_code != 0
        assert 'is not set' in result.output

    def test_get_redacts_api_key(self, tmp_path):
        """API key values do not echo plaintext via `config get`."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.OPENROUTER_API_KEY}=secret-token-xyz\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'get',
                  config.OPENROUTER_API_KEY])
        assert result.exit_code == 0
        assert 'secret-token-xyz' not in result.output


class TestConfigShowSurfaceAndAutoThreshold:
    """`config show` surfaces per-store SURFACE_ and AUTO_SEMANTIC_THRESHOLD_ keys."""

    def test_show_includes_per_store_surface(self, tmp_path):
        """`MEMMAN_SURFACE_<store>` appears in per_store output."""
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.SURFACE_FOR("mystore")}=claw\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'show'])
        assert result.exit_code == 0, result.output
        assert 'MEMMAN_SURFACE_mystore' in result.output
        assert 'claw' in result.output

    def test_show_includes_per_store_auto_threshold(self, tmp_path):
        """`MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>` appears in per_store output.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / config.ENV_FILENAME).write_text(
            f'{config.AUTO_THRESHOLD_FOR("tuned")}=0.72\n')
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--data-dir', str(data_dir), 'config', 'show'])
        assert result.exit_code == 0, result.output
        assert 'MEMMAN_AUTO_SEMANTIC_THRESHOLD_tuned' in result.output
        assert '0.72' in result.output


class TestConfigSetPgDsn:
    """`memman config set-pg-dsn` assembles a libpq URI from prompts."""

    def test_default_writes_assembled_uri(self, tmp_path):
        """`--default` writes MEMMAN_DEFAULT_POSTGRES_DSN built from five prompts."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir,
                  'config', 'set-pg-dsn', '--default'],
            input='db.internal\n5433\nmemman\ns3cret!@\nmemman_prod\n')
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        dsn = parsed[config.DEFAULT_PG_DSN]
        assert dsn == (
            'postgresql://memman:s3cret%21%40@db.internal:5433/memman_prod')
        assert 's3cret' not in result.output
        assert ':***@' in result.output

    def test_store_writes_per_store_key(self, tmp_path):
        """`--store NAME` writes MEMMAN_POSTGRES_DSN_<NAME>."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir,
                  'config', 'set-pg-dsn', '--store', 'work'],
            input='localhost\n5432\nmemman\n\nmemman\n')
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[config.env_key_for('postgres', 'DSN', 'work')] == (
            'postgresql://memman@localhost:5432/memman')
        assert config.DEFAULT_PG_DSN not in parsed

    def test_requires_default_or_store(self, tmp_path):
        """No flags = error; both flags = error."""
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        no_flags = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set-pg-dsn'])
        assert no_flags.exit_code != 0
        assert 'exactly one of --default or --store' in no_flags.output
        both = runner.invoke(
            cli, ['--data-dir', data_dir,
                  'config', 'set-pg-dsn',
                  '--default', '--store', 'work'])
        assert both.exit_code != 0
        assert 'exactly one of --default or --store' in both.output


def _write_env(path: Path, contents: str) -> None:
    """Write contents to an env file and reset the config cache."""
    path.write_text(contents)
    config.reset_file_cache()


class TestConfigResolver:
    """Env-var resolver: file-canonical keys, parser edge cases, cache."""

    @pytest.fixture
    def env_path(self, tmp_path, monkeypatch):
        """Pin MEMMAN_DATA_DIR to tmp and return the env-file path."""
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path))
        config.reset_file_cache()
        yield tmp_path / config.ENV_FILENAME
        config.reset_file_cache()

    def test_get_ignores_shell_env_for_installable_keys(self, env_path, monkeypatch):
        """Installable keys are file-canonical; shell env never overrides."""
        monkeypatch.setenv(config.LLM_MODEL_FAST, 'env-model')
        _write_env(env_path, f'{config.LLM_MODEL_FAST}=file-model\n')
        assert config.get(config.LLM_MODEL_FAST) == 'file-model'

    def test_get_returns_file_value(self, env_path, monkeypatch):
        """get() returns the value from the env file."""
        monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
        _write_env(env_path, f'{config.LLM_MODEL_FAST}=file-model\n')
        assert config.get(config.LLM_MODEL_FAST) == 'file-model'

    def test_get_returns_none_when_file_missing_key(self, env_path, monkeypatch):
        """get() returns None when the key is absent from the file."""
        monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
        assert config.get(config.LLM_MODEL_FAST) is None

    def test_get_returns_none_when_shell_env_set_but_file_missing(
            self, env_path, monkeypatch):
        """Shell-only value is invisible -- file is the only source."""
        monkeypatch.setenv(config.LLM_MODEL_FAST, 'env-only')
        assert config.get(config.LLM_MODEL_FAST) is None

    def test_parser_skips_blank_lines_and_comments(self, env_path):
        """Parser ignores blank lines and # comments."""
        contents = '\n'.join([
            '# This is a comment',
            '',
            f'{config.LLM_MODEL_FAST}=fast',
            '   ',
            '# Another comment',
            f'{config.LLM_MODEL_SLOW_CANONICAL}=slow',
            ])
        _write_env(env_path, contents + '\n')
        assert config.get(config.LLM_MODEL_FAST) == 'fast'
        assert config.get(config.LLM_MODEL_SLOW_CANONICAL) == 'slow'

    def test_parser_strips_quoted_values(self, env_path):
        """Parser strips single and double quotes from values."""
        contents = '\n'.join([
            f'{config.LLM_MODEL_FAST}="quoted-fast"',
            f"{config.LLM_MODEL_SLOW_CANONICAL}='quoted-slow'",
            ])
        _write_env(env_path, contents + '\n')
        assert config.get(config.LLM_MODEL_FAST) == 'quoted-fast'
        assert config.get(config.LLM_MODEL_SLOW_CANONICAL) == 'quoted-slow'

    def test_parser_does_not_expand_variables(self, env_path):
        """Parser does not expand shell variable syntax."""
        contents = f'{config.LLM_MODEL_FAST}=${{HOME}}/models\n'
        _write_env(env_path, contents)
        assert config.get(config.LLM_MODEL_FAST) == '${HOME}/models'

    def test_missing_file_returns_none(self, env_path, monkeypatch):
        """Missing env file returns None without error."""
        monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
        assert not env_path.exists()
        assert config.get(config.LLM_MODEL_FAST) is None

    def test_data_dir_change_invalidates_cache(self, tmp_path, monkeypatch):
        """Changing DATA_DIR causes the file cache to be invalidated."""
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

    def test_get_bool_resolves_through_file(self, env_path, monkeypatch):
        """get_bool() resolves installable keys through the file."""
        monkeypatch.delenv(config.LOG_LEVEL, raising=False)
        _write_env(env_path, f'{config.LOG_LEVEL}=on\n')
        assert config.get_bool(config.LOG_LEVEL) is True

    def test_get_bool_ignores_shell_env_for_installable_key(
            self, env_path, monkeypatch):
        """get_bool() reads file only for installable keys."""
        monkeypatch.setenv(config.LOG_LEVEL, 'on')
        _write_env(env_path, f'{config.LOG_LEVEL}=off\n')
        assert config.get_bool(config.LOG_LEVEL) is False

    def test_effective_source_reports_file_only_for_installable(
            self, env_path, monkeypatch):
        """Installable keys report 'file' or 'unset'; shell env is invisible."""
        monkeypatch.setenv(config.LLM_MODEL_FAST, 'env-val')
        assert config.effective_source(config.LLM_MODEL_FAST) == 'unset'

        _write_env(env_path, f'{config.LLM_MODEL_FAST}=file-val\n')
        assert config.effective_source(config.LLM_MODEL_FAST) == 'file'

        _write_env(env_path, '')
        assert config.effective_source(config.LLM_MODEL_FAST) == 'unset'

    def test_effective_source_reports_env_for_process_control(
            self, env_path, monkeypatch):
        """Process-control vars (DEBUG, WORKER) read os.environ directly."""
        monkeypatch.setenv(config.DEBUG, '1')
        assert config.effective_source(config.DEBUG) == 'env'
        monkeypatch.delenv(config.DEBUG, raising=False)
        assert config.effective_source(config.DEBUG) == 'unset'

    def test_enumerate_effective_config_redacts_secrets(self, env_path, monkeypatch):
        """enumerate_effective_config redacts and exposes secrets correctly."""
        _write_env(env_path, f'{config.OPENROUTER_API_KEY}=super-secret\n')
        out = config.enumerate_effective_config(redact=True)
        assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'

        out_unredacted = config.enumerate_effective_config(redact=False)
        assert out_unredacted[config.OPENROUTER_API_KEY] == 'super-secret'

    def test_enumerate_resolves_through_file(self, env_path, monkeypatch):
        """enumerate_effective_config returns file-only values."""
        monkeypatch.delenv(config.LLM_MODEL_FAST, raising=False)
        _write_env(env_path, f'{config.LLM_MODEL_FAST}=file-only\n')
        out = config.enumerate_effective_config(redact=False)
        assert out[config.LLM_MODEL_FAST] == 'file-only'

    def test_process_control_vars_bypass_file(self, env_path, monkeypatch):
        """Process-control vars written to file are invisible to enumerate."""
        monkeypatch.delenv(config.WORKER, raising=False)
        _write_env(env_path, f'{config.WORKER}=1\n')
        out = config.enumerate_effective_config()
        assert out[config.WORKER] is None

    def test_installable_keys_excludes_process_control(self):
        """Process-control vars are not in INSTALLABLE_KEYS."""
        process_control = {
            config.DATA_DIR,
            config.STORE,
            config.WORKER,
            config.DEBUG,
            }
        for var in process_control:
            assert var not in config.INSTALLABLE_KEYS

    def test_installable_keys_covers_secrets(self):
        """Every secret is in INSTALLABLE_KEYS so install can write it."""
        for secret in config.SECRET_VARS:
            assert secret in config.INSTALLABLE_KEYS
