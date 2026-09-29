"""Tests for memman.config -- variables, set command, and env-var resolver.
"""

from collections.abc import Iterator
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
    'MEMMAN_LLM_MODEL',
    'MEMMAN_LLM_PROVIDER_ONLY',
    'MEMMAN_LLM_DATA_COLLECTION',
    'MEMMAN_LLM_ZDR',
    'MEMMAN_EMBED_PROVIDER',
    'MEMMAN_OPENROUTER_ENDPOINT',
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
    'MEMMAN_VOYAGE_EMBED_MODEL',
    'MEMMAN_INTERVAL',
    'MEMMAN_BACKUP_CRON',
    'MEMMAN_BACKUP_TARGET',
    'MEMMAN_BACKUP_KEEP',
    'MEMMAN_SCHEDULER_KIND',
    'MEMMAN_AUTHOR',
    'MEMMAN_EMBED_SWAP_BATCH_SIZE',
    'MEMMAN_EMBED_SWAP_INDEX_TIMEOUT',
    'MEMMAN_REINDEX_TIMEOUT',
    }


def test_required_install_keys_returns_subset_of_installable():
    """Verify required_install_keys stays inside INSTALLABLE_KEYS.

    Mutation: A provider mapped to a key install cannot write, so the install
        check demands a variable it never persists.
    Oracle: Subset test against INSTALLABLE_KEYS for each curated provider.
    """
    for embed in ('voyage', 'openai', 'openrouter'):
        keys = config.required_install_keys(embed)
        assert keys <= set(config.INSTALLABLE_KEYS)


def test_required_install_keys_picks_curated_secrets():
    """Verify each curated embed provider maps to its own API key.

    Mutation: Two providers swapped in PROVIDER_REQUIRED_KEYS, so install
        verifies the wrong secret.
    Oracle: Hand-written provider to key pairs.
    """
    assert config.required_install_keys('voyage') == {config.VOYAGE_API_KEY}
    assert config.required_install_keys('openai') == {
        config.OPENAI_EMBED_API_KEY}
    assert config.required_install_keys('openrouter') == {
        config.OPENROUTER_API_KEY}


def test_required_install_keys_experimental_returns_empty():
    """Verify an unregistered embed provider needs no key.

    Mutation: required_install_keys indexing the map directly and raising
        KeyError for an unknown provider.
    Oracle: The empty set for `cohere`.
    """
    assert config.required_install_keys('cohere') == set()


class TestIsOpenrouterEndpoint:
    """`config.is_openrouter_endpoint` matches OR hosts robustly.
    """

    def test_canonical_url(self):
        """Verify the canonical OpenRouter URL matches.

        Mutation: The host test rejecting the bare `openrouter.ai` host.
        Oracle: The documented API URL.
        """
        assert config.is_openrouter_endpoint(
            'https://openrouter.ai/api/v1') is True

    def test_trailing_slash(self):
        """Verify a trailing slash does not change the match.

        Mutation: Matching the whole URL string instead of the parsed host.
        Oracle: The canonical URL with a trailing slash.
        """
        assert config.is_openrouter_endpoint(
            'https://openrouter.ai/api/v1/') is True

    def test_http_scheme(self):
        """Verify the http scheme still matches.

        Mutation: A `startswith("https://")` test in place of host parsing.
        Oracle: The canonical URL over http.
        """
        assert config.is_openrouter_endpoint(
            'http://openrouter.ai/api/v1') is True

    def test_regional_subdomain(self):
        """Verify a regional subdomain matches.

        Mutation: Dropping the `.openrouter.ai` suffix arm, leaving exact host
            equality.
        Oracle: The `eu.openrouter.ai` host.
        """
        assert config.is_openrouter_endpoint(
            'https://eu.openrouter.ai/api/v1') is True

    def test_www_prefix(self):
        """Verify a `www.` host matches.

        Mutation: Dropping the suffix arm, leaving exact host equality that
            rejects www.
        Oracle: The `www.openrouter.ai` host.
        """
        assert config.is_openrouter_endpoint(
            'https://www.openrouter.ai/api/v1') is True

    def test_other_endpoints_are_not_or(self):
        """Verify other hosts do not match.

        Mutation: A loose match, such as any host containing `open`, or an
            unconditional True.
        Oracle: OpenAI, Anthropic, and a localhost URL.
        """
        assert config.is_openrouter_endpoint(
            'https://api.openai.com/v1') is False
        assert config.is_openrouter_endpoint(
            'https://api.anthropic.com/v1') is False
        assert config.is_openrouter_endpoint(
            'http://localhost:11434/v1') is False


class TestIsLoopbackEndpoint:
    """`config.is_loopback_endpoint` matches the standard loopback hosts.
    """

    def test_localhost(self):
        """Verify `localhost` is loopback.

        Mutation: Dropping `localhost` from the loopback host set.
        Oracle: The Ollama default URL.
        """
        assert config.is_loopback_endpoint(
            'http://localhost:11434/v1') is True

    def test_loopback_ipv4(self):
        """Verify 127.0.0.1 is loopback.

        Mutation: Dropping `127.0.0.1` from the loopback host set.
        Oracle: The literal address with a port.
        """
        assert config.is_loopback_endpoint('http://127.0.0.1:1234') is True

    def test_loopback_ipv6(self):
        """Verify the IPv6 loopback `[::1]` is loopback.

        Mutation: Dropping `::1` from the set, or comparing the bracketed
            netloc instead of the parsed hostname.
        Oracle: The bracketed IPv6 literal with a port.
        """
        assert config.is_loopback_endpoint('http://[::1]:11434/v1') is True

    def test_dotted_localhost_subdomain(self):
        """Verify a `*.localhost` host is loopback.

        Mutation: Dropping the `.localhost` suffix arm.
        Oracle: The `api.localhost` host.
        """
        assert config.is_loopback_endpoint(
            'http://api.localhost:1234') is True

    def test_remote_endpoint_is_not_loopback(self):
        """Verify remote hosts are not loopback.

        Mutation: An unconditional True, which skips the API-key requirement
            for a remote endpoint.
        Oracle: The OpenAI and OpenRouter URLs.
        """
        assert config.is_loopback_endpoint(
            'https://api.openai.com/v1') is False
        assert config.is_loopback_endpoint(
            'https://openrouter.ai/api/v1') is False


def test_secret_vars_subset_of_installable():
    """Verify every secret is an installable key.

    Mutation: A new secret missing from INSTALLABLE_KEYS, which install cannot
        then write.
    Oracle: Subset test of SECRET_VARS against INSTALLABLE_KEYS.
    """
    assert config.SECRET_VARS <= set(config.INSTALLABLE_KEYS)


def test_install_defaults_keys_subset_of_installable():
    """Verify every install default is an installable key.

    Mutation: A default seeded for a key outside INSTALLABLE_KEYS.
    Oracle: Subset test of INSTALL_DEFAULTS keys against INSTALLABLE_KEYS.
    """
    assert set(config.INSTALL_DEFAULTS) <= set(config.INSTALLABLE_KEYS)


def test_all_vars_covers_installable_plus_direct_env_vars():
    """Verify _ALL_VARS is installable keys plus direct env vars.

    Mutation: A process-control or tuning var missing from _ALL_VARS, so
        enumerate_effective_config omits it.
    Oracle: A hand-listed union of INSTALLABLE_KEYS and the direct vars.
    """
    expected = set(config.INSTALLABLE_KEYS) | {
        config.DATA_DIR, config.STORE, config.WORKER, config.DEBUG,
        config.SCHEDULER_KIND, config.AUTHOR,
        config.EMBED_SWAP_BATCH_SIZE,
        config.EMBED_SWAP_INDEX_TIMEOUT,
        config.REINDEX_TIMEOUT,
        }
    assert set(config._ALL_VARS) == expected


def test_log_level_bootstrap_literal_matches_install_default():
    """Verify the LOG_LEVEL install default matches the CLI literal.

    `cli._configure_logging` falls back to the literal `WARNING` before
    install. A change to the default must change the literal too.

    Mutation: INSTALL_DEFAULTS[LOG_LEVEL] changing while the CLI literal stays.
    Oracle: The literal `WARNING`.
    """
    assert config.INSTALL_DEFAULTS[config.LOG_LEVEL] == 'WARNING'


def test_constants_match_expected_names():
    """Every memman env var the codebase uses has a constant here.

    Mutation: a config constant's literal env-var string drifting
        from its name here, or a new constant landing in config.py
        with no matching literal added to `ALL_EXPECTED_NAMES`.
    Oracle: `ALL_EXPECTED_NAMES`, a set of literal strings maintained
        independently of `config.py`.
    """
    actual = {
        config.DATA_DIR, config.STORE,
        config.LLM_ENDPOINT, config.LLM_API_KEY,
        config.LLM_MODEL,
        config.LLM_PROVIDER_ONLY,
        config.LLM_DATA_COLLECTION,
        config.LLM_ZDR,
        config.EMBED_PROVIDER,
        config.OPENROUTER_ENDPOINT,
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
        config.VOYAGE_EMBED_MODEL,
        config.DEFAULT_BACKEND,
        config.DEFAULT_PG_DSN,
        config.INTERVAL,
        config.BACKUP_CRON,
        config.BACKUP_TARGET,
        config.BACKUP_KEEP,
        config.SCHEDULER_KIND,
        config.AUTHOR,
        config.EMBED_SWAP_BATCH_SIZE,
        config.EMBED_SWAP_INDEX_TIMEOUT,
        config.REINDEX_TIMEOUT,
        }
    assert actual == ALL_EXPECTED_NAMES


def test_get_bool_truthy_values(env_file):
    """Verify get_bool accepts 1, true, yes, on in any case.

    Mutation: A case-sensitive compare, or a member dropped from TRUTHY.
    Oracle: Hand-listed truthy spellings in mixed case.
    """
    for val in ['1', 'true', 'TRUE', 'yes', 'ON', 'On']:
        env_file(config.LOG_LEVEL, val)
        assert config.get_bool(config.LOG_LEVEL) is True


def test_get_bool_falsy_values(env_file):
    """Verify get_bool is False for unset, empty, and other text.

    Mutation: Treating any non-empty text as truthy.
    Oracle: Unset plus `0`, `false`, `no`, `off`, empty, and `garbage`.
    """
    env_file(config.LOG_LEVEL, None)
    assert config.get_bool(config.LOG_LEVEL) is False
    for val in ['0', 'false', 'no', 'off', '', 'garbage']:
        env_file(config.LOG_LEVEL, val)
        assert config.get_bool(config.LOG_LEVEL) is False


def test_is_worker_detects_worker_env(monkeypatch):
    """Verify is_worker is True only for MEMMAN_WORKER=1.

    Mutation: A truthy-string test that accepts `true`, or a default of True
        when unset.
    Oracle: Unset, `1`, `0`, and `true`.
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
    """Verify enumerate_effective_config lists every var, unset as None.

    The `no_default_env` mark skips the INSTALL_DEFAULTS seed, so every var
    resolves to None.

    Mutation: A var missing from the result, or a default invented for an unset
        var.
    Oracle: `ALL_EXPECTED_NAMES`, kept apart from config.py.
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
    """Verify enumerate_effective_config returns live values.

    Mutation: Serving a stale value, or None for a set var.
    Oracle: Two values written to the env file.
    """
    env_file(config.LLM_ENDPOINT, 'https://openrouter.ai/api/v1')
    env_file(config.LLM_MODEL, 'anthropic/claude-sonnet-4.6')
    out = config.enumerate_effective_config()
    assert out[config.LLM_ENDPOINT] == 'https://openrouter.ai/api/v1'
    assert out[config.LLM_MODEL] == 'anthropic/claude-sonnet-4.6'


def test_enumerate_redacts_secrets_by_default(env_file):
    """Verify secrets are redacted by default.

    Mutation: Dropping the redact branch, so an API key prints in plain text.
    Oracle: The `***REDACTED***` marker for two secret keys.
    """
    env_file(config.OPENROUTER_API_KEY, 'sk-or-secret-value')
    env_file(config.VOYAGE_API_KEY, 'pa-secret')
    out = config.enumerate_effective_config()
    assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'
    assert out[config.VOYAGE_API_KEY] == '***REDACTED***'


def test_enumerate_redact_false_exposes_secrets(env_file):
    """Verify redact=False returns the raw secret.

    Mutation: Redacting regardless of the flag.
    Oracle: The plain value written to the env file.
    """
    env_file(config.OPENROUTER_API_KEY, 'sk-or-plaintext')
    out = config.enumerate_effective_config(redact=False)
    assert out[config.OPENROUTER_API_KEY] == 'sk-or-plaintext'


@pytest.mark.no_default_env
def test_enumerate_empty_string_is_unset(env_file):
    """Verify an empty value maps to None.

    Mutation: Returning the empty string for an empty setting.
    Oracle: An empty LLM model setting reads None.
    """
    env_file(config.LLM_MODEL, '')
    out = config.enumerate_effective_config()
    assert out[config.LLM_MODEL] is None


class TestConfigSet:
    """`memman config set` writes and validates env-file entries.
    """

    def test_writes_env_file(self, tmp_path):
        """Verify `config set` writes the key into the env file.

        Mutation: config_set exiting 0 without persisting, or writing to a
            different path.
        Oracle: The env file parsed back from disk.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set',
                  config.DEFAULT_BACKEND, 'postgres'])
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[config.DEFAULT_BACKEND] == 'postgres'

    def test_rejects_unknown_key(self, tmp_path):
        """Verify `config set` rejects a key outside the accepted shapes.

        Mutation: Dropping the accepted-key check, so any name lands in the env
            file.
        Oracle: Non-zero exit and the `not a recognized config key` text.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir, 'config', 'set',
                  'MEMMAN_BOGUS_KEY', 'value'])
        assert result.exit_code != 0
        assert 'not a recognized config key' in result.output

    def test_overrides_existing_value(self, tmp_path):
        """Verify `config set` overrides an existing value.

        Mutation: Sticky-seed behavior, which keeps the old value.
        Oracle: sqlite on disk replaced by postgres.
        """
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
        """Verify `config set` keeps the other rows of the env file.

        Mutation: Rewriting the file from the one key, which drops the rest.
        Oracle: Three seeded rows; the two untouched rows read back unchanged.
        """
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


class TestConfigGet:
    """`memman config get KEY` prints env-file values, exits 1 on unset.
    """

    def test_get_returns_value(self, tmp_path):
        """Verify `config get` prints the stored value.

        Mutation: config_get printing nothing, or a different key.
        Oracle: The seeded `sqlite` in the output.
        """
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
        """Verify `config get` fails with a message for an unset key.

        Mutation: Exiting 0 with empty output for an unset key.
        Oracle: Non-zero exit and `is not set` in the output.
        """
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
        """Verify `config get` does not print an API key.

        Mutation: Dropping the `API_KEY` redaction branch.
        Oracle: The seeded token is absent from the output.
        """
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


def test_config_models_command_removed():
    """Verify `memman config models` is not a command.

    Mutation: The candidate lister left registered.
    Oracle: Click's usage error: exit 2 with `No such command`.
    """
    result = CliRunner().invoke(cli, ['config', 'models', '--help'])
    assert result.exit_code == 2
    assert 'No such command' in result.output


class TestConfigSetPgDsn:
    """`memman config set-pg-dsn` assembles a libpq URI from prompts.
    """

    def test_default_writes_assembled_uri(self, tmp_path):
        """Verify `--default` writes the URI assembled from five prompts.

        Mutation: Leaving the password unencoded, or echoing it in plain text.
        Oracle: Hand-encoded URI (`!` as %21, `@` as %40), and `:***@` in the
            output.
        """
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
        """Verify `--store NAME` writes the per-store key.

        Mutation: Writing the default key, or keeping a `:` for an empty
            password.
        Oracle: The URI with no password, and no default key in the file.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        result = runner.invoke(
            cli, ['--data-dir', data_dir,
                  'config', 'set-pg-dsn', '--store', 'work'],
            input='localhost\n5432\nmemman\n\nmemman\n')
        assert result.exit_code == 0, result.output
        parsed = config.parse_env_file(config.env_file_path(data_dir))
        assert parsed[config.POSTGRES_DSN_FOR('work')] == (
            'postgresql://memman@localhost:5432/memman')
        assert config.DEFAULT_PG_DSN not in parsed

    def test_requires_default_or_store(self, tmp_path):
        """Verify set-pg-dsn needs exactly one of --default and --store.

        Mutation: Accepting no flag, or both, and writing a key.
        Oracle: Non-zero exit and the `exactly one of` text for each case.
        """
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
    """Write contents to an env file and reset the config cache.
    """
    path.write_text(contents)
    config.reset_file_cache()


class TestConfigResolver:
    """Env-var resolver: file-canonical keys, parser edge cases, cache.
    """

    @pytest.fixture
    def env_path(
            self, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
        """Pin MEMMAN_DATA_DIR to tmp and return the env-file path.
        """
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path))
        config.reset_file_cache()
        yield tmp_path / config.ENV_FILENAME
        config.reset_file_cache()

    def test_get_ignores_shell_env_for_installable_keys(self, env_path, monkeypatch):
        """Verify the shell environment never overrides the env file.

        Mutation: get() consulting os.environ before the file.
        Oracle: A file value that beats a conflicting shell value.
        """
        monkeypatch.setenv(config.LLM_MODEL, 'env-model')
        _write_env(env_path, f'{config.LLM_MODEL}=file-model\n')
        assert config.get(config.LLM_MODEL) == 'file-model'

    def test_get_returns_file_value(self, env_path, monkeypatch):
        """Verify get() returns the env file value.

        Mutation: get() returning None for a key the file holds.
        Oracle: The value written to the file.
        """
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        _write_env(env_path, f'{config.LLM_MODEL}=file-model\n')
        assert config.get(config.LLM_MODEL) == 'file-model'

    def test_get_returns_none_when_file_missing_key(self, env_path, monkeypatch):
        """Verify get() returns None for a key the file lacks.

        Mutation: get() raising KeyError or returning an empty string.
        Oracle: None for an unset key.
        """
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        assert config.get(config.LLM_MODEL) is None

    def test_get_returns_none_when_shell_env_set_but_file_missing(
            self, env_path, monkeypatch):
        """Verify a shell-only value is invisible to get().

        Mutation: get() falling back to os.environ.
        Oracle: None despite a set shell variable.
        """
        monkeypatch.setenv(config.LLM_MODEL, 'env-only')
        assert config.get(config.LLM_MODEL) is None

    def test_parser_skips_blank_lines_and_comments(self, env_path):
        """Verify the parser skips blank lines and comments.

        Mutation: Parsing a comment or blank line as a key, or stopping at the
            first one.
        Oracle: Two values that follow blanks and comments.
        """
        contents = '\n'.join([
            '# This is a comment',
            '',
            f'{config.LLM_MODEL}=model-a',
            '   ',
            '# Another comment',
            f'{config.LLM_ENDPOINT}=endpoint-b',
            ])
        _write_env(env_path, contents + '\n')
        assert config.get(config.LLM_MODEL) == 'model-a'
        assert config.get(config.LLM_ENDPOINT) == 'endpoint-b'

    def test_parser_strips_quoted_values(self, env_path):
        """Verify the parser strips matching quotes.

        Mutation: Keeping the quote characters in the value.
        Oracle: A double-quoted and a single-quoted value.
        """
        contents = '\n'.join([
            f'{config.LLM_MODEL}="quoted-model"',
            f"{config.LLM_ENDPOINT}='quoted-endpoint'",
            ])
        _write_env(env_path, contents + '\n')
        assert config.get(config.LLM_MODEL) == 'quoted-model'
        assert config.get(config.LLM_ENDPOINT) == 'quoted-endpoint'

    def test_parser_does_not_expand_variables(self, env_path):
        """Verify the parser leaves `${VAR}` as written.

        Mutation: Expanding shell variables in a value.
        Oracle: The literal `${HOME}/models`.
        """
        contents = f'{config.LLM_MODEL}=${{HOME}}/models\n'
        _write_env(env_path, contents)
        assert config.get(config.LLM_MODEL) == '${HOME}/models'

    def test_missing_file_returns_none(self, env_path, monkeypatch):
        """Verify a missing env file reads as unset.

        Mutation: The parser raising FileNotFoundError.
        Oracle: None from get() with no file on disk.
        """
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        assert not env_path.exists()
        assert config.get(config.LLM_MODEL) is None

    def test_data_dir_change_invalidates_cache(self, tmp_path, monkeypatch):
        """Verify a DATA_DIR change reloads the env file.

        Mutation: Caching by first read, so the second dir serves the first dir
            values.
        Oracle: Two dirs holding different values for one key.
        """
        dir_a = tmp_path / 'a'
        dir_b = tmp_path / 'b'
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / config.ENV_FILENAME).write_text(
            f'{config.LLM_MODEL}=from-a\n')
        (dir_b / config.ENV_FILENAME).write_text(
            f'{config.LLM_MODEL}=from-b\n')

        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        monkeypatch.setenv(config.DATA_DIR, str(dir_a))
        config.reset_file_cache()
        assert config.get(config.LLM_MODEL) == 'from-a'

        monkeypatch.setenv(config.DATA_DIR, str(dir_b))
        assert config.get(config.LLM_MODEL) == 'from-b'

    def test_get_bool_resolves_through_file(self, env_path, monkeypatch):
        """Verify get_bool reads the env file.

        Mutation: get_bool ignoring the file.
        Oracle: The value `on` in the file reads True.
        """
        monkeypatch.delenv(config.LOG_LEVEL, raising=False)
        _write_env(env_path, f'{config.LOG_LEVEL}=on\n')
        assert config.get_bool(config.LOG_LEVEL) is True

    def test_get_bool_ignores_shell_env_for_installable_key(
            self, env_path, monkeypatch):
        """Verify get_bool ignores the shell for installable keys.

        Mutation: get_bool reading os.environ before the file.
        Oracle: A file value `off` beats a shell value `on`.
        """
        monkeypatch.setenv(config.LOG_LEVEL, 'on')
        _write_env(env_path, f'{config.LOG_LEVEL}=off\n')
        assert config.get_bool(config.LOG_LEVEL) is False

    def test_enumerate_effective_config_redacts_secrets(self, env_path, monkeypatch):
        """Verify redact=True hides a secret and redact=False shows it.

        Mutation: Redaction skipped for file values, or applied when redact is
            False.
        Oracle: The marker for one call and the raw value for the other.
        """
        _write_env(env_path, f'{config.OPENROUTER_API_KEY}=super-secret\n')
        out = config.enumerate_effective_config(redact=True)
        assert out[config.OPENROUTER_API_KEY] == '***REDACTED***'

        out_unredacted = config.enumerate_effective_config(redact=False)
        assert out_unredacted[config.OPENROUTER_API_KEY] == 'super-secret'

    def test_enumerate_resolves_through_file(self, env_path, monkeypatch):
        """Verify enumerate_effective_config reads file-only values.

        Mutation: Enumerate reading os.environ for installable keys.
        Oracle: The value that only the file holds.
        """
        monkeypatch.delenv(config.LLM_MODEL, raising=False)
        _write_env(env_path, f'{config.LLM_MODEL}=file-only\n')
        out = config.enumerate_effective_config(redact=False)
        assert out[config.LLM_MODEL] == 'file-only'

    def test_process_control_vars_bypass_file(self, env_path, monkeypatch):
        """Verify a process-control var in the file is ignored.

        Mutation: Resolving WORKER through the file, so a file line could turn
            on worker mode.
        Oracle: None for MEMMAN_WORKER written to the file only.
        """
        monkeypatch.delenv(config.WORKER, raising=False)
        _write_env(env_path, f'{config.WORKER}=1\n')
        out = config.enumerate_effective_config()
        assert out[config.WORKER] is None

    def test_installable_keys_excludes_process_control(self):
        """Verify process-control vars are not installable.

        Mutation: A process-control var added to INSTALLABLE_KEYS, which lets
            `config set` persist it.
        Oracle: A hand-listed set of process-control names.
        """
        process_control = {
            config.DATA_DIR,
            config.STORE,
            config.WORKER,
            config.DEBUG,
            }
        for var in process_control:
            assert var not in config.INSTALLABLE_KEYS

    def test_installable_keys_covers_secrets(self):
        """Verify each secret is an installable key.

        Mutation: A secret missing from INSTALLABLE_KEYS.
        Oracle: Membership of each SECRET_VARS entry.
        """
        for secret in config.SECRET_VARS:
            assert secret in config.INSTALLABLE_KEYS
