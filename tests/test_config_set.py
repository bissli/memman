"""Tests for the `memman config set` command."""


from click.testing import CliRunner
from memman import config
from memman.cli import cli


def test_config_set_writes_env_file(tmp_path):
    """`config set` writes the key into the env file at mode 0600."""
    runner = CliRunner()
    data_dir = str(tmp_path / 'memman')
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'config', 'set',
              config.BACKEND, 'postgres'])
    assert result.exit_code == 0, result.output
    parsed = config.parse_env_file(config.env_file_path(data_dir))
    assert parsed[config.BACKEND] == 'postgres'


def test_config_set_rejects_unknown_key(tmp_path):
    """`config set` exits non-zero when KEY is not in INSTALLABLE_KEYS."""
    runner = CliRunner()
    data_dir = str(tmp_path / 'memman')
    result = runner.invoke(
        cli, ['--data-dir', data_dir, 'config', 'set',
              'MEMMAN_BOGUS_KEY', 'value'])
    assert result.exit_code != 0
    assert 'INSTALLABLE_KEYS' in result.output


def test_config_set_overrides_existing_value(tmp_path):
    """`config set` overrides an existing env-file value (explicit override).
    """
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.BACKEND}=sqlite\n')
    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', str(data_dir), 'config', 'set',
              config.BACKEND, 'postgres'])
    assert result.exit_code == 0, result.output
    parsed = config.parse_env_file(config.env_file_path(str(data_dir)))
    assert parsed[config.BACKEND] == 'postgres'


def test_config_set_preserves_other_rows(tmp_path):
    """`config set` merges -- other env-file rows are preserved verbatim."""
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.LLM_PROVIDER}=openrouter\n'
        f'{config.OPENROUTER_API_KEY}=keep-me\n'
        f'{config.BACKEND}=sqlite\n')
    runner = CliRunner()
    result = runner.invoke(
        cli, ['--data-dir', str(data_dir), 'config', 'set',
              config.BACKEND, 'postgres'])
    assert result.exit_code == 0, result.output
    parsed = config.parse_env_file(config.env_file_path(str(data_dir)))
    assert parsed[config.BACKEND] == 'postgres'
    assert parsed[config.LLM_PROVIDER] == 'openrouter'
    assert parsed[config.OPENROUTER_API_KEY] == 'keep-me'
