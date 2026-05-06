"""Tests for the install wizard (`memman.setup.wizard`)."""


import pytest
from memman import config
from memman.setup import wizard


@pytest.fixture
def tty(monkeypatch):
    """Force `sys.stdin.isatty()` True so wizard takes the interactive path."""
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)


@pytest.fixture
def no_tty(monkeypatch):
    """Force `sys.stdin.isatty()` False so wizard takes the headless path."""
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)


def _strip_default_secrets(monkeypatch, data_dir):
    """Remove the conftest-seeded mock secrets from the test env file."""
    path = config.env_file_path(str(data_dir))
    parsed = config.parse_env_file(path)
    for key in (config.OPENROUTER_API_KEY, config.VOYAGE_API_KEY):
        parsed.pop(key, None)
    contents = '\n'.join(f'{k}={v}' for k, v in parsed.items()) + '\n'
    path.write_text(contents)
    config.reset_file_cache()


class TestWizardFlow:
    """Top-level run_wizard return shape under headless / tty conditions."""

    @pytest.mark.parametrize('mode', ['no_tty', 'no_wizard_in_tty'])
    def test_no_prompts_returns_empty(self, monkeypatch, tmp_path, mode):
        """Headless mode and `--no-wizard` both short-circuit to empty."""
        if mode == 'no_tty':
            monkeypatch.setattr('sys.stdin.isatty', lambda: False)
            out = wizard.run_wizard(str(tmp_path / 'memman'))
        else:
            monkeypatch.setattr('sys.stdin.isatty', lambda: True)
            out = wizard.run_wizard(
                str(tmp_path / 'memman'), no_wizard=True)
        assert out == {}

    def test_explicit_backend_flag_bypasses_prompt(self, no_tty, tmp_path):
        """`--backend sqlite` is honored without any prompting."""
        out = wizard.run_wizard(str(tmp_path / 'memman'), backend='sqlite')
        assert out[config.BACKEND] == 'sqlite'

    def test_postgres_hidden_when_extras_unavailable(
            self, tty, tmp_path, monkeypatch):
        """Sqlite-only menu is the no-prompt fast path; wizard returns empty."""
        monkeypatch.setattr(
            'memman.setup.wizard.extras.is_available', lambda extra: False)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.BACKEND not in out

    def test_postgres_hidden_when_backend_module_missing(
            self, tty, tmp_path, monkeypatch):
        """Even with extras present, the postgres backend module gates visibility."""
        monkeypatch.setattr(
            'memman.setup.wizard.extras.is_available', lambda extra: True)
        monkeypatch.setattr(
            'memman.setup.wizard._backend_module_exists', lambda: False)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.BACKEND not in out


class TestSecretPrompts:
    """Secret-prompt logic for OPENROUTER_API_KEY / VOYAGE_API_KEY."""

    def test_secrets_prompt_fires_when_missing_in_tty(
            self, tty, tmp_path, monkeypatch):
        """Wizard prompts (masked) when mandatory secrets are absent in TTY mode."""
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.delenv(config.OPENROUTER_API_KEY, raising=False)
        monkeypatch.delenv(config.VOYAGE_API_KEY, raising=False)

        inputs = iter(['fresh-or-key', 'fresh-vy-key'])
        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt',
            lambda *a, **kw: next(inputs))
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert out[config.OPENROUTER_API_KEY] == 'fresh-or-key'
        assert out[config.VOYAGE_API_KEY] == 'fresh-vy-key'

    def test_secrets_prompt_skipped_when_present_in_file(
            self, tty, tmp_path):
        """Wizard does not prompt when both secrets are already in the file.

        The autouse `_isolate_env` fixture seeds both secrets, so a default
        test environment should not trigger any secret prompt.
        """
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.OPENROUTER_API_KEY not in out
        assert config.VOYAGE_API_KEY not in out

    def test_secrets_prompt_skipped_when_shell_has_them(
            self, tmp_path, monkeypatch):
        """Wizard does not prompt when secrets are exported in the shell.

        Even though the runtime resolver ignores `os.environ`, install will
        still seed the shell value into the file via `collect_install_knobs`,
        so the wizard considers a shell-set secret already-resolved.
        """
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.setenv(config.OPENROUTER_API_KEY, 'shell-or')
        monkeypatch.setenv(config.VOYAGE_API_KEY, 'shell-vy')

        def _should_not_be_called(*a, **kw):
            raise AssertionError('wizard prompted when shell already had values')

        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt', _should_not_be_called)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.OPENROUTER_API_KEY not in out
        assert config.VOYAGE_API_KEY not in out


class TestDsn:
    """`--pg-dsn` flag and probe behavior in headless mode."""

    def test_pg_dsn_required_in_non_interactive_postgres(
            self, no_tty, tmp_path):
        """Headless --backend postgres without --pg-dsn is a hard error."""
        import click as _click
        with pytest.raises(_click.ClickException, match='pg-dsn'):
            wizard.run_wizard(
                str(tmp_path / 'memman'), backend='postgres', no_wizard=True)

    def test_pg_dsn_flag_probed_and_returned(
            self, monkeypatch, no_tty, tmp_path):
        """A passing `--pg-dsn` is probed via psycopg.connect and returned."""
        probe_calls = []

        def _fake_probe(dsn):
            probe_calls.append(dsn)

        monkeypatch.setattr(
            'memman.setup.wizard._probe_dsn', _fake_probe)
        out = wizard.run_wizard(
            str(tmp_path / 'memman'), backend='postgres',
            pg_dsn='postgresql://u@h/db')
        assert probe_calls == ['postgresql://u@h/db']
        assert out[config.BACKEND] == 'postgres'
        assert out[config.PG_DSN] == 'postgresql://u@h/db'

    def test_pg_dsn_probe_failure_raises(
            self, monkeypatch, no_tty, tmp_path):
        """A failing `--pg-dsn` probe surfaces as a ClickException."""
        import click as _click

        def _fail(dsn):
            raise RuntimeError('connection refused')

        monkeypatch.setattr('memman.setup.wizard._probe_dsn', _fail)
        with pytest.raises(_click.ClickException, match='connection failed'):
            wizard.run_wizard(
                str(tmp_path / 'memman'), backend='postgres',
                pg_dsn='postgresql://u@h/db')


def test_run_install_rejects_flag_vs_file_conflict(tmp_path, monkeypatch):
    """`memman install --backend X` warns and exits when file holds Y."""
    import click as _click
    from memman.setup import claude as setup_claude
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.BACKEND}=sqlite\n'
        f'{config.OPENROUTER_API_KEY}=k\n'
        f'{config.VOYAGE_API_KEY}=v\n')
    config.reset_file_cache()
    monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(
        setup_claude, 'memman_binary_path', lambda: '/fake/bin/memman')
    monkeypatch.setattr(
        setup_claude, 'detect_environments', list)

    with pytest.raises(_click.ClickException, match='memman config set'):
        setup_claude.run_install(
            str(data_dir), backend='postgres', no_wizard=True)


def test_existing_stores_hint_only_when_switching_to_postgres(
        monkeypatch, no_tty, tmp_path, capsys):
    """The migration hint prints only when populated SQLite stores exist
    AND the file's current backend is not already postgres.
    """
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.OPENROUTER_API_KEY}=k\n'
        f'{config.VOYAGE_API_KEY}=v\n')
    stores_dir = data_dir / 'data'
    stores_dir.mkdir(parents=True, exist_ok=True)
    (stores_dir / 'default').mkdir()
    (stores_dir / 'default' / 'memman.db').write_text('fake')
    monkeypatch.setattr(
        'memman.setup.wizard._probe_dsn', lambda dsn: None)
    config.reset_file_cache()
    wizard.run_wizard(
        str(data_dir), backend='postgres',
        pg_dsn='postgresql://u@h/db')
    out = capsys.readouterr().out
    assert 'memman migrate' in out
    assert 'default' in out


@pytest.mark.postgres
class TestProbeDsn:
    """DSN probe correctness against a live pgvector container."""

    def test_probe_dsn_raises_when_pgvector_missing(self, pg_dsn):
        """Drop pgvector and verify the probe complains; restore after."""
        import psycopg
        from memman.setup.wizard import _probe_dsn

        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('DROP EXTENSION IF EXISTS vector CASCADE')
        try:
            with pytest.raises(RuntimeError, match='pgvector'):
                _probe_dsn(pg_dsn)
        finally:
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    def test_probe_dsn_emits_pgbouncer_hint_on_remote_dsn(
            self, pg_dsn, capsys, monkeypatch):
        """Probe of a remote-shaped DSN emits the PgBouncer hint.

        Monkeypatches `_is_remote_dsn` to return True for the test
        container's DSN and checks that the hint appears in stdout.
        """
        from memman.setup import wizard as wiz_mod
        monkeypatch.setattr(wiz_mod, '_is_remote_dsn', lambda _dsn: True)
        wiz_mod._probe_dsn(pg_dsn)
        captured = capsys.readouterr()
        assert 'PgBouncer' in captured.out
