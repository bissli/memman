"""Tests for the install wizard (`memman.setup.wizard`).
"""

import os

import click
import pytest
from memman import config
from memman.setup import claude as setup_claude
from memman.setup import wizard


@pytest.fixture
def tty(monkeypatch):
    """Force `sys.stdin.isatty()` True so wizard takes the interactive path.
    """
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)


@pytest.fixture
def no_tty(monkeypatch):
    """Force `sys.stdin.isatty()` False so wizard takes the headless path.
    """
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)


def _strip_default_secrets(monkeypatch, data_dir):
    """Remove the conftest-seeded mock secrets from the test env file.
    """
    path = config.env_file_path(str(data_dir))
    parsed = config.parse_env_file(path)
    parsed.pop(config.API_KEY, None)
    contents = '\n'.join(f'{k}={v}' for k, v in parsed.items()) + '\n'
    path.write_text(contents)
    config.reset_file_cache()


class TestWizardFlow:
    """Top-level run_wizard return shape under headless / tty conditions.
    """

    @pytest.mark.parametrize('mode', ['no_tty', 'no_wizard_in_tty'])
    def test_no_prompts_returns_empty(self, monkeypatch, tmp_path, mode):
        """Verify a headless or --no-wizard run with no flags returns nothing.

        Mutation: the wizard writes a backend or other row when no flag is
            given.
        Oracle: the empty dict.
        """
        if mode == 'no_tty':
            monkeypatch.setattr('sys.stdin.isatty', lambda: False)
            out = wizard.run_wizard(str(tmp_path / 'memman'))
        else:
            monkeypatch.setattr('sys.stdin.isatty', lambda: True)
            out = wizard.run_wizard(
                str(tmp_path / 'memman'), no_wizard=True)
        assert out == {}

    def test_explicit_backend_flag_bypasses_prompt(self, no_tty, tmp_path):
        """Verify --backend sqlite is recorded under both backend keys.

        Mutation: the flag is ignored, so the defaulted choice is not
            persisted.
        Oracle: literal 'sqlite' at MEMMAN_DEFAULT_BACKEND and its default row.
        """
        out = wizard.run_wizard(str(tmp_path / 'memman'), backend='sqlite')
        assert out[config.DEFAULT_BACKEND] == 'sqlite'
        assert out[config.BACKEND_FOR('default')] == 'sqlite'

    def test_postgres_hidden_when_extras_unavailable(
            self, tty, tmp_path, monkeypatch):
        """Verify a sqlite-only menu skips the prompt and persists nothing.

        Mutation: the menu offers postgres without the extras installed.
        Oracle: neither backend key appears in the returned dict.
        """
        monkeypatch.setattr(
            'memman.setup.wizard.extras.is_available', lambda extra: False)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.DEFAULT_BACKEND not in out
        assert config.BACKEND_FOR('default') not in out


class TestSecretPrompts:
    """Secret-prompt logic for MEMMAN_API_KEY.
    """

    def test_secrets_prompt_fires_when_missing_in_tty(
            self, tty, tmp_path, monkeypatch):
        """Verify a missing API key is prompted for in a TTY and returned.

        Mutation: the prompt is skipped, or the answer is dropped.
        Oracle: the scripted prompt answer compared to the returned row.
        """
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.delenv(config.API_KEY, raising=False)

        inputs = iter(['fresh-key'])
        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt',
            lambda *a, **kw: next(inputs))
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert out[config.API_KEY] == 'fresh-key'

    def test_secrets_prompt_skipped_when_present_in_file(
            self, tty, tmp_path):
        """Verify no secret prompt fires when the env file already holds it.

        Mutation: the file-layer check is dropped, so the key is asked for
            again.
        Oracle: the seeded secret key is absent from the returned dict.
        """
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.API_KEY not in out

    def test_secrets_prompt_skipped_when_shell_has_them(
            self, tmp_path, monkeypatch):
        """Verify no prompt fires for a MEMMAN secret exported in the shell.

        Mutation: the shell-export check is dropped, so the wizard prompts.
        Oracle: a prompt stub that raises if called.
        """
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.setenv(config.API_KEY, 'shell-key')

        def _should_not_be_called(*a, **kw):
            raise AssertionError('wizard prompted when shell already had values')

        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt', _should_not_be_called)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        assert config.API_KEY not in out


class TestNativeKeyDetection:
    """A native OPENROUTER_API_KEY triggers announce-then-prompt.
    """

    def test_native_openrouter_key_announces_and_prompts_with_default(
            self, tty, tmp_path, monkeypatch, capsys):
        """Verify a native vendor key is announced as a masked default.

        Mutation: the native value is captured silently, or the prompt echoes
            it.
        Oracle: captured prompt kwargs, stdout text, and the exported values.
        """
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.delenv(config.API_KEY, raising=False)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-value')

        captured: list[dict] = []

        def fake_prompt(*args, **kwargs):
            captured.append(dict(kwargs, _arg=args[0] if args else None))
            return kwargs.get('default', '')

        monkeypatch.setattr('memman.setup.wizard.click.prompt', fake_prompt)
        out = wizard.run_wizard(str(tmp_path / 'memman'))
        stdout = capsys.readouterr().out
        assert 'detected OPENROUTER_API_KEY in shell' in stdout
        assert out[config.API_KEY] == 'native-or-value'
        key_call = next(
            c for c in captured if c['_arg'].strip() == config.API_KEY)
        assert key_call['default'] == 'native-or-value'
        assert key_call['hide_input'] is True
        assert key_call['show_default'] is False

    def test_memman_prefixed_still_silent_skips(
            self, tty, tmp_path, monkeypatch, capsys):
        """Verify a MEMMAN-prefixed export is skipped with no announcement.

        Mutation: the native-key announcement or prompt fires for a MEMMAN-
            key.
        Oracle: a raising prompt stub and 'detected' absent from stdout.
        """
        _strip_default_secrets(monkeypatch, tmp_path / 'memman')
        monkeypatch.setenv(config.DATA_DIR, str(tmp_path / 'memman'))
        monkeypatch.setenv(config.API_KEY, 'memman-key')
        monkeypatch.setenv('OPENROUTER_API_KEY', 'native-or-value')

        def _should_not_be_called(*a, **kw):
            raise AssertionError('wizard prompted despite MEMMAN- key present')

        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt', _should_not_be_called)
        wizard.run_wizard(str(tmp_path / 'memman'))
        stdout = capsys.readouterr().out
        assert 'detected' not in stdout


class TestDsn:
    """`--pg-dsn` flag and probe behavior in headless mode.
    """

    def test_pg_dsn_required_in_non_interactive_postgres(
            self, no_tty, tmp_path):
        """Verify headless postgres without --pg-dsn is a hard error.

        Mutation: the missing DSN passes through and install continues.
        Oracle: ClickException whose message names pg-dsn.
        """
        with pytest.raises(click.ClickException, match='pg-dsn'):
            wizard.run_wizard(
                str(tmp_path / 'memman'), backend='postgres', no_wizard=True)

    def test_pg_dsn_flag_probed_and_returned(
            self, monkeypatch, no_tty, tmp_path):
        """Verify a --pg-dsn flag is probed once, recorded under both keys.

        Mutation: the flag DSN skips the probe, or lands under one key only.
        Oracle: a probe stub recording its calls, and the literal DSN.
        """
        probe_calls = []

        def _fake_probe(dsn):
            probe_calls.append(dsn)

        monkeypatch.setattr(
            'memman.setup.wizard._probe_dsn', _fake_probe)
        out = wizard.run_wizard(
            str(tmp_path / 'memman'), backend='postgres',
            pg_dsn='postgresql://u@h/db')
        assert probe_calls == ['postgresql://u@h/db']
        assert out[config.DEFAULT_BACKEND] == 'postgres'
        assert out[config.DEFAULT_PG_DSN] == 'postgresql://u@h/db'
        assert out[config.BACKEND_FOR('default')] == 'postgres'
        dsn_key = config.POSTGRES_DSN_FOR('default')
        assert out[dsn_key] == 'postgresql://u@h/db'

    def test_pg_dsn_probe_failure_raises(
            self, monkeypatch, no_tty, tmp_path):
        """Verify a failing --pg-dsn probe surfaces as a ClickException.

        Mutation: the probe error is swallowed or escapes as a bare
            RuntimeError.
        Oracle: ClickException whose message contains 'connection failed'.
        """

        def _fail(dsn):
            raise RuntimeError('connection refused')

        monkeypatch.setattr('memman.setup.wizard._probe_dsn', _fail)
        with pytest.raises(click.ClickException, match='connection failed'):
            wizard.run_wizard(
                str(tmp_path / 'memman'), backend='postgres',
                pg_dsn='postgresql://u@h/db')


def test_run_install_rejects_flag_vs_file_conflict(tmp_path, monkeypatch):
    """Verify install --backend X refuses when the env file holds Y.

    Mutation: the flag silently overrides the file value.
    Oracle: ClickException pointing at the 'memman config set' command.
    """
    data_dir = tmp_path / 'memman'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / config.ENV_FILENAME).write_text(
        f'{config.DEFAULT_BACKEND}=sqlite\n'
        f'{config.API_KEY}=k\n')
    config.reset_file_cache()
    monkeypatch.setattr(setup_claude, 'detect_scheduler', lambda: 'systemd')
    monkeypatch.setattr(
        setup_claude, 'memman_binary_path', lambda: '/fake/bin/memman')
    monkeypatch.setattr(
        setup_claude, 'detect_claude_code',
        lambda: {'display': 'Claude Code',
                 'detected': False, 'bin_path': '',
                 'version': '',
                 'config_dir': str(tmp_path / 'memman' / '.claude')})

    with pytest.raises(click.ClickException, match='memman config set'):
        setup_claude.run_install(
            str(data_dir), backend='postgres', no_wizard=True)


@pytest.mark.postgres
class TestProbeDsn:
    """DSN probe correctness against a live pgvector container.
    """

    def test_probe_dsn_raises_when_pgvector_missing(self, pg_dsn):
        """Verify the probe fails when the pgvector extension is absent.

        Mutation: the pg_extension check is dropped from the probe.
        Oracle: a live container with the extension dropped, restored after.
        """
        import psycopg

        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('drop extension if exists vector cascade')
        try:
            with pytest.raises(RuntimeError, match='pgvector'):
                wizard._probe_dsn(pg_dsn)
        finally:
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute('create extension if not exists vector')

    def test_probe_dsn_emits_pgbouncer_hint_on_remote_dsn(
            self, pg_dsn, capsys, monkeypatch):
        """Verify probing a remote-shaped DSN prints the PgBouncer hint.

        Mutation: the hint is dropped or gated on the wrong host test.
        Oracle: 'PgBouncer' in captured stdout, with _is_remote_dsn forced
            True.
        """
        monkeypatch.setattr(wizard, '_is_remote_dsn', lambda _dsn: True)
        wizard._probe_dsn(pg_dsn)
        captured = capsys.readouterr()
        assert 'PgBouncer' in captured.out


def _strip_model(data_dir):
    """Remove MEMMAN_LLM_MODEL from the seeded test env file.
    """
    path = config.env_file_path(data_dir)
    parsed = config.parse_env_file(path)
    parsed.pop(config.LLM_MODEL, None)
    path.write_text(''.join(f'{k}={v}\n' for k, v in parsed.items()))
    config.reset_file_cache()


class TestModelDefault:
    """The OpenRouter branch leaves the model to `INSTALL_DEFAULTS`.
    """

    def test_openrouter_install_asks_nothing_and_reads_no_catalog(
            self, tty, monkeypatch):
        """Verify an OpenRouter install with no model neither asks nor fetches.

        Mutation: the wizard still offers a pick, or still GETs the OpenRouter
            catalogs to build one.
        Oracle: an HTTP client and a prompt that each fail the test if called.
        """
        data_dir = os.environ[config.DATA_DIR]
        _strip_model(data_dir)

        def _no_client(*a, **k):
            raise AssertionError('read a catalog')

        def _no_prompt(*a, **k):
            raise AssertionError('prompted for a model')

        monkeypatch.setattr('httpx.Client', _no_client)
        monkeypatch.setattr('memman.setup.wizard.click.prompt', _no_prompt)
        out = wizard.run_wizard(data_dir)
        assert config.LLM_MODEL not in out


class TestModelPrompts:
    """A non-OpenRouter endpoint prompts for the three model ids.
    """

    def test_non_openrouter_endpoint_prompts_for_missing_models(
            self, tty, tmp_path, monkeypatch):
        """Verify a custom endpoint prompts for the LLM, embed and rerank ids.

        Mutation: the wizard leaves the models to INSTALL_DEFAULTS, which
            hold OpenRouter ids the custom endpoint rejects.
        Oracle: scripted answers for the key and three models, compared to
            the returned rows by key.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        config.env_file_path(str(data_dir)).write_text('')
        config.reset_file_cache()
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))
        monkeypatch.delenv(config.API_KEY, raising=False)
        answers = {
            f'  {config.API_KEY}': 'custom-key',
            f'  {config.LLM_MODEL}': 'llm-id',
            f'  {config.EMBED_MODEL}': 'embed-id',
            f'  {config.RERANK_MODEL}': 'rerank-id',
            }
        monkeypatch.setattr(
            'memman.setup.wizard.click.prompt',
            lambda text, **kw: answers[text])
        out = wizard.run_wizard(
            str(data_dir), backend='sqlite',
            endpoint='https://api.example.com/v1')
        assert out == {
            config.DEFAULT_BACKEND: 'sqlite',
            config.BACKEND_FOR('default'): 'sqlite',
            config.ENDPOINT: 'https://api.example.com/v1',
            config.API_KEY: 'custom-key',
            config.LLM_MODEL: 'llm-id',
            config.EMBED_MODEL: 'embed-id',
            config.RERANK_MODEL: 'rerank-id',
            }
