"""Interactive install wizard for `memman install`.

A click-only TUI that collects the values `memman install` needs and
returns them for the env file. It never overrides an existing env-file
value. `setup.claude.run_install` rejects flag-vs-file conflicts before
the wizard runs.

Notes
-----
- Endpoint: one URL (`MEMMAN_ENDPOINT`) for an OpenAI-compatible server
  that answers `/chat/completions`, `/embeddings` and `/rerank`.
  OpenRouter is the default and takes the shipped models from
  `INSTALL_DEFAULTS` with no prompt. Any other endpoint prompts for the
  LLM, embed, and rerank model ids.
- Secret: `MEMMAN_API_KEY` (required off loopback) is prompted with
  masked input when both the env file and the shell lack it.
- Backend: postgres is offered only when the `memman[postgres]` extras
  import. Without them the wizard writes `MEMMAN_DEFAULT_BACKEND=sqlite`
  with no prompt.
- Output keys: `MEMMAN_DEFAULT_BACKEND` and `MEMMAN_BACKEND_default`,
  plus `MEMMAN_DEFAULT_POSTGRES_DSN` and `MEMMAN_POSTGRES_DSN_default`
  for postgres.
- Non-TTY mode (`sys.stdin.isatty()` False or `--no-wizard`) skips every
  prompt and uses flag values and defaults.
"""

from __future__ import annotations

import os
import sys

import click
from memman import config, extras

DSN_MAX_ATTEMPTS = 3
DSN_PROBE_TIMEOUT_SEC = 5
ENDPOINT_MAX_ATTEMPTS = 3
API_KEY_MAX_ATTEMPTS = 3


def run_wizard(
        data_dir: str,
        *,
        backend: str | None = None,
        pg_dsn: str | None = None,
        endpoint: str | None = None,
        no_wizard: bool = False) -> dict[str, str]:
    """Drive the install wizard and return values to merge into the env file.

    The caller merges the result into `~/.memman/env` with
    `_write_env_keys` before `check_prereqs` runs, so the prereq check
    sees the collected secrets.

    Parameters
    ----------
    data_dir : str
        Base data directory; the env file lives at <data_dir>/env.
    backend : str | None
        Explicit `--backend` flag value, or None when unset.
    pg_dsn : str | None
        Explicit `--pg-dsn` flag value, or None when unset.
    endpoint : str | None
        Explicit `--endpoint` flag value, or None.
    no_wizard : bool
        True skips all prompts (flags and defaults only).

    Returns
    -------
    dict[str, str]
        Env-file rows the wizard collected, e.g.
        `{MEMMAN_DEFAULT_BACKEND: 'sqlite',
        MEMMAN_BACKEND_default: 'sqlite'}`.
    """
    interactive = sys.stdin.isatty() and not no_wizard
    file_values = config.parse_env_file(config.env_file_path(data_dir))

    out: dict[str, str] = {}

    endpoint, endpoint_user_supplied = _select_endpoint(
        flag=endpoint, file_values=file_values, interactive=interactive)
    if endpoint_user_supplied:
        out[config.ENDPOINT] = endpoint

    out.update(_collect_api_key(
        file_values, endpoint=endpoint, interactive=interactive))
    out.update(_collect_models(
        file_values, endpoint=endpoint, interactive=interactive))

    chosen_backend, backend_was_user_supplied = _select_backend(
        backend=backend, file_values=file_values, interactive=interactive)
    if backend_was_user_supplied:
        out[config.DEFAULT_BACKEND] = chosen_backend
        out[config.BACKEND_FOR('default')] = chosen_backend

    if chosen_backend == 'postgres':
        dsn = _collect_dsn(
            pg_dsn=pg_dsn, file_values=file_values,
            interactive=interactive)
        if dsn:
            out[config.DEFAULT_PG_DSN] = dsn
            out[config.POSTGRES_DSN_FOR('default')] = dsn

    if interactive and not out:
        click.echo(click.style(
            'wizard: using existing env-file values', dim=True))

    return out


def _prompt_with_native_default(key: str, native_value: str) -> str:
    """Announce a detected native shell key and prompt with it as default.

    The prompt is masked. The printed hint names the native variable, so
    a blank Enter is auditable.

    Parameters
    ----------
    key : str
        MEMMAN-prefixed env name being collected, shown in the prompt.
    native_value : str
        Non-empty `OPENROUTER_API_KEY` shell value, offered as the
        default.

    Returns
    -------
    str
        The stripped value the user entered or accepted.
    """
    click.echo('')
    click.echo(click.style(
        f'  detected {config.OPENROUTER_NATIVE_API_KEY} in shell;'
        ' press Enter to use it, or type a new value', dim=True))
    value = click.prompt(
        f'  {key}', hide_input=True, default=native_value,
        show_default=False, confirmation_prompt=False)
    return value.strip()


def _select_endpoint(
        *,
        flag: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the endpoint URL: flag, then file, then prompt or default.

    Returns
    -------
    tuple[str, bool]
        `(endpoint, user_supplied)`. `user_supplied` is True for the
        `--endpoint` flag or a prompt answer, which the wizard persists
        as `MEMMAN_ENDPOINT`. False for a file value or the
        headless default, which `INSTALL_DEFAULTS` writes later.

    Raises
    ------
    click.ClickException
        The flag is not an http(s) URL, or the prompt fails
        `ENDPOINT_MAX_ATTEMPTS` times.
    """
    if flag:
        if not _is_http_url(flag):
            raise click.ClickException(
                f'--endpoint must start with http:// or https://;'
                f' got {flag!r}')
        return flag, True
    existing = file_values.get(config.ENDPOINT, '').strip()
    if existing:
        return existing, False
    default = config.INSTALL_DEFAULTS[config.ENDPOINT]
    if not interactive:
        return default, False
    click.echo('')
    click.echo(click.style('Choose an endpoint URL:', bold=True))
    click.echo(click.style(
        '  OpenRouter is the default; the endpoint must serve'
        ' /chat/completions, /embeddings and /rerank.',
        dim=True))
    for attempt in range(1, ENDPOINT_MAX_ATTEMPTS + 1):
        candidate = click.prompt(
            '  endpoint URL', default=default,
            show_default=True).strip()
        if _is_http_url(candidate):
            return candidate, True
        click.echo(click.style(
            '  endpoint must start with http:// or https://', fg='red'))
        if attempt == ENDPOINT_MAX_ATTEMPTS:
            raise click.ClickException(
                'gave up resolving a valid endpoint URL after'
                f' {ENDPOINT_MAX_ATTEMPTS} attempts')


def _is_http_url(value: str) -> bool:
    """True when value starts with `http://` or `https://`.
    """
    lowered = value.lower()
    return lowered.startswith(('http://', 'https://'))


def _collect_api_key(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for `MEMMAN_API_KEY` when interactive and not already set.

    Parameters
    ----------
    file_values : dict[str, str]
        The env file as parsed before the wizard ran.
    endpoint : str
        The endpoint the install uses. A loopback endpoint may leave the
        key blank. On OpenRouter a shell `OPENROUTER_API_KEY` is offered
        as the default.
    interactive : bool
        False returns `{}`.

    Returns
    -------
    dict[str, str]
        `{MEMMAN_API_KEY: <key>}`, or `{}` when the file or shell has it.

    Raises
    ------
    click.ClickException
        A non-loopback endpoint gets a blank key `API_KEY_MAX_ATTEMPTS`
        times.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    if file_values.get(config.API_KEY, '').strip():
        return out
    if os.environ.get(config.API_KEY, '').strip():
        return out
    if config.is_openrouter_endpoint(endpoint):
        native = os.environ.get(config.OPENROUTER_NATIVE_API_KEY, '').strip()
        if native:
            out[config.API_KEY] = _prompt_with_native_default(
                config.API_KEY, native)
            return out
    loopback = config.is_loopback_endpoint(endpoint)
    click.echo('')
    if loopback:
        click.echo(click.style(
            f'{config.API_KEY} (optional for loopback endpoint;'
            ' leave blank to skip).', dim=True))
        value = click.prompt(
            f'  {config.API_KEY}',
            default='', show_default=False,
            hide_input=True, confirmation_prompt=False).strip()
        if value:
            out[config.API_KEY] = value
        return out
    click.echo(click.style(
        f'{config.API_KEY} is required for non-loopback endpoints.',
        fg='yellow'))
    for attempt in range(1, API_KEY_MAX_ATTEMPTS + 1):
        value = click.prompt(
            f'  {config.API_KEY}',
            hide_input=True, confirmation_prompt=False).strip()
        if value:
            out[config.API_KEY] = value
            return out
        click.echo(click.style(
            '  API key is required for non-loopback endpoints', fg='red'))
        if attempt == API_KEY_MAX_ATTEMPTS:
            raise click.ClickException(
                f'gave up collecting {config.API_KEY} after'
                f' {API_KEY_MAX_ATTEMPTS} attempts')


def _collect_models(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for the three model ids a non-OpenRouter endpoint needs.

    Parameters
    ----------
    file_values : dict[str, str]
        The env file as parsed before the wizard ran.
    endpoint : str
        The endpoint the install uses.
    interactive : bool
        False returns `{}` with no prompt.

    Returns
    -------
    dict[str, str]
        One row per model key neither the file nor the shell holds, or
        `{}` when the session is headless or the endpoint is OpenRouter.
        A headless install on a non-OpenRouter endpoint with a model
        missing is refused by `collect_install_knobs`.

    Raises
    ------
    click.ClickException
        The user enters a blank id `ENDPOINT_MAX_ATTEMPTS` times.
    """
    out: dict[str, str] = {}
    if not interactive or config.is_openrouter_endpoint(endpoint):
        return out
    missing = [
        key for key in (config.LLM_MODEL, config.EMBED_MODEL,
                        config.RERANK_MODEL)
        if not file_values.get(key, '').strip()
        and not os.environ.get(key, '').strip()]
    if not missing:
        return out
    click.echo('')
    click.echo(click.style(
        'Non-OpenRouter endpoint: enter the model ids to use.', bold=True))
    click.echo(click.style(
        "  Each passes through verbatim; consult the vendor's docs for"
        ' valid ids.', dim=True))
    for key in missing:
        for _attempt in range(ENDPOINT_MAX_ATTEMPTS):
            value = click.prompt(f'  {key}').strip()
            if value:
                out[key] = value
                break
            click.echo(click.style('  model id cannot be blank', fg='red'))
        else:
            raise click.ClickException(
                f'gave up collecting {key} after'
                f' {ENDPOINT_MAX_ATTEMPTS} attempts')
    return out


def _select_backend(
        *,
        backend: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the backend: flag, then file, then prompt or default.

    Returns
    -------
    tuple[str, bool]
        `(chosen_backend, user_supplied)`. `user_supplied` is True for
        the flag or a prompt answer, which the wizard persists as
        `MEMMAN_DEFAULT_BACKEND`. False for a file value or the
        `'sqlite'` default, which `INSTALL_DEFAULTS` writes later.
    """
    if backend:
        return backend, True
    file_backend = file_values.get(config.DEFAULT_BACKEND, '').strip()
    if file_backend:
        return file_backend, False
    options = ['sqlite']
    if extras.is_available('postgres'):
        options.append('postgres')
    if len(options) == 1 or not interactive:
        return options[0], False
    click.echo('')
    click.echo(click.style('Choose a memman storage backend:', bold=True))
    for opt in options:
        suffix = click.style(' (default)', dim=True) if opt == 'sqlite' else ''
        click.echo(f'  {click.style(opt, fg="cyan")}{suffix}')
    chosen = click.prompt(
        '  backend', type=click.Choice(options), default='sqlite',
        show_choices=False, show_default=False)
    return chosen, True


def _collect_dsn(
        *,
        pg_dsn: str | None,
        file_values: dict[str, str],
        interactive: bool) -> str | None:
    """Resolve a Postgres DSN: flag, then file, then prompt and probe.

    Returns
    -------
    str | None
        The DSN to persist, or None when the file already holds one.

    Raises
    ------
    click.ClickException
        The flag DSN fails the probe, a headless run has neither flag
        nor file value, or the prompt fails `DSN_MAX_ATTEMPTS` times.
    """
    if pg_dsn:
        try:
            _probe_dsn(pg_dsn)
        except Exception as exc:
            raise click.ClickException(
                f'postgres connection failed: {exc}')
        return pg_dsn
    if file_values.get(config.DEFAULT_PG_DSN, '').strip():
        return None
    if not interactive:
        raise click.ClickException(
            'postgres backend requires --pg-dsn in non-interactive mode')
    click.echo('')
    click.echo(click.style(
        'Enter a Postgres DSN (e.g. postgresql://user@host:5432/db).',
        bold=True))
    click.echo(click.style(
        '  Tip: omit the password and use ~/.pgpass for shared hosts.',
        dim=True))
    for attempt in range(1, DSN_MAX_ATTEMPTS + 1):
        candidate = click.prompt('  DSN', type=str).strip()
        try:
            _probe_dsn(candidate)
            return candidate
        except Exception as exc:
            click.echo(click.style(
                f'  connection failed: {exc}', fg='red'))
            if attempt == DSN_MAX_ATTEMPTS:
                raise click.ClickException(
                    f'gave up after {DSN_MAX_ATTEMPTS} attempts')


def _probe_dsn(dsn: str) -> None:
    """Connect, verify the pgvector extension, and hint PgBouncer.

    A non-localhost DSN also prints a PgBouncer recommendation.

    Parameters
    ----------
    dsn : str
        Postgres connection string.

    Raises
    ------
    RuntimeError
        The pgvector extension is not installed in the target database.
    Exception
        Any connection failure from the driver.
    """
    from memman.store.postgres import _connection
    with _connection(
            dsn, connect_timeout=DSN_PROBE_TIMEOUT_SEC,
            register_vector=False) as conn, \
            conn.cursor() as cur:
        cur.execute('select 1')
        cur.execute(
            "select 1 from pg_extension where extname = 'vector'")
        if cur.fetchone() is None:
            raise RuntimeError(
                'pgvector extension is not installed in the target '
                'database; run `create extension vector;` as a '
                'superuser, then retry')
    if _is_remote_dsn(dsn):
        click.echo(click.style(
            '  hint: non-localhost Postgres detected; consider'
            ' running through PgBouncer (transaction-pooling mode)'
            ' for connection-count safety in multi-agent'
            ' deployments.',
            dim=True))


def _is_remote_dsn(dsn: str) -> bool:
    """True when the DSN names a host other than localhost.

    Handles the `host=...` keyword form and the `postgresql://host/...`
    URI form. A DSN with neither form counts as local.
    """
    lowered = dsn.lower()
    local_markers = (
        'host=localhost', 'host=127.0.0.1', '@localhost', '@127.0.0.1')
    if any(marker in lowered for marker in local_markers):
        return False
    return 'host=' in lowered or '://' in lowered
