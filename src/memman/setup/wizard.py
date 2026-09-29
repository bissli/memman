"""Interactive install wizard for `memman install`.

A click-only TUI that collects the values `memman install` needs and
returns them for the env file. It never overrides an existing env-file
value. `setup.claude.run_install` rejects flag-vs-file conflicts before
the wizard runs.

Notes
-----
- LLM endpoint: one URL (`MEMMAN_LLM_ENDPOINT`) for any OpenAI-compatible
  `/chat/completions` server. OpenRouter is the default and takes the
  shipped model from `INSTALL_DEFAULTS` with no prompt. Any other
  endpoint prompts for a model slug.
- Secrets: the embed provider's API key (when one is required) and the
  LLM endpoint's API key (required off loopback) are prompted with
  masked input when both the env file and the shell lack them.
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
from memman.embed import SUPPORTED_EMBED_PROVIDERS

DSN_MAX_ATTEMPTS = 3
DSN_PROBE_TIMEOUT_SEC = 5
ENDPOINT_MAX_ATTEMPTS = 3
API_KEY_MAX_ATTEMPTS = 3


def run_wizard(
        data_dir: str,
        *,
        backend: str | None = None,
        pg_dsn: str | None = None,
        llm_endpoint: str | None = None,
        embed_provider: str | None = None,
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
    llm_endpoint : str | None
        Explicit `--llm-endpoint` flag value, or None.
    embed_provider : str | None
        Explicit `--embed-provider` flag value, or None.
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

    endpoint, endpoint_user_supplied = _select_llm_endpoint(
        flag=llm_endpoint, file_values=file_values, interactive=interactive)
    if endpoint_user_supplied:
        out[config.LLM_ENDPOINT] = endpoint

    chosen_embed, embed_user_supplied = _select_embed_provider(
        flag=embed_provider, file_values=file_values, interactive=interactive)
    if embed_user_supplied:
        out[config.EMBED_PROVIDER] = chosen_embed

    out.update(_collect_secrets(
        file_values, embed=chosen_embed, interactive=interactive))
    out.update(_collect_llm_api_key(
        file_values, endpoint=endpoint, interactive=interactive))
    out.update(_collect_llm_model(
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


def _collect_secrets(
        file_values: dict[str, str],
        *,
        embed: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for missing embed-side mandatory secrets when interactive.

    Parameters
    ----------
    file_values : dict[str, str]
        The env file as parsed before the wizard ran.
    embed : str
        Embed provider; `required_install_keys(embed)` names the
        mandatory keys. An experimental provider has none.
    interactive : bool
        False returns `{}`, and the prereq check raises
        `<KEY> is required ...` later.

    Returns
    -------
    dict[str, str]
        Secrets the user typed. A key already in the file, or exported
        in the shell under its MEMMAN-prefixed name, is skipped. A key
        exported only under its vendor-native name (e.g.
        `VOYAGE_API_KEY`) gets a masked prompt defaulting to that value.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    for key in sorted(config.required_install_keys(embed)):
        if file_values.get(key, '').strip():
            continue
        if os.environ.get(key, '').strip():
            continue
        native_value = _native_only_value(key)
        if native_value:
            out[key] = _prompt_with_native_default(key, native_value)
            continue
        click.echo(click.style(
            f'\n{key} is not set; install requires it.', fg='yellow'))
        value = click.prompt(
            f'  {key}', hide_input=True, confirmation_prompt=False)
        out[key] = value.strip()
    return out


def _native_only_value(key: str) -> str:
    """Shell value of `key`'s vendor-native fallback variable.

    Returns '' when the key has no registered native fallback or the
    native variable is empty. The MEMMAN-prefixed shell variable is not
    consulted, so a caller checks it first.
    """
    native_name = config.NATIVE_INSTALL_KEY_FALLBACKS.get(key)
    if not native_name:
        return ''
    return os.environ.get(native_name, '').strip()


def _prompt_with_native_default(
        key: str,
        native_value: str,
        *,
        source_key: str | None = None) -> str:
    """Announce a detected native shell key and prompt with it as default.

    The prompt is masked. The printed hint names the native variable, so
    a blank Enter is auditable.

    Parameters
    ----------
    key : str
        MEMMAN-prefixed env name being collected, shown in the prompt.
    native_value : str
        Non-empty native shell value, offered as the default.
    source_key : str | None
        MEMMAN-prefixed name whose native fallback supplied
        `native_value`. None means `key`. The OpenRouter cascade passes
        `OPENROUTER_API_KEY` because that native key seeds
        `MEMMAN_LLM_API_KEY`.

    Returns
    -------
    str
        The stripped value the user entered or accepted.
    """
    native_name = config.NATIVE_INSTALL_KEY_FALLBACKS[source_key or key]
    click.echo('')
    click.echo(click.style(
        f'  detected {native_name} in shell;'
        ' press Enter to use it, or type a new value', dim=True))
    value = click.prompt(
        f'  {key}', hide_input=True, default=native_value,
        show_default=False, confirmation_prompt=False)
    return value.strip()


def _select_llm_endpoint(
        *,
        flag: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the LLM endpoint URL: flag, then file, then prompt or default.

    Returns
    -------
    tuple[str, bool]
        `(endpoint, user_supplied)`. `user_supplied` is True for the
        `--llm-endpoint` flag or a prompt answer, which the wizard
        persists as `MEMMAN_LLM_ENDPOINT`. False for a file value or the
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
                f'--llm-endpoint must start with http:// or https://;'
                f' got {flag!r}')
        return flag, True
    existing = file_values.get(config.LLM_ENDPOINT, '').strip()
    if existing:
        return existing, False
    default = config.INSTALL_DEFAULTS[config.LLM_ENDPOINT]
    if not interactive:
        return default, False
    click.echo('')
    click.echo(click.style('Choose an LLM endpoint URL:', bold=True))
    click.echo(click.style(
        '  OpenRouter is the default; any OpenAI-compatible endpoint works'
        ' (Anthropic at /v1, OpenAI, Gemini /v1beta/openai, Ollama, ...).',
        dim=True))
    for attempt in range(1, ENDPOINT_MAX_ATTEMPTS + 1):
        candidate = click.prompt(
            '  LLM endpoint URL', default=default,
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


def _collect_llm_api_key(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for `MEMMAN_LLM_API_KEY` when interactive and not already set.

    Returns `{}` when the file or shell has the key. On OpenRouter it
    also returns `{}` when `MEMMAN_OPENROUTER_API_KEY` is present, since
    `collect_install_knobs` fills `LLM_API_KEY` from it.

    Parameters
    ----------
    file_values : dict[str, str]
        The env file as parsed before the wizard ran.
    endpoint : str
        The LLM endpoint the install uses. A loopback endpoint may
        leave the key blank.
    interactive : bool
        False returns `{}`.

    Returns
    -------
    dict[str, str]
        `{MEMMAN_LLM_API_KEY: <key>}`, or `{}`.

    Raises
    ------
    click.ClickException
        A non-loopback endpoint gets a blank key `API_KEY_MAX_ATTEMPTS`
        times.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    if file_values.get(config.LLM_API_KEY, '').strip():
        return out
    if os.environ.get(config.LLM_API_KEY, '').strip():
        return out
    if config.is_openrouter_endpoint(endpoint):
        if (file_values.get(config.OPENROUTER_API_KEY, '').strip()
                or os.environ.get(config.OPENROUTER_API_KEY, '').strip()):
            return out
        native_or = _native_only_value(config.OPENROUTER_API_KEY)
        if native_or:
            out[config.LLM_API_KEY] = _prompt_with_native_default(
                config.LLM_API_KEY, native_or,
                source_key=config.OPENROUTER_API_KEY)
            return out
    loopback = config.is_loopback_endpoint(endpoint)
    click.echo('')
    if loopback:
        click.echo(click.style(
            f'{config.LLM_API_KEY} (optional for loopback endpoint;'
            ' leave blank to skip).', dim=True))
        value = click.prompt(
            f'  {config.LLM_API_KEY}',
            default='', show_default=False,
            hide_input=True, confirmation_prompt=False).strip()
        if value:
            out[config.LLM_API_KEY] = value
        return out
    click.echo(click.style(
        f'{config.LLM_API_KEY} is required for non-loopback endpoints.',
        fg='yellow'))
    for attempt in range(1, API_KEY_MAX_ATTEMPTS + 1):
        value = click.prompt(
            f'  {config.LLM_API_KEY}',
            hide_input=True, confirmation_prompt=False).strip()
        if value:
            out[config.LLM_API_KEY] = value
            return out
        click.echo(click.style(
            '  API key is required for non-loopback endpoints', fg='red'))
        if attempt == API_KEY_MAX_ATTEMPTS:
            raise click.ClickException(
                f'gave up collecting {config.LLM_API_KEY} after'
                f' {API_KEY_MAX_ATTEMPTS} attempts')


def _collect_llm_model(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Collect `MEMMAN_LLM_MODEL` in a TTY when neither file nor shell has it.

    Parameters
    ----------
    file_values : dict[str, str]
        The env file as parsed before the wizard ran.
    endpoint : str
        The LLM endpoint the install uses.
    interactive : bool
        False returns `{}` with no prompt.

    Returns
    -------
    dict[str, str]
        `{MEMMAN_LLM_MODEL: <id>}`, or `{}` when a value exists, the
        session is headless, or the endpoint is OpenRouter. A headless
        install on a non-OpenRouter endpoint with no model is refused by
        `collect_install_knobs`.

    Raises
    ------
    click.ClickException
        The user enters a blank slug `ENDPOINT_MAX_ATTEMPTS` times.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    if file_values.get(config.LLM_MODEL, '').strip():
        return out
    if os.environ.get(config.LLM_MODEL, '').strip():
        return out
    if config.is_openrouter_endpoint(endpoint):
        return out
    click.echo('')
    click.echo(click.style(
        'Non-OpenRouter endpoint: enter the model slug to use.', bold=True))
    click.echo(click.style(
        '  It passes through verbatim to /chat/completions; consult the'
        " vendor's docs for valid ids.", dim=True))
    for attempt in range(1, ENDPOINT_MAX_ATTEMPTS + 1):
        value = click.prompt('  model slug').strip()
        if value:
            out[config.LLM_MODEL] = value
            return out
        click.echo(click.style('  model slug cannot be blank', fg='red'))
    raise click.ClickException(
        f'gave up collecting {config.LLM_MODEL} after'
        f' {ENDPOINT_MAX_ATTEMPTS} attempts')


def _select_embed_provider(
        *,
        flag: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the embed provider: flag, then file, then prompt or default.

    Returns
    -------
    tuple[str, bool]
        `(chosen, user_supplied)`. `user_supplied` is True for the
        `--embed-provider` flag or a prompt answer, which the wizard
        persists as `MEMMAN_EMBED_PROVIDER`. False for a file value or
        the headless default, which `INSTALL_DEFAULTS` writes later.
    """
    if flag:
        return flag, True
    existing = file_values.get(config.EMBED_PROVIDER, '').strip()
    if existing:
        if existing not in SUPPORTED_EMBED_PROVIDERS and interactive:
            click.echo(click.style(
                f'  note: {config.EMBED_PROVIDER}={existing!r} is an'
                ' experimental provider; doctor will warn.', dim=True))
        return existing, False
    default = config.INSTALL_DEFAULTS[config.EMBED_PROVIDER]
    if not interactive:
        return default, False
    click.echo('')
    click.echo(click.style('Choose an embed provider:', bold=True))
    chosen = click.prompt(
        '  embed provider',
        type=click.Choice(list(SUPPORTED_EMBED_PROVIDERS)),
        default=default, show_choices=True, show_default=True)
    return chosen, True


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
