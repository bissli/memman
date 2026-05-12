"""Interactive install wizard for `memman install`.

Pure-click TUI -- no questionary / prompt_toolkit. Three visible
features today:

1. LLM endpoint selection. memman speaks one wire protocol (OpenAI's
   `/chat/completions`). The wizard prompts for a single endpoint URL
   (`MEMMAN_LLM_ENDPOINT`); OpenRouter is the default, and any other
   OpenAI-compat endpoint (Anthropic at `/v1`, OpenAI, Gemini's
   OpenAI shim, Ollama, vLLM, LiteLLM, ...) is accepted. For non-OR
   endpoints the wizard also prompts for the three role-specific model
   slugs (no shared model catalog exists for non-OR vendors).

2. Mandatory-secret prompting. The embed provider's API key (when one
   is required) and the LLM endpoint's API key (required for any
   non-loopback endpoint) are prompted with masked input when missing
   from both the env file and the shell.

3. Backend selection (sqlite | postgres). Postgres is hidden until
   the `memman[postgres]` extras are importable AND the
   `memman.store.postgres` module exists. Until both checks pass,
   only sqlite is selectable -- the wizard skips the prompt entirely
   and writes `MEMMAN_DEFAULT_BACKEND=sqlite` straight through,
   avoiding a one-option confirmation prompt.

The wizard writes per-store dispatch keys: `MEMMAN_DEFAULT_BACKEND`
(and `MEMMAN_DEFAULT_POSTGRES_DSN` for postgres) plus
`MEMMAN_BACKEND_default` (and `MEMMAN_POSTGRES_DSN_default` for postgres)
for the freshly-created `default` store.

Non-TTY mode (`sys.stdin.isatty()` False or `--no-wizard`) skips all
prompts and uses flag values + defaults. The wizard never silently
overrides an existing env-file value; flag-vs-file conflicts are
handled by the caller in `setup.claude.run_install` before the wizard
is invoked, with a clear message pointing at `memman config set`.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec

import click
from memman import config, extras
from memman.embed import SUPPORTED_EMBED_PROVIDERS

DSN_MAX_ATTEMPTS = 3
DSN_PROBE_TIMEOUT_SEC = 5
ENDPOINT_MAX_ATTEMPTS = 3
API_KEY_MAX_ATTEMPTS = 3
MODEL_SLUG_PROMPTS: tuple[tuple[str, str], ...] = (
    ('fast', 'MEMMAN_LLM_MODEL_FAST'),
    ('slow canonical', 'MEMMAN_LLM_MODEL_SLOW_CANONICAL'),
    ('slow metadata', 'MEMMAN_LLM_MODEL_SLOW_METADATA'),
    )


def run_wizard(
        data_dir: str,
        *,
        backend: str | None = None,
        pg_dsn: str | None = None,
        llm_endpoint: str | None = None,
        embed_provider: str | None = None,
        no_wizard: bool = False) -> dict[str, str]:
    """Drive the install wizard and return values to merge into the env file.

    The returned dict contains only keys the wizard chose / collected.
    Caller is responsible for merging this dict into `~/.memman/env`
    via `_write_env_keys` BEFORE `check_prereqs` runs, so the prereq
    check sees the secrets in the file layer.

    Args:
        data_dir: base data directory; the env file lives at <data_dir>/env.
        backend: explicit `--backend` flag value, or None when unset.
        pg_dsn: explicit `--pg-dsn` flag value, or None when unset.
        llm_endpoint: explicit `--llm-endpoint` flag value, or None.
        embed_provider: explicit `--embed-provider` flag value, or None.
        no_wizard: when True, skip all prompts (use flags + defaults).

    Returns
        Dict of env-file rows the wizard collected, e.g.
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
    out.update(_collect_llm_model_slugs(
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
            out[config.env_key_for('postgres', 'DSN', 'default')] = dsn

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

    Skips a key when the file already has it OR when the shell exports
    it (which would seed the file via `collect_install_knobs` anyway).
    Non-interactive runs return an empty dict and let the existing
    prereq check raise `<KEY> is required ...`. The set of mandatory
    keys is computed from `required_install_keys(embed)`; experimental
    providers return an empty set (no prompts).
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    for key in sorted(config.required_install_keys(embed)):
        if file_values.get(key, '').strip():
            continue
        if os.environ.get(key, '').strip():
            continue
        click.echo(click.style(
            f'\n{key} is not set; install requires it.', fg='yellow'))
        value = click.prompt(
            f'  {key}', hide_input=True, confirmation_prompt=False)
        out[key] = value.strip()
    return out


def _select_llm_endpoint(
        *,
        flag: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the LLM endpoint URL and report whether the user supplied it.

    Returns `(endpoint, user_supplied)`. `user_supplied` is True when
    the value came from `--llm-endpoint` or an interactive prompt --
    i.e., something the wizard should persist to the env file as
    `MEMMAN_LLM_ENDPOINT`. False when the value already lived in the
    file or when the wizard is just naming the default headlessly (in
    which case `INSTALL_DEFAULTS` writes it later in the install flow).
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
    raise click.ClickException('unreachable')


def _is_http_url(value: str) -> bool:
    """Lightweight `http(s)://` prefix check used by the endpoint prompt."""
    lowered = value.lower()
    return lowered.startswith(('http://', 'https://'))


def _collect_llm_api_key(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for `MEMMAN_LLM_API_KEY` when interactive and not already set.

    Loopback endpoints (Ollama, local vLLM, LiteLLM proxy) may omit the
    key; non-loopback endpoints re-prompt up to `API_KEY_MAX_ATTEMPTS`
    times if the user enters a blank value before refusing the install.
    When the endpoint is OpenRouter and `MEMMAN_OPENROUTER_API_KEY` is
    already present, `collect_install_knobs` auto-fills `LLM_API_KEY`
    from it -- the wizard skips the prompt in that case.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    if file_values.get(config.LLM_API_KEY, '').strip():
        return out
    if os.environ.get(config.LLM_API_KEY, '').strip():
        return out
    if (config.is_openrouter_endpoint(endpoint)
            and (file_values.get(config.OPENROUTER_API_KEY, '').strip()
                 or os.environ.get(config.OPENROUTER_API_KEY, '').strip())):
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
    return out


def _collect_llm_model_slugs(
        file_values: dict[str, str],
        *,
        endpoint: str,
        interactive: bool) -> dict[str, str]:
    """Prompt for the three role-model slugs when the endpoint is non-OR.

    OpenRouter endpoints rely on `collect_install_knobs` plus the
    `openrouter_models` resolver to fill role slugs from OR's catalog.
    Any other endpoint has no shared catalog memman can introspect, so
    the wizard prompts interactively for each missing slug.
    Non-interactive non-OR installs leave the slugs unset; the user is
    expected to set them via `memman config set` or shell env.
    """
    out: dict[str, str] = {}
    if config.is_openrouter_endpoint(endpoint):
        return out
    if not interactive:
        return out
    click.echo('')
    click.echo(click.style(
        'Non-OpenRouter endpoint: enter the three model slugs to use.',
        bold=True))
    click.echo(click.style(
        '  These pass through verbatim to /chat/completions; consult the'
        " vendor's docs for valid ids.", dim=True))
    for label, env_key in MODEL_SLUG_PROMPTS:
        if file_values.get(env_key, '').strip():
            continue
        if os.environ.get(env_key, '').strip():
            continue
        for attempt in range(1, ENDPOINT_MAX_ATTEMPTS + 1):
            value = click.prompt(f'  {label} model slug').strip()
            if value:
                out[env_key] = value
                break
            click.echo(click.style(
                '  model slug cannot be blank', fg='red'))
            if attempt == ENDPOINT_MAX_ATTEMPTS:
                raise click.ClickException(
                    f'gave up collecting {env_key} after'
                    f' {ENDPOINT_MAX_ATTEMPTS} attempts')
    return out


def _select_embed_provider(
        *,
        flag: str | None,
        file_values: dict[str, str],
        interactive: bool) -> tuple[str, bool]:
    """Resolve the embed provider and report whether the user supplied it.

    Returns `(chosen, user_supplied)`. `user_supplied` is True when the
    value came from a `--embed-provider` flag or an interactive prompt --
    i.e., something the wizard should persist to the env file as
    `MEMMAN_EMBED_PROVIDER`. False when the value already lived in the
    file or when the wizard is just naming the default headlessly (in
    which case `INSTALL_DEFAULTS` writes it later in the install flow).
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
    """Resolve the backend choice and report whether the user supplied it.

    Returns `(chosen_backend, user_supplied)`. `user_supplied` is True
    when the value came from a flag, an existing file row, or an
    interactive prompt -- i.e., something the wizard should persist back
    to the env file as `MEMMAN_DEFAULT_BACKEND`. False when the wizard
    is just naming the default (`'sqlite'`) and `INSTALL_DEFAULTS` will
    write it later in the install flow.
    """
    if backend:
        return backend, True
    file_backend = file_values.get(config.DEFAULT_BACKEND, '').strip()
    if file_backend:
        return file_backend, False
    options = _selectable_backends()
    if len(options) <= 1 or not interactive:
        return (options[0] if options else 'sqlite'), False
    click.echo('')
    click.echo(click.style('Choose a memman storage backend:', bold=True))
    for opt in options:
        suffix = click.style(' (default)', dim=True) if opt == 'sqlite' else ''
        click.echo(f'  {click.style(opt, fg="cyan")}{suffix}')
    chosen = click.prompt(
        '  backend', type=click.Choice(options), default='sqlite',
        show_choices=False, show_default=False)
    return chosen, True


def _selectable_backends() -> list[str]:
    """Return the list of backends the wizard can offer.

    Sqlite is always available. Postgres is included only when both
    `memman[postgres]` extras are importable AND the runtime
    `memman.store.postgres` module exists.
    """
    out = ['sqlite']
    if extras.is_available('postgres') and _backend_module_exists():
        out.append('postgres')
    return out


def _backend_module_exists() -> bool:
    """Return True when `memman.store.postgres` can be imported."""
    try:
        return find_spec('memman.store.postgres') is not None
    except ModuleNotFoundError:
        return False


def _collect_dsn(
        *,
        pg_dsn: str | None,
        file_values: dict[str, str],
        interactive: bool) -> str | None:
    """Resolve a Postgres DSN: flag > file > interactive prompt + probe.

    The probe is `psycopg.connect(dsn, connect_timeout=N).close()`.
    Re-prompts up to `DSN_MAX_ATTEMPTS` on failure, then exits 1.
    Non-interactive runs require `pg_dsn` or a file value -- otherwise
    they error out telling the user to pass `--pg-dsn`.
    """
    if pg_dsn:
        _probe_dsn_or_die(pg_dsn)
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


def _probe_dsn_or_die(dsn: str) -> None:
    """Probe a DSN; raise click.ClickException on failure."""
    try:
        _probe_dsn(dsn)
    except Exception as exc:
        raise click.ClickException(f'postgres connection failed: {exc}')


def _probe_dsn(dsn: str) -> None:
    """Open + verify pgvector + emit PgBouncer hint on remote DSN.

    Asserts `select 1` and `pg_extension where extname = 'vector'`;
    non-localhost URLs emit a PgBouncer recommendation. Raises on
    hard failure (cannot connect, pgvector missing).

    Lazy-imports `psycopg` so users without `memman[postgres]` are
    not blocked from importing the wizard module itself.
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
    """Best-effort detection of a non-localhost host in a DSN.

    Handles both `host=...` keyword form and `postgresql://host/...`
    URI form. Returns False on parse failure (defensive: don't
    spam the hint for parse-edge-case DSNs).
    """
    lowered = dsn.lower()
    for marker in ('host=localhost', 'host=127.0.0.1', '@localhost', '@127.0.0.1'):
        if marker in lowered:
            return False
    return bool('host=' in lowered or '://' in lowered)
