"""Interactive install wizard for `memman install`.

Pure-click TUI -- no questionary / prompt_toolkit. Three visible
features today:

1. LLM endpoint selection. memman speaks one wire protocol (OpenAI's
   `/chat/completions`). The wizard prompts for a single endpoint URL
   (`MEMMAN_LLM_ENDPOINT`); OpenRouter is the default, and any other
   OpenAI-compat endpoint (Anthropic at `/v1`, OpenAI, Gemini's
   OpenAI shim, Ollama, vLLM, LiteLLM, ...) is accepted. When the env
   file has no model, the wizard offers up to three OpenRouter
   candidates to pick from, or prompts for the slug on any other
   endpoint (no shared model catalog exists for non-OR vendors).

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
import httpx
from memman import config, extras
from memman.embed import SUPPORTED_EMBED_PROVIDERS
from memman.llm import openrouter_models

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

    Skips silently when the file has the key OR when the shell exports
    the MEMMAN-prefixed name (which seeds the file via
    `collect_install_knobs` anyway). When only the vendor-native name
    is exported (e.g. `VOYAGE_API_KEY`), the wizard announces the
    detection and shows a masked prompt with the native value as the
    default -- never silently captures cross-tool shell variables.
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
    """Return the shell value of `key`'s native fallback when MEMMAN- is unset.

    Returns '' when the key has no registered native fallback or when
    the native shell variable is empty. The MEMMAN-prefixed shell var
    is NOT consulted -- callers must do that check first and treat a
    set MEMMAN- value as a silent-skip case.
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

    `key` is the memman-prefixed env name being collected (shown in the
    prompt). `source_key` is the memman-prefixed name whose native
    fallback supplied `native_value`; defaults to `key` for the common
    direct-mapping case (`MEMMAN_VOYAGE_API_KEY` <- `VOYAGE_API_KEY`).
    The OpenRouter LLM cascade passes `source_key=OPENROUTER_API_KEY`
    because the detected native key (`OPENROUTER_API_KEY`) seeds a
    different memman key (`MEMMAN_LLM_API_KEY`).

    Caller has already verified `native_value` is non-empty and the
    MEMMAN-prefixed name is unset. The prompt stays masked so the
    default value isn't echoed; the printed hint names the detected
    native variable so a blank Enter is auditable.
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
    return out


def pick_candidate(candidates: list[openrouter_models.Candidate]) -> str:
    """Number `candidates`, prompt for one, and return its model id.

    Parameters
    ----------
    candidates : list[openrouter_models.Candidate]
        Non-empty, in display order; the prompt defaults to the first.

    Returns
    -------
    str
        The picked candidate's `model_id`.
    """
    for number, candidate in enumerate(candidates, 1):
        click.echo(f'  {number}. {candidate.label()}')
    number = click.prompt(
        '  model', type=click.IntRange(1, len(candidates)), default=1)
    return candidates[number - 1].model_id


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
        session is headless, or OpenRouter offers no candidate.

    Notes
    -----
    - On OpenRouter the operator picks from `fetch_candidates`, fed the
      shipped model's family and the pin and ceilings from the file,
      the shell, or `INSTALL_DEFAULTS`. A failed fetch or an empty list
      leaves the key to `INSTALL_DEFAULTS` in `collect_install_knobs`.
    - Any other endpoint has no catalog memman can read, so the operator
      types the slug. A headless install there with no model is refused
      by `collect_install_knobs`.
    """
    out: dict[str, str] = {}
    if not interactive:
        return out
    if file_values.get(config.LLM_MODEL, '').strip():
        return out
    if os.environ.get(config.LLM_MODEL, '').strip():
        return out
    click.echo('')
    if config.is_openrouter_endpoint(endpoint):
        seeds = {
            key: (file_values.get(key, '').strip()
                  or os.environ.get(key, '').strip()
                  or config.INSTALL_DEFAULTS[key])
            for key in (config.LLM_PROVIDER_ONLY,
                        config.LLM_MAX_INPUT_PRICE,
                        config.LLM_MAX_OUTPUT_PRICE)}
        default_model = config.INSTALL_DEFAULTS[config.LLM_MODEL]
        try:
            candidates = openrouter_models.fetch_candidates(
                endpoint,
                family=default_model.split('/', 1)[0] + '/',
                max_input_per_m=float(seeds[config.LLM_MAX_INPUT_PRICE]),
                max_output_per_m=float(seeds[config.LLM_MAX_OUTPUT_PRICE]),
                vendors=frozenset(
                    name.strip()
                    for name in seeds[config.LLM_PROVIDER_ONLY].split(',')
                    if name.strip()))
        except (httpx.HTTPError, RuntimeError) as exc:
            click.echo(click.style(
                f'  cannot read the OpenRouter catalogs ({exc});'
                f' installing {default_model}', fg='yellow'))
            return out
        if not candidates:
            click.echo(click.style(
                f'  no OpenRouter candidate under the pin and ceilings;'
                f' installing {default_model}', fg='yellow'))
            return out
        click.echo(click.style('Pick the LLM model:', bold=True))
        out[config.LLM_MODEL] = pick_candidate(candidates)
        return out
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
