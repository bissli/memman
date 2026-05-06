"""Interactive install wizard for `memman install`.

Pure-click TUI -- no questionary / prompt_toolkit. Two visible
features today:

1. Mandatory-secret prompting. When `OPENROUTER_API_KEY` or
   `VOYAGE_API_KEY` is missing from both the env file and the shell,
   prompt with masked input. Eliminates the README's
   `export X=...; export Y=...; memman install` ceremony for
   first-time installs in a TTY.

2. Backend selection (sqlite | postgres). Postgres is hidden until
   the `memman[postgres]` extras are importable AND the
   `memman.store.postgres` module exists. Until both checks pass,
   only sqlite is selectable -- the wizard skips the prompt entirely
   and writes `MEMMAN_DEFAULT_BACKEND=sqlite` straight through,
   avoiding a one-option confirmation prompt.

The wizard writes per-store dispatch keys: `MEMMAN_DEFAULT_BACKEND`
(and `MEMMAN_DEFAULT_PG_DSN` for postgres) plus
`MEMMAN_BACKEND_default` (and `MEMMAN_PG_DSN_default` for postgres)
for the freshly-created `default` store.

Non-TTY mode (`sys.stdin.isatty()` False or `--no-wizard`) skips all
prompts and uses flag values + defaults. The wizard never silently
overrides an existing env-file value; flag-vs-file conflicts are
handled by the caller in `setup.claude.run_install` before the wizard
is invoked, with a clear message pointing at `memman config set`.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

import click
from memman import config, extras

DSN_MAX_ATTEMPTS = 3
DSN_PROBE_TIMEOUT_SEC = 5


def run_wizard(
        data_dir: str,
        *,
        backend: str | None = None,
        pg_dsn: str | None = None,
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
        no_wizard: when True, skip all prompts (use flags + defaults).

    Returns
        Dict of env-file rows the wizard collected, e.g.
        `{MEMMAN_DEFAULT_BACKEND: 'sqlite',
        MEMMAN_BACKEND_default: 'sqlite'}`.
    """
    interactive = sys.stdin.isatty() and not no_wizard
    file_values = config.parse_env_file(config.env_file_path(data_dir))

    out: dict[str, str] = {}
    out.update(_collect_secrets(file_values, interactive=interactive))

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
            out[config.PG_DSN_FOR('default')] = dsn

    if interactive and not out:
        click.echo(click.style(
            'wizard: using existing env-file values', dim=True))

    return out


def _collect_secrets(
        file_values: dict[str, str],
        *,
        interactive: bool) -> dict[str, str]:
    """Prompt for missing mandatory secrets when running interactively.

    Skips a key when the file already has it OR when the shell exports
    it (which would seed the file via `collect_install_knobs` anyway).
    Non-interactive runs return an empty dict and let the existing
    prereq check raise `OPENROUTER_API_KEY is required ...`.
    """
    import os
    out: dict[str, str] = {}
    if not interactive:
        return out
    for key in config.MANDATORY_INSTALL_KEYS:
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
    import psycopg
    with psycopg.connect(
            dsn, connect_timeout=DSN_PROBE_TIMEOUT_SEC) as conn:
        with conn.cursor() as cur:
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
