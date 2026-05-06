"""Convert legacy global-backend env keys to the per-store key shape.

The legacy install model wrote `MEMMAN_BACKEND` and `MEMMAN_PG_DSN`
as process-global keys in `<data_dir>/env`. The per-store routing
model replaces these with `MEMMAN_BACKEND_<store>` /
`MEMMAN_PG_DSN_<store>` plus `MEMMAN_DEFAULT_BACKEND` /
`MEMMAN_DEFAULT_PG_DSN` defaults.

`bootstrap_per_store_keys(data_dir)` runs once on a legacy install:
it enumerates the stores reachable through the legacy globals,
writes per-store keys for any store missing one, seeds the new
defaults, and strips the bare globals. Re-running on an already-
converted install is a no-op.

Operator-edited per-store keys are never overwritten -- the
algorithm only fills blanks.
"""

import logging
from pathlib import Path

from memman import config
from memman.setup.scheduler import _write_env_keys
from memman.store.db import valid_store_name

logger = logging.getLogger('memman')


def bootstrap_per_store_keys(data_dir: str) -> list[str]:
    """Idempotent legacy-install -> per-store-keys converter.

    Returns a human-readable action log. An empty list means nothing
    needed converting (already on the per-store shape).
    """
    file_values = config.parse_env_file(config.env_file_path(data_dir))
    legacy_backend = (file_values.get(config.BACKEND) or '').strip()
    legacy_dsn = (file_values.get(config.PG_DSN) or '').strip()
    if not legacy_backend and not legacy_dsn:
        return []

    actions: list[str] = []
    updates: dict[str, str] = {}
    removes: set[str] = set()

    backend_kind = legacy_backend.lower() or 'sqlite'

    sqlite_stores = _enumerate_sqlite_stores(data_dir)
    pg_stores: list[str] = []
    if backend_kind == 'postgres' and legacy_dsn:
        pg_stores, pg_action = _enumerate_postgres_stores(legacy_dsn)
        if pg_action:
            actions.append(pg_action)

    if config.DEFAULT_BACKEND not in file_values and backend_kind:
        updates[config.DEFAULT_BACKEND] = backend_kind
        actions.append(f'set {config.DEFAULT_BACKEND}={backend_kind}')
    if (backend_kind == 'postgres' and legacy_dsn
            and config.DEFAULT_PG_DSN not in file_values):
        updates[config.DEFAULT_PG_DSN] = legacy_dsn
        actions.append(f'set {config.DEFAULT_PG_DSN}=<dsn>')

    seen: set[str] = set()
    all_stores = sqlite_stores + [s for s in pg_stores if s not in sqlite_stores]
    for store in all_stores:
        if store in seen:
            continue
        seen.add(store)
        key = config.BACKEND_FOR(store)
        if key in file_values:
            continue
        if store in pg_stores:
            updates[key] = 'postgres'
            actions.append(f'set {key}=postgres')
            dsn_key = config.PG_DSN_FOR(store)
            if dsn_key not in file_values and legacy_dsn:
                updates[dsn_key] = legacy_dsn
                actions.append(f'set {dsn_key}=<dsn>')
        else:
            updates[key] = backend_kind
            actions.append(f'set {key}={backend_kind}')

    if config.BACKEND in file_values:
        removes.add(config.BACKEND)
        actions.append(f'removed legacy {config.BACKEND}')
    if config.PG_DSN in file_values:
        removes.add(config.PG_DSN)
        actions.append(f'removed legacy {config.PG_DSN}')

    if updates or removes:
        _write_env_keys(updates, removes=removes, data_dir=data_dir)
    return actions


def _enumerate_sqlite_stores(data_dir: str) -> list[str]:
    """Return store names with a `<data_dir>/data/<name>/memman.db` file.
    """
    base = Path(data_dir) / 'data'
    if not base.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if not valid_store_name(entry.name):
            continue
        if (entry / 'memman.db').exists():
            names.append(entry.name)
    return names


def _enumerate_postgres_stores(dsn: str) -> tuple[list[str], str | None]:
    """Return `(store_names, action_or_None)`.

    The action string is set when psycopg is missing or the connection
    fails; the caller surfaces it as a warning while still proceeding
    with the SQLite-side conversion.
    """
    try:
        from memman.store.postgres import _connection
    except ImportError as exc:
        msg = f'skipped postgres enumeration: psycopg unavailable ({exc})'
        logger.warning(msg)
        return [], msg
    try:
        with _connection(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "select nspname from pg_namespace"
                " where nspname like 'store_%' order by nspname")
            rows = cur.fetchall()
    except Exception as exc:
        msg = f'skipped postgres enumeration: {exc}'
        logger.warning(msg)
        return [], msg
    names = [r[0][len('store_'):] for r in rows]
    return [n for n in names if valid_store_name(n)], None
