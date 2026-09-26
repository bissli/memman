"""Active-store session helper.

`active_store` is the canonical "open the configured backend, run the
standard pre-flight, and ensure close on exit" entry point used
throughout the CLI and any other code path that needs the active
Backend for the duration of one operation.

Lives at the top level (not under `memman.store/`) because the
function deliberately composes two subsystems -- store dispatch
(`store.factory`) and embedding fingerprint (`embed.fingerprint`).
Putting the composition at the top level keeps each subsystem
self-contained.

Usage:

    from memman.session import active_store

    with active_store(data_dir=data_dir, store=name) as backend:
        ins = backend.nodes.get(id)

The context manager closes the Backend (and its underlying connection)
on `__exit__`, even when the body raises. `unchecked=True` skips the
fingerprint seed/assert, used by diagnostics
(`memman doctor`, `memman embed status`) that must run against a stale
or fresh store without being aborted by `EmbedFingerprintError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import click
from memman.store.backend import Backend
from memman.store.errors import BackendError


@contextmanager
def active_store(
        *, data_dir: str, store: str,
        unchecked: bool = False) -> Iterator[Backend]:
    """Yield the active Backend for one operation.

    Dispatches on the per-store keys (`MEMMAN_BACKEND_<store>` with
    fallback to `MEMMAN_DEFAULT_BACKEND`) via `factory.open_backend`,
    seeds + asserts the embedding fingerprint, then yields the
    Backend. Closes on exit even if the
    body raises.

    Parameters
    ----------
    data_dir : str
        Base data directory.
    store : str
        Resolved store name; the caller applies `_resolve_store_name`
        before invoking this helper.
    unchecked : bool, default False
        When True, skip the seed/assert. Used by diagnostics that
        must run against a stale or fresh store.

    Yields
    ------
    Backend
        The open per-store Backend, closed on exit.

    Raises
    ------
    click.ClickException
        When the fingerprint check or the backend open via
        `factory.open_backend` fails.

    Notes
    -----
    - One `except` covers `ConfigError` from the runtime layer and
      from the backend layer alike, because `store.errors.ConfigError`
      subclasses `memman.exceptions.ConfigError`.
    - `BackendError` is wrapped alongside it, so a store whose
      on-disk state cannot be opened exits with a message instead of
      a traceback.
    - Imports for the fingerprint helpers are deferred to call time
      so the test suite's monkeypatching of `memman.embed.fingerprint`
      (autouse seed-then-assert) is observed.
    """
    from memman.embed import fingerprint as fp_mod
    from memman.embed import get_client
    from memman.exceptions import ConfigError, EmbedFingerprintError
    from memman.store.factory import open_backend

    try:
        backend = open_backend(store, data_dir)
    except (ConfigError, BackendError) as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        if not unchecked:
            try:
                fp_mod.seed_if_fresh(backend, get_client())
                fp_mod.bound_embedder(backend)
            except (EmbedFingerprintError, ConfigError) as exc:
                raise click.ClickException(str(exc)) from exc
        yield backend
    finally:
        backend.close()
