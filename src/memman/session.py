"""Active-store session helper.

`active_store` is the canonical "open the configured backend, run the
standard pre-flight, and ensure close on exit" entry point used
throughout the CLI and any other code path that needs the active
Backend for the duration of one operation.

Lives at the top level (not under `memman.store/`) because the
function deliberately composes three subsystems -- store dispatch
(`store.factory`), embedding fingerprint (`embed.fingerprint`), and
graph constants (`graph.engine`). Putting the composition at the top
level keeps each subsystem self-contained.

Usage:

    from memman.session import active_store

    with active_store(data_dir=data_dir, store=name) as backend:
        ins = backend.nodes.get(id)

The context manager closes the Backend (and its underlying connection)
on `__exit__`, even when the body raises. `unchecked=True` skips the
fingerprint seed/assert and constants reindex, used by diagnostics
(`memman doctor`, `memman embed status`) that must run against a stale
or fresh store without being aborted by `EmbedFingerprintError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import click
from memman.store.backend import Backend


@contextmanager
def active_store(
        *, data_dir: str, store: str,
        unchecked: bool = False) -> Iterator[Backend]:
    """Yield the active Backend for one operation.

    Dispatches on the per-store keys (`MEMMAN_BACKEND_<store>` with
    fallback to `MEMMAN_DEFAULT_BACKEND`) via `factory.open_backend`,
    runs the constants reindex pass, seeds + asserts the embedding
    fingerprint, then yields the Backend. Closes on exit even if the
    body raises.

    Imports for the fingerprint helpers are deferred to call time so
    test-suite monkeypatching of `memman.embed.fingerprint` (autouse
    seed-then-assert) is observed.

    Args:
        data_dir: base data directory.
        store: resolved store name (the caller is expected to apply
            `_resolve_store_name` before invoking this helper).
        unchecked: when True, skip seed/assert/reindex. Used by
            diagnostics that must run against a stale or fresh store.

    Raises
        click.ClickException: if the fingerprint check fails. Other
            exceptions (cluster open failures, schema absence) propagate
            unchanged.
    """
    from memman.embed import fingerprint as fp_mod
    from memman.embed import get_client
    from memman.exceptions import ConfigError, EmbedFingerprintError
    from memman.graph.engine import reindex_if_constants_changed
    from memman.store.factory import open_backend

    backend = open_backend(store, data_dir)
    try:
        if not unchecked:
            reindex_if_constants_changed(backend)
            try:
                ec = get_client()
                fp_mod.seed_if_fresh(backend, ec)
                fp_mod.assert_consistent(backend, ec)
            except (EmbedFingerprintError, ConfigError) as exc:
                raise click.ClickException(str(exc)) from exc
        yield backend
    finally:
        backend.close()
