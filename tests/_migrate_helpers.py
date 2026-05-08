"""Test-only thin shim around the migrator classes.

Keeps existing migrate tests asserting the gather/apply contract
through `SqliteMigrator` and `PostgresMigrator` without rewriting
each test's call shape. Production callers (the `memman migrate`
CLI) instantiate the migrators directly; these helpers exist
solely so the test suite stays compact during the v3 clean-break
refactor that deleted the module-level
`migrate_store_to_postgres` / `migrate_store_to_sqlite` functions.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from memman.migrate import MigrateResult, SchemaState


def _canonical_data_dir(per_store_path: Path, store: str) -> tuple[str, str | None]:
    """Map an arbitrary per-store path to a `<data_dir>/data/<store>` layout.

    Returns `(data_dir, scratch_root)` where `scratch_root` is a
    tempdir to remove later (`None` when the input was already
    canonical). Migrators expect `<data_dir>/data/<store>/memman.db`;
    older tests pass flat `<tmp>/<store>` source dirs.
    """
    if (per_store_path.parent.name == 'data'
            and per_store_path.name == store):
        return str(per_store_path.parent.parent), None
    scratch = Path(tempfile.mkdtemp(prefix='memman-migrate-helper-'))
    (scratch / 'data').mkdir()
    Path(scratch / 'data' / store).symlink_to(per_store_path)
    return str(scratch), str(scratch)


def migrate_store_to_postgres(
        *, source_dir: str, dsn: str, store: str,
        dry_run: bool = False,
        state: SchemaState = SchemaState.ABSENT) -> MigrateResult:
    """Forward sqlite -> postgres via the migrator classes.

    Accepts both the canonical `<data_dir>/data/<store>` layout and
    legacy flat `<tmp>/<store>` test fixtures. After apply the
    helper performs the same destination-count verification the
    legacy module-level function did, raising `MigrateError` on
    mismatch so tests asserting verify behavior keep working.
    """
    from memman.migrate import _verify_destination_counts
    from memman.store.postgres import PostgresMigrator, _connection
    from memman.store.postgres import _store_schema, drop_postgres_store
    from memman.store.sqlite import SqliteMigrator

    data_dir, scratch = _canonical_data_dir(Path(source_dir), store)
    schema = _store_schema(store)

    try:
        src = SqliteMigrator(data_dir)
        src.preflight_source(store)
        payload = src.gather(store)

        if dry_run:
            return MigrateResult(
                store=store, schema=schema,
                insights=len(payload.insights),
                edges=len(payload.edges),
                oplog=len(payload.oplog),
                meta=len(payload.meta),
                dry_run=True)

        if state in {SchemaState.EMPTY, SchemaState.POPULATED}:
            drop_postgres_store(store, dsn)

        tgt = PostgresMigrator(data_dir, dsn=dsn)
        tgt.preflight_target(store)
        tgt.apply(store, payload)

        with _connection(dsn, autocommit=True) as conn:
            _verify_destination_counts(
                conn, schema, store,
                expected={
                    'insights': len(payload.insights),
                    'edges': len(payload.edges),
                    'oplog': len(payload.oplog),
                    'meta': len(payload.meta),
                    })
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    return MigrateResult(
        store=store, schema=schema,
        insights=len(payload.insights),
        edges=len(payload.edges),
        oplog=len(payload.oplog),
        meta=len(payload.meta),
        dry_run=False, verified=True)


def migrate_store_to_sqlite(
        *, dsn: str, target_dir: str, store: str) -> MigrateResult:
    """Reverse postgres -> sqlite via the migrator classes.

    Tests already use canonical `<data_dir>/data/<store>` target
    paths, so no scratch redirect is needed.
    """
    from memman.store.postgres import PostgresMigrator, _store_schema
    from memman.store.sqlite import SqliteMigrator

    target_path = Path(target_dir)
    if (target_path.parent.name != 'data'
            or target_path.name != store):
        raise ValueError(
            f'reverse-migrate test helper requires target_dir to be'
            f' <data_dir>/data/<store>, got {target_dir!r}')
    data_dir = str(target_path.parent.parent)
    schema = _store_schema(store)

    src = PostgresMigrator(data_dir, dsn=dsn)
    src.preflight_source(store)
    payload = src.gather(store)

    tgt = SqliteMigrator(data_dir)
    tgt.preflight_target(store)
    tgt.apply(store, payload)

    return MigrateResult(
        store=store, schema=schema,
        insights=len(payload.insights),
        edges=len(payload.edges),
        oplog=len(payload.oplog),
        meta=len(payload.meta),
        dry_run=False, verified=True)
