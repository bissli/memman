#!/usr/bin/env python3
r"""Whole-store schema rebuild via gather -> repair -> apply.

Rebuilds every SQLite store through `SqliteMigrator.gather()` ->
`repair_payload()` -> `apply()`, bringing the insights schema to the
current baseline. Retained from the 0.18.0 migration as the only
implementation of this machinery; the one-off 0.18.0 data repairs
(provenance backfill, backbone-to-proximity conversion) ran once and
were removed. `repair_payload` keeps only the STRUCTURAL orphan
filter -- gather selects every edge with no join, and an edge missing
a live endpoint would hit the FK-checked insert after the store was
already archived away.

A future schema change reuses this script wholesale: bump the payload
version it asserts, adjust guard 2's column probe to the new columns,
and put any new one-off repairs in `repair_payload`.

Usage
-----
    python scripts/rebuild_schema.py [STORE ...] [--data-dir PATH] \
                                     [--probe] [--dry-run] \
                                     [--log PATH] [--force]

With no STORE arguments, runs against every SQLite store directory
under `<data_dir>/data/`. `--probe` copies each store into a
throwaway `/tmp` data dir and rebuilds the copy (count parity proof,
zero risk); `--dry-run` gathers and repairs but writes nothing;
`--force` defeats the already-migrated skip (guard 2) and is only for
a store deliberately restored from `archive/` by hand.

Notes
-----
- Runs inside the shared drain lock; stop the scheduler first.
- On ANY exception after a store was archived, the rollback restores
  the pre-migration directory and the whole run aborts. A breadcrumb
  (`<data_dir>/rebuild_schema.inflight`) marks an in-flight store; a
  pre-existing breadcrumb is a hard abort, never a skip -- `apply`
  commits the new schema BEFORE its transaction opens, so a killed
  run leaves a valid-looking empty store the column probe alone
  would misread as migrated.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import memman
from memman.migrate import PAYLOAD_VERSION, MigrationPayload, held_drain_lock
from memman.store import node as _node
from memman.store.db import default_data_dir, list_local_store_dirs, open_db
from memman.store.db import store_dir
from memman.store.factory import resolve_store_backend
from memman.store.snapshot import delete_snapshot
from memman.store.sqlite import SqliteMigrator
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
BREADCRUMB_NAME = 'rebuild_schema.inflight'

_LOG_PATH: str | None = None


def _log(msg: str) -> None:
    """Print a line and append it to the log file when one is set."""
    print(msg)
    if _LOG_PATH:
        with Path(_LOG_PATH).open('a') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')


def assert_correct_interpreter() -> None:
    """Abort unless this process imports the editable-tree memman.

    The pipx interpreter would import 0.17.3's `SqliteMigrator` and
    rebuild the OLD schema, silently. Both conditions must hold: the
    import resolves inside this repo's `src/`, and its payload
    version is the 0.18.0 one.
    """
    mod_path = Path(memman.__file__).resolve()
    if not mod_path.is_relative_to(REPO / 'src'):
        raise SystemExit(
            f'wrong memman: imported {mod_path}, not the editable'
            f' tree under {REPO / "src"}. Run this script with the'
            ' editable-install interpreter:\n'
            f'  <editable venv>/bin/python'
            f' {REPO}/scripts/rebuild_schema.py')
    if PAYLOAD_VERSION != 2:
        raise SystemExit(
            f'wrong memman: PAYLOAD_VERSION is {PAYLOAD_VERSION},'
            ' expected 2 (the 0.18.0 schema). Refusing to rebuild'
            ' with a mismatched payload shape.')


@dataclass
class RepairResult:
    """Outcome of `repair_payload` -- the payload plus counters.

    Attributes
    ----------
    payload : MigrationPayload
        The repaired payload (mutated in place and returned).
    orphan_edges_dropped : int
        Edges missing a live endpoint, dropped by the orphan filter.
    touched_ids : set[str]
        Live, non-soft-deleted endpoints of dropped edges --
        `refresh_effective_importance` raises `ValueError` on a
        missing or soft-deleted row, after `apply` has committed.
    """

    payload: MigrationPayload
    orphan_edges_dropped: int = 0
    touched_ids: set = field(default_factory=set)


def repair_payload(payload: MigrationPayload) -> RepairResult:
    """Apply the structural repairs to a gathered payload.

    Pure and database-free: mutates and returns the payload inside a
    `RepairResult`. The 0.18.0 one-off repairs ran once and were
    removed; only the orphan filter remains, and it is NOT a one-off:
    `soft_delete` hard-deletes edges while leaving rows, so any
    future path that removes an insight without its edges recreates
    orphans, and `gather` selects every edge with no join.

    Parameters
    ----------
    payload : MigrationPayload
        As returned by `SqliteMigrator.gather`.

    Returns
    -------
    RepairResult
        Repaired payload plus the orphan counter and `touched_ids`.
    """
    result = RepairResult(payload)

    # orphans -- gather selects every edge with no join, and FK
    # enforcement would reject them at insert, after the store was
    # archived away
    live = {i.id for i in payload.insights}
    orphans = [e for e in payload.edges
               if e.source_id not in live or e.target_id not in live]
    payload.edges = [e for e in payload.edges
                     if e.source_id in live and e.target_id in live]
    result.orphan_edges_dropped = len(orphans)

    alive = {i.id for i in payload.insights if i.deleted_at is None}
    for e in orphans:
        result.touched_ids |= {e.source_id, e.target_id} & alive
    return result


def _insight_columns(db_path: Path) -> set[str]:
    """Return the insights table's column names, read-only."""
    with closing(sqlite3.connect(
            f'file:{db_path}?mode=ro', uri=True)) as conn:
        return {r[1] for r in conn.execute(
            'pragma table_info(insights)').fetchall()}


def _insight_count(db_path: Path) -> int:
    """Return `count(*)` of insights (including soft-deleted)."""
    with closing(sqlite3.connect(
            f'file:{db_path}?mode=ro', uri=True)) as conn:
        return int(conn.execute(
            'select count(*) from insights').fetchone()[0])


def store_gate(data_dir: str, store: str, force: bool) -> str:
    """Decide whether a store still needs migrating.

    Returns
    -------
    str
        'migrate' or 'skip'. Skipping requires ALL of: no breadcrumb
        (checked by the caller as a hard abort), both new columns
        present, and a non-zero insight count -- a new-schema db with
        zero rows is the killed-mid-apply signature, never a
        completed migration.
    """
    db_path = Path(store_dir(data_dir, store)) / 'memman.db'
    if not db_path.exists():
        return 'migrate'
    cols = _insight_columns(db_path)
    if not {'session_id', 'queue_uuid'} <= cols:
        return 'migrate'
    if _insight_count(db_path) == 0:
        return 'migrate'
    if force:
        _log(f'WARNING: --force re-running already-migrated store'
             f' {store!r}; repair_payload runs again against data it'
             ' already repaired, with nothing signalling any damage')
        return 'migrate'
    _log(f'{store}: already at 0.18.0 schema, skipping')
    return 'skip'


def verify_counts(
        db_path: Path, payload: MigrationPayload) -> list[str]:
    """Compare the applied store's row counts against the payload.

    Compares TOTAL insight rows (soft-deleted included), edges after
    repair, oplog rows, and embedded-row count. Returns a list of
    mismatch descriptions (empty = pass).
    """
    expected = {
        'insights': len(payload.insights),
        'edges': len(payload.edges),
        'oplog': len(payload.oplog),
        'embedded': sum(
            1 for i in payload.insights if i.embedding is not None),
        }
    with closing(sqlite3.connect(
            f'file:{db_path}?mode=ro', uri=True)) as conn:
        actual = {
            'insights': conn.execute(
                'select count(*) from insights').fetchone()[0],
            'edges': conn.execute(
                'select count(*) from edges').fetchone()[0],
            'oplog': conn.execute(
                'select count(*) from oplog').fetchone()[0],
            'embedded': conn.execute(
                'select count(*) from insights'
                ' where embedding is not null').fetchone()[0],
            }
    return [
        f'{k}: expected {expected[k]}, got {actual[k]}'
        for k in expected if expected[k] != actual[k]]


def migrate_store(
        m: SqliteMigrator, store: str, data_dir: str,
        *, dry_run: bool = False) -> RepairResult:
    """Run the per-store algorithm (already inside the drain lock).

    Steps: preflight -> gather -> repair -> archive -> drop ->
    apply -> count check -> effective-importance refresh. Steps after
    a successful archive are wrapped so ANY failure restores the
    pre-migration directory and re-raises -- the failure mode this
    guards is a valid, empty, doctor-clean store with the real rows
    stranded in `archive/`.
    """
    m.preflight_source(store)
    payload = m.gather(store)
    result = repair_payload(payload)
    _log(f'{store}:'
         f' orphan_edges_dropped={result.orphan_edges_dropped}'
         f' touched={len(result.touched_ids)}')
    if dry_run:
        return result

    breadcrumb = Path(data_dir) / BREADCRUMB_NAME
    breadcrumb.write_text(store)
    artifact = m.archive(store, data_dir)
    if artifact.kind == 'none' or not artifact.location:
        breadcrumb.unlink(missing_ok=True)
        raise RuntimeError(
            f'{store}: archive produced no artifact; without one'
            ' there is no rollback -- aborting before drop')
    try:
        # drop is a guaranteed no-op here (archive renamed the whole
        # directory away); kept only for order-independence
        m.drop(store)
        m.apply(store, result.payload)
        sdir = store_dir(data_dir, store)
        db_path = Path(sdir) / 'memman.db'
        mismatches = verify_counts(db_path, result.payload)
        if mismatches:
            raise RuntimeError(
                f'{store}: count check failed: {mismatches}')
        db = open_db(sdir)
        try:
            for iid in sorted(result.touched_ids):
                _node.refresh_effective_importance(db, iid)
        finally:
            db.close()
        delete_snapshot(sdir)
    except BaseException:
        sdir = Path(store_dir(data_dir, store))
        _log(f'{store}: FAILED after archive; rolling back from'
             f' {artifact.location}')
        shutil.rmtree(sdir, ignore_errors=True)
        shutil.move(str(artifact.location), str(sdir))
        # rollback restored the pre-migration directory, so nothing
        # is in flight any more; the breadcrumb survives only when
        # the move above itself blew up
        breadcrumb.unlink(missing_ok=True)
        raise
    breadcrumb.unlink(missing_ok=True)
    _log(f'{store}: rebuilt ok (archive at {artifact.location})')
    return result


def probe_store(store: str, data_dir: str) -> RepairResult:
    """Rebuild a copy of the store in a throwaway /tmp data dir.

    Never touches `data/` -- a `<store>__probe` directory there would
    become a real store to `list_local_store_dirs`, to backups and to
    the housekeeping loop. The throwaway dir registers nothing;
    cleanup is one rmtree.
    """
    tmp_root = tempfile.mkdtemp(prefix='memman-migrate-probe-')
    try:
        (Path(tmp_root) / 'data').mkdir()
        shutil.copytree(
            store_dir(data_dir, store),
            Path(tmp_root) / 'data' / store)
        return migrate_store(
            SqliteMigrator(tmp_root), store, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    """Entry point. Returns a process exit code."""
    global _LOG_PATH
    assert_correct_interpreter()

    parser = argparse.ArgumentParser(
        description='0.17.3 -> 0.18.0 store rebuild with data repair')
    parser.add_argument('stores', nargs='*', metavar='STORE')
    parser.add_argument('--data-dir', default=default_data_dir())
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log', default='/tmp/rebuild_schema.log')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    _LOG_PATH = args.log

    data_dir = args.data_dir
    stores = args.stores or list_local_store_dirs(data_dir)
    if not stores:
        _log(f'no stores found under {data_dir}/data')
        return 1

    breadcrumb = Path(data_dir) / BREADCRUMB_NAME
    if breadcrumb.exists():
        inflight = breadcrumb.read_text().strip()
        _log(f'ABORT: breadcrumb {breadcrumb} names in-flight store'
             f' {inflight!r} from an interrupted run. Restore it by'
             f' hand -- rm -rf {store_dir(data_dir, inflight)} &&'
             f' mv <newest archive/{inflight}/*> back, then delete'
             ' the breadcrumb -- before re-running.')
        return 1

    for s in stores:
        kind = resolve_store_backend(s, data_dir)
        if kind != 'sqlite':
            raise SystemExit(
                f'store {s!r} resolves to backend {kind!r};'
                ' this script is SQLite-only')

    failures = 0
    # Notes:
    # - expected records this run's applied insight count per store,
    #   so the final gate can pass a LEGITIMATELY empty store (five
    #   exist on this host) while still failing the killed-mid-apply
    #   signature (zero rows where the payload had rows). Stores
    #   skipped as already-migrated proved count > 0 in store_gate.
    expected: dict[str, int] = {}
    with held_drain_lock(data_dir):
        for s in tqdm(stores, unit='store', desc='rebuild'):
            if not args.probe and not args.dry_run:
                if store_gate(data_dir, s, args.force) == 'skip':
                    continue
            try:
                if args.probe:
                    probe_store(s, data_dir)
                else:
                    result = migrate_store(
                        SqliteMigrator(data_dir), s, data_dir,
                        dry_run=args.dry_run)
                    expected[s] = len(result.payload.insights)
            except Exception as exc:
                _log(f'{s}: ERROR {type(exc).__name__}: {exc}')
                failures += 1
                if not args.probe and not args.dry_run:
                    # never continue past a live-run failure: the
                    # rollback already ran, and blundering on risks
                    # compounding a half-migrated fleet
                    return 1

    if args.probe or args.dry_run:
        return 1 if failures else 0

    # Final gate: exit 0 only when every requested store reports the
    # new schema with its expected rows -- the scheduler must not
    # restart before this passes
    bad = []
    for s in stores:
        db_path = Path(store_dir(data_dir, s)) / 'memman.db'
        if not db_path.exists():
            bad.append(f'{s}: missing')
            continue
        cols = _insight_columns(db_path)
        count = _insight_count(db_path)
        if not {'session_id', 'queue_uuid'} <= cols:
            bad.append(f'{s}: old schema')
        elif s in expected and count != expected[s]:
            bad.append(
                f'{s}: {count} rows, expected {expected[s]}')
        elif s not in expected and count == 0:
            bad.append(f'{s}: zero rows and not rebuilt this run')
    if bad:
        _log(f'final gate FAILED: {bad}')
        return 1
    _log(f'all {len(stores)} stores at 0.18.0 schema')
    return 0


if __name__ == '__main__':
    sys.exit(main())
