#!/usr/bin/env python3
r"""Move or copy chosen rows from one memman store to another, by hand.

memman has no verb for a store split. The script copies each row with
its id, dates, author, summary and embedding, which a fresh `memman
remember` would lose.

Usage
-----
    python scripts/move_rows.py export SOURCE OUTDIR
    python scripts/move_rows.py apply SOURCE TARGET PLAN [--dry-run]

`export` only reads SOURCE. It writes `<SOURCE>_payload.pkl`, the full
payload as a backup, and `<SOURCE>_current.json`, the current rows.

`apply` reads PLAN, a TSV of `<id>\t<move|copy>` lines naming current
rows of SOURCE. TARGET must already exist (`memman store create`).

Notes
-----
- A planned row reaches TARGET with every row it replaced, so
  `memman insights show <id> --history` works in TARGET.
- A `move` row is then forgotten in SOURCE with the rows it replaced,
  so `memman forget` would accept each step.
- A re-run finishes a run that stopped part way: the copy skips ids
  TARGET holds, and the forget skips forgotten rows.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

from memman import config, drain_lock
from memman.fork import _acquire_drain_lock, _migrator_for
from memman.migrate import MigrateInsight, MigrationPayload
from memman.queue import queue_db
from memman.store import factory
from memman.store.db import default_data_dir
from memman.store.model import insight_to_delta_dict


def export(source: str, outdir: Path, data_dir: str) -> None:
    """Write the SOURCE payload backup and its current rows as JSON.
    """
    payload = _migrator_for(source, data_dir).gather(source)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f'{source}_payload.pkl').open('wb') as f:
        pickle.dump(payload, f)
    current_list = sorted((
        {
            'id': ins.id,
            'created_at': ins.created_at.isoformat(),
            'author': ins.author,
            'content': ins.content,
            }
        for ins in payload.insights
        if ins.deleted_at is None and ins.replaced_by is None
        ), key=lambda row: row['created_at'])
    with (outdir / f'{source}_current.json').open('w') as f:
        json.dump(current_list, f, indent=1)
    print(json.dumps({
        'source': source,
        'rows': len(payload.insights),
        'current': len(current_list),
        }))


def apply(
        source: str, target: str, plan_path: Path, data_dir: str,
        dry_run: bool) -> None:
    r"""Copy the planned rows to TARGET, then forget the `move` rows.

    Parameters
    ----------
    source : str
        Store the rows leave.
    target : str
        Existing store with the same embed model as SOURCE.
    plan_path : Path
        TSV of `<id>\t<move|copy>`. Every id must be a current row of
        SOURCE.
    data_dir : str
        Base memman data directory.
    dry_run : bool
        Print the counts and write nothing.

    Raises
    ------
    SystemExit
        A bad plan line, an id not current in SOURCE, an embed model
        mismatch, or a queued replace in SOURCE of a planned row.
        Nothing is written.
    """
    action_by_id = {}
    for line in plan_path.read_text().splitlines():
        row_id, action = line.split('\t')
        if action not in {'move', 'copy'}:
            raise SystemExit(f'bad action {action!r} for {row_id}')
        action_by_id[row_id] = action
    if source == target:
        raise SystemExit('source and target are the same store')

    lock_fd = _acquire_drain_lock(data_dir)
    try:
        payload = _migrator_for(source, data_dir).gather(source)
        insight_by_id = {ins.id: ins for ins in payload.insights}
        not_current = [
            row_id for row_id in action_by_id
            if row_id not in insight_by_id
            or insight_by_id[row_id].deleted_at is not None
            or insight_by_id[row_id].replaced_by is not None
            ]
        if not_current:
            raise SystemExit(f'not current in {source}: {not_current}')
        with factory.open_backend(target, data_dir, read_only=True) as tb:
            target_fp = tb.meta.get('embed_fingerprint')
        source_fp = payload.meta.get('embed_fingerprint')
        if target_fp != source_fp:
            raise SystemExit(
                f'embed model differs: {source} {source_fp},'
                f' {target} {target_fp}')

        replaced_by_successor: dict[str, list[MigrateInsight]] = {}
        for ins in payload.insights:
            if ins.replaced_by:
                replaced_by_successor.setdefault(
                    ins.replaced_by, []).append(ins)
        chain_by_id = {}
        for row_id in action_by_id:
            chain_list = []
            pending = [row_id]
            while pending:
                for ins in replaced_by_successor.get(pending.pop(), []):
                    chain_list.append(ins)
                    pending.append(ins.id)
            chain_by_id[row_id] = chain_list

        planned_ids = set(action_by_id) | {
            ins.id for chain_list in chain_by_id.values()
            for ins in chain_list
            }
        queued_sql = """
select replaced_id
from queue
where store = ? and status in ('pending', 'failed')
  and replaced_id is not null
"""
        with queue_db(data_dir) as conn:
            queued_ids = {
                row[0] for row in conn.execute(queued_sql, (source,))}
        if queued_ids & planned_ids:
            raise SystemExit(
                f'queued replace in {source} names planned rows:'
                f' {sorted(queued_ids & planned_ids)}')

        copied_list = [insight_by_id[row_id] for row_id in action_by_id] + [
            ins for chain_list in chain_by_id.values() for ins in chain_list
            ]
        forget_ids = [
            ins.id for row_id, action in action_by_id.items()
            if action == 'move'
            for ins in [*chain_by_id[row_id], insight_by_id[row_id]]
            if ins.deleted_at is None
            ]
        report = {
            'source': source,
            'target': target,
            'move': sum(a == 'move' for a in action_by_id.values()),
            'copy': sum(a == 'copy' for a in action_by_id.values()),
            'chain_rows': len(copied_list) - len(action_by_id),
            'forget_in_source': len(forget_ids),
            'dry_run': dry_run,
            }
        if dry_run:
            print(json.dumps(report))
            return

        _migrator_for(target, data_dir).apply(target, MigrationPayload(
            fingerprint=payload.fingerprint,
            embedding_dim=payload.embedding_dim,
            insights=copied_list,
            oplog=[],
            meta={}))
        # The copy skips an id TARGET already holds, whatever its state.
        with factory.open_backend(target, data_dir, read_only=True) as tb:
            not_copied = [
                row_id for row_id, action in action_by_id.items()
                if action == 'move'
                and ((ins := tb.nodes.get_include_deleted(row_id)) is None
                     or ins.deleted_at is not None
                     or ins.replaced_by is not None)
                ]
        if not_copied:
            raise SystemExit(
                f'not current in {target}, so not forgotten in {source}:'
                f' {not_copied}')
        forgotten_cnt = 0
        with factory.open_backend(source, data_dir) as backend, \
                backend.transaction():
            for row_id in forget_ids:
                before = backend.nodes.get_include_deleted(row_id)
                if backend.nodes.soft_delete(row_id):
                    backend.oplog.log(
                        operation='forget',
                        insight_id=row_id,
                        detail=f'moved to {target}',
                        before=insight_to_delta_dict(before))
                    forgotten_cnt += 1
        report['forgotten'] = forgotten_cnt
        print(json.dumps(report))
    finally:
        drain_lock.release(lock_fd)


def main() -> int:
    """Parse the command line and run one subcommand.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export_parser = sub.add_parser('export')
    export_parser.add_argument('source')
    export_parser.add_argument('outdir', type=Path)
    apply_parser = sub.add_parser('apply')
    apply_parser.add_argument('source')
    apply_parser.add_argument('target')
    apply_parser.add_argument('plan', type=Path)
    apply_parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data_dir = os.environ.get(config.DATA_DIR, default_data_dir())
    if args.command == 'export':
        export(args.source, args.outdir, data_dir)
    else:
        apply(args.source, args.target, args.plan, data_dir, args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
