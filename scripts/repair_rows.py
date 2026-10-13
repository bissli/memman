r"""Hard-delete or rewrite rows of one memman store, keeping every chain whole.

Usage
-----
    python scripts/repair_rows.py audit STORE
    python scripts/repair_rows.py delete STORE ID [ID ...] [OPTIONS]
    python scripts/repair_rows.py rewrite STORE ID TEXT [TEXT ...] [OPTIONS]
    python scripts/repair_rows.py apply STORE PLAN [OPTIONS]

OPTIONS are `--dry-run` and `--backup-dir DIR`. PLAN is a JSON-lines
file, or `-` for stdin, of the ops `delete` and `rewrite` take:

    {"action": "delete", "id": "<id>"}
    {"action": "rewrite", "id": "<id>", "content": ["<text>", ...]}

`audit` prints one JSON line per stored row, any state. Every other
command prints one JSON report. A refusal exits 2 with `{"refused": ...}`
on stderr and leaves the store unchanged. Any other failure exits 1
with `{"error": ...}` on stderr.

Notes
-----
- `delete` of a row with a successor splices it out of the chain: its
  predecessors then point at that successor. A row with none (current,
  forgotten, or pointing at a missing row) goes with every row it
  replaced. The oplog loses every entry holding any text a deleted row
  held that no surviving row holds. A delete of a row the same run
  already deleted does nothing.
- `rewrite` leaves the row current with the first text. It keeps the
  id, created_at, author, meta and the rows it replaced, and clears the
  summary, vector and enrich stamps. A replaced row leaves its
  successor's history. Each further text becomes a new current row
  with the same created_at, author and meta.
- A rewritten or new row has no vector, so vector recall misses it
  until enriched. The scheduler's drain enriches pending rows only in a
  store it wrote a queued row to, a few per drain.
- Ops run in order in one transaction, each seeing what the ones
  before it left. A refused op, a queued `replace` of a deleted or
  rewritten row, or a run that would leave a new broken chain rolls
  back the whole batch.
- The run holds the drain lock and, on Postgres, the store's enrich
  lock, and refuses during an embedding swap. On SQLite nothing locks
  out a manual `memman enrich`, which can stamp a rewritten row with
  the old text's summary and vector. Never run the two at once.
- A run that writes first builds a `memman backup` bundle in a new
  directory under `--backup-dir`, else `MEMMAN_BACKUP_TARGET`. The
  bundle holds every store as it stood before the run. `memman backup
  restore` of it rewinds the whole data dir, not this store alone.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memman import config, drain_lock
from memman.backup import build_bundle
from memman.cli import _content_refusal_message
from memman.embed.swap import META_STATE as EMBED_SWAP_STATE
from memman.migrate import MigrateInsight
from memman.queue import queue_db
from memman.search.keyword import insight_tokens
from memman.store import factory
from memman.store.backend import Backend
from memman.store.db import default_data_dir
from memman.store.model import Insight, format_timestamp, insight_to_delta_dict
from memman.store.overlay import BRANCH_PARENT, BRANCH_TOKEN
from memman.store.sqlite import SqliteBackend

_OP_KEYS = {'delete': {'action', 'id'}, 'rewrite': {'action', 'id', 'content'}}


class Refused(Exception):
    """An op or a run this script will not carry out.
    """


class _DryRun(Exception):
    """Carries a dry run's report out of the transaction it rolls back.
    """

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__('dry run')
        self.report = report


def _connection(backend: Backend, sql: str) -> tuple[Any, str]:
    """The backend's open connection and `sql` in its dialect.

    Parameters
    ----------
    sql : str
        SQLite-dialect SQL with `?` placeholders. `{t}` stands before
        each table name, for the Postgres store schema.
    """
    if isinstance(backend, SqliteBackend):
        return backend._db._conn, sql.format(t='')
    schema_sql = sql.format(t=f'{backend.nodes._schema}.')
    return backend.nodes._conn, schema_sql.replace('?', '%s')


def fetch(backend: Backend, sql: str, params: tuple = ()) -> list[tuple]:
    """Every row `sql` selects, on the backend's own connection.

    Parameters
    ----------
    sql : str
        SQLite dialect, `{t}` before each table name (see `_connection`).
    """
    conn, text = _connection(backend, sql)
    if isinstance(backend, SqliteBackend):
        return conn.execute(text, params).fetchall()
    with conn.cursor() as cur:
        cur.execute(text, params)
        return cur.fetchall()


def execute(backend: Backend, sql: str, params: tuple = ()) -> int:
    """Rows `sql` changed, on the backend's own connection.

    Parameters
    ----------
    sql : str
        SQLite dialect, `{t}` before each table name (see `_connection`).
    """
    conn, text = _connection(backend, sql)
    if isinstance(backend, SqliteBackend):
        return conn.execute(text, params).rowcount
    with conn.cursor() as cur:
        cur.execute(text, params)
        return cur.rowcount


def _refuse_branch(backend: Backend, store: str) -> None:
    """Refuse a branch store, or a store that has branches.

    Raises
    ------
    Refused
        The store's meta holds a branch parent or a branch token.
    """
    if backend.meta.get(BRANCH_PARENT) is not None or any(
            key.startswith(f'{BRANCH_TOKEN}:')
            for key in backend.meta.keys()):  # noqa: SIM118
        raise Refused(f'{store} is a branch or has branches')


def check_ops(op_list: list[dict[str, Any]]) -> None:
    """Refuse a malformed op, or rewrite text `memman remember` refuses.

    Raises
    ------
    Refused
        The first bad op, by its position in `op_list`.
    """
    if not op_list:
        raise Refused('no ops')
    for idx, op in enumerate(op_list):
        if not isinstance(op, dict) or op.get('action') not in _OP_KEYS:
            raise Refused(f'op {idx}: action must be delete or rewrite')
        if set(op) != _OP_KEYS[op['action']]:
            raise Refused(
                f'op {idx}: {op["action"]} takes exactly the keys'
                f' {sorted(_OP_KEYS[op["action"]])}')
        if not isinstance(op['id'], str) or not op['id']:
            raise Refused(f'op {idx}: id must be a non-empty string')
        if op['action'] == 'delete':
            continue
        content_list = op['content']
        if (not isinstance(content_list, list) or not content_list
            or not all(isinstance(text, str) for text in content_list)):
            raise Refused(f'op {idx}: content must be a non-empty list of text')
        for text in content_list:
            refusal = _content_refusal_message(text)
            if refusal:
                raise Refused(f'op {idx} ({op["id"]}): {refusal}')


def _delete(backend: Backend, row_id: str, report: dict[str, Any]) -> None:
    """Delete one row, splicing it out or taking its history along.
    """
    ((replaced_by,),) = fetch(
        backend, 'select replaced_by from {t}insights where id = ?', (row_id,))
    successor_found = replaced_by is not None and fetch(
        backend, 'select 1 from {t}insights where id = ?', (replaced_by,))
    predecessor_ids = [
        row[0] for row in fetch(
            backend, 'select id from {t}insights where replaced_by = ?',
            (row_id,))
        ]
    if successor_found:
        deleted_ids = [row_id]
        for pred_id in predecessor_ids:
            execute(
                backend, 'update {t}insights set replaced_by = ? where id = ?',
                (replaced_by, pred_id))
        report['repointed'].extend(predecessor_ids)
    else:
        deleted_ids = [row_id]
        pending_ids = list(predecessor_ids)
        while pending_ids:
            pred_id = pending_ids.pop()
            deleted_ids.append(pred_id)
            pending_ids.extend(
                row[0] for row in fetch(
                    backend,
                    'select id from {t}insights where replaced_by = ?',
                    (pred_id,)))
    if isinstance(backend, SqliteBackend):
        text_sql = """
delete from {t}oplog
where json_extract(before, '$.content') = ?
   or json_extract(after, '$.content') = ?
"""
    else:
        text_sql = """
delete from {t}oplog
where before->>'content' = ? or after->>'content' = ?
"""
    held_texts = set()
    for gone_id in deleted_ids:
        ((content,),) = fetch(
            backend, 'select content from {t}insights where id = ?', (gone_id,))
        held_texts.add(content)
        for delta_pair in fetch(
                backend, 'select before, after from {t}oplog where insight_id = ?',
                (gone_id,)):
            for delta in delta_pair:
                if isinstance(delta, str):
                    delta = json.loads(delta)
                if delta and delta.get('content'):
                    held_texts.add(delta['content'])
    for gone_id in deleted_ids:
        execute(backend, 'delete from {t}insights where id = ?', (gone_id,))
        execute(backend, 'delete from {t}oplog where insight_id = ?', (gone_id,))
    for text in held_texts:
        # A `replace` entry of a deleted row holds its successor's text,
        # which a surviving row still holds.
        if not fetch(
                backend, 'select 1 from {t}insights where content = ?', (text,)):
            execute(backend, text_sql, (text, text))
    backend.oplog.log(
        operation='purge',
        insight_id=row_id,
        detail=json.dumps({
            'deleted': deleted_ids,
            'repointed': predecessor_ids if successor_found else [],
            }))
    report['deleted'].extend(deleted_ids)


def _rewrite(
        backend: Backend, row_id: str, content_list: list[str],
        report: dict[str, Any]) -> None:
    """Make one row current with the first text, inserting the rest.
    """
    row = backend.nodes.get_include_deleted(row_id)
    now = format_timestamp(datetime.now(timezone.utc))
    text = content_list[0]
    set_sql = """
update {t}insights
set content = ?, summary = null, summary_model = null,
    embedding = null, embedding_model = null,
    enrich_attempted_at = null, enriched_at = null,
    deleted_at = null, replaced_by = null, updated_at = ?
"""
    if isinstance(backend, SqliteBackend):
        execute(
            backend, set_sql + ', embedding_pending = null where id = ?',
            (text, now, row_id))
    else:
        execute(
            backend, set_sql + ', kw_tokens = ? where id = ?',
            (text, now,
             sorted(insight_tokens(Insight(content=text, meta=row.meta))),
             row_id))

    inserted_ids = []
    for extra_text in content_list[1:]:
        new_id = str(uuid.uuid4())
        if not backend.nodes.insert_raw(MigrateInsight(
                id=new_id,
                content=extra_text,
                summary=None,
                embedding=None,
                enrich_attempted_at=None,
                enriched_at=None,
                created_at=row.created_at,
                updated_at=datetime.now(timezone.utc),
                deleted_at=None,
                embedding_model=None,
                queue_uuid=None,
                replaced_by=None,
                author=row.author,
                summary_model=None,
                meta=row.meta)):
            raise Refused(f'new id {new_id} already stored')
        backend.oplog.log(
            operation='rewrite',
            insight_id=new_id,
            detail=f'split from {row_id}',
            after=insight_to_delta_dict(Insight(content=extra_text)))
        inserted_ids.append(new_id)

    backend.oplog.log(
        operation='rewrite',
        insight_id=row_id,
        detail=json.dumps({
            'detached_from': row.replaced_by,
            'revived': row.deleted_at is not None,
            'split_into': inserted_ids,
            }),
        before=insight_to_delta_dict(row),
        after=insight_to_delta_dict(Insight(content=text)))
    report['rewritten'].append(row_id)
    report['inserted'].extend(inserted_ids)
    if row.replaced_by is not None:
        report['detached'].append(row_id)
    if row.deleted_at is not None:
        report['revived'].append(row_id)


def run_ops(backend: Backend, op_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Run checked ops in order. The caller holds the transaction.

    Parameters
    ----------
    backend : Backend
        A SQLite or Postgres store, never a branch.
    op_list : list[dict[str, Any]]
        Ops `check_ops` passed.

    Returns
    -------
    dict[str, Any]
        Ids by outcome: `deleted`, `repointed` (now pointing past a
        deleted row), `rewritten`, `detached`, `revived` and
        `inserted`.

    Raises
    ------
    Refused
        An id no row holds when its op runs, an id on a cycle, or a
        run that leaves a row in a replacement integrity population
        that did not hold it before. The caller rolls back.
    """
    integrity_before = backend.nodes.replacement_integrity()
    on_cycle_ids = set(integrity_before['self_pointer']) | set(
        integrity_before['unterminated'])
    report: dict[str, Any] = {
        'deleted': [], 'repointed': [], 'rewritten': [],
        'detached': [], 'revived': [], 'inserted': [],
        }
    for op in op_list:
        row_id = op['id']
        if op['action'] == 'delete' and row_id in report['deleted']:
            continue
        if row_id in on_cycle_ids:
            raise Refused(f'{row_id} is on a replacement cycle')
        if not fetch(backend, 'select 1 from {t}insights where id = ?', (row_id,)):
            raise Refused(f'no row {row_id}')
        if op['action'] == 'delete':
            _delete(backend, row_id, report)
        else:
            _rewrite(backend, row_id, op['content'], report)

    integrity_after = backend.nodes.replacement_integrity()
    for population, ids in integrity_after.items():
        new_ids = set(ids) - set(integrity_before[population])
        if new_ids:
            raise Refused(f'run would add {population} rows: {sorted(new_ids)}')
    return report


def apply(
        store: str, op_list: list[dict[str, Any]], data_dir: str, *,
        dry_run: bool, backup_dir: str | None) -> dict[str, Any]:
    """Back up, then run the ops on `store` in one transaction.

    Parameters
    ----------
    store : str
        A SQLite or Postgres store, neither a branch nor a branch parent.
    op_list : list[dict[str, Any]]
        Ops in the PLAN shape.
    data_dir : str
        Base memman data directory.
    dry_run : bool
        Run every op, report, and roll back. No backup is built.
    backup_dir : str or None
        Directory for the bundle. Required unless `dry_run`.

    Returns
    -------
    dict[str, Any]
        `run_ops`' report, plus `store`, `dry_run` and `backup`, the
        bundle path or None.

    Raises
    ------
    Refused
        Before any write: a bad op, no backup directory, a running
        drain, a branch store, an embedding swap, a bundle missing the
        store, or an enrich run holding the store. Inside
        the rolled-back transaction: any `run_ops` refusal, or a queued
        `replace` naming a deleted or rewritten row.
    """
    check_ops(op_list)
    if not dry_run and not backup_dir:
        raise Refused(
            'no backup directory: pass --backup-dir or set'
            f' {config.BACKUP_TARGET}')
    try:
        lock_fd = drain_lock.acquire(data_dir)
    except drain_lock.DrainLockBusy as exc:
        raise Refused('a drain is running; re-run in a moment') from exc
    try:
        with factory.open_backend(store, data_dir) as backend:
            _refuse_branch(backend, store)
            if backend.meta.get(EMBED_SWAP_STATE):
                raise Refused(f'{store} has an embedding swap in progress')
            backup_path = None
            if not dry_run:
                run_dir = Path(backup_dir).expanduser() / (
                    f'repair-{store}-'
                    f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
                run_dir.mkdir(parents=True)
                bundle = build_bundle(data_dir, str(run_dir))
                if not any(
                        entry['name'] == store and entry['status'] == 'ok'
                        for entry in bundle['stores']):
                    raise Refused(
                        f'backup {bundle["bundle"]} holds no snapshot of {store}')
                backup_path = bundle['bundle']
            queued_sql = """
select replaced_id
from queue
where store = ? and status in ('pending', 'failed')
  and replaced_id is not null
"""
            with backend.reembed_lock('rebuild') as held:
                if not held:
                    raise Refused(f'an enrich run holds {store}')
                try:
                    with backend.transaction():
                        report = run_ops(backend, op_list)
                        with queue_db(data_dir) as conn:
                            queued_ids = {
                                row[0]
                                for row in conn.execute(queued_sql, (store,))}
                        blocked_ids = queued_ids & set(
                            report['deleted'] + report['rewritten'])
                        if blocked_ids:
                            raise Refused(
                                f'queued replace names {sorted(blocked_ids)}')
                        if dry_run:
                            raise _DryRun(report)
                except _DryRun as exc:
                    report = exc.report
    finally:
        drain_lock.release(lock_fd)
    return {**report, 'store': store, 'dry_run': dry_run, 'backup': backup_path}


def audit(store: str, data_dir: str) -> None:
    """Print every stored row of `store` as one JSON line, oldest first.

    Each line holds `id`, `state` (current, replaced or forgotten; a
    forgotten replaced row is forgotten), `replaced_by`, `integrity`
    (the replacement integrity populations holding the row), `refusal`
    (the `memman remember` refusal of its text, or None), `created_at`,
    `author`, `meta` and `content`.

    Raises
    ------
    Refused
        `store` is a branch or has branches.
    """
    with factory.open_backend(store, data_dir, read_only=True) as backend:
        _refuse_branch(backend, store)
        integrity = backend.nodes.replacement_integrity()
        row_list = fetch(backend, """
select id
from {t}insights
order by created_at, id
""")
        for (row_id,) in row_list:
            row = backend.nodes.get_include_deleted(row_id)
            if row.deleted_at is not None:
                state = 'forgotten'
            elif row.replaced_by is not None:
                state = 'replaced'
            else:
                state = 'current'
            print(json.dumps({
                'id': row.id,
                'state': state,
                'replaced_by': row.replaced_by,
                'integrity': sorted(
                    population for population, ids in integrity.items()
                    if row.id in ids),
                'refusal': _content_refusal_message(row.content),
                'created_at': row.created_at.isoformat(),
                'author': row.author,
                'meta': row.meta,
                'content': row.content,
                }))


def main() -> int:
    """Run one command against MEMMAN_DATA_DIR, else ~/.memman.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit').add_argument('store')
    delete_parser = sub.add_parser('delete')
    delete_parser.add_argument('store')
    delete_parser.add_argument('ids', nargs='+')
    rewrite_parser = sub.add_parser('rewrite')
    rewrite_parser.add_argument('store')
    rewrite_parser.add_argument('id')
    rewrite_parser.add_argument('content', nargs='+')
    apply_parser = sub.add_parser('apply')
    apply_parser.add_argument('store')
    apply_parser.add_argument('plan')
    for write_parser in (delete_parser, rewrite_parser, apply_parser):
        write_parser.add_argument('--dry-run', action='store_true')
        write_parser.add_argument(
            '--backup-dir', default=config.get(config.BACKUP_TARGET))
    args = parser.parse_args()
    data_dir = os.environ.get(config.DATA_DIR, default_data_dir())

    try:
        if args.command == 'audit':
            audit(args.store, data_dir)
            return 0
        if args.command == 'delete':
            op_list = [{'action': 'delete', 'id': row_id} for row_id in args.ids]
        elif args.command == 'rewrite':
            op_list = [
                {'action': 'rewrite', 'id': args.id, 'content': args.content}]
        else:
            plan_text = (
                sys.stdin.read() if args.plan == '-'
                else Path(args.plan).read_text())
            try:
                op_list = [
                    json.loads(line)
                    for line in plan_text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise Refused(f'bad plan line: {exc}') from exc
        report = apply(
            args.store, op_list, data_dir,
            dry_run=args.dry_run, backup_dir=args.backup_dir)
    except Refused as exc:
        sys.stderr.write(json.dumps({'refused': str(exc)}) + '\n')
        return 2
    except Exception as exc:
        sys.stderr.write(
            json.dumps({'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        return 1
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
