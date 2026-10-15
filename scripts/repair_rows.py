r"""Hard-delete or rewrite memman rows, keeping every chain whole.

Usage
-----
Run from the memman repo, so the dev install of memman loads:

    cd <memman repo> && poetry run python scripts/repair_rows.py COMMAND ...

    audit STORE
    delete STORE ID [ID ...] [OPTIONS]
    rewrite STORE ID TEXT [TEXT ...] [OPTIONS]
    apply STORE PLAN [OPTIONS]

OPTIONS:

    --dry-run           run every op, report, and roll back
    --backup-dir DIR    bundle directory, else MEMMAN_BACKUP_TARGET from
                        the data dir's `env` file, never the shell

PLAN is a JSON-lines file, or `-` for stdin, of the ops the other
commands take:

    {"action": "delete", "id": "<id>"}
    {"action": "rewrite", "id": "<id>", "content": ["<text>", ...]}

The data dir is `MEMMAN_DATA_DIR`, else `~/.memman`. The endpoint, key
and model come from its `env` file, as for every memman command, so a
caller exports nothing.

Output
------
`audit` prints one JSON line per stored row, in any state, oldest first:

    id, state, replaced_by, integrity, refusal, created_at, author, meta,
    content

- `state`: current, replaced or forgotten. A forgotten replaced row is
  forgotten.
- `integrity`: the replacement integrity populations holding the row,
  from dangling, self_pointer and unterminated.
- `refusal`: the `memman remember` refusal of the text, or null.

Every other command prints one JSON report on stdout. It holds `store`,
`dry_run`, `backup` (the bundle path, or null on a dry run) and these
id lists:

- `deleted`: every row a delete removed, history rows included.
- `spliced`: the named rows cut out of a chain, a subset of `deleted`.
- `repointed`: rows now pointing past a spliced row.
- `rewritten`: rows a `rewrite` set to its first text.
- `detached`: rewritten rows that were replaced.
- `revived`: rewritten rows that were forgotten.
- `inserted`: new rows holding a `rewrite`'s further texts.
- `enriched`: rewritten and new rows stamped enriched. Empty on a dry
  run.
- `enrich_error`: present only when enrichment raised after the commit.

Exit Status
-----------
- 0: the run finished.
- 1: `enrich_error` is set, with the report still on stdout, or any
  other failure, with `{"error": ...}` on stderr.
- 2: a refusal, with `{"refused": ...}` on stderr. The store is
  unchanged.

Delete
------
- A row with a successor is spliced out. Its predecessors then point at
  that successor.
- A row with none (current, forgotten, or pointing at a missing row)
  goes with every row it replaced.
- A delete of a row the same run already deleted does nothing.
- To collapse a chain to its head and its first version, delete each
  row between them.
- Every oplog entry holding a text that a deleted row held and no
  surviving row holds goes too, so the oplog keeps no copy of it.
  `memman insights show --history` follows `replaced_by` alone, so the
  surviving chain still reads whole.
- In the dry-run report, `deleted` beyond the named ids is history a
  head delete took, and `spliced` names the deletes that kept their
  chain. Read it before the real run.

Rewrite
-------
- The row becomes current with the first text. It keeps its id,
  created_at, author, meta and the rows it replaced.
- The row loses its summary, vector and enrich stamps.
- A forgotten row is revived.
- A replaced row leaves its successor's history, which then starts at
  that successor.
- Each further text becomes a new current row with the same created_at,
  author and meta.
- Every text must pass the `memman remember` refusals.
- A real run enriches every rewritten and new row before it exits,
  through the drain's own `enrich_pending`. That takes the store's
  pending rows oldest first, so older pending rows are enriched with
  them.
- A row missing from `enriched` is stranded, or still pending when
  `enrich_error` is set. `memman enrich --stranded-only` retries a
  stranded row, and the drain takes a pending one.
- A real rewrite run refuses first when the store's embedder is
  unavailable. A dry run skips this check.

Safety
------
- Ops run in order in one transaction, each seeing what the ones before
  it left.
- The whole batch rolls back on a refused op, on a pending or failed
  queued `replace` naming a deleted or rewritten row, or on a run that
  would leave a new broken chain. Let the drain finish the queued writes
  before such a run.
- The run holds the drain lock and, on Postgres, the store's enrich
  lock.
- The run refuses during an embedding swap, and on a branch store or a
  branch parent.
- On SQLite nothing locks out a manual `memman enrich`, which can stamp
  a rewritten row with the old text's summary and vector. Never run the
  two at once.
- Each real run first builds a `memman backup` bundle of every store, in
  a new directory under `--backup-dir`, and reports its path.
- `memman backup restore` of that bundle rewinds the whole data dir:
  every store, the queue and the non-secret config.
- A batch in one call costs one bundle, so send batches.
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
from memman.embed import fingerprint
from memman.embed.swap import META_STATE as EMBED_SWAP_STATE
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.llm.client import get_llm_client
from memman.migrate import MigrateInsight
from memman.pipeline.enrich import enrich_pending
from memman.queue import queue_db
from memman.search.keyword import insight_tokens
from memman.store import factory
from memman.store.backend import Backend
from memman.store.db import default_data_dir
from memman.store.model import Insight, format_timestamp, insight_to_delta_dict
from memman.store.overlay import BRANCH_PARENT, BRANCH_TOKEN
from memman.store.sqlite import SqliteBackend

_OP_KEYS = {
    'delete': {'action', 'id'},
    'rewrite': {'action', 'id', 'content'},
    }


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
    backend : Backend
        A SQLite or Postgres store.
    sql : str
        SQLite-dialect SQL with `?` placeholders. `{t}` stands before
        each table name, for the Postgres store schema.

    Returns
    -------
    tuple[Any, str]
        The connection, and `sql` with `{t}` filled. On Postgres every
        `?` becomes `%s`, one inside a string literal included.
    """
    if isinstance(backend, SqliteBackend):
        return backend._db._conn, sql.format(t='')
    schema_sql = sql.format(t=f'{backend.nodes._schema}.')
    return backend.nodes._conn, schema_sql.replace('?', '%s')


def fetch(backend: Backend, sql: str, params: tuple = ()) -> list[tuple]:
    """Every row `sql` selects, on the backend's own connection.

    Parameters
    ----------
    backend : Backend
        Store whose connection, and so whose open transaction, runs `sql`.
    sql : str
        SQLite dialect, `{t}` before each table name (see `_connection`).
    params : tuple, default ()
        Values for the `?` placeholders.

    Returns
    -------
    list[tuple]
        One tuple per selected row.
    """
    conn, text = _connection(backend, sql)
    if isinstance(backend, SqliteBackend):
        return conn.execute(text, params).fetchall()
    with conn.cursor() as cur:
        cur.execute(text, params)
        return cur.fetchall()


def execute(backend: Backend, sql: str, params: tuple = ()) -> int:
    """Run `sql` on the backend's own connection.

    Parameters
    ----------
    backend : Backend
        Store whose connection, and so whose open transaction, runs `sql`.
    sql : str
        SQLite dialect, `{t}` before each table name (see `_connection`).
    params : tuple, default ()
        Values for the `?` placeholders.

    Returns
    -------
    int
        Rows `sql` changed.
    """
    conn, text = _connection(backend, sql)
    if isinstance(backend, SqliteBackend):
        return conn.execute(text, params).rowcount
    with conn.cursor() as cur:
        cur.execute(text, params)
        return cur.rowcount


def _refuse_branch(backend: Backend, store: str) -> None:
    """Refuse a branch store, or a store that has branches.

    Parameters
    ----------
    backend : Backend
        The open store.
    store : str
        Store name, for the message.

    Raises
    ------
    Refused
        The store's meta holds a branch parent, or a branch token. The
        message names each branch, since `memman store drop` also clears
        the token of one removed by hand.
    """
    if backend.meta.get(BRANCH_PARENT) is not None:
        raise Refused(f'{store} is a branch')
    prefix = f'{BRANCH_TOKEN}:'
    branches = sorted(
        key.removeprefix(prefix)
        for key in backend.meta.keys()  # noqa: SIM118
        if key.startswith(prefix))
    if branches:
        raise Refused(
            f'{store} has branches: {", ".join(branches)}; merge or drop'
            ' each first')


def check_ops(op_list: list[dict[str, Any]]) -> None:
    """Refuse a malformed op, or rewrite text `memman remember` refuses.

    Parameters
    ----------
    op_list : list[dict[str, Any]]
        Ops in the PLAN shape.

    Raises
    ------
    Refused
        `op_list` is empty, or holds a bad op. The message names the
        first bad op by its position in `op_list`.
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

    Parameters
    ----------
    backend : Backend
        Store inside the caller's open transaction.
    row_id : str
        A stored row whose chain ends. `run_ops` checks both, and a
        cycle would loop the history walk forever.
    report : dict[str, Any]
        The `run_ops` report, extended in place: `deleted`, and on a
        splice `spliced` and `repointed`.
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
        report['spliced'].append(row_id)
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

    Parameters
    ----------
    backend : Backend
        Store inside the caller's open transaction.
    row_id : str
        A stored row in any state.
    content_list : list[str]
        Texts `check_ops` passed. The first replaces the row's text, and
        each further one becomes a new row.
    report : dict[str, Any]
        The `run_ops` report, extended in place: `rewritten`, `inserted`,
        and `detached` or `revived` for a replaced or forgotten row.

    Raises
    ------
    Refused
        A new row's id is already stored.
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
        Id lists by outcome, as the module docstring's Output section
        names them: `deleted`, `spliced`, `repointed`, `rewritten`,
        `detached`, `revived` and `inserted`.

    Raises
    ------
    Refused
        An id no row holds when its op runs, an id whose chain never
        ends (a self-pointer, or on or leading into a cycle), or a
        run that leaves a row in a replacement integrity population
        that did not hold it before. The caller rolls back.
    """
    integrity_before = backend.nodes.replacement_integrity()
    on_cycle_ids = set(integrity_before['self_pointer']) | set(
        integrity_before['unterminated'])
    report: dict[str, Any] = {
        'deleted': [], 'spliced': [], 'repointed': [], 'rewritten': [],
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
        The `run_ops` report, plus `store`, `dry_run`, `backup` (the
        bundle path, or None on a dry run) and `enriched` (the rewritten
        and new rows stamped enriched, empty on a dry run).
        `enrich_error` is present only when enrichment raised after the
        commit.

    Raises
    ------
    Refused
        Before any store write: a bad op, no backup directory, a running
        drain, a branch store, an embedding swap, no LLM client or no
        available embedder for a real run that rewrites, a bundle missing
        the store, or an enrich run holding the store.
        Inside the rolled-back transaction: any `run_ops` refusal, or a
        queued `replace` naming a deleted or rewritten row.
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
            enrich_clients = None
            if not dry_run and any(op['action'] == 'rewrite' for op in op_list):
                try:
                    enrich_clients = (
                        get_llm_client(), fingerprint.bound_embedder(backend))
                except (ConfigError, EmbedFingerprintError) as exc:
                    raise Refused(f'cannot enrich rewritten rows: {exc}') from exc
                if not enrich_clients[1].available():
                    raise Refused(
                        f'the embedder of {store} is unavailable;'
                        ' run memman doctor')
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
                enriched_ids = []
                if enrich_clients is not None:
                    llm_client, embed_client = enrich_clients
                    target_ids = list(dict.fromkeys(
                        report['rewritten'] + report['inserted']))
                    pending_sql = """
select 1
from {t}insights
where id = ? and enrich_attempted_at is null
"""
                    try:
                        while any(
                                fetch(backend, pending_sql, (row_id,))
                                for row_id in target_ids):
                            if not enrich_pending(
                                    backend, llm_client=llm_client,
                                    embed_client=embed_client):
                                break
                    except Exception as exc:
                        report['enrich_error'] = f'{type(exc).__name__}: {exc}'
                    enriched_ids = [
                        row_id for row_id in target_ids
                        if fetch(
                            backend,
                            'select 1 from {t}insights'
                            ' where id = ? and enriched_at is not null',
                            (row_id,))
                        ]
                report['enriched'] = enriched_ids
    finally:
        drain_lock.release(lock_fd)
    return {**report, 'store': store, 'dry_run': dry_run, 'backup': backup_path}


def audit(store: str, data_dir: str) -> None:
    """Print every stored row of `store` as one JSON line, oldest first.

    Each line takes the keys the module docstring's Output section lists.

    Parameters
    ----------
    store : str
        Store name, not a branch. A branch parent is read like any store.
    data_dir : str
        Base memman data directory.

    Raises
    ------
    Refused
        `store` is a branch.
    """
    with factory.open_backend(store, data_dir, read_only=True) as backend:
        if backend.meta.get(BRANCH_PARENT) is not None:
            raise Refused(f'{store} is a branch')
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

    Returns
    -------
    int
        Exit status: 0 done, 1 an error or `enrich_error`, 2 a refusal.
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
    return 1 if 'enrich_error' in report else 0


if __name__ == '__main__':
    sys.exit(main())
