"""SQLite implementation of the Backend Protocol surface.

Thin facade. Most Protocol verbs bind 1:1 to a free function in
`store/{node,oplog,db}.py`.

The recall path goes through `Backend.recall_session()`, which yields
a `SqliteRecallSession` holding one in-process embedding matrix for
the life of a single request. There is no persisted read cache: the
pipeline reads live SQL, so recall's candidate universe is the
store's active set by construction.
"""

import contextlib
import fcntl
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self
from urllib.parse import quote

import numpy as np
from memman.embed.fingerprint import Fingerprint
from memman.embed.swap import swap_remedy
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.migrate import Artifact, MigrateError, MigrateInsight
from memman.migrate import MigrateOpLog, MigrationPayload, Migrator
from memman.migrate import sanitize_identifier
from memman.setup.archive import archive_store_dir
from memman.store import db as _db
from memman.store import node as _node
from memman.store import oplog as _oplog
from memman.store.backend import Backend, MetaStore, NodeStore, Oplog
from memman.store.backend import RecallSession
from memman.store.db import DB
from memman.store.model import EnrichmentCoverage, Id, Insight, NodeStats
from memman.store.model import OpLogEntry, OpLogStats, ProvenanceCount
from memman.store.model import WorkerRun, format_timestamp, parse_timestamp

logger = logging.getLogger('memman')


class SqliteNodeStore(NodeStore):
    """Bindings from NodeStore Protocol verbs to `store.node` functions.
    """

    def __init__(self, db: DB) -> None:
        self._db = db

    def insert(self, ins: Insight) -> None:
        _node.insert_insight(self._db, ins)

    def get(self, id: Id) -> Insight | None:
        return _node.get_insight_by_id(self._db, id)

    def get_include_deleted(self, id: Id) -> Insight | None:
        return _node.get_insight_by_id_include_deleted(self._db, id)

    def resolve_id(self, id_or_prefix: str) -> str:
        """Resolve an exact id or unambiguous prefix to a full id.
        """
        exact = self._db._query(
            'select id from insights where id = ?',
            (id_or_prefix,)).fetchone()
        if exact is not None:
            return exact[0]
        rows = self._db._query(
            'select id from insights'
            ' where substr(id, 1, length(?)) = ?',
            (id_or_prefix, id_or_prefix)).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) == 0:
            return id_or_prefix
        raise ValueError(
            f'prefix {id_or_prefix!r} matches {len(rows)} rows')

    def query(
            self, *, keyword: str = '', limit: int = 20) -> list[Insight]:
        return _node.query_insights(self._db, keyword=keyword, limit=limit)

    def soft_delete(self, id: Id) -> bool:
        return _node.soft_delete_insight(self._db, id)

    def mark_replaced(self, predecessor_id: Id, successor_id: Id) -> bool:
        return _node.mark_insight_replaced(
            self._db, predecessor_id, successor_id)

    def predecessors(self, successor_id: Id) -> list[Insight]:
        return _node.get_predecessors(self._db, successor_id)

    def replacement_integrity(self) -> dict[str, list[Id]]:
        return _node.replacement_integrity(self._db)

    def update_enrichment(self, id: Id, *, summary: str) -> None:
        _node.update_enrichment(self._db, id, summary)

    def count_active(self) -> int:
        return _node.count_active_insights(self._db)

    def count_total(self) -> int:
        return _node.count_total_insights(self._db)

    def has_row_with_queue_uuid(self, queue_uuid: str) -> bool:
        return _node.has_row_with_queue_uuid(self._db, queue_uuid)

    def provenance_distribution(self) -> list[ProvenanceCount]:
        rows = _node.provenance_distribution(self._db)
        return [
            ProvenanceCount(prompt_version=r[0], count=r[1])
            for r in rows
            ]

    def get_all_active(self) -> list[Insight]:
        return _node.get_all_active_insights(self._db)

    def stats(self) -> NodeStats:
        d = _node.get_stats(self._db)
        return NodeStats(
            total_insights=d.get('total_insights', 0),
            replaced_insights=d.get('replaced_insights', 0),
            deleted_insights=d.get('deleted_insights', 0),
            oplog_count=d.get('oplog_count', 0))

    def update_embedding(
            self, id: Id, vec: list[float], model: str) -> None:
        _node.update_embedding(
            self._db, id, serialize_vector(vec), model)

    def embedding_stats(self) -> tuple[int, int]:
        return _node.embedding_stats(self._db)

    def enrichment_coverage(self) -> EnrichmentCoverage:
        sql = """
select count(*),
       sum(case when embedding is null then 1 else 0 end),
       sum(case when (summary is null or summary = '')
                 and enriched_at is null
                then 1 else 0 end),
       sum(case when enrich_attempted_at is not null
                 and enriched_at is null
                then 1 else 0 end)
from insights
where deleted_at is null and replaced_by is null
"""
        row = self._db._query(sql).fetchone()
        if row is None:
            return EnrichmentCoverage()
        total, miss_emb, miss_sum, stranded = row
        return EnrichmentCoverage(
            total_active=int(total or 0),
            missing_embedding=int(miss_emb or 0),
            missing_summary=int(miss_sum or 0),
            stranded=int(stranded or 0))

    def embedding_size_distribution(self) -> dict[int, int]:
        sql = """
select length(embedding), count(*)
from insights
where deleted_at is null and replaced_by is null and embedding is not null
group by length(embedding)
"""
        rows = self._db._query(sql).fetchall()
        return {int(size): int(count) for size, count in rows}

    def stamp_enrich_attempted(self, id: Id) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_enrich_attempted(self._db, id, ts)

    def stamp_enriched(
            self, id: Id, *,
            prompt_version: str | None = None) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_enriched(
            self._db, id, ts, prompt_version=prompt_version)

    def get_pending_enrich_ids(self, *, limit: int) -> list[Id]:
        return _node.get_pending_enrich_ids(self._db, limit)

    def get_active_ids(self) -> list[Id]:
        return _node.get_active_insight_ids(self._db)

    def count_pending_enrich(self) -> int:
        return _node.count_pending_enrich(self._db)

    def get_unenriched_attempted_ids(self, *, limit: int) -> list[Id]:
        return _node.get_unenriched_attempted_ids(self._db, limit)

    def iter_stale_insight_ids(self, active_pv: str) -> list[Id]:
        return _node.iter_stale_insight_ids(self._db, active_pv)

    def count_stale_insights(self, active_pv: str) -> int:
        return _node.count_stale_insights(self._db, active_pv)

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        _node.reset_for_rebuild(self._db, ids)


class SqliteMetaStore(MetaStore):
    """Bindings from MetaStore Protocol verbs to `store.db` get/set.
    """

    def __init__(self, db: DB) -> None:
        self._db = db

    def get(self, key: str) -> str | None:
        return _db.get_meta(self._db, key)

    def set(self, key: str, value: str) -> None:
        _db.set_meta(self._db, key, value)

    def delete(self, key: str) -> None:
        self._db._exec('delete from meta where key = ?', (key,))

    def keys(self) -> list[str]:
        rows = self._db._query('select key from meta').fetchall()
        return [r[0] for r in rows]


class SqliteOplog(Oplog):
    """Bindings from Oplog Protocol verbs to `store.oplog` functions.
    """

    def __init__(self, db: DB) -> None:
        self._db = db

    def log(
            self, *, operation: str, insight_id: Id,
            detail: str,
            before: dict[str, Any] | None = None,
            after: dict[str, Any] | None = None) -> None:
        _oplog.log_op(
            self._db, operation, insight_id, detail,
            before=before, after=after)

    def maintenance_step(self) -> None:
        _oplog.maintenance_step(self._db)

    def trim_by_age(self) -> int:
        return _oplog.trim_oplog_by_age(self._db)

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        rows = _oplog.get_oplog(self._db, limit=limit, since=since)
        return [
            OpLogEntry(
                id=r['id'], operation=r['operation'],
                insight_id=r['insight_id'], detail=r['detail'],
                created_at=parse_timestamp(r['created_at']),
                before=r.get('before'),
                after=r.get('after'))
            for r in rows
            ]

    def stats(self, *, since: str = '') -> OpLogStats:
        d = _oplog.get_oplog_stats(self._db, since=since)
        return OpLogStats(
            operation_counts=d.get('operation_counts', {}),
            total_active=d.get('total_active', 0))


@dataclass
class SqliteRecallSession(RecallSession):
    """Read-side session for one recall request.

    Owns an in-process embedding matrix, built lazily on first vector
    use so a keyword-only recall pays nothing for it, and dropped on
    context exit.

    Attributes
    ----------
    db : DB
        Live handle the matrix is built from. Reading it per request
        is what makes recall's candidate universe equal the store's
        active set.

    Notes
    -----
    - A row scores only against a query of its own blob width and
      scores 0.0 against any other, so a half-finished `embed swap`
      never raises on a ragged `np.array`.
    """

    db: DB
    _groups: dict[int, tuple[list[Id], Any, Any]] | None = None

    def close(self) -> None:
        """Drop the matrices so they do not outlive the request.
        """
        self._groups = None

    def _load(self) -> None:
        """Build one embedding matrix per stored width, once.
        """
        if self._groups is not None:
            return
        sql = """
select id, embedding
from insights
where deleted_at is null and replaced_by is null and embedding is not null
"""
        rows = [(rid, blob) for rid, blob in self.db._query(sql) if blob]

        by_width: dict[int, list[tuple[Id, bytes]]] = {}
        malformed = 0
        for rid, blob in rows:
            # A float64 vector is a whole number of 8-byte doubles.
            # np.frombuffer would raise on anything else and take the
            # whole channel down with it.
            if len(blob) % 8:
                malformed += 1
                continue
            by_width.setdefault(len(blob), []).append((rid, blob))
        if malformed:
            logger.warning(
                f'{malformed} embedding blob(s) are not a whole number'
                f' of float64 values and were skipped; run'
                f' `memman embed reembed` to repair')

        # One matrix per width: a store mid-`embed reembed` holds two
        # widths, and scoring only the modal one would blank the
        # vector channel for a query at the other width.
        groups: dict[int, tuple[list[Id], Any, Any]] = {}
        for width, entries in by_width.items():
            dim = width // 8
            # float64 matches the precision cosines use store-wide.
            matrix = np.empty((len(entries), dim), dtype=np.float64)
            for row, (rid, blob) in enumerate(entries):
                matrix[row] = np.frombuffer(blob, dtype='<f8')
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0
            groups[dim] = ([rid for rid, _b in entries], matrix, norms)

        self._groups = groups

    def _cosines(
            self, query_vec: list[float]) -> tuple[list[Id], Any]:
        """Ids and cosines for the rows matching the query's width.

        Rows stored at any other width are absent from the result,
        which the callers read as similarity 0.0.
        """
        empty: tuple[list[Id], Any] = ([], np.zeros((0,), dtype=np.float64))
        self._load()
        if not self._groups:
            return empty
        query = np.asarray(query_vec, dtype=np.float64)
        if query.ndim != 1:
            return empty
        group = self._groups.get(int(query.shape[0]))
        if group is None:
            return empty
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return empty
        ids, matrix, norms = group
        return ids, (matrix @ query) / (norms * query_norm)

    def similarities(
            self, query_vec: list[float]) -> dict[Id, float]:
        """Cosine per id, positives only. See the Protocol docstring.
        """
        row_ids, sims = self._cosines(query_vec)
        return {
            row_ids[row]: float(sims[row])
            for row in np.nonzero(sims > 0.0)[0]
            }

    def keyword_counts(
            self, query_tokens: set[str]) -> dict[Id, int]:
        """Match count per active insight id, from FTS5 probes.

        The contract, including the non-ASCII divergence from
        `keyword.insight_tokens`, is in the Protocol docstring.
        """
        if not query_tokens:
            return {}
        sql = """
select i.id
from insights_fts f
join insights i on i.rowid = f.rowid
where insights_fts match ? and i.deleted_at is null and i.replaced_by is null
"""
        counts: dict[Id, int] = {}
        # Notes:
        # - One probe per token: an `OR` expression returns the union
        #   of rows but not which token matched which row, and the
        #   per-token count is `kw_score`'s numerator.
        # - The match expression is built from the token alone, never
        #   from user text, since FTS5 `match` takes a query language.
        #   The quotes keep the probe valid if `tokenize` stops
        #   guaranteeing `[a-zA-Z0-9]+`.
        for token in query_tokens:
            for (iid,) in self.db._query(sql, (f'"{token}"',)):
                counts[iid] = counts.get(iid, 0) + 1
        return counts

    def vector_anchors(
            self, query_vec: list[float], *,
            k: int = 10) -> list[tuple[Id, float]]:
        """Return top-k (id, similarity) matches. Cosine in (0, 1].

        Ties break on id descending.
        """
        row_ids, sims = self._cosines(query_vec)
        scored = [
            (float(sims[row]), row_ids[row])
            for row in np.nonzero(sims > 0.0)[0]
            ]
        scored.sort(reverse=True)
        return [(rid, sim) for sim, rid in scored[:k]]


class SqliteBackend(Backend):
    """Per-store backend wrapping a SQLite `DB`.

    Wraps an already-open `DB`; `open_sqlite_backend` opens one.
    """

    nodes: SqliteNodeStore
    meta: SqliteMetaStore
    oplog: SqliteOplog

    def __init__(self, db: DB) -> None:
        self._db = db
        self.nodes = SqliteNodeStore(db)
        self.meta = SqliteMetaStore(db)
        self.oplog = SqliteOplog(db)

    @property
    def path(self) -> str:
        """Path of the `memman.db` file.
        """
        return self._db.path

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block in one `begin immediate` write transaction.

        Commits when the block returns and rolls back when it raises.
        A nested entry joins the outer transaction and opens no second
        one.
        """
        if self._db._in_tx:
            yield
            return
        self._db._in_tx = True
        try:
            self._db._conn.execute('begin immediate')
            yield
            self._db._conn.execute('commit')
        except Exception:
            try:
                self._db._conn.execute('rollback')
            except sqlite3.OperationalError as rollback_exc:
                logger.debug(f'rollback skipped: {rollback_exc}')
            raise
        finally:
            self._db._in_tx = False

    @contextmanager
    def reembed_lock(self, name: str) -> Iterator[bool]:
        """Yield True: SQLite runs single-process, so the lock is free.
        """
        yield True

    @contextmanager
    def swap_lock(self) -> Iterator[bool]:
        """Yield whether this handle holds the store's swap lock.

        An exclusive, non-blocking flock on `swap.lock` beside the
        store's `memman.db`. The kernel releases it when the holder
        exits, so a crash leaves no stale lock. `_require_stopped`
        already excludes the drain; this excludes a second swap or an
        abort from another shell.
        """
        path = Path(self._db.path).parent / 'swap.lock'
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def swap_prepare(self, target_dim: int) -> None:
        """No-op: `embedding_pending` is in the baseline schema.
        """
        return

    def iter_for_swap(
            self, cursor: str, batch: int) -> list[tuple[str, str]]:
        """Return rows still needing `embedding_pending`.
        """
        return _node.iter_for_swap(self._db, cursor, batch)

    def write_swap_batch(
            self, items: list[tuple[str, list[float]]]) -> None:
        """Bulk-update `embedding_pending` for the given (id, vec) items.
        """
        blobs = [(rid, serialize_vector(vec)) for (rid, vec) in items]
        _node.write_swap_batch(self._db, blobs)

    def swap_cutover(self, target: Fingerprint) -> None:
        """Copy `embedding_pending` into `embedding`, set model, null shadow.

        Runs in its own transaction. The caller writes the fingerprint
        afterward.
        """
        with self.transaction():
            _node.swap_cutover_sqlite(self._db, target.model)

    def swap_abort(self) -> None:
        """Null `embedding_pending` on every row.
        """
        with self.transaction():
            _node.swap_abort_sqlite(self._db)

    @contextmanager
    def recall_session(self) -> Iterator[SqliteRecallSession]:
        """Yield a session that reads the live database.
        """
        session = SqliteRecallSession(db=self._db)
        try:
            yield session
        finally:
            session.close()

    def integrity_check(self) -> dict[str, Any]:
        """Report page-level integrity and keyword-index drift.

        Returns
        -------
        dict[str, Any]
            `{'ok': bool, 'detail': str}`. `detail` is the pragma's
            own word when the pages are bad, otherwise names the
            keyword index when its terms no longer match the text.

        Notes
        -----
        - The probe scans every indexed row, so it belongs in the
          `doctor` path and never on the recall path.
        """
        row = self._db._query('pragma integrity_check').fetchone()
        result = row[0] if row else 'unknown'
        if result != 'ok':
            return {'ok': False, 'detail': result}
        # Notes:
        # - Rank 1 is the only form that reads the content table:
        #   `pragma integrity_check` and the default FTS5
        #   `'integrity-check'` both pass on a drifted index.
        # - The probe needs a write transaction, so a read-only handle
        #   or a busy writer raises `OperationalError` and says
        #   nothing about the index. Real drift raises
        #   `DatabaseError`. Reporting "not run" beats reporting
        #   corruption that is not there.
        try:
            self._db._query(
                "insert into insights_fts(insights_fts, rank)"
                " values('integrity-check', 1)")
        except sqlite3.OperationalError as exc:
            return {
                'ok': True,
                'detail': f'{result}; insights_fts not checked: {exc}',
                }
        except sqlite3.DatabaseError as exc:
            return {
                'ok': False,
                'detail': (
                    f'insights_fts does not match insights: {exc};'
                    f" repair with: insert into"
                    f" insights_fts(insights_fts) values('rebuild')"),
                }
        return {'ok': True, 'detail': result}

    def start_run(self) -> int | None:
        """No-op: drain hangs are observable at the foreground prompt.
        """
        return None

    def beat_run(self, run_id: int | None) -> None:
        """No-op for SQLite mode.
        """
        return

    def finish_run(self, run_id: int | None) -> None:
        """No-op for SQLite mode.
        """
        return

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        """No-op: SQLite drain has no per-store worker_runs table.
        """
        return []

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        self.close()


def open_sqlite_backend(
        store: str, data_dir: str, *,
        read_only: bool = False) -> 'SqliteBackend':
    """Open or create the per-store SQLite backend.

    Materializes `<data_dir>/data/<store>/memman.db` on demand;
    `read_only=True` opens the existing DB in `mode=ro` without
    creating it.
    """
    sdir = _db.store_dir(data_dir, store)
    if read_only:
        return SqliteBackend(_db.open_read_only(sdir))
    return SqliteBackend(_db.open_db(sdir))


def drop_sqlite_store(store: str, data_dir: str) -> None:
    """Remove the SQLite store directory for `store` if it exists.
    """
    sdir = _db.store_dir(data_dir, store)
    if Path(sdir).is_dir():
        shutil.rmtree(sdir)


class SqliteMigrator(Migrator):
    """SQLite implementation of the Migrator surface.

    `gather(store)` opens `<data_dir>/data/<store>/memman.db`
    read-only and materializes every row into a backend-agnostic
    `MigrationPayload`. `apply(store, payload)` writes the payload
    into a fresh sqlite store at the same path. Self-round-trip
    (gather -> apply -> gather) is the equality invariant; the
    cross-backend round-trip via `PostgresMigrator` is locked by
    test_migrator_classes.py.
    """

    backend_name: ClassVar[str] = 'sqlite'

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def _store_path(self, store: str) -> Path:
        return Path(_db.store_dir(self.data_dir, store)) / 'memman.db'

    def _connect_ro(self, store: str, path: Path) -> sqlite3.Connection:
        """Open one store's SQLite file read-only for a migration read.

        Parameters
        ----------
        store : str
            Store name, for the error message only.
        path : Path
            Full path to the store's `memman.db`.

        Returns
        -------
        sqlite3.Connection
            An open read-only connection the caller closes.

        Raises
        ------
        MigrateError
            When the file cannot be opened or read as a database.
            Never a bare `sqlite3.Error`: `memman migrate` catches
            `MigrateError`, and the CLI root group catches
            `BackendError`, so an untranslated driver error reaches
            the operator as a traceback.
        """
        # Notes:
        # - The probe read forces the header: `connect` is lazy, so
        #   corrupt bytes surface only at the first statement.
        # - The path is percent-encoded: an unescaped `#` or `?` would
        #   silently open a different file read-write.
        uri = f'file:{quote(str(path))}?mode=ro'
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.execute('pragma schema_version')
        except sqlite3.Error as exc:
            if conn is not None:
                conn.close()
            raise MigrateError(
                f'cannot read sqlite store {store!r} at {path}:'
                f' {exc}') from exc
        return conn

    def preflight_source(self, store: str) -> None:
        path = self._store_path(store)
        if not path.exists():
            raise MigrateError(
                f'sqlite store not found: {path}')
        try:
            with contextlib.closing(
                    self._connect_ro(store, path)) as conn:
                n = conn.execute(
                    'select count(*) from insights').fetchone()[0]
                fp = conn.execute(
                    "select 1 from meta where key ="
                    " 'embed_fingerprint'").fetchone()
                swap_state = conn.execute(
                    "select value from meta where key ="
                    " 'embed_swap_state'").fetchone()
        except sqlite3.Error as exc:
            raise MigrateError(
                f'cannot read sqlite store {store!r} at {path}:'
                f' {exc}') from exc
        if n == 0 and fp is None:
            raise MigrateError(
                f'sqlite store {store!r} is empty (no insights,'
                f' no embed fingerprint); nothing to migrate')
        if swap_state and swap_state[0]:
            raise MigrateError(
                f'sqlite store {store!r} has an embed swap in'
                f' flight (state={swap_state[0]!r});'
                f' {swap_remedy(store, swap_state[0])}')

    def preflight_target(self, store: str) -> None:
        sanitize_identifier(store)
        target_root = Path(self.data_dir) / 'data'
        target_root.mkdir(mode=0o755, exist_ok=True, parents=True)

    def gather(self, store: str) -> MigrationPayload:
        path = self._store_path(store)
        if not path.exists():
            raise MigrateError(
                f'sqlite store not found: {path}')

        with contextlib.closing(
                self._connect_ro(store, path)) as conn:
            meta_dict = dict(conn.execute(
                'select key, value from meta').fetchall())

            fp_str = meta_dict.get('embed_fingerprint')
            if not fp_str:
                raise MigrateError(
                    f'sqlite store {store!r} has no'
                    f' embed_fingerprint meta key')
            fingerprint = Fingerprint.from_json(fp_str)

            rows = conn.execute("""
select id, content, summary, embedding,
       enrich_attempted_at, enriched_at, created_at, updated_at,
       deleted_at, prompt_version, embedding_model,
       queue_uuid, replaced_by, author
from insights
order by id
""").fetchall()
            insights: list[MigrateInsight] = []
            for r in rows:
                emb = deserialize_vector(r[3]) if r[3] else None
                insights.append(MigrateInsight(
                    id=r[0], content=r[1],
                    summary=r[2],
                    embedding=emb,
                    enrich_attempted_at=(
                        parse_timestamp(r[4]) if r[4] else None),
                    enriched_at=(
                        parse_timestamp(r[5]) if r[5] else None),
                    created_at=parse_timestamp(r[6]),
                    updated_at=parse_timestamp(r[7]),
                    deleted_at=(
                        parse_timestamp(r[8]) if r[8] else None),
                    prompt_version=r[9],
                    embedding_model=r[10],
                    queue_uuid=r[11],
                    replaced_by=r[12],
                    author=r[13]))

            op_rows = conn.execute("""
select id, operation, insight_id, detail, created_at,
       before, after
from oplog
order by id
""").fetchall()
            oplog = [
                MigrateOpLog(
                    id=int(o[0]), operation=o[1],
                    insight_id=o[2], detail=o[3] or '',
                    created_at=parse_timestamp(o[4]),
                    before=json.loads(o[5]) if o[5] else None,
                    after=json.loads(o[6]) if o[6] else None,
                    legacy_id=int(o[0]))
                for o in op_rows]

        return MigrationPayload(
            fingerprint=fingerprint,
            embedding_dim=fingerprint.dim,
            insights=insights,
            oplog=oplog,
            meta=meta_dict)

    def apply(
            self, store: str, payload: MigrationPayload) -> None:
        target_dir = _db.store_dir(self.data_dir, store)
        Path(target_dir).mkdir(
            mode=0o755, exist_ok=True, parents=True)
        db = _db.open_db(target_dir)
        try:
            conn = db.conn
            try:
                conn.execute('begin')

                insight_rows = []
                for ins in payload.insights:
                    emb_blob = (
                        serialize_vector(ins.embedding)
                        if ins.embedding is not None else None)
                    insight_rows.append((
                        ins.id, ins.content,
                        ins.summary,
                        emb_blob,
                        format_timestamp(ins.enrich_attempted_at)
                        if ins.enrich_attempted_at else None,
                        format_timestamp(ins.enriched_at)
                        if ins.enriched_at else None,
                        format_timestamp(ins.created_at),
                        format_timestamp(ins.updated_at),
                        format_timestamp(ins.deleted_at)
                        if ins.deleted_at else None,
                        ins.prompt_version,
                        ins.embedding_model,
                        ins.queue_uuid,
                        ins.replaced_by,
                        ins.author))
                if insight_rows:
                    conn.executemany(
                        'insert into insights ('
                        ' id, content, summary,'
                        ' embedding,'
                        ' enrich_attempted_at, enriched_at, created_at,'
                        ' updated_at, deleted_at, prompt_version,'
                        ' embedding_model,'
                        ' queue_uuid,'
                        ' replaced_by, author)'
                        ' values (?, ?, ?, ?,'
                        ' ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        insight_rows)

                max_oplog_id = 0
                for op in payload.oplog:
                    row = (
                        op.legacy_id, op.operation, op.insight_id,
                        op.detail,
                        format_timestamp(op.created_at),
                        json.dumps(op.before)
                        if op.before is not None else None,
                        json.dumps(op.after)
                        if op.after is not None else None)
                    try:
                        conn.execute(
                            'insert into oplog ('
                            ' id, operation, insight_id, detail,'
                            ' created_at, before, after)'
                            ' values (?, ?, ?, ?, ?, ?, ?)',
                            row)
                        max_oplog_id = max(max_oplog_id, op.legacy_id)
                    except sqlite3.IntegrityError:
                        conn.execute(
                            'insert into oplog ('
                            ' operation, insight_id, detail,'
                            ' created_at, before, after)'
                            ' values (?, ?, ?, ?, ?, ?)',
                            row[1:])
                if max_oplog_id > 0:
                    conn.execute(
                        "insert or replace into sqlite_sequence"
                        " (name, seq) values ('oplog', ?)",
                        (max_oplog_id,))

                meta_rows = list(payload.meta.items())
                if meta_rows:
                    conn.executemany(
                        'insert or replace into meta'
                        ' (key, value) values (?, ?)',
                        meta_rows)

                conn.execute('commit')
            except Exception as exc:
                try:
                    conn.execute('rollback')
                except sqlite3.Error:
                    pass
                raise MigrateError(
                    f'sqlite apply for store {store!r} failed:'
                    f' {type(exc).__name__}: {exc}') from exc
        finally:
            db.close()

    def archive(self, store: str, data_dir: str) -> Artifact:
        path = archive_store_dir(data_dir, store)
        if path is None:
            return Artifact(kind='none', location=None)
        return Artifact(kind='filesystem', location=str(path))
