"""SQLite implementation of the Backend Protocol surface.

Phase 1a: thin facade. Each Protocol verb binds 1:1 to an existing
free function in `store/{node,edge,oplog,db,snapshot}.py`. The legacy
function call sites (cli.py, maintenance.py, doctor.py) keep working
unchanged through Phase 4; pipeline call sites flip to `Backend`
verbs as part of Phase 1a's rewire step.

This module is the only file allowed to import
`memman.store.snapshot` outside of `store/`. The recall path goes
through `Backend.recall_session()` which yields a
`SqliteRecallSession`; pipeline code reads `session.snapshot` for
its current SQL-or-snapshot branching logic and falls through to
the active backend's verbs when the snapshot is absent. Phase 1b
moves that logic into `SqliteRecallSession` methods.
"""

import logging
import shutil
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memman.embed.vector import deserialize_vector, serialize_vector
from memman.store import db as _db
from memman.store import edge as _edge
from memman.store import node as _node
from memman.store import oplog as _oplog
from memman.store import snapshot as _snapshot
from memman.store.backend import Backend, Cluster, EdgeStore, MetaStore
from memman.store.backend import NodeStore, Oplog, RecallSession
from memman.store.db import DB
from memman.store.model import Edge, Id, Insight, NodeStats, OpLogEntry
from memman.store.model import OpLogStats, ProvenanceCount, ReembedRow
from memman.store.model import format_timestamp, parse_timestamp

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

    def get_many(self, ids: Sequence[Id]) -> list[Insight]:
        if not ids:
            return []
        by_id: dict[Id, Insight] = {}
        for iid in ids:
            ins = _node.get_insight_by_id(self._db, iid)
            if ins is not None:
                by_id[iid] = ins
        return [by_id[i] for i in ids if i in by_id]

    def query(
            self, *, keyword: str = '', category: str = '',
            min_importance: int = 0, source: str = '',
            limit: int = 20) -> list[Insight]:
        return _node.query_insights(
            self._db, keyword=keyword, category=category,
            min_importance=min_importance, source=source, limit=limit)

    def soft_delete(
            self, id: Id, *, tolerate_missing: bool = False) -> bool:
        return _node.soft_delete_insight(
            self._db, id, tolerate_missing=tolerate_missing)

    def update_entities(self, id: Id, entities: list[str]) -> None:
        _node.update_entities(self._db, id, entities)

    def update_enrichment(
            self, id: Id, *, keywords: list[str], summary: str,
            semantic_facts: list[str]) -> None:
        _node.update_enrichment(
            self._db, id, keywords, summary, semantic_facts)

    def increment_access_count(self, id: Id) -> None:
        _node.increment_access_count(self._db, id)

    def refresh_effective_importance(self, id: Id) -> float:
        return _node.refresh_effective_importance(self._db, id)

    def get_retention_candidates(
            self, *, threshold: float,
            limit: int) -> tuple[list[dict[str, Any]], int]:
        return _node.get_retention_candidates(
            self._db, threshold, limit)

    def count_active(self) -> int:
        return _node.count_active_insights(self._db)

    def count_total(self) -> int:
        return _node.count_total_insights(self._db)

    def has_active_with_source(self, source: str) -> bool:
        return _node.has_active_with_source(self._db, source)

    def iter_for_reembed(
            self, cursor: Id, batch: int) -> list[ReembedRow]:
        rows = _node.iter_for_reembed(self._db, cursor, batch)
        return [
            ReembedRow(
                id=r[0], content=r[1], embedding_model=r[2],
                blob_length=r[3])
            for r in rows
            ]

    def count_orphans(self) -> tuple[int, int]:
        return _node.count_orphans(self._db)

    def provenance_distribution(self) -> list[ProvenanceCount]:
        rows = _node.provenance_distribution(self._db)
        return [
            ProvenanceCount(
                prompt_version=r[0], model_id=r[1], count=r[2])
            for r in rows
            ]

    def auto_prune(
            self, *, max_insights: int,
            exclude_ids: list[Id] | None = None) -> int:
        return _node.auto_prune(
            self._db, max_insights, exclude_ids)

    def boost_retention(self, id: Id) -> None:
        _node.boost_retention(self._db, id)

    def get_recent_in_window(
            self, *, exclude_id: Id, window_hours: float,
            limit: int) -> list[Insight]:
        return _node.get_recent_insights_in_window(
            self._db, exclude_id, window_hours, limit)

    def get_latest_by_source(
            self, *, source: str, exclude_id: Id) -> Insight | None:
        return _node.get_latest_insight_by_source(
            self._db, source, exclude_id)

    def get_recent_active(
            self, *, exclude_id: Id, limit: int) -> list[Insight]:
        return _node.get_recent_active_insights(
            self._db, exclude_id, limit)

    def get_all_active(self) -> list[Insight]:
        return _node.get_all_active_insights(self._db)

    def stats(self) -> NodeStats:
        d = _node.get_stats(self._db)
        return NodeStats(
            total_insights=d.get('total_insights', 0),
            deleted_insights=d.get('deleted_insights', 0),
            edge_count=d.get('edge_count', 0),
            oplog_count=d.get('oplog_count', 0),
            by_category=d.get('by_category', {}),
            top_entities=d.get('top_entities', []))

    def update_embedding(
            self, id: Id, vec: list[float], model: str) -> None:
        _node.update_embedding(
            self._db, id, serialize_vector(vec), model)

    def get_embedding(self, id: Id) -> bytes | None:
        return _node.get_embedding(self._db, id)

    def get_all_embeddings(self) -> list[tuple[Id, str, bytes]]:
        return _node.get_all_embeddings(self._db)

    def iter_embeddings_as_vecs(
            self) -> Iterator[tuple[Id, list[float]]]:
        for eid, _content, blob in _node.get_all_embeddings(self._db):
            v = deserialize_vector(blob)
            if v is not None:
                yield eid, v

    def embedding_stats(self) -> tuple[int, int]:
        return _node.embedding_stats(self._db)

    def get_without_embedding(self, *, limit: int = 100) -> list[Insight]:
        return _node.get_insights_without_embedding(self._db, limit)

    def stamp_linked(self, id: Id) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_linked(self._db, id, ts)

    def stamp_enriched(self, id: Id) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_enriched(self._db, id, ts)

    def get_pending_link_ids(self, *, limit: int) -> list[Id]:
        return _node.get_pending_link_ids(self._db, limit)

    def get_active_ids(self) -> list[Id]:
        return _node.get_active_insight_ids(self._db)

    def count_pending_links(self) -> int:
        return _node.count_pending_links(self._db)

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        _node.reset_for_rebuild(self._db, ids)

    def clear_linked_at(self) -> None:
        _node.clear_linked_at(self._db)

    def review_content_quality(
            self, *, limit: int = 50) -> list[dict[str, Any]]:
        return _node.review_content_quality(self._db, limit)


class SqliteEdgeStore(EdgeStore):
    """Bindings from EdgeStore Protocol verbs to `store.edge` functions.
    """

    def __init__(self, db: DB) -> None:
        self._db = db

    def upsert(self, edge: Edge) -> None:
        _edge.insert_edge(self._db, edge)

    def by_node(self, node_id: Id) -> list[Edge]:
        return _edge.get_edges_by_node(self._db, node_id)

    def by_node_and_type(
            self, node_id: Id, edge_type: str) -> list[Edge]:
        return _edge.get_edges_by_node_and_type(
            self._db, node_id, edge_type)

    def by_source_and_type(
            self, source_id: Id, edge_type: str) -> list[Edge]:
        return _edge.get_edges_by_source_and_type(
            self._db, source_id, edge_type)

    def find_with_entity(
            self, entity: str, *, exclude_id: Id,
            limit: int) -> list[Id]:
        return _edge.find_insights_with_entity(
            self._db, entity, exclude_id, limit)

    def count_with_entity(
            self, entity: str, *, exclude_id: Id) -> int:
        return _edge.count_insights_with_entity(
            self._db, entity, exclude_id)

    def all(self) -> list[Edge]:
        return _edge.get_all_edges(self._db)

    def delete_by_node(self, node_id: Id) -> None:
        _edge.delete_edges_by_node(self._db, node_id)

    def delete_auto_for_node(
            self, node_id: Id, edge_type: str) -> None:
        _edge.delete_auto_edges_for_node(
            self._db, node_id, edge_type)

    def delete_auto_by_type(self, edge_type: str) -> None:
        _edge.delete_auto_edges_by_type(self._db, edge_type)

    def count_auto_by_type(self, edge_type: str) -> int:
        return _edge.count_auto_edges_by_type(self._db, edge_type)

    def delete_low_weight_temporal_proximity(
            self, *, min_weight: float) -> None:
        _edge.delete_low_weight_temporal_proximity(self._db, min_weight)

    def count_low_weight_temporal_proximity(
            self, *, min_weight: float) -> int:
        return _edge.count_low_weight_temporal_proximity(
            self._db, min_weight)

    def get_weight(
            self, source_id: Id, target_id: Id,
            edge_type: str) -> float | None:
        return _edge.get_edge_weight(
            self._db, source_id, target_id, edge_type)

    def count_dangling_by_type(self) -> dict[str, int]:
        return _edge.count_dangling_by_type(self._db)

    def degree_distribution(self) -> dict[Id, int]:
        return _edge.degree_distribution(self._db)

    def get_neighborhood(
            self, seed_id: Id, *, depth: int,
            edge_filter: str = '') -> list[tuple[Id, int, str]]:
        active_ids = set(_node.get_active_insight_ids(self._db))
        edges = _edge.get_all_edges(self._db)
        adj: dict[Id, list[Edge]] = {}
        for e in edges:
            adj.setdefault(e.source_id, []).append(e)
            if e.source_id != e.target_id:
                adj.setdefault(e.target_id, []).append(e)

        visited = {seed_id}
        queue: deque[tuple[Id, int]] = deque([(seed_id, 0)])
        out: list[tuple[Id, int, str]] = []

        while queue:
            cur_id, hop = queue.popleft()
            if hop >= depth:
                continue
            for edge in adj.get(cur_id, []):
                if edge_filter and edge.edge_type != edge_filter:
                    continue
                neighbor_id = (
                    edge.target_id if edge.target_id != cur_id
                    else edge.source_id)
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                if neighbor_id not in active_ids:
                    continue
                out.append((neighbor_id, hop + 1, edge.edge_type))
                queue.append((neighbor_id, hop + 1))
        return out


class SqliteMetaStore(MetaStore):
    """Bindings from MetaStore Protocol verbs to `store.db` get/set."""

    def __init__(self, db: DB) -> None:
        self._db = db

    def get(self, key: str) -> str | None:
        return _db.get_meta(self._db, key)

    def set(self, key: str, value: str) -> None:
        _db.set_meta(self._db, key, value)


class SqliteOplog(Oplog):
    """Bindings from Oplog Protocol verbs to `store.oplog` functions."""

    def __init__(self, db: DB) -> None:
        self._db = db

    def log(
            self, *, operation: str, insight_id: Id,
            detail: str) -> None:
        _oplog.log_op(self._db, operation, insight_id, detail)

    def maintenance_step(self) -> None:
        _oplog.maintenance_step(self._db)

    def trim_by_age(
            self, *,
            retention_days: int = _oplog.OPLOG_RETENTION_DAYS) -> int:
        return _oplog.trim_oplog_by_age(self._db, retention_days)

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        rows = _oplog.get_oplog(self._db, limit=limit, since=since)
        return [
            OpLogEntry(
                id=r['id'], operation=r['operation'],
                insight_id=r['insight_id'], detail=r['detail'],
                created_at=parse_timestamp(r['created_at']))
            for r in rows
            ]

    def stats(self, *, since: str = '') -> OpLogStats:
        d = _oplog.get_oplog_stats(self._db, since=since)
        return OpLogStats(
            operation_counts=d.get('operation_counts', {}),
            never_accessed=d.get('never_accessed', 0),
            total_active=d.get('total_active', 0))


@dataclass
class SqliteRecallSession(RecallSession):
    """Read-side session for the recall pipeline.

    Phase 1a: exposes `snapshot` directly. Recall pipeline code reads
    it to take the snapshot-or-SQL branching decision that lives in
    `search.recall.intent_aware_recall`. Phase 1b moves that branching
    into Protocol-typed verbs on `RecallSession`.

    Construction reads the snapshot eagerly so the lifecycle is clear:
    enter the context, read the snapshot once, fall through to SQL on
    miss, exit the context. Concurrent readers each take their own
    session; SQLite snapshots are file-backed (POSIX rename semantics
    guarantee atomicity).
    """

    snapshot: _snapshot.Snapshot | None = None

    def close(self) -> None:
        """No-op for SQLite (snapshot is in-memory, file-backed).

        Postgres' RecallSession in Phase 2 will close the connection
        here.
        """


class SqliteBackend(Backend):
    """Per-store backend wrapping a SQLite `DB`.

    Construction takes an already-open `DB`. `SqliteCluster.open()`
    calls `_db.open_db(...)` and wraps the result; tests / cli code
    that already have a `DB` can wrap it directly:
    `SqliteBackend(db)`.

    This keeps the Phase 1a non-goal "no backend dispatch in cli"
    intact: cli's `open_db` -> `SqliteBackend(db)` insertion is the
    cheapest possible threading of `Backend` through pipeline calls.
    """

    nodes: SqliteNodeStore
    edges: SqliteEdgeStore
    meta: SqliteMetaStore
    oplog: SqliteOplog

    def __init__(self, db: DB) -> None:
        self._db = db
        self.nodes = SqliteNodeStore(db)
        self.edges = SqliteEdgeStore(db)
        self.meta = SqliteMetaStore(db)
        self.oplog = SqliteOplog(db)

    @property
    def path(self) -> str:
        """Backing file path. SQLite-specific; pipeline code that needs
        a directory derives it via `pathlib.Path(backend.path).parent`.
        """
        return self._db.path

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block in a write transaction.

        Phase 1a delegates to `db.in_transaction` for parity with the
        existing `BEGIN IMMEDIATE` semantics. Nested entry currently
        raises (see `db.DB.in_transaction`). Phase 1b lifts the
        nested-call SAVEPOINT contract per the Protocol docstring.
        """
        if self._db._in_tx:
            yield
            return
        self._db._in_tx = True
        try:
            self._db._conn.execute('BEGIN IMMEDIATE')
            yield
            self._db._conn.execute('COMMIT')
        except Exception:
            self._db._conn.execute('ROLLBACK')
            raise
        finally:
            self._db._in_tx = False

    @contextmanager
    def write_lock(self, name: str) -> Iterator[None]:
        """No-op on SQLite -- `BEGIN IMMEDIATE` already serializes
        per-process. Postgres uses `pg_advisory_xact_lock`.
        """
        yield

    @contextmanager
    def reembed_lock(self, name: str) -> Iterator[bool]:
        """Always yields True on SQLite (single-process by definition).

        Postgres acquires `pg_try_advisory_lock` on a dedicated
        connection so concurrent sweeps fail-fast instead of
        racing.
        """
        yield True

    @contextmanager
    def readonly_context(self) -> Iterator['SqliteBackend']:
        """Yield a read-only Backend bound to a separate connection.

        Opens the database with `mode=ro` and closes on exit. The
        new SqliteBackend wraps the same DB path with a different
        underlying connection.
        """
        ro_db = _db.open_read_only(str(Path(self._db.path).parent))
        try:
            yield SqliteBackend(ro_db)
        finally:
            ro_db.close()

    @contextmanager
    def recall_session(self) -> Iterator[SqliteRecallSession]:
        """Yield a SqliteRecallSession for one recall request.

        Reads the snapshot eagerly using the active embed fingerprint;
        falls through to SQL when the snapshot is missing or
        fingerprint-mismatched.
        """
        snap: _snapshot.Snapshot | None = None
        try:
            from memman.embed.fingerprint import active_fingerprint
            store_dir_path = str(Path(self._db.path).parent)
            snap = _snapshot.read_snapshot(
                store_dir_path, active_fingerprint())
        except Exception as exc:
            logger.warning(f'snapshot load failed, using SQL: {exc}')
            snap = None
        session = SqliteRecallSession(snapshot=snap)
        try:
            yield session
        finally:
            session.close()

    def write_snapshot(self, fingerprint: Any) -> bool:
        """Materialize the recall snapshot.

        Phase 1a: this is the SQLite-specific snapshot writer. Not on
        the Backend Protocol surface (snapshots are SQLite-only). cli
        and worker callers reach this via `backend.write_snapshot(...)`
        rather than importing `store.snapshot.write_snapshot`.
        """
        return _snapshot.write_snapshot(
            self._db, str(Path(self._db.path).parent), fingerprint)

    def storage_summary(self) -> dict[str, Any]:
        return _db.storage_summary(self._db)

    def close(self) -> None:
        self._db.close()


def open_ro_db(sdir: str) -> DB:
    """Open the SQLite store at `sdir` in read-only mode.

    cli-internal helper that puts the only `open_read_only` literal
    inside the SQLite backend module. Callers that need a raw DB
    handle (because they do not yet take a Backend) reach this
    instead of importing `store.db.open_read_only` directly. The
    helper disappears when cli.py is rewired through Protocol verbs
    in Phase 4.
    """
    return _db.open_read_only(sdir)


class SqliteCluster(Cluster):
    """Cluster implementation for the SQLite backend.

    Stateless: `open()` opens or creates the per-store DB on demand;
    nothing is cached at the cluster level.
    """

    def open(self, *, store: str, data_dir: str) -> SqliteBackend:
        sdir = _db.store_dir(data_dir, store)
        return SqliteBackend(_db.open_db(sdir))

    def open_read_only(
            self, *, store: str, data_dir: str) -> SqliteBackend:
        sdir = _db.store_dir(data_dir, store)
        return SqliteBackend(_db.open_read_only(sdir))

    def list_stores(self, *, data_dir: str) -> list[str]:
        return _db.list_stores(data_dir)

    def drop_store(self, *, store: str, data_dir: str) -> None:
        sdir = _db.store_dir(data_dir, store)
        if Path(sdir).is_dir():
            shutil.rmtree(sdir)

    def close(self) -> None:
        return None
