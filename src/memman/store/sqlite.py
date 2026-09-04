"""SQLite implementation of the Backend Protocol surface.

Thin facade. Each Protocol verb binds 1:1 to an existing free
function in `store/{node,edge,oplog,db}.py`.

The recall path goes through `Backend.recall_session()`, which yields
a `SqliteRecallSession` holding one in-process embedding matrix for
the life of a single request. There is no persisted read cache: the
pipeline reads live SQL, so recall's candidate universe is the
store's active set by construction.
"""

import contextlib
import json
import logging
import shutil
import sqlite3
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self
from urllib.parse import quote

import numpy as np
from memman.embed.fingerprint import Fingerprint
from memman.embed.vector import deserialize_vector, serialize_vector
from memman.migrate import PAYLOAD_VERSION, Artifact, BackendFeatures
from memman.migrate import MigrateEdge, MigrateError, MigrateInsight
from memman.migrate import MigrateOpLog, MigrationPayload, Migrator
from memman.migrate import PendingReembed, SwapState, sanitize_identifier
from memman.store import db as _db
from memman.store import edge as _edge
from memman.store import node as _node
from memman.store import oplog as _oplog
from memman.store.backend import Backend, EdgeStore, MetaStore, NodeStore
from memman.store.backend import Oplog, RecallSession
from memman.store.base import BaseNodeStore
from memman.store.db import DB
from memman.store.model import Edge, EnrichmentCoverage, Id, Insight
from memman.store.model import NodeStats, OpLogEntry, OpLogStats
from memman.store.model import ProvenanceCount, ReembedRow, WorkerRun
from memman.store.model import format_timestamp, parse_timestamp

logger = logging.getLogger('memman')


class SqliteNodeStore(BaseNodeStore, NodeStore):
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
            source: str = '', limit: int = 20) -> list[Insight]:
        return _node.query_insights(
            self._db, keyword=keyword, category=category, source=source,
            limit=limit)

    def soft_delete(self, id: Id) -> bool:
        return _node.soft_delete_insight(self._db, id)

    def supersede(self, predecessor_id: Id, successor_id: Id) -> bool:
        return _node.supersede_insight(
            self._db, predecessor_id, successor_id)

    def update_entities(self, id: Id, entities: list[str]) -> None:
        _node.update_entities(self._db, id, entities)

    def update_enrichment(
            self, id: Id, *, keywords: list[str], summary: str,
            semantic_facts: list[str]) -> None:
        _node.update_enrichment(
            self._db, id, keywords, summary, semantic_facts)

    def increment_access_count(self, id: Id) -> None:
        _node.increment_access_count(self._db, id)

    def increment_corroboration(
            self, id: Id, *, queue_uuid: str | None = None) -> bool:
        return _node.increment_corroboration(self._db, id, queue_uuid)

    def count_active(self) -> int:
        return _node.count_active_insights(self._db)

    def count_total(self) -> int:
        return _node.count_total_insights(self._db)

    def has_active_with_queue_uuid(self, queue_uuid: str) -> bool:
        return _node.has_active_with_queue_uuid(self._db, queue_uuid)

    def get_by_queue_uuid(self, queue_uuid: str) -> list[Insight]:
        return _node.get_by_queue_uuid(self._db, queue_uuid)

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

    def get_recent_in_window(
            self, *, exclude_id: Id, window_hours: float,
            limit: int) -> list[Insight]:
        return _node.get_recent_insights_in_window(
            self._db, exclude_id, window_hours, limit)

    def get_latest_by_session(
            self, *, session_id: str | None,
            exclude_id: Id) -> Insight | None:
        return _node.get_latest_insight_by_session(
            self._db, session_id, exclude_id)

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
            superseded_insights=d.get('superseded_insights', 0),
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

    def embedding_stats(self) -> tuple[int, int]:
        return _node.embedding_stats(self._db)

    def enrichment_coverage(self) -> EnrichmentCoverage:
        sql = """
select count(*),
       sum(case when embedding is null then 1 else 0 end),
       sum(case when keywords is null or keywords = '' then 1 else 0 end),
       sum(case when (summary is null or summary = '')
                 and enriched_at is null
                then 1 else 0 end),
       sum(case when semantic_facts is null
                 or semantic_facts = ''
                then 1 else 0 end)
from insights
where deleted_at is null and superseded_by is null
"""
        row = self._db._query(sql).fetchone()
        if row is None:
            return EnrichmentCoverage()
        total, miss_emb, miss_kw, miss_sum, miss_sf = row
        return EnrichmentCoverage(
            total_active=int(total or 0),
            missing_embedding=int(miss_emb or 0),
            missing_keywords=int(miss_kw or 0),
            missing_summary=int(miss_sum or 0),
            missing_semantic_facts=int(miss_sf or 0))

    def embedding_size_distribution(self) -> dict[int, int]:
        sql = """
select length(embedding), count(*)
from insights
where deleted_at is null and superseded_by is null and embedding is not null
group by length(embedding)
"""
        rows = self._db._query(sql).fetchall()
        return {int(size): int(count) for size, count in rows}

    def stamp_linked(self, id: Id) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_linked(self._db, id, ts)

    def stamp_enriched(
            self, id: Id, *,
            prompt_version: str | None = None) -> None:
        ts = format_timestamp(datetime.now(timezone.utc))
        _node.stamp_enriched(
            self._db, id, ts, prompt_version=prompt_version)

    def get_pending_link_ids(self, *, limit: int) -> list[Id]:
        return _node.get_pending_link_ids(self._db, limit)

    def get_active_ids(self) -> list[Id]:
        return _node.get_active_insight_ids(self._db)

    def count_pending_links(self) -> int:
        return _node.count_pending_links(self._db)

    def get_unenriched_linked_ids(self, *, limit: int) -> list[Id]:
        return _node.get_unenriched_linked_ids(self._db, limit)

    def count_unenriched_linked(self) -> int:
        return _node.count_unenriched_linked(self._db)

    def iter_stale_insight_ids(self, active_pv: str) -> list[Id]:
        return _node.iter_stale_insight_ids(self._db, active_pv)

    def count_stale_insights(self, active_pv: str) -> int:
        return _node.count_stale_insights(self._db, active_pv)

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        _node.reset_for_rebuild(self._db, ids)

    def clear_linked_at(self) -> None:
        _node.clear_linked_at(self._db)


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

    def adjacency(self) -> dict[Id, list[tuple[Id, str, float]]]:
        return _edge.get_adjacency(self._db)

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

    def delete(self, key: str) -> None:
        self._db._exec('delete from meta where key = ?', (key,))

    def keys(self) -> list[str]:
        rows = self._db._query('select key from meta').fetchall()
        return [r[0] for r in rows]


class SqliteOplog(Oplog):
    """Bindings from Oplog Protocol verbs to `store.oplog` functions."""

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
                created_at=parse_timestamp(r['created_at']),
                before=r.get('before'),
                after=r.get('after'))
            for r in rows
            ]

    def stats(self, *, since: str = '') -> OpLogStats:
        d = _oplog.get_oplog_stats(self._db, since=since)
        return OpLogStats(
            operation_counts=d.get('operation_counts', {}),
            never_accessed=d.get('never_accessed', 0),
            total_active=d.get('total_active', 0))

    def delta_coverage(self) -> tuple[int, int]:
        sql = """
select count(*),
       sum(case when before is not null
                 or after is not null
                then 1 else 0 end)
from oplog
"""
        row = self._db._query(sql).fetchone()
        if row is None:
            return (0, 0)
        return (int(row[0] or 0), int(row[1] or 0))


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
    - The matrix is float64 because `embed.vector.cosine_similarity`
      promotes to float64, so a float64 matmul keeps this path within
      a float ulp of the per-pair helper it replaces. The speed comes
      from `np.frombuffer` replacing `struct.unpack` (measured
      12.4 ms -> 1.0 ms at N=1053), not from a narrower dtype.
    - Rows whose blob width differs from the store's modal width are
      left out of the matrix, so a half-finished `embed swap` scores
      them 0.0 instead of raising on a ragged `np.array`.
    """

    db: DB
    _groups: dict[int, tuple[list[Id], Any, Any]] | None = None
    _row_of: dict[Id, tuple[int, int]] | None = None
    _meta: dict[Id, tuple[str, str]] | None = None

    def close(self) -> None:
        """Drop the matrices so they do not outlive the request."""
        self._groups = None
        self._row_of = None
        self._meta = None

    def _load(self) -> None:
        """Build one embedding matrix per stored width, once.

        Notes
        -----
        - Grouped by width rather than reduced to a single modal
          width: a store mid-`embed reembed` holds two widths, and
          scoring only the modal group would blank the whole vector
          channel for a query at the other width -- including the
          rows that query CAN score. Each row is compared only
          against a query of its own width, which is
          `cosine_similarity`'s own 0.0-on-mismatch rule applied per
          row rather than per store.
        - One pass reads category and source beside the blob, so the
          eligibility filter in `vector_anchors` needs no second
          query and no cache handed in by the pipeline.
        """
        if self._groups is not None:
            return
        sql = """
select id, category, source, embedding
from insights
where deleted_at is null and superseded_by is null and embedding is not null
"""
        rows = [
            (rid, cat, src, blob)
            for rid, cat, src, blob in self.db._query(sql)
            if blob]
        self._meta = {rid: (cat, src) for rid, cat, src, _b in rows}

        by_width: dict[int, list[tuple[Id, bytes]]] = {}
        malformed = 0
        for rid, _cat, _src, blob in rows:
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

        groups: dict[int, tuple[list[Id], Any, Any]] = {}
        row_of: dict[Id, tuple[int, int]] = {}
        for width, entries in by_width.items():
            dim = width // 8
            matrix = np.empty((len(entries), dim), dtype=np.float64)
            for row, (rid, blob) in enumerate(entries):
                matrix[row] = np.frombuffer(blob, dtype='<f8')
                row_of[rid] = (dim, row)
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0
            groups[dim] = ([rid for rid, _b in entries], matrix, norms)

        self._groups = groups
        self._row_of = row_of

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
        """Cosine per id, positives only. See the Protocol docstring."""
        row_ids, sims = self._cosines(query_vec)
        return {
            row_ids[row]: float(sims[row])
            for row in np.nonzero(sims > 0.0)[0]
            }

    def vectors_for_ids(
            self, ids: list[Id]) -> dict[Id, list[float]]:
        """Embeddings for specific ids.

        Reads whatever width each id was stored at, so a store mid
        -reembed returns both.
        """
        self._load()
        if not self._groups or self._row_of is None:
            return {}
        out: dict[Id, list[float]] = {}
        for rid in ids:
            found = self._row_of.get(rid)
            if found is None:
                continue
            dim, row = found
            group = self._groups.get(dim)
            if group is not None:
                out[rid] = group[1][row].tolist()
        return out

    def keyword_counts(
            self, query_tokens: set[str]) -> dict[Id, int]:
        """Match count per active insight id, from FTS5 probes.

        See the Protocol docstring for the contract.

        Notes
        -----
        - Agrees with `keyword.insight_tokens` on ASCII text and
          diverges on non-ASCII; see the Protocol docstring for the
          class and the measured rate. Do not "fix" it here -- the
          tokenizers differ by construction.
        - One probe per token rather than one `OR` expression: the
          combined form returns the union of the rows but not which
          token matched which row, and the per-token count IS
          `kw_score`'s numerator. Measured at the same cost either
          way (3.1 ms against 3.0 ms for 50 tokens at N=1054).
        - The probe expression is built here and never from user
          text: FTS5 `match` takes a query language, and 8 of 11
          realistic queries handed to it raw raise a syntax error.
          Quoting the token costs nothing and makes the probe hold
          even if `tokenize` ever stops guaranteeing `[a-zA-Z0-9]+`.
        """
        if not query_tokens:
            return {}
        sql = """
select i.id
from insights_fts f
join insights i on i.rowid = f.rowid
where insights_fts match ? and i.deleted_at is null and i.superseded_by is null
"""
        counts: dict[Id, int] = {}
        for token in query_tokens:
            for (iid,) in self.db._query(sql, (f'"{token}"',)):
                counts[iid] = counts.get(iid, 0) + 1
        return counts

    def vector_anchors(
            self, query_vec: list[float], *, k: int = 10,
            category: str = '', source: str = '') -> list[tuple[Id, float]]:
        """Return top-k (id, similarity) matches. Cosine in (0, 1].

        Notes
        -----
        - `category` / `source` restrict eligibility BEFORE the top-k
          cut, so a filtered recall still returns k anchors where k
          exist. Post-filtering the hits would under-fill k.
        - Ties break on id descending, matching the previous
          `sort(reverse=True)` over `(sim, id)` pairs.
        """
        row_ids, sims = self._cosines(query_vec)
        meta = self._meta or {}
        scored: list[tuple[float, Id]] = []
        for row in np.nonzero(sims > 0.0)[0]:
            rid = row_ids[row]
            if category or source:
                cat, src = meta.get(rid, ('', ''))
                if category and cat != category:
                    continue
                if source and src != source:
                    continue
            scored.append((float(sims[row]), rid))
        scored.sort(reverse=True)
        return [(rid, sim) for sim, rid in scored[:k]]


class SqliteBackend(Backend):
    """Per-store backend wrapping a SQLite `DB`.

    Construction takes an already-open `DB`. `open_sqlite_backend`
    calls `_db.open_db(...)` and wraps the result; tests / cli code
    that already have a `DB` can wrap it directly:
    `SqliteBackend(db)`.
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

        Delegates to `db.in_transaction` for `begin immediate`
        semantics. Nested entry currently raises (see
        `db.DB.in_transaction`).
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
    def write_lock(self, name: str) -> Iterator[None]:
        """No-op on SQLite -- `begin immediate` already serializes
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
    def swap_lock(self) -> Iterator[bool]:
        """Always yields True on SQLite (single-process by definition).

        `_require_stopped('swap')` already excludes the drain at the
        CLI boundary. Postgres acquires `pg_try_advisory_lock` on
        the `embed_swap:<schema>` key so cross-process swaps fail-
        fast instead of racing.
        """
        yield True

    def swap_prepare(self, target_dim: int) -> None:
        """No-op on SQLite; `embedding_pending` blob is in the baseline
        schema. Dim is not enforced at the column level on SQLite.
        """
        return

    def iter_for_swap(
            self, cursor: str, batch: int) -> list[tuple[str, str]]:
        """Return rows still needing `embedding_pending`."""
        return _node.iter_for_swap(self._db, cursor, batch)

    def write_swap_batch(
            self, items: list[tuple[str, list[float]]]) -> None:
        """Bulk-update `embedding_pending` for the given (id, vec) items."""
        blobs = [(rid, serialize_vector(vec)) for (rid, vec) in items]
        _node.write_swap_batch(self._db, blobs)

    def swap_cutover(self, target: Fingerprint) -> None:
        """Copy `embedding_pending` into `embedding`, set model, null shadow.
        Runs in its own transaction; orchestrator records cutover state
        before invoking and writes the fingerprint after.
        """
        with self.transaction():
            _node.swap_cutover_sqlite(self._db, target.model)

    def swap_abort(self) -> None:
        """Null `embedding_pending` on every row."""
        with self.transaction():
            _node.swap_abort_sqlite(self._db)

    @contextmanager
    def drain_lock(
            self, store: str | None = None) -> Iterator[bool]:
        """Always yields True on SQLite (single-process by definition).

        SQLite drains are gated by the process-global fcntl
        `drain.lock` file at the queue layer (`src/memman/drain_lock.py`),
        not at the Backend level. This verb is a no-op for Backend
        Protocol parity; Postgres opens a dedicated connection with
        keepalives and acquires a per-store advisory lock.
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

        The session builds its embedding matrix from the live
        database, so there is no stored artifact whose embedding model
        could disagree with the caller's.
        """
        session = SqliteRecallSession(db=self._db)
        try:
            yield session
        finally:
            session.close()

    def storage_summary(self) -> dict[str, Any]:
        return _db.storage_summary(self._db)

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
        - The rank-1 form is the only one that reads the content
          table: `pragma integrity_check` and FTS5's own default
          `'integrity-check'` both pass on an index whose terms have
          drifted from the rows they index. Measured on a store
          edited behind the index -- both blind, rank 1 raises.
        - It scans every indexed row, so it belongs here in the
          `doctor` path and never on the recall path.
        - The probe needs a write transaction, so a read-only handle
          or a busy writer makes it raise without saying anything
          about the index. Those arrive as `OperationalError` while
          real drift arrives as `DatabaseError`, which is why the
          two are caught separately -- reporting "not run" beats
          reporting corruption that is not there.
        """
        row = self._db._query('pragma integrity_check').fetchone()
        result = row[0] if row else 'unknown'
        if result != 'ok':
            return {'ok': False, 'detail': result}
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

    def introspect_columns(self, table: str) -> set[str]:
        from memman.store.backend import _check_identifier
        _check_identifier(table)
        rows = self._db._query(
            f'pragma table_info({table})').fetchall()
        return {row[1] for row in rows}

    def start_run(self) -> int | None:
        """No-op: drain hangs are observable at the foreground prompt.
        """
        return None

    def beat_run(self, run_id: int | None) -> None:
        """No-op for SQLite mode."""
        return

    def finish_run(self, run_id: int | None) -> None:
        """No-op for SQLite mode."""
        return

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        """No-op: SQLite drain has no per-store worker_runs table."""
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


def open_ro_db(sdir: str) -> DB:
    """Open the SQLite store at `sdir` in read-only mode.

    cli-internal helper that puts the only `open_read_only` literal
    inside the SQLite backend module. Callers that need a raw DB
    handle reach this instead of importing
    `store.db.open_read_only` directly.
    """
    return _db.open_read_only(sdir)


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
    """Remove the SQLite store directory for `store` if it exists."""
    sdir = _db.store_dir(data_dir, store)
    if Path(sdir).is_dir():
        shutil.rmtree(sdir)


_SQLITE_MIGRATOR_FEATURES = BackendFeatures(
    supports_edges=True,
    supports_oplog=True,
    supports_reembed=True,
    supports_drain_heartbeat=False,
    supports_filesystem_artifacts=True,
    supports_dry_run=True,
    accepted_embedding_dtypes=frozenset({'float32', 'float64'}))


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
    snapshot_features: ClassVar[BackendFeatures] = _SQLITE_MIGRATOR_FEATURES

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
        # Percent-encode, and probe with a read that forces the header:
        # `connect` is lazy, so corrupt bytes surface at the first
        # statement, and an unescaped `#` or `?` in the path would
        # silently open a different file read-write.
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
        except sqlite3.Error as exc:
            raise MigrateError(
                f'cannot read sqlite store {store!r} at {path}:'
                f' {exc}') from exc
        if n == 0 and fp is None:
            raise MigrateError(
                f'sqlite store {store!r} is empty (no insights,'
                f' no embed fingerprint); nothing to migrate')

    def preflight_target(self, store: str) -> None:
        sanitize_identifier(
            store, max_len=63, allowed_chars=r'[A-Za-z0-9_]')
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
select id, content, category, importance, entities,
       source, access_count, keywords, summary, semantic_facts,
       last_accessed_at, embedding,
       linked_at, enriched_at, created_at, updated_at,
       deleted_at, prompt_version, model_id, embedding_model,
       embedding_pending, session_id, queue_uuid,
       corroboration_count, superseded_by
from insights
order by id
""").fetchall()
            insights: list[MigrateInsight] = []
            pending: list[PendingReembed] = []
            for r in rows:
                emb = deserialize_vector(r[11]) if r[11] else None
                insights.append(MigrateInsight(
                    id=r[0], content=r[1], category=r[2],
                    importance=int(r[3]),
                    entities=json.loads(r[4]) if r[4] else [],
                    source=r[5], access_count=int(r[6]),
                    keywords=json.loads(r[7]) if r[7] else None,
                    summary=r[8],
                    semantic_facts=(
                        json.loads(r[9]) if r[9] else None),
                    last_accessed_at=(
                        parse_timestamp(r[10]) if r[10] else None),
                    embedding=emb,
                    linked_at=(
                        parse_timestamp(r[12]) if r[12] else None),
                    enriched_at=(
                        parse_timestamp(r[13]) if r[13] else None),
                    created_at=parse_timestamp(r[14]),
                    updated_at=parse_timestamp(r[15]),
                    deleted_at=(
                        parse_timestamp(r[16]) if r[16] else None),
                    prompt_version=r[17], model_id=r[18],
                    embedding_model=r[19],
                    session_id=r[21], queue_uuid=r[22],
                    corroboration_count=int(r[23]),
                    superseded_by=r[24]))
                if r[20] is not None:
                    pv = deserialize_vector(r[20])
                    if pv is not None:
                        pending.append(PendingReembed(
                            insight_id=r[0], vector=pv))

            edge_rows = conn.execute("""
select source_id, target_id, edge_type, weight,
       metadata, created_at
from edges
order by source_id, target_id, edge_type
""").fetchall()
            edges = [
                MigrateEdge(
                    source_id=e[0], target_id=e[1],
                    edge_type=e[2], weight=float(e[3]),
                    metadata=json.loads(e[4]) if e[4] else {},
                    created_at=parse_timestamp(e[5]))
                for e in edge_rows]

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

        swap_state = None
        if 'embed_swap_state' in meta_dict:
            try:
                dim = int(meta_dict.get(
                    'embed_swap_target_dim', '0'))
            except ValueError:
                dim = 0
            swap_state = SwapState(
                target_provider=meta_dict.get(
                    'embed_swap_target_provider', ''),
                target_model=meta_dict.get(
                    'embed_swap_target_model', ''),
                target_dim=dim,
                cursor=meta_dict.get('embed_swap_cursor') or None,
                started_at=None)

        stripped_meta = {
            k: v for k, v in meta_dict.items()
            if not k.startswith('embed_swap_')}

        return MigrationPayload(
            payload_version=PAYLOAD_VERSION,
            fingerprint=fingerprint,
            embedding_dim=fingerprint.dim,
            embedding_dtype='float64',
            insights=insights,
            edges=edges,
            oplog=oplog,
            embedding_pending=pending,
            swap_state=swap_state,
            meta=stripped_meta)

    def apply(
            self, store: str, payload: MigrationPayload) -> None:
        if payload.payload_version != PAYLOAD_VERSION:
            raise MigrateError(
                f'payload version {payload.payload_version} does not'
                f' match this build ({PAYLOAD_VERSION}); re-gather'
                ' with the matching memman')
        if payload.embedding_dtype not in (
                self.snapshot_features.accepted_embedding_dtypes):
            raise MigrateError(
                f'sqlite cannot accept embedding_dtype'
                f' {payload.embedding_dtype!r}; accepted:'
                f' {sorted(self.snapshot_features.accepted_embedding_dtypes)}')

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
                        ins.id, ins.content, ins.category,
                        ins.importance,
                        json.dumps(ins.entities),
                        ins.source, ins.access_count,
                        json.dumps(ins.keywords)
                        if ins.keywords is not None else None,
                        ins.summary,
                        json.dumps(ins.semantic_facts)
                        if ins.semantic_facts is not None
                        else None,
                        format_timestamp(ins.last_accessed_at)
                        if ins.last_accessed_at else None,
                        emb_blob,
                        format_timestamp(ins.linked_at)
                        if ins.linked_at else None,
                        format_timestamp(ins.enriched_at)
                        if ins.enriched_at else None,
                        format_timestamp(ins.created_at),
                        format_timestamp(ins.updated_at),
                        format_timestamp(ins.deleted_at)
                        if ins.deleted_at else None,
                        ins.prompt_version, ins.model_id,
                        ins.embedding_model,
                        ins.session_id, ins.queue_uuid,
                        ins.corroboration_count, ins.superseded_by))
                if insight_rows:
                    conn.executemany(
                        'insert into insights ('
                        ' id, content, category, importance,'
                        ' entities, source, access_count,'
                        ' keywords, summary, semantic_facts,'
                        ' last_accessed_at, embedding,'
                        ' linked_at, enriched_at, created_at,'
                        ' updated_at, deleted_at, prompt_version,'
                        ' model_id, embedding_model, session_id,'
                        ' queue_uuid, corroboration_count,'
                        ' superseded_by)'
                        ' values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                        ' ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        insight_rows)

                edge_rows = [(
                    e.source_id, e.target_id, e.edge_type, e.weight,
                    json.dumps(e.metadata),
                    format_timestamp(e.created_at))
                    for e in payload.edges]
                if edge_rows:
                    conn.executemany(
                        'insert into edges ('
                        ' source_id, target_id, edge_type, weight,'
                        ' metadata, created_at)'
                        ' values (?, ?, ?, ?, ?, ?)',
                        edge_rows)

                max_oplog_id = 0
                for op in payload.oplog:
                    desired_id = op.legacy_id or op.id
                    row = (
                        desired_id, op.operation, op.insight_id,
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
                        max_oplog_id = max(max_oplog_id, desired_id)
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

                for p in payload.embedding_pending:
                    blob = serialize_vector(p.vector)
                    conn.execute(
                        'update insights'
                        ' set embedding_pending = ?'
                        ' where id = ?',
                        (blob, p.insight_id))

                meta_rows = list(payload.meta.items())
                if payload.swap_state:
                    s = payload.swap_state
                    meta_rows.extend([
                        ('embed_swap_target_provider',
                         s.target_provider),
                        ('embed_swap_target_model', s.target_model),
                        ('embed_swap_target_dim', str(s.target_dim)),
                        ('embed_swap_cursor', s.cursor or ''),
                        ])
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
                if isinstance(exc, MigrateError):
                    raise
                raise MigrateError(
                    f'sqlite apply for store {store!r} failed:'
                    f' {type(exc).__name__}: {exc}') from exc
        finally:
            db.close()

    def archive(self, store: str, data_dir: str) -> Artifact:
        from memman.setup.archive import archive_store_dir
        path = archive_store_dir(data_dir, store)
        if path is None:
            return Artifact(
                kind='none', location=None,
                metadata={'reason': 'no source dir to archive'})
        return Artifact(
            kind='filesystem',
            location=str(path), metadata={})

    def drop(self, store: str) -> None:
        drop_sqlite_store(store, self.data_dir)
