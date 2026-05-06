"""Backend Protocol surface.

Defines `Cluster`, `Backend`, the four sub-store Protocols (`NodeStore`,
`EdgeStore`, `MetaStore`, `Oplog`), `RecallSession`, and `QueueBackend`.
SQLite implements them in `store/sqlite.py`; Postgres in `store/postgres.py`.

Distributed-shaping commitments baked into this Protocol surface:

1. **Timestamp ownership at the boundary.** `nodes.insert(insight)`,
   `edges.upsert(edge)`, `oplog.log(...)`, `nodes.stamp_linked(id)`,
   `nodes.stamp_enriched(id)` accept no `created_at` argument.
   Backends stamp these server-side -- SQLite via Python `datetime.now`,
   Postgres via `now()`. Pipeline code never produces a timestamp that
   lands in a database write.

2. **`Backend.write_lock(name)` is a Protocol verb.** SQLite's
   implementation is a no-op (`BEGIN IMMEDIATE` already serializes
   per-process). Postgres uses `pg_advisory_xact_lock`.

3. **`Backend.transaction()` nesting contract.** Nested calls reuse
   the outer transaction (SAVEPOINT-like or no-op). Required by the
   existing `apply_all -> auto_prune` nested pattern.

4. **`Backend.readonly_context()` semantics.** SQLite spawns a separate
   read-only connection. Postgres MUST yield a connection in autocommit
   mode so reader threads see main-thread commits as they land.
"""

import re
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol, Self, runtime_checkable

from memman.store.errors import ConfigError
from memman.store.model import Edge, EnrichmentCoverage, Id, Insight
from memman.store.model import IntegrityReport, NodeStats, OpLogEntry
from memman.store.model import OpLogStats, ProvenanceCount, QueueRow
from memman.store.model import QueueStats, ReembedRow, WorkerRun

_VALID_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _check_identifier(name: str) -> None:
    """Reject SQL identifiers that are not safe to interpolate.

    Some SQL constructs (`pragma table_info(<table>)` on SQLite, schema
    and table names on Postgres) cannot be parameterized; the value is
    interpolated as a literal. Reject anything that is not a plain
    identifier so an unsanitized name cannot inject DDL. Shared by
    both backends so the validation contract is one place.
    """
    if not _VALID_IDENTIFIER_RE.match(name):
        raise ConfigError(f'invalid SQL identifier: {name!r}')


@runtime_checkable
class NodeStore(Protocol):
    """Insight CRUD + lifecycle + statistics."""

    def insert(self, ins: Insight) -> None:
        """Insert a new insight. Backend stamps timestamps server-side.
        """
        ...

    def get(self, id: Id) -> Insight | None:
        """Return one active insight by id, or None when absent."""
        ...

    def get_include_deleted(self, id: Id) -> Insight | None:
        """Return one insight by id, including soft-deleted rows."""
        ...

    def get_many(self, ids: Sequence[Id]) -> list[Insight]:
        """Return active insights for the given ids, in input order.

        Missing ids are silently dropped from the result. Used by the
        bfs caller to hydrate insights for the bounded neighborhood
        returned by `EdgeStore.get_neighborhood`.
        """
        ...

    def query(
            self, *, keyword: str = '', category: str = '',
            min_importance: int = 0, source: str = '',
            limit: int = 20) -> list[Insight]:
        """Filtered insight query, ordered by importance/created_at desc.
        """
        ...

    def soft_delete(
            self, id: Id, *, tolerate_missing: bool = False) -> bool:
        """Soft-delete an insight and remove its edges."""
        ...

    def update_entities(self, id: Id, entities: list[str]) -> None:
        """Replace the entities array for an insight."""
        ...

    def update_enrichment(
            self, id: Id, *, keywords: list[str], summary: str,
            semantic_facts: list[str]) -> None:
        """Update LLM enrichment columns for an insight."""
        ...

    def increment_access_count(self, id: Id) -> None:
        """Bump access_count and refresh last_accessed_at."""
        ...

    def refresh_effective_importance(self, id: Id) -> float:
        """Recompute and persist effective_importance for one insight."""
        ...

    def get_retention_candidates(
            self, *, threshold: float,
            limit: int) -> tuple[list[dict[str, Any]], int]:
        """Return non-immune insights sorted by effective_importance asc.
        """
        ...

    def count_active(self) -> int:
        """Count non-deleted insights."""
        ...

    def count_total(self) -> int:
        """Count all insights, including soft-deleted."""
        ...

    def has_active_with_source(self, source: str) -> bool:
        """Return True if any active insight uses the given source."""
        ...

    def iter_for_reembed(
            self, cursor: Id, batch: int) -> list[ReembedRow]:
        """Return a batch of (id, content, embedding_model, blob_length).
        """
        ...

    def count_orphans(self) -> tuple[int, int]:
        """Return (orphan_count, total_active)."""
        ...

    def provenance_distribution(self) -> list[ProvenanceCount]:
        """Return (prompt_version, model_id, count) for active rows."""
        ...

    def auto_prune(
            self, *, max_insights: int,
            exclude_ids: list[Id] | None = None) -> int:
        """Soft-delete the lowest-EI non-immune insights when over cap.
        """
        ...

    def boost_retention(self, id: Id) -> None:
        """Boost an insight's retention: access_count +3."""
        ...

    def get_recent_in_window(
            self, *, exclude_id: Id, window_hours: float,
            limit: int) -> list[Insight]:
        """Return recent active insights inside a time window."""
        ...

    def get_latest_by_source(
            self, *, source: str, exclude_id: Id) -> Insight | None:
        """Return the most-recent active insight for a given source."""
        ...

    def get_recent_active(
            self, *, exclude_id: Id, limit: int) -> list[Insight]:
        """Return the N most recent active insights, any source."""
        ...

    def get_all_active(self) -> list[Insight]:
        """Return all active insights ordered by created_at desc."""
        ...

    def stats(self) -> NodeStats:
        """Aggregate statistics."""
        ...

    def update_embedding(
            self, id: Id, vec: list[float], model: str) -> None:
        """Persist an embedding vector + its model name.

        Backends bind the vector to their native storage type
        (BLOB on SQLite via `serialize_vector`; pgvector(512) on
        Postgres). `serialize_vector` / `deserialize_vector` stay
        confined to the SqliteBackend.
        """
        ...

    def get_embedding(self, id: Id) -> bytes | None:
        """Return the raw embedding blob for an active insight."""
        ...

    def get_all_embeddings(self) -> list[tuple[Id, str, bytes]]:
        """Return all (id, content, blob) triples for active insights.

        Used by snapshot writes and the `BaseNodeStore` defaults for
        `get_without_embedding` / `iter_embeddings_as_vecs`. Pipeline
        and recall paths prefer `iter_embeddings_as_vecs` directly.
        """
        ...

    def iter_embeddings_as_vecs(
            self) -> Iterator[tuple[Id, list[float]]]:
        """Yield (id, vec) for every active insight with an embedding.

        SQLite implementation deserializes the blob inside the
        backend so callers see only `list[float]`. Postgres implements
        this as a server-side cursor over `select id, embedding ...`,
        binding pgvector(512) directly.

        Use `dict(backend.nodes.iter_embeddings_as_vecs())` when a
        cache is needed; iterate directly when memory matters.
        """
        ...

    def embedding_stats(self) -> tuple[int, int]:
        """Return (total_active, embedded_count)."""
        ...

    def enrichment_coverage(self) -> EnrichmentCoverage:
        """Per-field NULL counts on the enrichment columns.

        Returns total_active + missing_{embedding,keywords,summary,
        semantic_facts}. Doctor consumes this for the
        enrichment-coverage check.
        """
        ...

    def embedding_size_distribution(self) -> dict[int, int]:
        """Histogram of stored embedding sizes for active insights.

        SQLite: keyed by `LENGTH(embedding)` byte count. Postgres:
        keyed by `vector_dims(embedding)` (the pgvector dim). A
        healthy store has one bucket. More than one bucket means a
        dim mismatch -- the doctor consistency check flags this.
        """
        ...

    def get_without_embedding(self, *, limit: int = 100) -> list[Insight]:
        """Return active insights that lack embeddings."""
        ...

    def stamp_linked(self, id: Id) -> None:
        """Mark an insight as linked. Backend stamps `linked_at` now.
        """
        ...

    def stamp_enriched(self, id: Id) -> None:
        """Mark an insight as enriched. Backend stamps `enriched_at` now.
        """
        ...

    def get_pending_link_ids(self, *, limit: int) -> list[Id]:
        """Return ids of insights with NULL linked_at."""
        ...

    def get_active_ids(self) -> list[Id]:
        """Return all active insight ids in creation order."""
        ...

    def count_pending_links(self) -> int:
        """Count active insights with NULL linked_at."""
        ...

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        """Clear enriched_at and linked_at for the given ids."""
        ...

    def clear_linked_at(self) -> None:
        """Set linked_at to NULL for every active insight."""
        ...

    def review_content_quality(
            self, *, limit: int) -> list[dict[str, Any]]:
        """Return active insights flagged by content-quality checks."""
        ...


@runtime_checkable
class EdgeStore(Protocol):
    """Edge CRUD + traversal."""

    def upsert(self, edge: Edge) -> None:
        """Insert or merge an edge, keeping the higher weight.

        Backend stamps `created_at` server-side.
        """
        ...

    def by_node(self, node_id: Id) -> list[Edge]:
        """All edges where node_id is source or target."""
        ...

    def by_node_and_type(
            self, node_id: Id, edge_type: str) -> list[Edge]:
        """Edges for a node filtered by type."""
        ...

    def by_source_and_type(
            self, source_id: Id, edge_type: str) -> list[Edge]:
        """Edges where source_id is source, filtered by type."""
        ...

    def find_with_entity(
            self, entity: str, *, exclude_id: Id,
            limit: int) -> list[Id]:
        """Insight ids that have the given entity."""
        ...

    def count_with_entity(
            self, entity: str, *, exclude_id: Id) -> int:
        """Count distinct insights that contain the entity."""
        ...

    def all(self) -> list[Edge]:
        """Return every edge in the graph."""
        ...

    def delete_by_node(self, node_id: Id) -> None:
        """Remove all edges referencing a node."""
        ...

    def delete_auto_for_node(
            self, node_id: Id, edge_type: str) -> None:
        """Delete auto-generated edges for a node, keeping manual."""
        ...

    def delete_auto_by_type(self, edge_type: str) -> None:
        """Delete auto-generated edges globally for reindex."""
        ...

    def count_auto_by_type(self, edge_type: str) -> int:
        """Count auto-generated edges by type using reindex filters."""
        ...

    def delete_low_weight_temporal_proximity(
            self, *, min_weight: float) -> None:
        """Delete temporal-proximity edges below the weight floor."""
        ...

    def count_low_weight_temporal_proximity(
            self, *, min_weight: float) -> int:
        """Count temporal-proximity edges below the weight floor."""
        ...

    def get_weight(
            self, source_id: Id, target_id: Id,
            edge_type: str) -> float | None:
        """Return one directed edge's weight, or None when absent."""
        ...

    def count_dangling_by_type(self) -> dict[str, int]:
        """{edge_type: count} for edges referencing missing/deleted nodes.
        """
        ...

    def degree_distribution(self) -> dict[Id, int]:
        """{insight_id: total_degree} for all active insights."""
        ...

    def get_neighborhood(
            self, seed_id: Id, *, depth: int,
            edge_filter: str = '') -> list[tuple[Id, int, str]]:
        """Bounded BFS neighborhood from one seed.

        Returns `(neighbor_id, hop, via_edge_type)` triples ordered by
        traversal arrival. Depth bound is enforced inside the verb so
        Postgres can emit just the bounded subgraph via a recursive CTE
        instead of streaming the full edge set to the client.

        `edge_filter`: when non-empty, only edges of that `edge_type`
        are followed.
        """
        ...


@runtime_checkable
class MetaStore(Protocol):
    """Key-value metadata table."""

    def get(self, key: str) -> str | None:
        """Read a meta value, or None when absent."""
        ...

    def set(self, key: str, value: str) -> None:
        """Write a meta value."""
        ...

    def delete(self, key: str) -> None:
        """Remove a meta key entirely. No-op when absent."""
        ...

    def keys(self) -> list[str]:
        """Return all meta keys in arbitrary order."""
        ...


@runtime_checkable
class Oplog(Protocol):
    """Operation log."""

    def log(
            self, *, operation: str, insight_id: Id, detail: str,
            before: dict[str, Any] | None = None,
            after: dict[str, Any] | None = None) -> None:
        """Record one operation. Backend stamps `created_at` now.

        Insert-only on both backends; trimming is performed by
        `maintenance_step`. `before` / `after` carry pre/post
        insight content for reconcile / replace / forget /
        auto_prune so the oplog alone is forensic-complete.
        """
        ...

    def maintenance_step(self) -> None:
        """Per-store backend maintenance pass (vacuum/trim)."""
        ...

    def trim_by_age(self, *, retention_days: int = 180) -> int:
        """Delete oplog rows older than retention_days. Returns count."""
        ...

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        """Return the most-recent N oplog entries."""
        ...

    def delta_coverage(self) -> tuple[int, int]:
        """Return (total_rows, rows_with_before_or_after)."""
        ...

    def stats(self, *, since: str = '') -> OpLogStats:
        """Operation counts + never-accessed insight count."""
        ...


@runtime_checkable
class RecallSession(Protocol):
    """Read-side handle for the recall pipeline.

    `Backend.recall_session()` yields one of these in a context. The
    session owns the read-side cache (snapshot, in-memory matrices,
    or a postgres connection in autocommit mode) for the duration of
    a single recall request. Closes deterministically on context exit.

    `vector_anchors` is the high-level verb the pipeline consumes
    inside the `with recall_session()` block. SQLite serves it from
    the snapshot's pre-loaded `embeddings` (or a lazily-populated
    `_embed_cache` on the snapshot-miss fallback path). Postgres
    serves it via HNSW with `embedding <=>`. The pipeline still
    reads `session.snapshot` directly when present for the broader
    recall flow (keyword_anchors, neighbors, hydrate, similarity,
    causal_neighbors).
    """

    snapshot: Any

    def vector_anchors(
            self, query_vec: list[float], *, k: int = 10,
            min_sim: float = 0.0) -> list[tuple[Id, float]]:
        """Top-k (id, similarity) anchors. Cosine in [-1, 1]."""
        ...


@runtime_checkable
class Backend(Protocol):
    """Per-store handle exposing the verb surface.

    Yielded by `Cluster.open(name)`. Owns its own connection (SQLite
    file / Postgres connection from a pool). Sub-stores
    (`nodes`/`edges`/`meta`/`oplog`) are bound to the same connection
    so they share the active transaction and read-after-write
    visibility.
    """

    nodes: NodeStore
    edges: EdgeStore
    meta: MetaStore
    oplog: Oplog

    @property
    def path(self) -> str:
        """Backend-specific identifier (file path on SQLite, DSN+schema
        on Postgres). Used for log lines and `memman status`.
        """
        ...

    def transaction(self) -> AbstractContextManager[None]:
        """Run a block inside a write transaction.

        Nesting reuses the outer transaction (SAVEPOINT or no-op);
        nested rollback is unsupported. Required by the existing
        `apply_all -> auto_prune` pattern.
        """
        ...

    def write_lock(
            self, name: str) -> AbstractContextManager[None]:
        """Acquire a named exclusive write lock for the duration of
        the block.

        SQLite: no-op (`BEGIN IMMEDIATE` already serializes
        per-process). Postgres: `pg_advisory_xact_lock` (transaction-
        scoped, reentrant within the same session). Used by
        `reindex_auto_edges` and `PostgresNodeStore.auto_prune` to
        serialize read-then-write paths against concurrent writers.
        Wrong primitive for sweeps that span minutes-to-hours; see
        `reembed_lock` for that case.
        """
        ...

    def drain_lock(
            self, store: str | None = None
            ) -> AbstractContextManager[bool]:
        """Acquire a per-store advisory drain lock.

        SQLite: yields True (the SQLite drain path is gated by the
        process-global fcntl `drain.lock` file; this verb is a no-op
        for backend-Protocol parity). Postgres: opens a dedicated
        connection outside any pool with `keepalives_idle=30`, runs
        `pg_try_advisory_lock`. Yields True when acquired, False
        otherwise. Lock auto-releases on connection close so a
        hung worker is detected by TCP keepalives rather than
        holding the lock indefinitely.
        """
        ...

    def reembed_lock(
            self, name: str) -> AbstractContextManager[bool]:
        """Acquire a session-scoped sweep lock for hours-long batch work.

        SQLite: yields True (single-process). Postgres:
        `pg_try_advisory_lock` on a dedicated connection outside any
        pool, with TCP keepalives so a hung sweep is detected by the
        kernel. Yields True when acquired, False otherwise (caller
        prints "another <name> in progress" and exits non-zero).
        Used by `embed reembed` and `graph rebuild`; do NOT use
        `write_lock` for these because `pg_advisory_xact_lock` would
        pin a transaction for the entire sweep duration and block
        autovacuum.
        """
        ...

    def readonly_context(
            self) -> AbstractContextManager['Backend']:
        """Yield a read-only Backend bound to a separate connection.

        SQLite: opens the database with `mode=ro`. Postgres: yields
        a connection in autocommit mode so writes from another
        thread/process are visible immediately.
        """
        ...

    def recall_session(
            self) -> AbstractContextManager[RecallSession]:
        """Yield a `RecallSession` bound to a read-side cache."""
        ...

    def storage_summary(self) -> dict[str, Any]:
        """Backend-specific storage info for `memman status`."""
        ...

    def integrity_check(self) -> dict[str, Any]:
        """Run a backend-specific integrity probe for `memman doctor`.

        SQLite: `pragma integrity_check`. Postgres: connectivity
        probe + schema-presence verification (HNSW index validity is
        checked separately at reindex time).

        Returns a dict shaped `{'ok': bool, 'detail': str}` -- doctor
        composes this with sub-store verbs to assemble its overall
        report.
        """
        ...

    def introspect_columns(self, table: str) -> set[str]:
        """Return the column names on a named table in this store.

        SQLite: `pragma table_info(<table>)`. Postgres:
        `information_schema.columns` filtered by the store's schema.
        Returns an empty set when the table does not exist (rather
        than raising) so doctor's schema-columns check can compute a
        symmetric difference cleanly across backends.
        """
        ...

    def close(self) -> None:
        """Close the backend's connection."""
        ...

    def __enter__(self) -> Self:
        """Return self so `with cluster.open(...) as backend:` works."""
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the backend on context exit."""
        ...


@runtime_checkable
class Cluster(Protocol):
    """Per-data-dir entry point.

    `open_cluster()` returns one of these. Cluster opens / creates
    per-store backends, lists stores, and is the natural home for
    multi-store coordination (drop, list).
    """

    def open(self, *, store: str, data_dir: str) -> Backend:
        """Open or create the named store and return its Backend."""
        ...

    def open_read_only(
            self, *, store: str, data_dir: str) -> Backend:
        """Open the named store in read-only mode (no migration)."""
        ...

    def list_stores(self, *, data_dir: str) -> list[str]:
        """Return sorted store names under data_dir."""
        ...

    def drop_store(self, *, store: str, data_dir: str) -> None:
        """Drop a named store (delete files / drop schema)."""
        ...

    def close(self) -> None:
        """Release any cluster-level resources (connection pool)."""
        ...


@runtime_checkable
class QueueBackend(Protocol):
    """Cross-store work queue."""

    def enqueue(
            self, *, store: str, op: str, payload: str) -> int:
        """Append one row, return its id."""
        ...

    def claim_batch(self, *, limit: int) -> list[QueueRow]:
        """Claim up to `limit` rows for processing."""
        ...

    def mark_done(self, ids: Sequence[int]) -> None:
        """Mark rows as completed."""
        ...

    def mark_failed(
            self, ids: Sequence[int], *, error: str) -> None:
        """Mark rows as failed (with retry accounting)."""
        ...

    def purge_store(self, store: str) -> int:
        """Delete all rows for a store, return deleted count."""
        ...

    def stats(self) -> QueueStats:
        """Aggregate queue statistics."""
        ...

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        """Return recent worker drain runs."""
        ...

    def start_run(self) -> int:
        """Insert a new in-progress `worker_runs` row and return its id.

        The row is stamped with `started_at = now()` and
        `last_heartbeat_at = now()` server-side; `ended_at` stays
        NULL until the drain completes. Used by the drain loop and
        consumed by `memman doctor` for hung-worker detection.
        """
        ...

    def beat_run(self, run_id: int) -> None:
        """Advance `last_heartbeat_at = now()` on a specific run.

        Called inline from the drain loop (one update per row
        processed) so that a worker stuck mid-row is detectable
        within a few enrichment cycles.
        """
        ...

    def integrity_report(self) -> IntegrityReport:
        """Aggregate integrity findings used by `memman doctor`."""
        ...
