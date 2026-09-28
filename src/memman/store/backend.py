"""Backend Protocol surface.

Defines `Backend`, the three sub-store Protocols (`NodeStore`,
`MetaStore`, `Oplog`), and `RecallSession`. SQLite
implements them in `store/sqlite.py`; Postgres in `store/postgres.py`.
The work queue is process-global and SQLite-only (see
`memman.queue`).

Distributed-shaping commitments baked into this Protocol surface:

1. **Timestamp ownership at the boundary.** `nodes.insert(insight)`,
   `oplog.log(...)`, `nodes.stamp_enrich_attempted(id)`,
   `nodes.stamp_enriched(id)` accept no `created_at` argument.
   Backends stamp these server-side -- SQLite via Python `datetime.now`,
   Postgres via `now()`. Pipeline code never produces a timestamp that
   lands in a database write.

2. **`Backend.transaction()` nesting contract.** Nested calls reuse
   the outer transaction (SAVEPOINT-like or no-op). Required by the
   nested `apply_all` write pattern.

"""

import re
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from memman.store.errors import ConfigError
from memman.store.model import EnrichmentCoverage, Id, Insight, NodeStats
from memman.store.model import OpLogEntry, OpLogStats, ProvenanceCount
from memman.store.model import WorkerRun

if TYPE_CHECKING:
    from memman.embed.fingerprint import Fingerprint

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

    def resolve_id(self, id_or_prefix: str) -> str:
        """Resolve an exact id or an unambiguous prefix to a full id.

        Parameters
        ----------
        id_or_prefix : str
            A full insight id or a prefix of one.

        Returns
        -------
        str
            The full id when the argument is an exact id or a prefix
            that matches exactly one row (including deleted and
            replaced rows). Returns the argument unchanged when no
            row matches, so the caller's existing not-found path fires.

        Raises
        ------
        ValueError
            When the prefix matches two or more rows; the message names
            the prefix and the match count.

        Notes
        -----
        - Exact match takes priority over prefix match: a full id that
          is also a prefix of another row resolves to itself.
        - Resolution scans all rows including deleted and replaced
          ones; each command's own get applies its state filter.
        """
        ...

    def query(
            self, *, keyword: str = '', limit: int = 20) -> list[Insight]:
        """Current insights holding every keyword word, newest first.
        """
        ...

    def soft_delete(self, id: Id) -> bool:
        """Soft-delete a non-deleted insight.

        Returns False when the row is missing or already deleted; a
        replaced row may still be deleted.
        """
        ...

    def mark_replaced(self, predecessor_id: Id, successor_id: Id) -> bool:
        """Point a current insight at its successor.

        Parameters
        ----------
        predecessor_id : Id
            The row being replaced; must be neither deleted nor
            already replaced.
        successor_id : Id
            The row that replaces it; not checked, since the pipeline
            writes the pointer before the successor row exists.

        Returns
        -------
        bool
            True when the pointer was written; False when the
            predecessor is not current, and the caller degrades to a
            plain add.

        Notes
        -----
        - The guard makes a row replaced at most once, which rules
          out forks in the chain.
        - A replaced row leaves every active read exactly as a
          soft-deleted one does; `get_include_deleted` and
          `has_row_with_queue_uuid` still see it.
        """
        ...

    def predecessors(self, successor_id: Id) -> list[Insight]:
        """Every row whose `replaced_by` names `successor_id`.

        Deleted rows included, oldest first; the history walk's
        backward step.
        """
        ...

    def replacement_integrity(self) -> dict[str, list[Id]]:
        """The three pointer populations a healthy store leaves empty.

        Keys: `dangling` (pointer at an id absent from the table),
        `self_pointer`, `unterminated` (a chain that never
        reaches a row without a pointer, so a cycle). A successor with
        two predecessors is a join, not a defect. The column has no
        foreign key, so the doctor check over this verb is the only
        pointer enforcement.
        """
        ...

    def update_enrichment(self, id: Id, *, summary: str) -> None:
        """Store the enrichment summary for an insight."""
        ...

    def count_active(self) -> int:
        """Count current insights: neither deleted nor replaced."""
        ...

    def count_total(self) -> int:
        """Count all insights, including soft-deleted."""
        ...

    def has_row_with_queue_uuid(self, queue_uuid: str) -> bool:
        """Return True if any insight, forgotten ones included, carries it.

        The idempotency check for queue replays; runs unconditionally
        for every drained row and answers "did this write land", so a
        replaced or forgotten row counts. Backends implement it in
        SQL so a null `queue_uuid` can never match.
        """
        ...

    def provenance_distribution(self) -> list[ProvenanceCount]:
        """Return (prompt_version, count) for active rows."""
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

    def embedding_stats(self) -> tuple[int, int]:
        """Return (total_active, embedded_count)."""
        ...

    def enrichment_coverage(self) -> EnrichmentCoverage:
        """Per-field NULL counts on the enrichment columns.

        Returns
        -------
        EnrichmentCoverage
            `total_active`, `missing_embedding`, `missing_summary`
            and `stranded` over active rows; doctor's
            enrichment-coverage check reads it.
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

    def stamp_enrich_attempted(self, id: Id) -> None:
        """Mark an insight enrich-attempted. Backend stamps
        `enrich_attempted_at` now.
        """
        ...

    def stamp_enriched(
            self, id: Id, *,
            prompt_version: str | None = None) -> None:
        """Mark an insight as enriched. Backend stamps `enriched_at` now.

        Parameters
        ----------
        id : Id
            Row to stamp.
        prompt_version : str or None, default None
            The `compute_prompt_version()` key this enrichment ran
            under. Pass it from the driver that knows the active
            config (the enrich/rebuild path) so the re-enrichment
            clears the row's staleness, not just its timestamp. The
            write path omits it, having set it at insert.
        """
        ...

    def get_pending_enrich_ids(self, *, limit: int) -> list[Id]:
        """Return ids of insights with NULL enrich_attempted_at."""
        ...

    def get_active_ids(self) -> list[Id]:
        """Return all active insight ids in creation order."""
        ...

    def count_pending_enrich(self) -> int:
        """Count active insights with NULL enrich_attempted_at."""
        ...

    def get_unenriched_attempted_ids(self, *, limit: int) -> list[Id]:
        """Return ids of attempted-but-unenriched (stranded) active
        insights.
        """
        ...

    def iter_stale_insight_ids(self, active_pv: str) -> list[Id]:
        """Return ids of the active insights `enrich --stale-only` replays.

        Stale means a present `prompt_version` that differs from
        `active_pv`, or a stranded row (attempted, never enriched)
        whatever its key. There is no model argument: `active_pv`
        folds in the LLM model already, and that is the only model a
        rebuild re-runs.
        """
        ...

    def count_stale_insights(self, active_pv: str) -> int:
        """Count the rows `iter_stale_insight_ids` returns."""
        ...

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        """Clear enriched_at and enrich_attempted_at for the given ids."""
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
        `maintenance_step`. `before` carries the prior insight
        content on a replace or forget row, and `after` the new
        content on a remember, replace or target-gone row, so the
        oplog alone is forensic-complete.
        """
        ...

    def maintenance_step(self) -> None:
        """Per-store backend maintenance pass (vacuum/trim)."""
        ...

    def trim_by_age(self) -> int:
        """Delete oplog rows older than 180 days. Returns count."""
        ...

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        """Return the most-recent N oplog entries."""
        ...

    def stats(self, *, since: str = '') -> OpLogStats:
        """Operation counts plus the current insight count."""
        ...


@runtime_checkable
class RecallSession(Protocol):
    """Read-side handle for the recall pipeline.

    `Backend.recall_session()` yields one of these in a context. The
    session owns the read-side cache (an in-process embedding
    matrix, or a postgres connection in autocommit mode) for the
    duration of a single recall request. Closes deterministically on
    context exit.

    `vector_anchors` is the high-level verb the pipeline consumes
    inside the `with recall_session()` block. SQLite serves it from
    an in-process embedding matrix built once per session; Postgres
    serves it via HNSW with `embedding <=>`. Similarity for
    non-anchor nodes comes from `similarities` -- the pipeline never
    holds a whole-store embedding dict. `keyword_counts` is the same story
    for tokens: the pipeline never tokenizes the store.

    Notes
    -----
    - There is deliberately no persisted read cache behind this
      Protocol. A materialized snapshot shipped once and froze
      permanently, because its writer stopped above a row cap while
      its reader had no staleness check. Any future cache here needs
      a refresh trigger on every mutation path AND a reader-side
      validity check that detects drift, or it does not get to be
      persisted.
    """

    def vector_anchors(
            self, query_vec: list[float], *,
            k: int = 10) -> list[tuple[Id, float]]:
        """Top-k (id, similarity) anchors. Cosine in (0, 1].

        Notes
        -----
        - Positives only, matching `similarities`: a row pointing
          away from the query is not an entry point into the graph.
          The sign boundary is the ONLY floor here, and it is the
          only one that can be, because a fixed cosine means
          different things under different embedding models while an
          orthogonal row is orthogonal under all of them.
        - So a store with fewer than k positive-cosine rows returns
          fewer than k anchors, by design.
        """
        ...

    def similarities(
            self, query_vec: list[float]) -> dict[Id, float]:
        """Cosine of `query_vec` against stored embeddings.

        Parameters
        ----------
        query_vec : list[float]
            Query embedding.

        Returns
        -------
        dict[Id, float]
            Cosine in (0, 1] per id. Non-positive similarities are
            omitted, so a missing key means "not similar", and
            callers read it with `.get(id, 0.0)`.

        Notes
        -----
        - Computed where the vectors already live -- one matmul on
          SQLite, one `embedding <=>` query on Postgres -- so the
          pipeline never ships N x dim floats to compute N scalars.
        """
        ...

    def keyword_counts(
            self, query_tokens: set[str]) -> dict[Id, int]:
        r"""Distinct query tokens present in each active insight.

        Parameters
        ----------
        query_tokens : set[str]
            Tokens from `search.keyword.tokenize`, so each is
            `[a-zA-Z0-9]+` and none is a stopword.

        Returns
        -------
        dict[Id, int]
            Match count per active insight id, in `[1, len(tokens)]`.
            An id with no matching token is omitted, so callers read
            it with `.get(id, 0)`.

        Notes
        -----
        - The count is over the insight's content, the same set
          `keyword.insight_tokens` builds, and it is the numerator of
          `kw_score`. A backend that returns a
          different count changes `signals.keyword` and the rerank
          blend together.
        - NON-ASCII TEXT DIVERGES ON SQLITE, deliberately and
          measurably. `keyword._WORD_RE` is `[a-zA-Z0-9]+`, so it
          splits a run at any other character; FTS5 `unicode61`
          keeps a whole Unicode word. `naive` spelled with an
          i-diaeresis is one FTS term and two Python tokens. A stored
          row is affected only if it carries such a run, so the reach is narrow, but it is not nil. Postgres
          matches Python exactly, and by construction rather than by
          agreement: it stores the set `insight_tokens` built at
          write time. Closing the gap means changing
          `_WORD_RE`, which restales every stored `kw_tokens` set, so
          it is its own change with its own sweep -- not this one.
        - Counted where the text already lives -- k index probes on
          SQLite, one indexed query on Postgres -- so the pipeline
          never tokenizes the whole store to score one query, and
          neither backend tokenizes a row at recall time at all.
        """
        ...


@runtime_checkable
class Backend(Protocol):
    """Per-store handle exposing the verb surface.

    Yielded by `factory.open_backend(store, data_dir)`. Owns its own
    connection (SQLite file / Postgres connection from a pool). Sub-stores
    (`nodes`/`meta`/`oplog`) are bound to the same connection
    so they share the active transaction and read-after-write
    visibility.
    """

    nodes: NodeStore
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
        nested rollback is unsupported. Required because `apply_all`
        runs inside a caller-opened transaction.
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
        Used by `embed reembed` and `memman enrich`. Session-scoped
        rather than `pg_advisory_xact_lock`, which would pin a
        transaction for the entire sweep duration and block
        autovacuum.
        """
        ...

    def swap_lock(self) -> AbstractContextManager[bool]:
        """Acquire a session-scoped swap lock for the embedding swap flow.

        SQLite: yields True (single-process; `_require_stopped`
        already excludes the drain). Postgres:
        `pg_try_advisory_lock` on a dedicated `embed_swap:<schema>`
        key, mirroring `reembed_lock` so swaps and reembeds do not
        contend on the same key. Held continuously across
        prepare -> backfill -> cutover.
        """
        ...

    def swap_prepare(self, target_dim: int) -> None:
        """Add the `embedding_pending` shadow column for a swap.

        SQLite: no-op; `embedding_pending BLOB` already sits in the
        baseline schema. Postgres: `alter table {schema}.insights add
        column embedding_pending vector(N)` plus a
        CONCURRENTLY-built HNSW index. Idempotent.
        """
        ...

    def iter_for_swap(
            self, cursor: str, batch: int) -> list[tuple[Id, str]]:
        """Return up to `batch` (id, content) pairs needing pending vectors.

        Filtered to `embedding_pending IS NULL` and `id > cursor`,
        ordered by id ascending. Used by the swap orchestrator to
        page through rows resumably.
        """
        ...

    def write_swap_batch(
            self, items: list[tuple[Id, list[float]]]) -> None:
        """Persist a batch of (id, new_vec) pairs into `embedding_pending`.
        """
        ...

    def swap_cutover(self, target: 'Fingerprint') -> None:
        """Atomically promote `embedding_pending` to `embedding`.

        SQLite: `update insights set embedding = embedding_pending,
        embedding_pending = null`. Postgres: drop `embedding`,
        rename `embedding_pending` to `embedding` in one
        transaction. Writes the new fingerprint as part of the
        same transaction.
        """
        ...

    def swap_abort(self) -> None:
        """Drop or null `embedding_pending` and clear all swap meta.
        """
        ...

    def recall_session(
            self) -> AbstractContextManager[RecallSession]:
        """Yield a `RecallSession` for one recall request.

        The session reads live storage, so it needs no key: there is
        no stored per-model artifact for a fingerprint to select.
        """
        ...

    def integrity_check(self) -> dict[str, Any]:
        """Run a backend-specific integrity probe for `memman doctor`.

        SQLite: `pragma integrity_check`, then a rank-1 FTS5
        `'integrity-check'` that detects a keyword index whose terms
        have drifted from the rows they index; a handle that cannot
        write reports the probe as not run rather than as drift.
        Postgres: connectivity probe + schema-presence verification
        (HNSW index validity is checked separately at reindex time).

        Returns a dict shaped `{'ok': bool, 'detail': str}` -- doctor
        composes this with sub-store verbs to assemble its overall
        report.
        """
        ...

    def close(self) -> None:
        """Close the backend's connection."""
        ...

    def __enter__(self) -> Self:
        """Return self so `with open_backend(...) as backend:` works."""
        ...

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        """Close the backend on context exit."""
        ...

    def start_run(self) -> int | None:
        """Open a new drain `worker_runs` row, return its id.

        SQLite returns None (single-process; drain hangs are
        observable at the foreground prompt). Postgres inserts into
        the per-store `worker_runs` table with `started_at = now()`
        and `last_heartbeat_at = now()` server-side and returns the
        new row id. `memman doctor` consumes the heartbeat field for
        hung-worker detection.
        """
        ...

    def beat_run(self, run_id: int | None) -> None:
        """Advance `last_heartbeat_at = now()` on a specific run.

        Called inline from the drain loop (one update per row
        processed) so a worker stuck mid-row is detectable within a
        few enrichment cycles. SQLite is a no-op; Postgres updates
        the per-store `worker_runs` row. `run_id=None` is a no-op
        for drains opened in SQLite mode.
        """
        ...

    def finish_run(self, run_id: int | None) -> None:
        """Stamp `ended_at = now()` on a specific run.

        Called once when the drain context closes so
        `check_drain_heartbeat` sees the run as completed rather
        than perpetually in-progress. SQLite is a no-op;
        `run_id=None` is a no-op.
        """
        ...

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        """Return recent worker drain runs (most recent first).

        SQLite returns an empty list. Postgres queries the per-store
        `worker_runs` table.
        """
        ...
