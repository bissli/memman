"""Shared dataclasses for backend implementations and pipeline code.

Domain types (Insight, Edge) plus DTOs returned by Backend Protocol
verbs (Neighbor, ScoredId, OpLogEntry, OpLogStats, NodeStats,
ProvenanceCount, IntegrityReport, QueueRow, QueueHints, QueueStats,
WorkerRun, ReembedRow). Includes the timestamp helper and importance
helpers used across the package.

Protocol commitment: `Insight.created_at`, `Insight.updated_at`,
and `Edge.created_at` carry no `default_factory` -- backends stamp
these server-side at the verb boundary. In-memory construction without
a value yields `None`; backends fill them in on insert and reads
return them populated.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger('memman')

Id = str
Score = float

MAX_INSIGHTS = 1000

VALID_CATEGORIES = {
    'preference', 'decision', 'fact',
    'insight', 'context', 'general',
    }

VALID_EDGE_TYPES = {'temporal', 'semantic', 'causal', 'entity'}


@dataclass
class Insight:
    """A memory node in the memman graph."""

    id: str = ''
    content: str = ''
    category: str = 'general'
    importance: int = 3
    entities: list[str] = field(default_factory=list)
    source: str = 'user'
    access_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None
    last_accessed_at: datetime | None = None
    effective_importance: float = 0.0
    prompt_version: str | None = None
    model_id: str | None = None
    embedding_model: str | None = None
    summary: str = ''

    def entities_json(self) -> str:
        """Return entities as a JSON string for storage."""
        return json.dumps(self.entities, sort_keys=True)

    def parse_entities(self, s: str) -> None:
        """Parse a JSON string into the entities field."""
        try:
            self.entities = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            self.entities = []
        if self.entities is None:
            self.entities = []


@dataclass
class Edge:
    """A directed relationship between two insights."""

    source_id: str = ''
    target_id: str = ''
    edge_type: str = 'semantic'
    weight: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    def metadata_json(self) -> str:
        """Return metadata as a JSON string for storage."""
        return json.dumps(self.metadata, sort_keys=True)

    def parse_metadata(self, s: str) -> None:
        """Parse a JSON string into the metadata field."""
        try:
            self.metadata = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            self.metadata = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Neighbor:
    """A graph neighbor: target id + edge type + weight."""

    target_id: Id
    edge_type: str
    weight: float


@dataclass
class ScoredId:
    """An id with an associated score (similarity, anchor weight, etc)."""

    id: Id
    score: Score


@dataclass
class OpLogEntry:
    """One row from the oplog table.

    `before` and `after` capture the insight content before and
    after the logged operation. Populated by reconcile, replace,
    forget, and auto_prune so forensic questions can be answered
    from the oplog alone. Pre-Slice-D rows have both as None.
    """

    id: int
    operation: str
    insight_id: str
    detail: str
    created_at: datetime
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


def insight_to_delta_dict(ins: 'Insight') -> dict[str, Any]:
    """Return the content fields of an insight for oplog deltas.

    Excludes embedding (it is not on the dataclass anyway), the
    surrogate `id`, and timestamps -- the surrounding oplog row
    already carries `insight_id` and `created_at`.
    """
    return {
        'content': ins.content,
        'category': ins.category,
        'importance': ins.importance,
        'entities': list(ins.entities or []),
        'source': ins.source,
        'summary': ins.summary,
        }


@dataclass
class OpLogStats:
    """Aggregated oplog statistics."""

    operation_counts: dict[str, int] = field(default_factory=dict)
    never_accessed: int = 0
    total_active: int = 0


@dataclass
class NodeStats:
    """Aggregate node statistics returned by `backend.nodes.stats`."""

    total_insights: int = 0
    deleted_insights: int = 0
    edge_count: int = 0
    oplog_count: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    top_entities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvenanceCount:
    """One (prompt_version, model_id, count) tuple from provenance distribution.
    """

    prompt_version: str | None
    model_id: str | None
    count: int


@dataclass
class IntegrityReport:
    """Aggregate integrity findings used by `memman doctor`."""

    orphan_count: int = 0
    total_active: int = 0
    dangling_by_type: dict[str, int] = field(default_factory=dict)
    degree_distribution: dict[str, int] = field(default_factory=dict)
    provenance: list[ProvenanceCount] = field(default_factory=list)


@dataclass
class EnrichmentCoverage:
    """Per-field NULL counts for the enrichment columns on `insights`.

    `memman doctor` consumes this to report which enrichment fields
    (embedding, keywords, summary, semantic_facts) have unfilled
    values among active insights.
    """

    total_active: int = 0
    missing_embedding: int = 0
    missing_keywords: int = 0
    missing_summary: int = 0
    missing_semantic_facts: int = 0


@dataclass
class QueueRow:
    """One claimable row from the per-host queue."""

    id: int
    store: str
    op: str
    payload: str
    attempts: int
    created_at: datetime


@dataclass
class QueueHints:
    """Hints attached to a queue row (per-store recency, etc)."""

    store: str
    last_seen: datetime | None = None
    pending_count: int = 0


@dataclass
class QueueStats:
    """Aggregate queue statistics."""

    total: int = 0
    by_store: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkerRun:
    """One worker drain run record."""

    id: int
    started_at: datetime
    ended_at: datetime | None
    rows_processed: int
    error: str = ''
    last_heartbeat_at: datetime | None = None


@dataclass
class ReembedRow:
    """One row returned by `nodes.iter_for_reembed`."""

    id: Id
    content: str
    embedding_model: str | None
    blob_length: int | None


def format_timestamp(dt: datetime) -> str:
    """Format datetime as RFC3339 with Z suffix (Go-compatible)."""
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_timestamp(s: str) -> datetime:
    """Parse RFC3339 timestamp, accepting both Z and +00:00 suffixes."""
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)


def format_float(value: float) -> str:
    """Format float to 4 decimal places (Go parity)."""
    return f'{value:.4f}'


def base_weight(importance: int) -> float:
    """Map importance (1-5) to a base weight."""
    weights = {5: 1.0, 4: 0.8, 3: 0.5, 2: 0.3}
    return weights.get(importance, 0.15)


def is_immune(importance: int, access_count: int) -> bool:
    """Check if an insight is immune to auto-pruning."""
    return importance >= 4 or access_count >= 3
