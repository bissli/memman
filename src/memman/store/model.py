"""Shared dataclasses for backend implementations and pipeline code.

The domain type (Insight) plus DTOs returned by Backend Protocol
verbs (OpLogEntry, OpLogStats, NodeStats, ProvenanceCount, QueueRow,
WorkerRun, ReembedRow). Includes the timestamp helper and importance
helpers used across the package.

Protocol commitment: `Insight.created_at` and `Insight.updated_at`
carry no `default_factory` -- backends stamp
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

VALID_CATEGORIES = {
    'preference', 'decision', 'fact',
    'insight', 'context',
    }


@dataclass
class Insight:
    """One stored memory."""

    id: str = ''
    content: str = ''
    category: str = 'fact'
    importance: int = 3
    entities: list[str] = field(default_factory=list)
    source: str = 'user'
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None
    prompt_version: str | None = None
    embedding_model: str | None = None
    summary: str = ''
    linked_at: datetime | None = None
    enriched_at: datetime | None = None
    queue_uuid: str | None = None
    superseded_by: str | None = None
    author: str | None = None

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
class OpLogEntry:
    """One row from the oplog table.

    `before` and `after` capture the insight content before and
    after the logged operation. Populated by replace, supersede,
    unsupersede and forget so forensic questions can be answered
    from the oplog alone. Older rows may have both as None.
    """

    id: int
    operation: str
    insight_id: str
    detail: str
    created_at: datetime
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


MAX_ROW_ENTITIES = 50


def dedupe_entities(entities: list[str]) -> list[str]:
    """Fold case and whitespace variants of one entity name into one.

    Parameters
    ----------
    entities : list[str]
        Entity names as the caller typed them or the replaced row
        carried them.

    Returns
    -------
    list[str]
        The names in input order, each stripped, none empty, one per
        name compared case-insensitively. The FIRST form of a name
        decides its casing.

    Notes
    -----
    - The stored column, the result JSON and the oplog delta all read
      this list. Folding in only one of them makes a write report an
      entity the store does not hold.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for entity in entities:
        name = entity.strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


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


BRIEF_CONTENT_CHARS = 200


def insight_to_recall_line(ins: 'Insight', score: float | None) -> str:
    """Return the one recall page line that shows an insight.

    Parameters
    ----------
    ins : Insight
        The row to show.
    score : float | None
        The row's rank score, printed to two decimals; None omits the
        field, since `recall --basic` computes no score.

    Returns
    -------
    str
        `<id8> <score> <created_at> <author> <category> | <text>`, with
        `-` for an unset author and `_` joining any whitespace inside
        one, so every field before `|` is one space-free token.

    Notes
    -----
    - `text` is the summary when the row has one, else the first
      `BRIEF_CONTENT_CHARS` characters of `content`, ending in `...`
      when the cut dropped anything.
    - Every run of whitespace, line breaks included, folds to one
      space in both, so a row never spans two lines.
    - `id8` is the first 8 characters of the id; every id-taking
      command resolves an unambiguous prefix.
    """
    if ins.summary.strip():
        text = ' '.join(ins.summary.split())
    else:
        text = ' '.join(ins.content.split())
        if len(text) > BRIEF_CONTENT_CHARS:
            text = text[:BRIEF_CONTENT_CHARS] + '...'
    fields = [ins.id[:8]]
    if score is not None:
        fields.append(f'{score:.2f}')
    fields += [
        format_timestamp(ins.created_at),
        '_'.join((ins.author or '').split()) or '-',
        ins.category]
    return f"{' '.join(fields)} | {text}"


def insight_to_full_dict(ins: 'Insight') -> dict[str, Any]:
    """Return the user-visible fields of an insight for JSON output.

    Used by CLI commands that emit Insight objects to stdout.
    Timestamps are formatted with
    `format_timestamp`; `updated_at` falls back to `created_at` so
    consumers always see a populated value. Optional fields
    (`deleted_at`, `superseded_by`, `summary`, `linked_at`,
    `enriched_at`) are emitted only when populated; the plumbing key
    `queue_uuid` is deliberately omitted.
    """
    out: dict[str, Any] = {
        'id': ins.id,
        'content': ins.content,
        'category': ins.category,
        'importance': ins.importance,
        'entities': list(ins.entities or []),
        'source': ins.source,
        'created_at': format_timestamp(ins.created_at),
        'updated_at': format_timestamp(ins.updated_at or ins.created_at),
        }
    if ins.deleted_at:
        out['deleted_at'] = format_timestamp(ins.deleted_at)
    if ins.superseded_by:
        out['superseded_by'] = ins.superseded_by
    if ins.summary:
        out['summary'] = ins.summary
    if ins.linked_at:
        out['linked_at'] = format_timestamp(ins.linked_at)
    if ins.enriched_at:
        out['enriched_at'] = format_timestamp(ins.enriched_at)
    if ins.author:
        out['author'] = ins.author
    return out


@dataclass
class OpLogStats:
    """Aggregated oplog statistics."""

    operation_counts: dict[str, int] = field(default_factory=dict)
    total_active: int = 0


@dataclass
class NodeStats:
    """Aggregate node statistics returned by `backend.nodes.stats`.

    Attributes
    ----------
    total_insights : int
        Current rows: neither deleted nor superseded.
    superseded_insights : int
        Rows with `superseded_by` set and `deleted_at` null.
    deleted_insights : int
        Rows with `deleted_at` set, superseded or not. The three
        counts partition `count_total`.
    """

    total_insights: int = 0
    superseded_insights: int = 0
    deleted_insights: int = 0
    oplog_count: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    top_entities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvenanceCount:
    """One (prompt_version, count) tuple from provenance distribution.
    """

    prompt_version: str | None
    count: int


@dataclass
class EnrichmentCoverage:
    """Per-field NULL counts for the enrichment columns on `insights`.

    `memman doctor` consumes this to report which enrichment fields
    (embedding, keywords, summary) have unfilled values among active
    insights.
    """

    total_active: int = 0
    missing_embedding: int = 0
    missing_keywords: int = 0
    missing_summary: int = 0


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
