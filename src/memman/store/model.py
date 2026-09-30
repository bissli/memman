"""Shared dataclasses for backend implementations and pipeline code.

The domain type (Insight) plus DTOs returned by Backend Protocol
verbs (OpLogEntry, OpLogStats, NodeStats, ProvenanceCount,
WorkerRun). Includes the timestamp helper used across the
package.

Protocol commitment: `Insight.created_at` and `Insight.updated_at`
carry no `default_factory`. Backends stamp them server-side at the verb
boundary. In-memory construction without a value yields `None`;
backends fill them in on insert and reads return them populated.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger('memman')

Id = str


@dataclass
class Insight:
    """One stored memory.
    """

    id: str = ''
    content: str = ''
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None
    prompt_version: str | None = None
    embedding_model: str | None = None
    summary: str = ''
    enrich_attempted_at: datetime | None = None
    enriched_at: datetime | None = None
    queue_uuid: str | None = None
    replaced_by: str | None = None
    author: str | None = None


@dataclass
class OpLogEntry:
    """One row from the oplog table.

    `before` carries the prior insight content on a replace or forget
    row, and `after` the new content on a remember, replace or
    target-gone row. A `recall:basic`, `recall-detail`, `rebuild` or
    `embed_reembed` row carries no deltas, so both stay None.
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

    Parameters
    ----------
    ins : Insight
        The row to snapshot.

    Returns
    -------
    dict[str, Any]
        `content` and `summary`. The id and timestamps
        are left out because the oplog row carries them.
    """
    return {
        'content': ins.content,
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
        `<id8> <score> <created_at> <author> | <text>`.
        `id8` is the first 8 characters of the id. `created_at` is the
        UTC date, `YYYY-MM-DD`, with no time of day. `-` stands for an
        unset author, and `_` joins any whitespace inside one, so
        every field before `|` is one space-free token. `text` is the
        summary, else the first `BRIEF_CONTENT_CHARS` characters of
        `content` with `...` when cut. Whitespace runs fold to one
        space, so a row never spans two lines.
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
        ins.created_at.astimezone(timezone.utc).strftime('%Y-%m-%d'),
        '_'.join((ins.author or '').split()) or '-']
    return f"{' '.join(fields)} | {text}"


def insight_to_full_dict(ins: 'Insight') -> dict[str, Any]:
    """Return the user-visible fields of an insight for JSON output.

    Timestamps use `format_timestamp`; `updated_at` falls back to
    `created_at`. Optional fields (`deleted_at`, `replaced_by`,
    `summary`, `enrich_attempted_at`, `enriched_at`) appear only when
    populated. `queue_uuid` is omitted.
    """
    out: dict[str, Any] = {
        'id': ins.id,
        'content': ins.content,
        'created_at': format_timestamp(ins.created_at),
        'updated_at': format_timestamp(ins.updated_at or ins.created_at),
        }
    if ins.deleted_at:
        out['deleted_at'] = format_timestamp(ins.deleted_at)
    if ins.replaced_by:
        out['replaced_by'] = ins.replaced_by
    if ins.summary:
        out['summary'] = ins.summary
    if ins.enrich_attempted_at:
        out['enrich_attempted_at'] = format_timestamp(
            ins.enrich_attempted_at)
    if ins.enriched_at:
        out['enriched_at'] = format_timestamp(ins.enriched_at)
    if ins.author:
        out['author'] = ins.author
    return out


@dataclass
class OpLogStats:
    """Aggregated oplog statistics.
    """

    operation_counts: dict[str, int] = field(default_factory=dict)
    total_active: int = 0


@dataclass
class NodeStats:
    """Aggregate node statistics returned by `backend.nodes.stats`.

    Attributes
    ----------
    total_insights : int
        Current rows: neither deleted nor replaced.
    replaced_insights : int
        Rows with `replaced_by` set and `deleted_at` null.
    deleted_insights : int
        Rows with `deleted_at` set, replaced or not. The three
        counts partition `count_total`.
    """

    total_insights: int = 0
    replaced_insights: int = 0
    deleted_insights: int = 0
    oplog_count: int = 0


@dataclass
class ProvenanceCount:
    """One (prompt_version, count) tuple from provenance distribution.
    """

    prompt_version: str | None
    count: int


@dataclass
class EnrichmentCoverage:
    """Enrichment gaps among active insights, read by `memman doctor`.

    Attributes
    ----------
    total_active : int
        Current rows: not forgotten, not replaced.
    missing_embedding : int
        Current rows with no vector.
    missing_summary : int
        Current rows with an empty summary and no `enriched_at`.
    stranded : int
        Current rows with `enrich_attempted_at` set and `enriched_at`
        null: an enrichment call failed and the pending path skips
        them.
    """

    total_active: int = 0
    missing_embedding: int = 0
    missing_summary: int = 0
    stranded: int = 0


@dataclass
class WorkerRun:
    """One worker drain run record.
    """

    id: int
    started_at: datetime
    ended_at: datetime | None
    last_heartbeat_at: datetime | None = None


def format_timestamp(dt: datetime) -> str:
    """Format datetime as RFC3339 with Z suffix (Go-compatible).
    """
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_timestamp(s: str) -> datetime:
    """Parse RFC3339 timestamp, accepting both Z and +00:00 suffixes.
    """
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)
