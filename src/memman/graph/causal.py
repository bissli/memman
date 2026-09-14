"""Causal edge creation and causal candidate discovery."""

import logging
import re
from typing import Any

from memman import trace
from memman.llm import usage as llm_usage
from memman.llm.shared import parse_json_list_response
from memman.search.keyword import tokenize
from memman.store.backend import Backend
from memman.store.model import Edge, Insight

logger = logging.getLogger('memman')

MIN_CAUSAL_OVERLAP = 0.15
MAX_CAUSAL_CANDIDATES = 10
LLM_CONFIDENCE_FLOOR = 0.75
LLM_BFS_NEIGHBORS = 10
LLM_RECENT_COUNT = 20

CAUSAL_PATTERN = re.compile(
    r'\b(because|therefore|due to|caused by|as a result|decided to|'
    r'chosen because|so that|in order to|leads to|results in|'
    r'enables|prevents|consequently|hence|thus)\b|'
    r'\bthis (ensures|means)\b',
    re.IGNORECASE)

ENABLES_PATTERN = re.compile(
    r'(?i)\b(so that|in order to|enables|leads to)\b')
PREVENTS_PATTERN = re.compile(
    r'(?i)\b(despite|prevented|prevents|blocked)\b')


def suggest_sub_type(text: str) -> str:
    """Guess a causal sub_type from the content text."""
    if PREVENTS_PATTERN.search(text):
        return 'prevents'
    if ENABLES_PATTERN.search(text):
        return 'enables'
    return 'causes'


def find_causal_signal(text: str) -> str:
    """Return the first matching causal keyword in the text."""
    m = CAUSAL_PATTERN.search(text)
    return m.group(0) if m else ''


def token_overlap(a: set[str], b: set[str]) -> float:
    """Compute |intersection| / max(|a|, |b|)."""
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    intersection = sum(1 for k in small if k in big)
    return intersection / max(len(a), len(b))


def find_causal_candidates(
        backend: Backend, insight: Insight) -> list[dict[str, Any]]:
    """Return insights with potential causal relationships via 2-hop BFS."""
    from memman.graph.bfs import BFSOptions, bfs
    nodes = bfs(backend, insight.id, BFSOptions(
        max_depth=2, max_nodes=MAX_CAUSAL_CANDIDATES))
    if not nodes:
        return []

    candidates = []
    for n in nodes:
        signal = find_causal_signal(n['insight'].content)
        if not signal:
            signal = find_causal_signal(insight.content)

        combined_text = insight.content + ' ' + n['insight'].content
        sub_type = suggest_sub_type(combined_text)

        candidates.append({
            'id': n['insight'].id,
            'content': n['insight'].content,
            'category': n['insight'].category,
            'hop': n['hop'],
            'via_edge': n['via_edge'],
            'causal_signal': signal,
            'suggested_sub_type': sub_type,
            })

    return candidates


LLM_SYSTEM_PROMPT = (
    'You are a causal relationship detector for a memory graph. '
    'Given a NEW insight and CONTEXT insights, identify causal '
    'relationships the NEW insight has with any context insight. '
    'Return a JSON array of objects with fields: '
    '"source_id", "target_id", "confidence" (0.0-1.0), '
    '"sub_type" ("causes"|"enables"|"prevents"), "rationale" (one sentence). '
    'Only include relationships with confidence >= 0.75. '
    'Return [] if no causal relationships exist.'
    )


def _build_llm_prompt(
        insight: Insight,
        neighbors: list[dict[str, Any]],
        recent: list[Insight],
        ) -> str:
    """Build the user prompt for LLM causal inference.

    Parameters
    ----------
    insight : Insight
        The new row, rendered first under its own header.
    neighbors : list[dict[str, Any]]
        Graph-reached candidate rows, each `{'insight': Insight, ...}`.
    recent : list[Insight]
        Recency-reached candidate rows, disjoint from `neighbors`.

    Returns
    -------
    str
        The prompt body.

    Notes
    -----
    - Both lists carry candidates only. A row the caller will later
      reject renders text no edge can use and is paid for per token.
    """
    parts = [f'NEW INSIGHT (id={insight.id}):\n{insight.content}\n']

    if neighbors:
        parts.append('GRAPH NEIGHBORS:')
        parts.extend(f'- id={n["insight"].id}: {n["insight"].content}' for n in neighbors[:LLM_BFS_NEIGHBORS])

    if recent:
        parts.append('\nRECENT INSIGHTS:')
        parts.extend(f'- id={r.id}: {r.content}' for r in recent[:LLM_RECENT_COUNT])

    return '\n'.join(parts)


def infer_llm_causal_edges(
        backend: Backend, insight: Insight,
        llm_client: Any) -> list[Edge]:
    """Infer causal edges via LLM, returning Edge objects without inserting.

    Parameters
    ----------
    backend : Backend
        Store the graph neighbors and the recent rows are read from.
    insight : Insight
        The newly written row every returned edge touches.
    llm_client : Any
        Client exposing `complete(system, user, stage=...)`.

    Returns
    -------
    list[Edge]
        Causal edges at or above `LLM_CONFIDENCE_FLOOR`, unsaved. Empty
        when no row clears `MIN_CAUSAL_OVERLAP`, the call fails, or the
        response does not parse.

    Notes
    -----
    - The prompt renders exactly the rows that may become an edge
      endpoint. `valid_ids` bars a row below `MIN_CAUSAL_OVERLAP`, so
      rendering one buys no edge and is paid for per token.
    - A row both the graph walk and the recent query reach is rendered
      once, as a graph neighbor.
    """
    from memman.graph.bfs import BFSOptions, bfs

    neighbors = bfs(backend, insight.id, BFSOptions(
        max_depth=2, max_nodes=LLM_BFS_NEIGHBORS))
    recent = backend.nodes.get_recent_active(
        exclude_id=insight.id, limit=LLM_RECENT_COUNT)

    new_tokens = tokenize(insight.content)
    neighbor_candidates = []
    for n in neighbors:
        prev_tokens = tokenize(n['insight'].content)
        overlap = token_overlap(new_tokens, prev_tokens)
        if overlap >= MIN_CAUSAL_OVERLAP:
            neighbor_candidates.append(n)

    neighbor_ids = {n['insight'].id for n in neighbor_candidates}
    recent_candidates = []
    for r in recent:
        if r.id in neighbor_ids:
            continue
        prev_tokens = tokenize(r.content)
        overlap = token_overlap(new_tokens, prev_tokens)
        if overlap >= MIN_CAUSAL_OVERLAP:
            recent_candidates.append(r)

    candidates = neighbor_candidates + [
        {'insight': r, 'hop': 0, 'via_edge': ''}
        for r in recent_candidates
        ]

    if not candidates:
        trace.event(
            'causal_infer_skipped',
            insight_id=insight.id,
            reason='no_candidates')
        return []

    prompt = _build_llm_prompt(
        insight, neighbor_candidates, recent_candidates)
    trace.event(
        'causal_infer_start',
        insight_id=insight.id,
        candidate_count=len(candidates),
        recent_count=len(recent_candidates))

    try:
        raw = llm_client.complete(
            LLM_SYSTEM_PROMPT, prompt, stage=llm_usage.STAGE_CAUSAL)
    except Exception as exc:
        logger.debug(f'LLM causal inference unavailable for {insight.id}')
        trace.event(
            'causal_infer_result',
            insight_id=insight.id,
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return []

    edges = parse_json_list_response(raw)
    if edges is None:
        trace.event(
            'causal_infer_result',
            insight_id=insight.id,
            outcome='parse_error',
            raw=raw)
        return []

    result = []
    valid_ids = {insight.id} | {
        n['insight'].id for n in candidates}

    for edge_data in edges:
        if not isinstance(edge_data, dict):
            continue
        try:
            confidence = float(edge_data.get('confidence', 0))
        except (ValueError, TypeError):
            continue
        if confidence < LLM_CONFIDENCE_FLOOR:
            continue

        source_id = edge_data.get('source_id', '')
        target_id = edge_data.get('target_id', '')
        if source_id not in valid_ids or target_id not in valid_ids:
            continue
        if source_id == target_id:
            continue

        sub_type = edge_data.get('sub_type', 'causes')
        if sub_type not in {'causes', 'enables', 'prevents'}:
            sub_type = 'causes'

        result.append(Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type='causal',
            weight=confidence,
            metadata={
                'created_by': 'llm',
                'confidence': confidence,
                'rationale': edge_data.get('rationale', ''),
                'sub_type': sub_type,
                }))

    logger.debug(
        f'LLM causal inference for {insight.id}: {len(result)} edges')
    trace.event(
        'causal_infer_result',
        insight_id=insight.id,
        outcome='ok',
        edge_count=len(result),
        edges=[{
            'source_id': e.source_id,
            'target_id': e.target_id,
            'sub_type': e.metadata.get('sub_type'),
            'confidence': e.metadata.get('confidence'),
            'rationale': e.metadata.get('rationale'),
            } for e in result])
    return result
