"""Causal edge creation and causal candidate discovery."""

import json
import logging
import re
from datetime import datetime, timezone

from mnemon.model import Edge, Insight, format_float
from mnemon.search.keyword import tokenize
from mnemon.store.edge import insert_edge
from mnemon.store.node import get_recent_active_insights

logger = logging.getLogger('mnemon')

MIN_CAUSAL_OVERLAP = 0.15
CAUSAL_LOOKBACK = 20
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

CAUSES_PATTERN = re.compile(r'(?i)\b(because|caused by|due to)\b')
ENABLES_PATTERN = re.compile(
    r'(?i)\b(so that|in order to|enables|leads to)\b')
PREVENTS_PATTERN = re.compile(
    r'(?i)\b(despite|prevented|prevents|blocked)\b')


def has_causal_signal(text: str) -> bool:
    """Return True if the text contains causal keywords."""
    return bool(CAUSAL_PATTERN.search(text))


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


def create_causal_edges(
        db: 'DB', insight: Insight, dry_run: bool = False) -> int:
    """Create causal edges when insights share token overlap and causal signals."""
    recent = get_recent_active_insights(
        db, insight.id, CAUSAL_LOOKBACK)
    if not recent:
        return 0

    new_tokens = tokenize(insight.content)
    if not new_tokens:
        return 0

    new_has_signal = has_causal_signal(insight.content)
    now = datetime.now(timezone.utc)
    count = 0

    for prev in recent:
        prev_has_signal = has_causal_signal(prev.content)
        if not new_has_signal and not prev_has_signal:
            continue

        prev_tokens = tokenize(prev.content)
        overlap = token_overlap(new_tokens, prev_tokens)
        if overlap < MIN_CAUSAL_OVERLAP:
            continue

        source_id = prev.id
        target_id = insight.id
        if not new_has_signal and prev_has_signal:
            if CAUSES_PATTERN.search(prev.content):
                source_id = insight.id
                target_id = prev.id

        sub_type = suggest_sub_type(insight.content + ' ' + prev.content)

        if not dry_run:
            try:
                insert_edge(db, Edge(
                    source_id=source_id, target_id=target_id,
                    edge_type='causal', weight=overlap,
                    metadata={
                        'overlap': format_float(overlap),
                        'sub_type': sub_type,
                        },
                    created_at=now))
            except Exception:
                pass
        count += 1

    return count


def find_causal_candidates(
        db: 'DB', insight: Insight) -> list[dict]:
    """Return insights with potential causal relationships via 2-hop BFS."""
    from mnemon.graph.bfs import BFSOptions, bfs
    nodes = bfs(db, insight.id, BFSOptions(
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
        neighbors: list[dict],
        recent: list[Insight],
        ) -> str:
    """Build the user prompt for LLM causal inference."""
    parts = [f'NEW INSIGHT (id={insight.id}):\n{insight.content}\n']

    if neighbors:
        parts.append('GRAPH NEIGHBORS:')
        parts.extend(f'- id={n["insight"].id}: {n["insight"].content}' for n in neighbors[:LLM_BFS_NEIGHBORS])

    if recent:
        parts.append('\nRECENT INSIGHTS:')
        parts.extend(f'- id={r.id}: {r.content}' for r in recent[:LLM_RECENT_COUNT])

    return '\n'.join(parts)


def create_llm_causal_edges(
        db: 'DB', insight: Insight, llm_client: object) -> int:
    """Create causal edges using LLM inference on 2-hop neighborhood."""
    from mnemon.graph.bfs import BFSOptions, bfs

    neighbors = bfs(db, insight.id, BFSOptions(
        max_depth=2, max_nodes=LLM_BFS_NEIGHBORS))
    recent = get_recent_active_insights(
        db, insight.id, LLM_RECENT_COUNT)

    new_tokens = tokenize(insight.content)
    candidates = []
    for n in neighbors:
        prev_tokens = tokenize(n['insight'].content)
        overlap = token_overlap(new_tokens, prev_tokens)
        if overlap >= MIN_CAUSAL_OVERLAP:
            candidates.append(n)

    for r in recent:
        prev_tokens = tokenize(r.content)
        overlap = token_overlap(new_tokens, prev_tokens)
        if overlap >= MIN_CAUSAL_OVERLAP:
            if not any(n['insight'].id == r.id for n in candidates):
                candidates.append({'insight': r, 'hop': 0, 'via_edge': ''})

    if not candidates:
        return 0

    prompt = _build_llm_prompt(insight, candidates, [])

    try:
        raw = llm_client.complete(LLM_SYSTEM_PROMPT, prompt)
    except Exception:
        logger.debug(f'LLM causal inference unavailable for {insight.id}')
        return 0

    try:
        edges = json.loads(raw)
        if not isinstance(edges, list):
            return 0
    except (json.JSONDecodeError, ValueError):
        return 0

    now = datetime.now(timezone.utc)
    count = 0
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

        try:
            insert_edge(db, Edge(
                source_id=source_id,
                target_id=target_id,
                edge_type='causal',
                weight=confidence,
                metadata={
                    'created_by': 'llm',
                    'confidence': confidence,
                    'rationale': edge_data.get('rationale', ''),
                    'sub_type': sub_type,
                    },
                created_at=now))
            count += 1
        except Exception:
            pass

    logger.debug(
        f'LLM causal inference for {insight.id}: {count} edges created')
    return count
