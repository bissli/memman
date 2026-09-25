"""Remember pipeline - single entry point shared by sync CLI and worker.

Structure:

1. Quality check - advisory warnings only.
2. Planning phase - enrich the row, then embed it once:
   keyword-enriched text when the enrichment carries keywords, else
   the content alone. **No DB writes.**
3. Apply phase - one transaction commits the replace link, insert,
   edges, enrichment update, and stamp.

A write adds one row, or replaces the row `replace <id>` names.
Nothing else retires a row.

The apply phase runs only after all LLM + embed work has returned.
Crashes during planning leave the DB untouched; the retry path
re-runs the whole pipeline cleanly. This closes the partial-write
fact-loss gap for a single queue row.
"""

import functools
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from memman.embed import EmbeddingProvider
from memman.exceptions import EmbedCredentialError
from memman.graph.engine import _resolve_semantic_threshold, fast_edges
from memman.graph.enrichment import build_enriched_text, enrich_with_llm
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import create_semantic_edges
from memman.llm.client import get_llm_client
from memman.search.quality import check_content_quality
from memman.store.backend import Backend
from memman.store.model import Edge, Insight, dedupe_entities
from memman.store.model import format_timestamp, insight_to_delta_dict

logger = logging.getLogger('memman')


@functools.lru_cache(maxsize=1)
def compute_prompt_version() -> str:
    """Return a 16-char SHA-256 hash of what a rebuild can replay.

    Returns
    -------
    str
        First 16 hex chars of a SHA-256 over the enrichment prompt
        and the resolved `MEMMAN_LLM_MODEL_SLOW` id.

    Notes
    -----
    - THE INVARIANT: this hashes exactly the inputs `link_pending`
      (`graph/engine.py`) re-runs, and nothing else. It is both the
      value `stamp_enriched` writes and the key
      `count_stale_insights` compares, so a key covering more than
      the remedy replays reports rows stale for a change
      re-enrichment cannot address - and `graph rebuild --stale`
      then clears the report by doing unrelated work, which is worse
      than having no remedy at all.
    - The slow model id IS folded in, because `link_pending` runs
      the enrichment call on `slow`.
    - An unresolvable slow model hashes as the empty string, so a
      store with no model configured still yields a stable key rather
      than raising on the `status` path.
    - Cached for the life of the process. Every consumer - `status`,
      one drain tick, one rebuild - is a fresh process; tests that
      vary the inputs call `cache_clear()`.
    """
    # Imported here, not at module top, so the hash reads each prompt
    # from its defining module at CALL time. A top-level `from x
    # import y` would bind a copy and make the invariant above
    # untestable.
    from memman import config
    from memman.exceptions import ConfigError
    from memman.graph.enrichment import ENRICHMENT_SYSTEM_PROMPT

    try:
        metadata_model = config.require(config.LLM_MODEL_SLOW)
    except ConfigError:
        metadata_model = ''
    blob = f'{ENRICHMENT_SYSTEM_PROMPT}\x00{metadata_model}'
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class FactPlan:
    """Planned write for one queued write.

    Attributes
    ----------
    action : str
        `add` or `replace`.
    fact_insight : Insight
        The row the apply phase inserts.
    targets : list[tuple[str, str]]
        `(insight_id, relation)`: the `replace` target; empty for an
        add.
    embed_vec : list[float] or None
        Vector of `build_enriched_text(content, keywords)`; None when
        the embed failed.
    enrichment : dict[str, Any]
        `keywords` and `summary`; empty when enrichment failed.
    """

    action: str
    fact_insight: Insight
    targets: list[tuple[str, str]] = field(default_factory=list)
    embed_vec: list[float] | None = None
    enrichment: dict[str, Any] = field(default_factory=dict)


def run_remember(
        backend: Backend,
        insight: Insight,
        content: str,
        ec: EmbeddingProvider,
        replaced_id: str = '',
        embed_cache: dict[str, list[float]] | None = None,
        *,
        store_name: str,
        ) -> dict[str, Any]:
    """Run the full remember pipeline and return the result dict.

    See module docstring for the overall shape.

    `ec` is the store-bound embed client (resolved from the store's
    `meta.embed_fingerprint` via `bound_embedder`); production callers
    pass `_StoreContext.ec`. `embed_cache` is optional drain-scope
    state hoisted by `_drain_queue` to amortize setup across rows in
    one drain pass. When omitted (e.g., direct test use), the function
    builds it from the backend itself.

    `content` is stored as written: no model judges it, rewords it,
    or picks its category, which is `insight.category`.

    `store_name` selects the per-store surface
    (`MEMMAN_SURFACE_<store>`) for the threshold lookup. It is
    keyword-only and required: an omitted store name silently
    resolves the code-surface row and skips the
    `MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>` override branch
    entirely, which is a wrong threshold rather than a missing one.
    """
    quality_warnings = check_content_quality(content)

    metadata_llm_client = get_llm_client('slow')
    if embed_cache is None:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())

    plan, llm_calls = _plan_fact(
        content, insight, replaced_id, metadata_llm_client, ec)
    plan.fact_insight.prompt_version = compute_prompt_version()
    plan.fact_insight.embedding_model = ec.model
    if plan.action == 'replace':
        embed_cache.pop(replaced_id, None)

    with backend.transaction():
        result = _apply_plan(
            backend, plan, embed_cache, store_name=store_name)

    return {
        'facts': [result],
        'quality_warnings': quality_warnings,
        'llm_calls': llm_calls,
        }


def _plan_fact(
        fact_text: str,
        parent: Insight,
        replaced_id: str,
        metadata_llm_client: Any,
        ec: Any,
        ) -> tuple[FactPlan, int]:
    """Plan one write without touching the DB.

    Parameters
    ----------
    fact_text : str
        The write's text, as the agent wrote it.
    parent : Insight
        The queued write; its category, entities and other metadata
        are inherited by the planned row.
    replaced_id : str
        A `replace` target, or `''`.
    metadata_llm_client : Any
        The slow client for enrichment.
    ec : Any
        The store's bound embed provider.

    Returns
    -------
    tuple[FactPlan, int]
        The plan - an add, or a replace of `replaced_id` - and the
        LLM calls made, 0 or 1.

    Raises
    ------
    EmbedCredentialError
        The store's embed provider has no credentials. An HTTP or
        runtime embed failure is logged instead, and the row is
        planned without a vector.
    """
    fact_insight = Insight(
        id=str(uuid.uuid4()), content=fact_text,
        category=parent.category, importance=parent.importance,
        entities=list(parent.entities), source=parent.source,
        created_at=parent.created_at, updated_at=parent.updated_at,
        session_id=parent.session_id, queue_uuid=parent.queue_uuid,
        author=parent.author)

    calls = 0
    try:
        enrichment = enrich_with_llm(fact_insight, metadata_llm_client)
        calls += 1
    except Exception:
        enrichment = {}

    fact_vec = None
    try:
        fact_vec = ec.embed(
            build_enriched_text(fact_text, enrichment.get('keywords', [])))
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

    return FactPlan(
        action='replace' if replaced_id else 'add',
        fact_insight=fact_insight,
        targets=[(replaced_id, 'replace')] if replaced_id else [],
        embed_vec=fact_vec,
        enrichment=enrichment,
        ), calls


def move_edges(
        backend: Backend, from_id: str, to_id: str,
        carried: list[Edge]) -> int:
    """Re-point a snapshot of a predecessor's edges onto its successor.

    Parameters
    ----------
    backend : Backend
        Open store; the caller holds the transaction.
    from_id : str
        The predecessor whose edges were snapshotted.
    to_id : str
        The successor that inherits them.
    carried : list[Edge]
        The predecessor's edges as read BEFORE its pointer was
        written, since `supersede` removes them.

    Returns
    -------
    int
        Edges written onto the successor. An edge whose far endpoint
        is the predecessor itself or the successor is dropped rather
        than re-pointed into a self-edge.
    """
    moved = 0
    for edge in carried:
        far_id = edge.target_id if edge.source_id == from_id else edge.source_id
        if far_id in {from_id, to_id}:
            continue
        backend.edges.upsert(Edge(
            source_id=to_id if edge.source_id == from_id else edge.source_id,
            target_id=to_id if edge.target_id == from_id else edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            metadata=dict(edge.metadata)))
        moved += 1
    return moved


def _apply_plan(
        backend: Backend,
        plan: FactPlan,
        embed_cache: dict[str, list[float]],
        *,
        store_name: str,
        ) -> dict[str, Any]:
    """Apply one planned write. Must be invoked inside a transaction.

    `store_name` selects the per-store surface for the calibrated
    semantic-edge threshold lookup. It is keyword-only and required
    for the same reason as on `run_remember`: an omitted store name
    resolves the code-surface row and skips the per-store override,
    giving a wrong threshold rather than none.

    Notes
    -----
    - A `replace` supersedes its target (never deletes it), moves the
      target's edges to the successor, and carries the target's recall
      history onto it. The entity list is the caller's as given.
    - A target that is not current (forgotten, or superseded by an
      earlier write) is dropped into `targets_gone`, and the plan
      degrades to a plain add.
    """
    fi = plan.fact_insight

    linking = plan.action == 'replace' and bool(plan.targets)
    linked_targets: list[tuple[str, str]] = []
    targets_gone: list[dict[str, str | None]] = []
    carried: list[tuple[str, list[Edge]]] = []
    predecessors: list[tuple[str, str, Insight]] = []
    if linking:
        for target_id, relation in plan.targets:
            before_target = backend.nodes.get_include_deleted(target_id)
            # Snapshot before the pointer is written: `supersede` removes
            # the predecessor's edges, and a later snapshot would also
            # scoop up the successor's own freshly minted edges.
            carried_edges = backend.edges.by_node(target_id)
            # The pointer is written BEFORE `nodes.insert`, and the
            # position is load-bearing: `create_temporal_edge` reads
            # `get_latest_by_session` and `get_recent_in_window`, so every
            # predecessor must already be out of the active set or the
            # successor chains its backbone to a row it replaced.
            linked = backend.nodes.supersede(target_id, fi.id)
            if not linked or before_target is None:
                targets_gone.append({
                    'id': target_id,
                    'superseded_by': (before_target.superseded_by
                                      if before_target is not None else None),
                    })
                logger.warning(
                    f'{relation} target {target_id} is not current;'
                    ' dropped from the plan')
                continue
            linked_targets.append((target_id, relation))
            carried.append((target_id, carried_edges))
            predecessors.append((target_id, relation, before_target))
        # Every predecessor keeps its content behind `superseded_by`,
        # and the successor copies nothing from it: the CLI already
        # seeded the caller's entity list from the target when the
        # flag was omitted.
        for target_id, _relation, before_target in predecessors:
            backend.oplog.log(
                operation='replace', insight_id=target_id,
                detail=f'replaced by {fi.id}',
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(fi))
        # Notes:
        # - The row is stored either way, so nothing is lost, but a
        #   caller who ran `replace` to correct one row otherwise gets
        #   a new unlinked row and no sign the correction missed.
        # - The row is filed against the SUCCESSOR, which is readable;
        #   the requested target may be gone from the table entirely,
        #   and it is named in the detail instead.
        for gone in targets_gone:
            backend.oplog.log(
                operation='target-gone', insight_id=fi.id,
                detail=f'{plan.action} target {gone["id"]} was not current;'
                f' stored without the link',
                after=insight_to_delta_dict(fi))
        if not linked_targets:
            logger.warning(
                f'{plan.action}: every target is gone; degrading to add')

    backend.nodes.insert(fi)
    stored = backend.nodes.get(fi.id)
    if stored is not None and stored.created_at is not None:
        fi.created_at = stored.created_at
        fi.updated_at = stored.updated_at

    final_vec = plan.embed_vec
    embedded = final_vec is not None
    if final_vec is not None:
        # The new row is not in the cache yet, and the semantic-edge
        # builder reads its vector from there, so the inserted row
        # registers itself here with the vector it stores.
        embed_cache[fi.id] = final_vec
        backend.nodes.update_embedding(
            fi.id, final_vec, fi.embedding_model or '')
    if fi.entities:
        # Normalize before the column, the edge builder and the result
        # dict read it: folding only on the way into the store makes
        # the write report an entity the store does not hold.
        fi.entities = dedupe_entities(fi.entities)
        backend.nodes.update_entities(fi.id, fi.entities)

    backend.oplog.log(
        operation='remember', insight_id=fi.id, detail=fi.content,
        after=insight_to_delta_dict(fi))

    semantic_threshold = _resolve_semantic_threshold(
        backend, store_name=store_name)
    edge_stats = fast_edges(backend, fi)
    edge_stats['entity'] = create_entity_edges(backend, fi)
    edge_stats['semantic'] = create_semantic_edges(
        backend, fi, embed_cache, threshold=semantic_threshold)

    if linking:
        for target_id, carried_edges in carried:
            move_edges(backend, target_id, fi.id, carried_edges)
        # Notes:
        # - Sweeps the edges this write just minted that name a
        #   target; `supersede` removed only the edges that existed
        #   before the plan ran, and a target already superseded must
        #   stay edgeless too.
        # - The target leaves the drain cache with its edges, or the
        #   next row of the same drain finds it as a semantic
        #   neighbor and mints the edge straight back.
        for target_id, _relation in plan.targets:
            backend.edges.delete_by_node(target_id)
            embed_cache.pop(target_id, None)

    backend.nodes.stamp_linked(fi.id)
    if plan.enrichment:
        backend.nodes.update_enrichment(
            fi.id,
            keywords=plan.enrichment.get('keywords', []),
            summary=plan.enrichment.get('summary', ''))
        backend.nodes.stamp_enriched(fi.id)

    if linking and not linked_targets:
        reported_action = 'add'
    else:
        reported_action = plan.action
    result: dict[str, Any] = {
        'id': fi.id,
        'content': fi.content,
        'category': fi.category,
        'importance': fi.importance,
        'entities': fi.entities,
        'action': reported_action,
        'created_at': (
            format_timestamp(fi.created_at)
            if fi.created_at is not None else ''),
        'edges_created': dict(edge_stats),
        'enrichment': {
            'keywords': plan.enrichment.get('keywords', []),
            'summary': plan.enrichment.get('summary', ''),
            },
        'embedded': embedded,
        }
    if linking:
        # `replaced_ids` names what this write linked; `targets_gone`
        # names the rows that now hold the topic, one read away, so a
        # degraded add cannot hide them.
        if linked_targets:
            result['replaced_ids'] = [t for t, _relation in linked_targets]
        if targets_gone:
            result['targets_gone'] = targets_gone
    return result
