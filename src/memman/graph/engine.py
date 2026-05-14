"""Graph engine: orchestrates automatic edge creation when insights are stored."""

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from memman.embed import EmbeddingProvider
from memman.embed import thresholds as embed_thresholds
from memman.embed._thresholds_generated import _THRESHOLDS as _CALIBRATED
from memman.embed.fingerprint import stored_fingerprint
from memman.graph.entity import MAX_ENTITY_LINKS, MAX_TOTAL_ENTITY_EDGES
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import create_semantic_edges
from memman.graph.temporal import MAX_PROXIMITY_EDGES, MIN_PROXIMITY_WEIGHT
from memman.graph.temporal import TEMPORAL_WINDOW_HOURS, create_temporal_edge
from memman.llm.client import MemmanLLMClient
from memman.store.backend import Backend
from memman.store.model import Edge, Insight

logger = logging.getLogger('memman')


def fast_edges(backend: Backend, insight: Insight) -> dict[str, int]:
    """Run cheap edge generators (temporal + entity).

    LLM causal inference is deferred to link_pending().
    Semantic edges are deferred to link_pending().
    """
    return {
        'temporal': create_temporal_edge(backend, insight),
        'entity': create_entity_edges(backend, insight),
        }


MAX_LINK_BATCH = 20


def link_pending(
        backend: Backend,
        embed_cache: dict[str, list[float]] | None = None,
        llm_client: MemmanLLMClient | None = None,
        metadata_llm_client: MemmanLLMClient | None = None,
        embed_client: EmbeddingProvider | None = None,
        max_batch: int = MAX_LINK_BATCH,
        on_progress: Callable[[str, Insight], None] | None = None,
        store_name: str | None = None,
        ) -> int:
    """Process insights where linked_at IS NULL.

    Creates semantic edges (and optionally LLM causal/enrichment edges)
    for pending insights. Returns the number of insights processed.

    `store_name` plumbs through to `_resolve_semantic_threshold` so the
    per-store surface (`MEMMAN_SURFACE_<store>`) selects the right row
    of `_thresholds_generated.py`. Defaults to None for back-compat;
    `None` resolves the code-surface threshold (the soft fallback in
    `config.get_store_surface`).
    """
    pending_ids = backend.nodes.get_pending_link_ids(limit=max_batch)
    if not pending_ids:
        return 0

    if embed_cache is None:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())

    semantic_threshold = _resolve_semantic_threshold(
        backend, store_name=store_name)

    if metadata_llm_client is None:
        metadata_llm_client = llm_client

    from memman import config
    from memman.graph.causal import infer_llm_causal_edges
    from memman.graph.enrichment import enrich_with_llm
    from memman.pipeline.remember import compute_prompt_version

    try:
        active_pv: str | None = compute_prompt_version()
    except Exception:
        active_pv = None
    try:
        active_model: str | None = config.require(
            config.LLM_MODEL_SLOW_CANONICAL)
    except Exception:
        active_model = None

    processed = 0

    for insight_id in pending_ids:
        insight = backend.nodes.get(insight_id)
        if insight is None:
            continue

        if on_progress:
            on_progress('enrich', insight)

        def _do_causal() -> list[Edge]:
            with backend.readonly_context() as ro:
                return infer_llm_causal_edges(
                    ro, insight, metadata_llm_client)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_enrich = pool.submit(
                enrich_with_llm, insight, metadata_llm_client)
            fut_causal = pool.submit(_do_causal)
            try:
                enrichment = fut_enrich.result()
            except Exception:
                enrichment = {}
            if on_progress:
                on_progress('causal', insight)
            try:
                causal_edges = fut_causal.result()
            except Exception:
                causal_edges = []

        keywords = enrichment.get('keywords', [])
        new_vec = None
        new_vec_model = ''
        reembed_failed = False
        if (keywords
                and embed_client is not None
                and embed_client.available()):
            from memman.graph.enrichment import build_enriched_text
            enriched_text = build_enriched_text(
                insight.content, keywords)
            try:
                new_vec = embed_client.embed(enriched_text)
                new_vec_model = embed_client.model or ''
            except Exception as exc:
                reembed_failed = True
                logger.warning(
                    'Re-embed failed for %s: %s; insight will not flip'
                    ' to enriched (retry on next pass)',
                    insight.id, exc)

        def _write_results() -> int:
            if enrichment:
                backend.nodes.update_enrichment(
                    insight.id,
                    keywords=enrichment.get('keywords', []),
                    summary=enrichment.get('summary', ''),
                    semantic_facts=enrichment.get('semantic_facts', []))
                backend.nodes.update_entities(
                    insight.id, enrichment.get('entities', []))
                insight.entities = enrichment.get('entities', [])

            if new_vec is not None:
                if embed_cache is not None:
                    embed_cache[insight.id] = new_vec
                backend.nodes.update_embedding(
                    insight.id, new_vec, new_vec_model)

            backend.edges.delete_auto_for_node(insight.id, 'entity')
            create_entity_edges(backend, insight)

            backend.edges.delete_auto_for_node(insight.id, 'semantic')
            sem_count = create_semantic_edges(
                backend, insight, embed_cache,
                threshold=semantic_threshold)

            backend.edges.delete_auto_for_node(insight.id, 'causal')
            for edge in causal_edges:
                backend.edges.upsert(edge)

            backend.nodes.stamp_linked(insight_id)
            if enrichment and not reembed_failed:
                backend.nodes.stamp_enriched(
                    insight_id,
                    prompt_version=active_pv,
                    model_id=active_model)
            return sem_count

        with backend.transaction():
            semantic_count = _write_results()
        if on_progress:
            on_progress('done', insight)
        processed += 1
        logger.debug(
            f'Linked {insight_id}: {semantic_count} semantic edges')

    return processed


def compute_constants_hash() -> str:
    """Return a short SHA-256 hash of all edge-relevant constants.

    Embed provider/model/dim drift is owned per-store by the stored
    `meta.embed_fingerprint` (resolved via `bound_embedder`) and not
    folded into this hash, which only governs silent edge-reindex
    on graph-shape constant changes.
    """
    blob = json.dumps({
        'calibrated_thresholds': sorted(
            (p, m, s, t) for (p, m, s), t in _CALIBRATED.items()),
        'min_proximity_weight': MIN_PROXIMITY_WEIGHT,
        'temporal_window_hours': TEMPORAL_WINDOW_HOURS,
        'max_entity_links': MAX_ENTITY_LINKS,
        'max_total_entity_edges': MAX_TOTAL_ENTITY_EDGES,
        'max_proximity_edges': MAX_PROXIMITY_EDGES,
        }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _resolve_semantic_threshold(
        backend: Backend,
        store_name: str | None = None) -> float | None:
    """Return the AUTO_SEMANTIC_THRESHOLD for this store.

    Precedence (returns the first that applies):
      1. `MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>` env-file override
         (numeric, or `'skip'`/`'none'` sentinel). Consulted before
         the fingerprint check so an operator-set override applies
         on a fresh store with no stamped fingerprint yet.
      2. Calibrated table row keyed by `(provider, model, surface)`.
      3. Surface-wide median fallback when the triple is uncalibrated.

    Returns `None` only when (a) the override is `'skip'`, or (b)
    no override is set AND no fingerprint is stamped. Callers
    propagate `None` to `create_semantic_edges` which then skips
    semantic-edge creation.

    Invalid env-file overrides are logged at WARNING and treated as
    no-override; doctor's `check_per_store_keys` is the validation
    entry point where operators see the structured error.
    """
    from memman import config
    if store_name is not None:
        try:
            override = config.get_store_auto_threshold(store_name)
        except ValueError as err:
            logger.warning(
                'invalid %s; ignoring override (%s)',
                config.AUTO_THRESHOLD_FOR(store_name), err)
            override = None
        if override == 'skip':
            return None
        if isinstance(override, float):
            return override
    fp = stored_fingerprint(backend)
    if fp is None:
        return None
    surface = ('code' if store_name is None
               else config.get_store_surface(store_name))
    threshold, _source = embed_thresholds.resolve_with_fallback(
        fp.provider, fp.model, surface)
    return threshold


REINDEX_CHUNK_SIZE = 10


def reindex_auto_edges(
        backend: Backend, dry_run: bool = False,
        store_name: str | None = None,
        chunk_size: int = REINDEX_CHUNK_SIZE) -> dict[str, int]:
    """Delete auto-created edges and re-create semantic/entity edges.

    Heuristic causal edges are deleted (replaced by LLM in Tier 3).
    LLM/manual causal edges are preserved.

    `store_name` plumbs through to `_resolve_semantic_threshold` for
    per-store surface dispatch; None resolves the code-surface default.

    Chunking: the per-insight delete-then-create loop runs in
    transactions of `chunk_size` insights so each lock-hold stays
    sub-second. A reader (e.g. recall on the hot path) waiting on the
    write lock sees at most ~200-400ms of blocking per chunk, well
    inside the 5s `busy_timeout`. The bulk deletes of `semantic`,
    `entity`, `causal`, and low-weight temporal edges happen in a
    short global pre-pass; per-chunk per-insight delete-then-create
    is idempotent because `backend.edges.upsert` is additive.

    Crash semantics: if reindex crashes mid-chunk, the caller's
    `reindex_if_constants_changed` will NOT stamp the new
    constants_hash, so the next drain retries. Per-insight
    delete-then-create makes the retry idempotent.
    """
    semantic_threshold = _resolve_semantic_threshold(
        backend, store_name=store_name)
    semantic_del = backend.edges.count_auto_by_type('semantic')
    entity_del = backend.edges.count_auto_by_type('entity')
    temporal_del = backend.edges.count_low_weight_temporal_proximity(
        min_weight=MIN_PROXIMITY_WEIGHT)
    causal_del = backend.edges.count_auto_by_type('causal')

    if dry_run:
        dry_stats: dict[str, int] = {
            'semantic_deleted': semantic_del,
            'entity_deleted': entity_del,
            'temporal_pruned': temporal_del,
            'causal_deleted': causal_del,
            'semantic_created': 0,
            'entity_created': 0,
            'dry_run': 1,
            }
        dry_insights = backend.nodes.get_all_active()
        if dry_insights:
            dry_embed_cache = dict(
                backend.nodes.iter_embeddings_as_vecs())
            for ins in dry_insights:
                dry_stats['entity_created'] += create_entity_edges(
                    backend, ins, dry_run=True)
                dry_stats['semantic_created'] += create_semantic_edges(
                    backend, ins, dry_embed_cache, dry_run=True,
                    threshold=semantic_threshold)
        return dry_stats

    stats: dict[str, int] = {
        'semantic_deleted': semantic_del,
        'entity_deleted': entity_del,
        'temporal_pruned': temporal_del,
        'causal_deleted': causal_del,
        'semantic_created': 0,
        'entity_created': 0,
        }

    with backend.transaction(), backend.write_lock('reindex'):
        backend.edges.delete_auto_by_type('semantic')
        backend.edges.delete_auto_by_type('entity')
        backend.edges.delete_low_weight_temporal_proximity(
            min_weight=MIN_PROXIMITY_WEIGHT)
        backend.edges.delete_auto_by_type('causal')

    insights = backend.nodes.get_all_active()
    if insights:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())
        for start in range(0, len(insights), chunk_size):
            chunk = insights[start:start + chunk_size]
            with backend.transaction(), backend.write_lock('reindex'):
                for insight in chunk:
                    backend.edges.delete_auto_for_node(
                        insight.id, 'semantic')
                    backend.edges.delete_auto_for_node(
                        insight.id, 'entity')
                    create_entity_edges(backend, insight)
                    create_semantic_edges(
                        backend, insight, embed_cache,
                        threshold=semantic_threshold)

    with backend.transaction(), backend.write_lock('reindex'):
        stats['entity_created'] = backend.edges.count_auto_by_type(
            'entity')
        stats['semantic_created'] = backend.edges.count_auto_by_type(
            'semantic')
        backend.nodes.clear_linked_at()
        backend.oplog.log(
            operation='reindex', insight_id='',
            detail=json.dumps(stats))

    return stats


def reindex_if_constants_changed(
        backend: Backend,
        store_name: str | None = None) -> dict[str, int] | None:
    """Reindex auto-edges when the stored constants hash is stale.

    Returns the reindex stats dict on reindex, or None when the stored
    hash already matched (common case). Emits a WARNING when constants
    drifted (operator hint to run `memman graph rebuild`) and a DEBUG
    line on first-time initialization. `store_name` plumbs through to
    `reindex_auto_edges` for per-store surface dispatch.
    """
    current_hash = compute_constants_hash()
    stored_hash = backend.meta.get('constants_hash')
    if stored_hash == current_hash:
        return None

    stats = reindex_auto_edges(backend, store_name=store_name)
    backend.meta.set('constants_hash', current_hash)
    if stored_hash is not None:
        logger.warning(
            f'edge constants changed (hash {stored_hash} ->'
            f' {current_hash}); reindexed edges and cleared'
            ' linked_at. Run `memman graph rebuild` to re-embed'
            ' and re-enrich.')
    else:
        logger.debug(
            f'initial constants hash set: {current_hash}')
    return stats
