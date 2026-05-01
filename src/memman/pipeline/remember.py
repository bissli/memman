"""Remember pipeline — single entry point shared by sync CLI and worker.

Structure:

1. Quality check — early return on reject.
2. LLM fact extraction (unless `no_reconcile`).
3. Read-only snapshot of embeddings + active insights.
4. Planning phase — for each fact: embed, reconcile (LLM), decide
   action, enrich + causal (parallel LLM), re-embed if keywords.
   **No DB writes.**
5. Apply phase — one transaction commits every planned soft-delete,
   insert, edge, enrichment update, and stamp.

The apply phase runs only after all LLM + embed work has returned.
Crashes during planning leave the DB untouched; the retry path
re-runs the whole pipeline cleanly. This closes the partial-write
fact-loss gap for a single queue row.
"""

import functools
import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from memman.embed import get_client
from memman.embed.vector import cosine_similarity
from memman.graph.causal import infer_llm_causal_edges
from memman.graph.engine import fast_edges
from memman.graph.enrichment import build_enriched_text, enrich_with_llm
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import create_semantic_edges
from memman.llm import extract as llm_extract
from memman.llm.client import get_llm_client
from memman.search.keyword import keyword_search
from memman.search.quality import check_content_quality
from memman.store.backend import Backend
from memman.store.model import MAX_INSIGHTS, Edge, Insight, format_timestamp

logger = logging.getLogger('memman')


@functools.lru_cache(maxsize=1)
def compute_prompt_version() -> str:
    """Return a 16-char SHA-256 hash of the write-path system prompts.

    Covers every system prompt that can mutate what lands in the store
    (fact extraction, reconciliation, LLM enrichment, LLM causal
    inference). Query-time prompts (QUERY_EXPANSION) are excluded
    because they don't affect stored content. The hash is cached for
    the life of the process — the prompts are module-level constants.

    Note: the slow-role model ids are *not* part of this hash. Swapping
    `MEMMAN_LLM_MODEL_SLOW_CANONICAL` or `MEMMAN_LLM_MODEL_SLOW_METADATA`
    to a model that produces structurally different facts will not
    invalidate stored insights. Run `memman graph rebuild` after a
    model swap to re-enrich.
    """
    from memman.graph.causal import LLM_SYSTEM_PROMPT as CAUSAL_PROMPT
    from memman.graph.enrichment import ENRICHMENT_SYSTEM_PROMPT
    from memman.llm.extract import FACT_EXTRACTION_SYSTEM
    from memman.llm.extract import RECONCILIATION_SYSTEM

    blob = f'{FACT_EXTRACTION_SYSTEM}\x00{RECONCILIATION_SYSTEM}\x00{ENRICHMENT_SYSTEM_PROMPT}\x00{CAUSAL_PROMPT}'
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


SIMILARITY_RECONCILE_THRESHOLD = 0.5
MAX_SIMILAR_FOR_RECONCILE = 10
KEYWORD_HITS_LIMIT = 5


@dataclass
class FactPlan:
    """Planned write for one extracted fact.
    """

    action: str
    fact_text: str
    fact_insight: Insight | None = None
    target_id: str | None = None
    embed_vec: list[float] | None = None
    enrichment: dict = field(default_factory=dict)
    causal_edges: list[Edge] = field(default_factory=list)
    enriched_vec: list[float] | None = None
    skip_reason: str = ''


def run_remember(
        backend: Backend,
        insight: Insight,
        content: str,
        no_reconcile: bool = False,
        replaced_id: str = '',
        cat_explicit: bool = False,
        imp_explicit: bool = False,
        embed_cache: dict[str, list[float]] | None = None,
        insights_by_id: dict[str, Insight] | None = None,
        executor: ThreadPoolExecutor | None = None,
        llm_client=None,
        ec=None,
        ) -> dict:
    """Run the full remember pipeline and return the result dict.

    See module docstring for the overall shape.

    `embed_cache`, `insights_by_id`, `executor`, `llm_client`, `ec`
    are optional drain-scope state hoisted by `_drain_queue` to amortize
    setup across rows in one drain pass. When omitted (e.g., direct
    test use), the function builds them from the backend itself.
    """
    quality_warnings = check_content_quality(content)

    if llm_client is None:
        llm_client = get_llm_client('slow_canonical')
        metadata_llm_client = get_llm_client('slow_metadata')
    else:
        metadata_llm_client = llm_client
    if ec is None:
        ec = get_client()
    llm_calls = 0

    if no_reconcile:
        facts = [{
            'text': content,
            'category': insight.category,
            'importance': insight.importance,
            'entities': [],
            }]
    else:
        facts = llm_extract.extract_facts(llm_client, content)
        llm_calls += 1
        if not facts:
            return {
                'id': insight.id,
                'content': content,
                'action': 'skipped',
                'skip_reason': 'trivial content',
                'quality_warnings': quality_warnings,
                'llm_calls': llm_calls,
                }

    if embed_cache is None:
        embed_cache = dict(backend.nodes.iter_embeddings_as_vecs())
    if insights_by_id is None:
        all_insights = backend.nodes.get_all_active()
        insights_by_id = {i.id: i for i in all_insights}

    owned_executor: ThreadPoolExecutor | None = None
    if executor is None:
        owned_executor = ThreadPoolExecutor(max_workers=2)
        executor = owned_executor

    deleted_in_batch: set[str] = set()

    plans: list[FactPlan] = []
    pending_replaced_id = replaced_id
    prompt_version = compute_prompt_version()
    llm_model_id = getattr(llm_client, 'model', None)
    embed_model = getattr(ec, 'model', None)
    try:
        for fact in facts:
            plan, calls = _plan_fact(
                fact, insight, pending_replaced_id, no_reconcile,
                cat_explicit, imp_explicit, insights_by_id,
                embed_cache, deleted_in_batch, llm_client,
                metadata_llm_client, ec,
                backend, executor)
            llm_calls += calls
            pending_replaced_id = ''

            if plan.fact_insight is not None:
                plan.fact_insight.prompt_version = prompt_version
                plan.fact_insight.model_id = llm_model_id
                plan.fact_insight.embedding_model = embed_model

            if plan.target_id and plan.action in {
                    'delete', 'update', 'replace'}:
                deleted_in_batch.add(plan.target_id)
                insights_by_id.pop(plan.target_id, None)
                embed_cache.pop(plan.target_id, None)

            if plan.fact_insight and plan.action != 'skipped':
                insights_by_id[plan.fact_insight.id] = plan.fact_insight
                vec = plan.enriched_vec or plan.embed_vec
                if vec is not None:
                    embed_cache[plan.fact_insight.id] = vec

            plans.append(plan)

        _batch_enriched_embeds(plans, ec)

        fact_results: list[dict] = []

        def apply_all() -> None:
            new_ids: list[str] = []
            for plan in plans:
                result = _apply_plan(backend, plan, embed_cache)
                fact_results.append(result)
                if plan.fact_insight and plan.action not in {
                        'skipped', 'deleted'}:
                    new_ids.append(plan.fact_insight.id)

            for nid in new_ids:
                try:
                    ei = backend.nodes.refresh_effective_importance(nid)
                except Exception:
                    ei = 0.0
                for r in fact_results:
                    if r.get('id') == nid:
                        r['effective_importance'] = ei
                        break

            if new_ids:
                try:
                    pruned = backend.nodes.auto_prune(
                        max_insights=MAX_INSIGHTS, exclude_ids=new_ids)
                except Exception:
                    pruned = 0
                for r in fact_results:
                    if r.get('id') in new_ids:
                        r['auto_pruned'] = pruned
                        break

        with backend.transaction():
            apply_all()
    finally:
        if owned_executor is not None:
            owned_executor.shutdown(wait=True)

    return {
        'facts': fact_results,
        'quality_warnings': quality_warnings,
        'llm_calls': llm_calls,
        }


def _batch_enriched_embeds(
        plans: list[FactPlan], ec: object) -> None:
    """Embed every plan's enriched text in one HTTP round-trip.

    Called once per row after planning completes. Plans whose
    enrichment yielded keywords get an enriched-text embedding
    written back into `enriched_vec`. Plans without keywords are
    untouched.
    """
    if ec is None or not ec.available():
        return
    pending: list[tuple[FactPlan, str]] = []
    for plan in plans:
        if plan.fact_insight is None:
            continue
        if plan.enriched_vec is not None:
            continue
        keywords = plan.enrichment.get('keywords', [])
        if not keywords:
            continue
        enriched_text = build_enriched_text(
            plan.fact_insight.content, keywords)
        pending.append((plan, enriched_text))

    if not pending:
        return

    texts = [t for _p, t in pending]
    try:
        vectors = ec.embed_batch(texts)
    except Exception as exc:
        logger.warning(f'enriched-text embed_batch failed: {exc}')
        return

    if len(vectors) != len(pending):
        logger.warning(
            f'embed_batch returned {len(vectors)} for {len(pending)} inputs')
        return

    for (plan, _t), vec in zip(pending, vectors):
        plan.enriched_vec = vec


def _plan_fact(
        fact: dict,
        parent: Insight,
        replaced_id: str,
        no_reconcile: bool,
        cat_explicit: bool,
        imp_explicit: bool,
        insights_by_id: dict[str, Insight],
        embed_cache: dict[str, list[float]],
        deleted_in_batch: set[str],
        llm_client: object,
        metadata_llm_client: object,
        ec: object,
        backend: Backend,
        executor: ThreadPoolExecutor,
        ) -> tuple[FactPlan, int]:
    """Plan a single fact without touching the DB. Returns (plan, llm_calls).

    Enriched-text re-embeds are deferred to a row-level batch pass
    (`_batch_enriched_embeds`) so multiple facts in one row collapse
    into one HTTP round-trip.
    """
    calls = 0
    fact_text = fact['text']
    fact_category = (parent.category if cat_explicit
                     else fact.get('category', parent.category))
    fact_importance = (parent.importance if imp_explicit
                       else fact.get('importance', parent.importance))
    fact_entities = fact.get('entities', [])

    fact_vec = None
    try:
        fact_vec = ec.embed(fact_text)
    except Exception:
        pass

    action = 'ADD'
    target_id: str | None = None
    merged_text: str | None = None

    if replaced_id:
        action = 'REPLACE'
        target_id = replaced_id
    elif not no_reconcile:
        snapshot = list(insights_by_id.values())
        keyword_hits = keyword_search(
            snapshot, fact_text, limit=KEYWORD_HITS_LIMIT)
        similar: list[tuple[str, str]] = []
        seen: set[str] = set()

        for hit_ins, _score in keyword_hits:
            if hit_ins.id in seen or hit_ins.id in deleted_in_batch:
                continue
            similar.append((hit_ins.id, hit_ins.content))
            seen.add(hit_ins.id)

        if fact_vec is not None:
            for eid, evec in embed_cache.items():
                if eid in seen or eid in deleted_in_batch:
                    continue
                ins = insights_by_id.get(eid)
                if ins is None:
                    continue
                sim = cosine_similarity(fact_vec, evec)
                if sim >= SIMILARITY_RECONCILE_THRESHOLD:
                    similar.append((ins.id, ins.content))
                    seen.add(eid)
                if len(similar) >= MAX_SIMILAR_FOR_RECONCILE:
                    break

        if similar:
            recon = llm_extract.reconcile_memories(
                llm_client, [fact], similar)
            calls += 1
            if recon:
                r = recon[0]
                action = r['action']
                target_id = r.get('target_id')
                merged_text = r.get('merged_text')

    if (action in {'UPDATE', 'REPLACE'}
            and target_id
            and target_id in deleted_in_batch):
        return FactPlan(
            action='skipped',
            fact_text=fact_text,
            fact_insight=Insight(
                id=str(uuid.uuid4()), content=merged_text or fact_text,
                category=fact_category, importance=fact_importance,
                entities=fact_entities + list(parent.entities),
                source=parent.source, access_count=parent.access_count,
                created_at=parent.created_at,
                updated_at=parent.updated_at),
            skip_reason='target already deleted',
            ), calls

    fact_id = str(uuid.uuid4())
    effective_text = merged_text or fact_text

    fact_insight = Insight(
        id=fact_id,
        content=effective_text,
        category=fact_category,
        importance=fact_importance,
        entities=fact_entities + list(parent.entities),
        source=parent.source,
        access_count=parent.access_count,
        created_at=parent.created_at,
        updated_at=parent.updated_at)

    embed_vec = fact_vec
    if merged_text:
        try:
            embed_vec = ec.embed(effective_text)
        except Exception:
            pass

    if action == 'NONE':
        return FactPlan(
            action='skipped',
            fact_text=fact_text,
            fact_insight=fact_insight,
            skip_reason='already captured',
            ), calls

    if action == 'DELETE' and target_id:
        if target_id in deleted_in_batch:
            return FactPlan(
                action='skipped',
                fact_text=fact_text,
                fact_insight=fact_insight,
                skip_reason='target already deleted',
                ), calls
        return FactPlan(
            action='delete',
            fact_text=fact_text,
            fact_insight=fact_insight,
            target_id=target_id,
            embed_vec=embed_vec,
            ), calls

    enrichment: dict = {}
    causal_edges: list[Edge] = []

    def _do_enrich() -> dict:
        return enrich_with_llm(fact_insight, metadata_llm_client)

    def _do_causal() -> list[Edge]:
        with backend.readonly_context() as ro:
            return infer_llm_causal_edges(
                ro, fact_insight, metadata_llm_client)

    fut_e = executor.submit(_do_enrich)
    fut_c = executor.submit(_do_causal)
    try:
        enrichment = fut_e.result()
        calls += 1
    except Exception:
        enrichment = {}
    try:
        causal_edges = fut_c.result()
        calls += 1
    except Exception:
        causal_edges = []

    if enrichment:
        fact_insight.entities = enrichment.get('entities', [])

    return FactPlan(
        action=action.lower(),
        fact_text=fact_text,
        fact_insight=fact_insight,
        target_id=target_id,
        embed_vec=embed_vec,
        enrichment=enrichment,
        causal_edges=causal_edges,
        enriched_vec=None,
        ), calls


def _apply_plan(
        backend: Backend,
        plan: FactPlan,
        embed_cache: dict[str, list[float]],
        ) -> dict:
    """Apply one planned fact. Must be invoked inside a transaction.
    """
    fi = plan.fact_insight
    if plan.action == 'skipped':
        return {
            'id': fi.id if fi else str(uuid.uuid4()),
            'content': fi.content if fi else plan.fact_text,
            'action': 'skipped',
            'reason': plan.skip_reason,
            }

    if plan.action == 'delete' and plan.target_id:
        deleted_now = backend.nodes.soft_delete(
            plan.target_id, tolerate_missing=True)
        if deleted_now:
            backend.oplog.log(
                operation='reconcile-delete',
                insight_id=plan.target_id,
                detail=f'contradicted by: {plan.fact_text[:200]}')
        else:
            logger.warning(
                f'reconcile-delete target {plan.target_id} already gone;'
                ' skipping')
        return {
            'id': fi.id,
            'content': fi.content,
            'action': 'deleted' if deleted_now else 'skipped',
            'target_id': plan.target_id,
            }

    target_already_gone = False
    if plan.action in {'update', 'replace'} and plan.target_id:
        op_name = ('replace' if plan.action == 'replace'
                   else 'reconcile-update')
        deleted_now = backend.nodes.soft_delete(
            plan.target_id, tolerate_missing=True)
        if deleted_now:
            backend.oplog.log(
                operation=op_name, insight_id=plan.target_id,
                detail=f'replaced by {fi.id}')
        else:
            target_already_gone = True
            logger.warning(
                f'{plan.action} target {plan.target_id} already deleted;'
                ' degrading to add')

    backend.nodes.insert(fi)

    final_vec = plan.enriched_vec or plan.embed_vec
    embedded = final_vec is not None
    if final_vec is not None:
        backend.nodes.update_embedding(
            fi.id, final_vec, fi.embedding_model or '')
    if fi.entities:
        backend.nodes.update_entities(fi.id, fi.entities)

    backend.oplog.log(
        operation='remember', insight_id=fi.id, detail=fi.content)

    edge_stats = fast_edges(backend, fi)
    edge_stats['entity'] = create_entity_edges(backend, fi)
    edge_stats['semantic'] = create_semantic_edges(
        backend, fi, embed_cache)

    for edge in plan.causal_edges:
        try:
            backend.edges.upsert(edge)
        except Exception:
            pass

    if (plan.action in {'update', 'replace'}
            and plan.target_id and not target_already_gone):
        backend.edges.delete_by_node(plan.target_id)

    backend.nodes.stamp_linked(fi.id)
    if plan.enrichment:
        backend.nodes.update_enrichment(
            fi.id,
            keywords=plan.enrichment.get('keywords', []),
            summary=plan.enrichment.get('summary', ''),
            semantic_facts=plan.enrichment.get('semantic_facts', []))
        backend.nodes.stamp_enriched(fi.id)

    reported_action = 'add' if target_already_gone else plan.action
    result: dict = {
        'id': fi.id,
        'content': fi.content,
        'category': fi.category,
        'importance': fi.importance,
        'entities': fi.entities,
        'action': reported_action,
        'created_at': format_timestamp(fi.created_at),
        'edges_created': {
            **edge_stats,
            'causal': len(plan.causal_edges),
            },
        'enrichment': {
            'keywords': plan.enrichment.get('keywords', []),
            'summary': plan.enrichment.get('summary', ''),
            'entities': plan.enrichment.get('entities', []),
            'semantic_facts': plan.enrichment.get('semantic_facts', []),
            },
        'embedded': embedded,
        }
    if plan.target_id and not target_already_gone:
        result['replaced_id'] = plan.target_id
    return result
