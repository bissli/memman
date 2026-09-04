"""Remember pipeline — single entry point shared by sync CLI and worker.

Structure:

1. Quality check — early return on reject.
2. LLM fact extraction (unless `no_reconcile`).
3. Read-only snapshot of embeddings + active insights.
4. Planning phase — for each fact: embed, reconcile (LLM), decide
   action, enrich + causal (parallel LLM), re-embed if keywords.
   **No DB writes.**
5. Apply phase — one transaction commits every planned supersession,
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
from typing import Any

import httpx
from memman import trace
from memman.embed import EmbeddingProvider
from memman.embed.vector import cosine_similarity
from memman.exceptions import EmbedCredentialError
from memman.graph.causal import infer_llm_causal_edges
from memman.graph.engine import _resolve_semantic_threshold, fast_edges
from memman.graph.enrichment import build_enriched_text, enrich_with_llm
from memman.graph.entity import create_entity_edges
from memman.graph.semantic import create_semantic_edges
from memman.llm import extract as llm_extract
from memman.llm.client import MemmanLLMClient, get_llm_client
from memman.llm.extract import _WS_COLLAPSE_RE
from memman.search.keyword import keyword_search
from memman.search.quality import check_content_quality
from memman.store.backend import Backend
from memman.store.model import Edge, Insight, format_timestamp
from memman.store.model import insight_to_delta_dict

logger = logging.getLogger('memman')


@functools.lru_cache(maxsize=1)
def compute_prompt_version() -> str:
    """Return a 16-char SHA-256 hash of what a rebuild can replay.

    Returns
    -------
    str
        First 16 hex chars of a SHA-256 over the enrichment prompt,
        the causal-inference prompt, and the resolved
        `MEMMAN_LLM_MODEL_SLOW_METADATA` id.

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
    - Extraction and reconciliation prompts are EXCLUDED. A stored
      row cannot be re-extracted: the source blob leaves the queue
      about a minute after its drain, so there is nothing to replay
      and nothing to report.
    - The metadata model id IS folded in, because `link_pending`
      runs both the enrichment and the causal call on
      `slow_metadata`. The canonical model is excluded for the same
      reason extraction is - it shapes content no rebuild rewrites.
    - An unresolvable metadata model hashes as the empty string, so a
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
    from memman.graph.causal import LLM_SYSTEM_PROMPT as CAUSAL_PROMPT
    from memman.graph.enrichment import ENRICHMENT_SYSTEM_PROMPT

    try:
        metadata_model = config.require(config.LLM_MODEL_SLOW_METADATA)
    except ConfigError:
        metadata_model = ''
    blob = (f'{ENRICHMENT_SYSTEM_PROMPT}\x00{CAUSAL_PROMPT}'
            f'\x00{metadata_model}')
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Notes:
# - SIMILARITY_RECONCILE_THRESHOLD gates which stored rows become
#   reconciliation candidates. It is MEASURED INERT over the range it
#   acts on rather than swept for an optimum, which is the honest
#   claim: a restatement's cosine against the row it restates is
#   0.7593 to 1.0000, so the floor has 0.26 of headroom even at the
#   hardest rung and never decides an outcome.
# - The ladder is eight restatements of one fact at increasing
#   distance, byte-identical out to a fully abstract reframing, judged
#   5 times each against sonnet-4.6 with zero disagreement; the
#   reconciler answered NONE on all eight. Four different facts at
#   high lexical overlap were also correct. See
#   experiments/recall_bench/dedup_residual.py and its
#   results/dedup_residual.json (2026-09-02).
# - Do NOT reason about this floor from the RECALL-side cosine
#   distribution, whose median is 0.2424 and whose 99th percentile
#   sits below 0.5. That distribution is query-to-row, where a short
#   query meets a long document; this gate is row-to-row between two
#   paraphrases of comparable length, and the two distributions are
#   nowhere near each other. Reading across them predicts the exact
#   opposite of what the gate does.
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
    enrichment: dict[str, Any] = field(default_factory=dict)
    causal_edges: list[Edge] = field(default_factory=list)
    enriched_vec: list[float] | None = None
    skip_reason: str = ''


def run_remember(
        backend: Backend,
        insight: Insight,
        content: str,
        ec: EmbeddingProvider,
        no_reconcile: bool = False,
        replaced_id: str = '',
        cat_explicit: bool = False,
        imp_explicit: bool = False,
        embed_cache: dict[str, list[float]] | None = None,
        insights_by_id: dict[str, Insight] | None = None,
        executor: ThreadPoolExecutor | None = None,
        llm_client: MemmanLLMClient | None = None,
        *,
        store_name: str,
        ) -> dict[str, Any]:
    """Run the full remember pipeline and return the result dict.

    See module docstring for the overall shape.

    `ec` is the store-bound embed client (resolved from the store's
    `meta.embed_fingerprint` via `bound_embedder`); production callers
    pass `_StoreContext.ec`. `embed_cache`, `insights_by_id`,
    `executor`, `llm_client` are optional drain-scope state hoisted
    by `_drain_queue` to amortize setup across rows in one drain pass.
    When omitted (e.g., direct test use), the function builds them
    from the backend itself.

    `store_name` selects the per-store surface
    (`MEMMAN_SURFACE_<store>`) for the threshold lookup. It is
    keyword-only and required: an omitted store name silently
    resolves the code-surface row and skips the
    `MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>` override branch
    entirely, which is a wrong threshold rather than a missing one.
    """
    quality_warnings = check_content_quality(content)

    if llm_client is None:
        llm_client = get_llm_client('slow_canonical')
        metadata_llm_client = get_llm_client('slow_metadata')
    else:
        metadata_llm_client = llm_client
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

    superseded_in_batch: set[str] = set()

    plans: list[FactPlan] = []
    pending_replaced_id = replaced_id
    prompt_version = compute_prompt_version()
    llm_model_id = llm_client.model
    embed_model = ec.model
    try:
        for fact in facts:
            plan, calls = _plan_fact(
                fact, insight, pending_replaced_id, no_reconcile,
                cat_explicit, imp_explicit, insights_by_id,
                embed_cache, superseded_in_batch, llm_client,
                metadata_llm_client, ec,
                backend, executor)
            llm_calls += calls
            pending_replaced_id = ''

            if plan.fact_insight is not None:
                plan.fact_insight.prompt_version = prompt_version
                plan.fact_insight.model_id = llm_model_id
                plan.fact_insight.embedding_model = embed_model

            if plan.target_id and plan.action in {
                    'update', 'replace', 'supersede'}:
                superseded_in_batch.add(plan.target_id)
                insights_by_id.pop(plan.target_id, None)
                embed_cache.pop(plan.target_id, None)

            if plan.fact_insight and plan.action != 'skipped':
                insights_by_id[plan.fact_insight.id] = plan.fact_insight
                vec = plan.enriched_vec or plan.embed_vec
                if vec is not None:
                    embed_cache[plan.fact_insight.id] = vec

            plans.append(plan)

        _batch_enriched_embeds(plans, ec)

        fact_results: list[dict[str, Any]] = []

        def apply_all() -> None:
            corroborated_ids: set[str] = set()
            for plan in plans:
                result = _apply_plan(
                    backend, plan, embed_cache, store_name=store_name,
                    corroborated_ids=corroborated_ids)
                fact_results.append(result)
                if (plan.action == 'skipped'
                        and result.get('action') == 'add'
                        and plan.fact_insight is not None):
                    # Repair the drain-scoped caches the planning
                    # loop never touched for a skipped plan: evict
                    # the dead target and register the inserted
                    # copy, or every later row exact-matches the
                    # same stale entry and inserts another copy.
                    if plan.target_id:
                        insights_by_id.pop(plan.target_id, None)
                        embed_cache.pop(plan.target_id, None)
                    insights_by_id[plan.fact_insight.id] = (
                        plan.fact_insight)
                    if plan.embed_vec is not None:
                        embed_cache[plan.fact_insight.id] = (
                            plan.embed_vec)

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


def skip_reason_for_result(result: Any) -> str:
    """Return why a `run_remember` result stored nothing, or `''`.

    Parameters
    ----------
    result : Any
        A `run_remember` return value, in either of its two shapes:
        the result-level skip (`action='skipped'`, `skip_reason`) the
        empty extractor produces, or the normal `facts` list whose
        entries each carry `action` and, when skipped, `reason`.
        Typed `Any` rather than `dict` because the sole caller is the
        drain loop, where anything else must read as "stored
        something" instead of raising.

    Returns
    -------
    str
        The reason nothing was stored -- the reasons joined by
        `'; '` when several facts each skipped for their own -- or
        the empty string when the write stored something.

    Notes
    -----
    - A write is lost only when NOTHING landed. A result mixing an
      add with a skip stored the add, so it returns `''`.
    - The two shapes spell the reason differently (`skip_reason` at
      the result level, `reason` per fact). Both must be read, or the
      reconcile skip -- every fact deduped onto an existing insight --
      stays silent.
    - A result of any other type reads as "stored something". The
      caller is the drain loop, where raising would send a row that
      actually succeeded to `mark_failed` and a retry.
    """
    if not isinstance(result, dict):
        return ''
    if result.get('action') == 'skipped':
        return result.get('skip_reason') or 'skipped'
    facts = result.get('facts') or []
    if not facts:
        return ''
    if any(f.get('action') != 'skipped' for f in facts):
        return ''
    reasons = sorted({f.get('reason', '') for f in facts if f.get('reason')})
    return '; '.join(reasons) or 'skipped'


def _batch_enriched_embeds(
        plans: list[FactPlan], ec: Any) -> None:
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
    except EmbedCredentialError:
        raise
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
        fact: dict[str, Any],
        parent: Insight,
        replaced_id: str,
        no_reconcile: bool,
        cat_explicit: bool,
        imp_explicit: bool,
        insights_by_id: dict[str, Insight],
        embed_cache: dict[str, list[float]],
        superseded_in_batch: set[str],
        llm_client: Any,
        metadata_llm_client: Any,
        ec: Any,
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
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

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
            if hit_ins.id in seen or hit_ins.id in superseded_in_batch:
                continue
            similar.append((hit_ins.id, hit_ins.content))
            seen.add(hit_ins.id)

        if fact_vec is not None:
            cosine_cands: list[tuple[float, str, str]] = []
            for eid, evec in embed_cache.items():
                if eid in seen or eid in superseded_in_batch:
                    continue
                ins = insights_by_id.get(eid)
                if ins is None:
                    continue
                sim = cosine_similarity(fact_vec, evec)
                if sim >= SIMILARITY_RECONCILE_THRESHOLD:
                    cosine_cands.append((sim, ins.id, ins.content))
            cosine_cands.sort(key=lambda c: c[0], reverse=True)
            for _sim, cid, ccontent in cosine_cands:
                if len(similar) >= MAX_SIMILAR_FOR_RECONCILE:
                    break
                similar.append((cid, ccontent))
                seen.add(cid)

        if similar:
            # Exact-match rung: byte-identical content (modulo case
            # and whitespace) needs no LLM judgement when exactly ONE
            # stored row matches. Two identical stored rows mean the
            # store is already inconsistent, and which one to merge
            # into is exactly the judgement worth an LLM call. Full
            # normalized equality only -- `in` would swallow every
            # superset fact.
            normalized = _WS_COLLAPSE_RE.sub(
                ' ', fact_text).strip().lower()
            exact_ids = [
                sid for sid, scontent in similar
                if _WS_COLLAPSE_RE.sub(' ', scontent).strip().lower()
                == normalized]
            if len(exact_ids) == 1:
                return FactPlan(
                    action='skipped',
                    fact_text=fact_text,
                    fact_insight=Insight(
                        id=str(uuid.uuid4()), content=fact_text,
                        category=fact_category,
                        importance=fact_importance,
                        entities=fact_entities + list(parent.entities),
                        source=parent.source,
                        access_count=parent.access_count,
                        created_at=parent.created_at,
                        updated_at=parent.updated_at,
                        session_id=parent.session_id,
                        queue_uuid=parent.queue_uuid),
                    target_id=exact_ids[0],
                    # Carry the already-computed vector so a target
                    # soft-deleted between planning and apply can
                    # degrade to an embedded add at no extra cost.
                    embed_vec=fact_vec,
                    skip_reason='exact duplicate',
                    ), calls
            recon = llm_extract.reconcile_memories(
                llm_client, [fact], similar)
            calls += 1
            if recon:
                r = recon[0]
                action = r['action']
                target_id = r.get('target_id')
                merged_text = r.get('merged_text')

    if (action in {'UPDATE', 'REPLACE', 'SUPERSEDE'}
            and target_id
            and target_id in superseded_in_batch):
        # An earlier fact in this write already took the target. A
        # second pointer would fork the chain and a skip would drop
        # the fact, so it lands as a plain add.
        trace.event(
            'batch_target_taken', target_id=target_id, action=action)
        action, target_id, merged_text = 'ADD', None, None

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
        updated_at=parent.updated_at,
        session_id=parent.session_id,
        queue_uuid=parent.queue_uuid)

    embed_vec = fact_vec
    if merged_text:
        try:
            embed_vec = ec.embed(effective_text)
        except EmbedCredentialError:
            raise
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning(
                f'merged embed failed; falling back to fact vector:'
                f' {exc}')

    if action == 'NONE':
        # Carry the target the model named, and the vector alongside
        # it for the same reason the exact-match rung does: a target
        # soft-deleted between planning and apply degrades to an add,
        # which reads `plan.embed_vec`.
        return FactPlan(
            action='skipped',
            fact_text=fact_text,
            fact_insight=fact_insight,
            target_id=target_id,
            embed_vec=embed_vec,
            skip_reason='already captured',
            ), calls

    enrichment: dict[str, Any] = {}
    causal_edges: list[Edge] = []

    def _do_enrich() -> dict[str, Any]:
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
        corroborated_ids: set[str] | None = None,
        ) -> dict[str, Any]:
    """Apply one planned fact. Must be invoked inside a transaction.

    `store_name` selects the per-store surface for the calibrated
    semantic-edge threshold lookup. It is keyword-only and required
    for the same reason as on `run_remember`: an omitted store name
    resolves the code-surface row and skips the per-store override,
    giving a wrong threshold rather than none. `corroborated_ids` is
    the caller's per-invocation dedup set: an extractor emitting the
    same fact twice in one row must bump its target once, not per
    occurrence.

    Notes
    -----
    - `update`, `replace` and `supersede` share one path: the target
      is superseded (never deleted), its edges move to the successor,
      and the successor inherits the entity union and recall history.
      They differ only in the oplog operation name and in whether the
      corroboration count carries, which `supersede` withholds.
    - A target that is not current (forgotten, or superseded by an
      earlier write) degrades the plan to a plain add whose result
      names the target and its successor.
    """
    corroborate_degraded = False
    if plan.action == 'skipped':
        skip_fi = plan.fact_insight
        # The exact-match rung and the reconciler's NONE verdict both
        # set target_id; the dedup-sibling and target-deleted skips
        # carry none.
        corroborated = False
        already_counted = (
            corroborated_ids is not None
            and plan.target_id in corroborated_ids)
        if plan.target_id and not already_counted:
            corroborated = backend.nodes.increment_corroboration(
                plan.target_id,
                queue_uuid=skip_fi.queue_uuid if skip_fi else None)
            if corroborated:
                if corroborated_ids is not None:
                    corroborated_ids.add(plan.target_id)
                backend.oplog.log(
                    operation='reconcile-corroborate',
                    insight_id=plan.target_id,
                    detail=f'restated by: {plan.fact_text[:200]}')
        if not plan.target_id or already_counted or corroborated:
            return {
                'id': skip_fi.id if skip_fi else str(uuid.uuid4()),
                'content': (skip_fi.content if skip_fi
                            else plan.fact_text),
                'action': 'skipped',
                'reason': plan.skip_reason,
                'target_id': plan.target_id,
                }
        # The exact-match target was soft-deleted between planning
        # and apply (an external forget); a skip here
        # would store the fact nowhere, so fall through to a plain
        # add carrying the vector computed before the rung. Mark the
        # dead target counted so a duplicate fact in the same row
        # skips against the copy this add inserts.
        corroborate_degraded = True
        if corroborated_ids is not None and plan.target_id:
            corroborated_ids.add(plan.target_id)
        logger.warning(
            f'corroborate target {plan.target_id} already deleted;'
            ' degrading to add')

    assert plan.fact_insight is not None, (
        'non-skipped FactPlan must carry a fact_insight')
    fi = plan.fact_insight

    target_already_gone = False
    target_superseded_by: str | None = None
    carried_edges: list[Edge] = []
    if plan.action in {'update', 'replace', 'supersede'} and plan.target_id:
        op_name = {
            'replace': 'replace',
            'update': 'reconcile-update',
            'supersede': 'reconcile-supersede',
            }[plan.action]
        before_target = backend.nodes.get_include_deleted(plan.target_id)
        # Snapshot before the pointer is written: `supersede` removes
        # the predecessor's edges, and a later snapshot would also
        # scoop up the plan's causal edges and the successor's own
        # freshly minted ones.
        carried_edges = backend.edges.by_node(plan.target_id)
        # Notes:
        # - The pointer is written BEFORE `nodes.insert`, and the
        #   position is load-bearing: `create_temporal_edge` reads
        #   `get_latest_by_session` and `get_recent_in_window`, so the
        #   predecessor must already be out of the active set or the
        #   successor chains its backbone to the row it replaced.
        # - The predecessor keeps its content behind `superseded_by`;
        #   what the successor copies is what the CURRENT view keeps.
        #   Entities union rather than overwrite because the extractor
        #   sees only the incoming text and would narrow the merged
        #   row's entity set on every pass; recall history carries.
        # - Corroboration carries on a refinement, not on a
        #   contradiction: it counts restatements of the claim the
        #   supersede just falsified.
        linked = backend.nodes.supersede(plan.target_id, fi.id)
        if not linked or before_target is None:
            target_already_gone = True
            carried_edges = []
            target_superseded_by = (
                before_target.superseded_by
                if before_target is not None else None)
            logger.warning(
                f'{plan.action} target {plan.target_id} is not current;'
                ' degrading to add')
        else:
            fi.entities = list(dict.fromkeys(
                list(fi.entities) + list(before_target.entities)))
            fi.access_count = max(
                fi.access_count, before_target.access_count)
            if plan.action != 'supersede':
                fi.corroboration_count = max(
                    fi.corroboration_count,
                    before_target.corroboration_count)
            detail = f'replaced by {fi.id}'
            if plan.action == 'supersede' and fi.content == plan.fact_text:
                # The model supplied no merged text, so the successor
                # may have dropped clauses of the predecessor that are
                # still true; the marker makes that rate measurable.
                detail += ' (unmerged)'
                trace.event(
                    'supersede_unmerged', target_id=plan.target_id)
            backend.oplog.log(
                operation=op_name, insight_id=plan.target_id,
                detail=detail,
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(fi))

    backend.nodes.insert(fi)
    stored = backend.nodes.get(fi.id)
    if stored is not None and stored.created_at is not None:
        fi.created_at = stored.created_at
        fi.updated_at = stored.updated_at

    final_vec = plan.enriched_vec or plan.embed_vec
    embedded = final_vec is not None
    if final_vec is not None:
        backend.nodes.update_embedding(
            fi.id, final_vec, fi.embedding_model or '')
    if fi.entities:
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

    for edge in plan.causal_edges:
        backend.edges.upsert(edge)

    if (plan.action in {'update', 'replace', 'supersede'}
            and plan.target_id and not target_already_gone):
        move_edges(backend, plan.target_id, fi.id, carried_edges)
        # Sweeps the causal edges the plan itself aimed at the
        # predecessor; `supersede` removed only the edges that existed
        # before the plan ran.
        backend.edges.delete_by_node(plan.target_id)

    backend.nodes.stamp_linked(fi.id)
    if plan.enrichment:
        backend.nodes.update_enrichment(
            fi.id,
            keywords=plan.enrichment.get('keywords', []),
            summary=plan.enrichment.get('summary', ''),
            semantic_facts=plan.enrichment.get('semantic_facts', []))
        backend.nodes.stamp_enriched(fi.id)

    reported_action = ('add' if target_already_gone
                       or corroborate_degraded else plan.action)
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
    if corroborate_degraded:
        # The degraded add supersedes nothing -- naming the dead
        # target as `replaced_id` would claim a replace that never
        # happened; `target_id` still names the row that vanished.
        result['target_id'] = plan.target_id
    elif target_already_gone:
        # The row that now holds the topic is one read away; a silent
        # add would hide it.
        result['target_id'] = plan.target_id
        result['target_superseded_by'] = target_superseded_by
    elif plan.target_id:
        result['replaced_id'] = plan.target_id
    return result
