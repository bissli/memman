"""Remember pipeline — single entry point shared by sync CLI and worker.

Structure:

1. Quality check — early return on reject.
2. LLM fact extraction (unless `no_reconcile`).
3. Read-only snapshot of embeddings + active insights.
4. Planning phase - for each fact: embed, shortlist, screen every
   shortlisted row (one LLM call per row, in parallel), show the kept
   rows to the verdict call one per row, assemble the verdicts, write
   one merge text per retiring target (one call each), then enrich +
   causal (parallel LLM) per planned row and re-embed if keywords.
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
import json
import logging
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
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
from memman.llm.extract import _WS_COLLAPSE_RE, UNJUDGED
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
MAX_SIMILAR_FOR_RECONCILE = 20
KEYWORD_HITS_LIMIT = 5


# Notes:
# - The write path's disposition of every screen relation: a kept row
#   is shown to the verdict call; `fallback` rows are shown only when
#   nothing is kept, the first alone; a dropped row is never shown.
# - REFINES rows are half of all candidates and carried the batch's
#   volume; an UNJUDGED row is kept because a screen the model could
#   not answer must not hide a contradiction.
SCREEN_DISPOSITION = {
    'CONTRADICTS': 'keep',
    'RESTATES': 'keep',
    UNJUDGED: 'keep',
    'REFINES': 'fallback',
    'UNRELATED': 'drop',
    }


@dataclass
class Candidate:
    """One reconcile shortlist row and what each stage saw and decided.

    Attributes
    ----------
    id : str
        The stored row.
    rung : str
        `keyword` or `cosine`, the rung that shortlisted it.
    score : float
        The rung's score.
    relation : str | None
        The screen's relation, one of `SCREEN_DISPOSITION`; None when
        the exact-match rung answered before the screen ran.
    screened : bool
        True when the row reached the verdict call.
    verdict : str | None
        The verdict call's answer, `supersede | update | none | keep`;
        None when the row was not screened in.
    """

    id: str
    rung: str
    score: float
    relation: str | None = None
    screened: bool = False
    verdict: str | None = None


def screened_rows(
        similar: list[tuple[str, str]],
        relation_by_id: dict[str, str]) -> list[tuple[str, str]]:
    """The shortlisted rows the verdict call is shown, in shortlist order.

    Parameters
    ----------
    similar : list[tuple[str, str]]
        The shortlist as `(insight_id, content)`.
    relation_by_id : dict[str, str]
        The screen's relation per shortlisted id.

    Returns
    -------
    list[tuple[str, str]]
        Every row whose relation is `keep` in `SCREEN_DISPOSITION`;
        when there is none, the first `fallback` row alone; else empty.
    """
    kept = [row for row in similar
            if SCREEN_DISPOSITION[relation_by_id[row[0]]] == 'keep']
    if kept:
        return kept
    return [row for row in similar
            if SCREEN_DISPOSITION[relation_by_id[row[0]]] == 'fallback'][:1]


def assemble_verdicts(
        kept: list[tuple[str, str]],
        verdict_by_id: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """One fact's action and targets from its per-row verdicts.

    Parameters
    ----------
    kept : list[tuple[str, str]]
        The screened rows as `(insight_id, content)`, in shortlist order.
    verdict_by_id : dict[str, str]
        The verdict call's answer per kept id.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        The action, `SUPERSEDE | UPDATE | NONE | ADD`, and the targets
        as `(insight_id, relation)`.

    Notes
    -----
    - Every `supersede` row is a target. A `none` row folds to `update`
      when another row supersedes: the verdict text's own rule that a
      restating memory is folded into the successor beside a
      contradicted one. The update slot takes the first `update` or
      folded row in shortlist order, alone.
    - With no linking row, the first `none` row is the answer; with
      none of those, ADD.
    """
    kept_ids = [row_id for row_id, _content in kept]
    supersedes = [rid for rid in kept_ids if verdict_by_id[rid] == 'supersede']
    updates = [rid for rid in kept_ids
               if verdict_by_id[rid] == 'update'
               or (verdict_by_id[rid] == 'none' and supersedes)][:1]
    nones = [rid for rid in kept_ids if verdict_by_id[rid] == 'none']
    if supersedes or updates:
        targets = ([(rid, 'supersede') for rid in supersedes]
                   + [(rid, 'update') for rid in updates])
        return ('SUPERSEDE' if supersedes else 'UPDATE'), targets
    if nones:
        return 'NONE', [(nones[0], 'none')]
    return 'ADD', []


@dataclass
class FactPlan:
    """Planned write for one extracted fact, or one of its successors.

    Attributes
    ----------
    action : str
        `add`, `update`, `supersede`, `replace` or `skipped`.
    fact_text : str
        The fact as extracted; a retiring plan's row stores the merge
        text written for its target when one came back.
    fact_insight : Insight | None
        The row the apply phase inserts; None only on a skip that
        carries nothing to degrade into.
    targets : list[tuple[str, str]]
        `(insight_id, relation)` per affected row, relation in
        `update | supersede | replace | none`. A fact that retires
        several rows plans one successor per row, so a retiring plan
        carries one target.
    candidates : list[Candidate]
        The reconcile shortlist with each stage's reading, logged by
        the apply phase for replay; carried by the first plan of a fact
        alone, so the row is logged once per fact.
    """

    action: str
    fact_text: str
    fact_insight: Insight | None = None
    targets: list[tuple[str, str]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
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
            fact_plans, calls = _plan_fact(
                fact, insight, pending_replaced_id, no_reconcile,
                cat_explicit, insights_by_id,
                embed_cache, superseded_in_batch, llm_client,
                metadata_llm_client, ec,
                backend, executor)
            llm_calls += calls
            pending_replaced_id = ''

            for plan in fact_plans:
                if plan.fact_insight is not None:
                    plan.fact_insight.prompt_version = prompt_version
                    plan.fact_insight.model_id = llm_model_id
                    plan.fact_insight.embedding_model = embed_model

                if plan.targets and plan.action in {
                        'update', 'replace', 'supersede'}:
                    for target_id, _relation in plan.targets:
                        superseded_in_batch.add(target_id)
                        insights_by_id.pop(target_id, None)
                        embed_cache.pop(target_id, None)

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
                    if plan.targets:
                        insights_by_id.pop(plan.targets[0][0], None)
                        embed_cache.pop(plan.targets[0][0], None)
                    insights_by_id[plan.fact_insight.id] = (
                        plan.fact_insight)
                    if plan.embed_vec is not None:
                        embed_cache[plan.fact_insight.id] = (
                            plan.embed_vec)

        # Notes:
        # - Planned rows entered the caches so later facts of this write
        #   could shortlist them, but the apply phase inserts them one
        #   at a time and the semantic-edge builder reads the cache: a
        #   planned row still in it is an edge target that does not
        #   exist yet, and the insert fails on the foreign key.
        # - Each plan re-registers its row, with the vector it stored,
        #   once inserted.
        for plan in plans:
            if plan.fact_insight is not None:
                embed_cache.pop(plan.fact_insight.id, None)

        with backend.transaction():
            apply_all()
            # Notes:
            # - A later fact's causal edges were planned while an
            #   earlier fact's target was still current and may name
            #   it; every row this write superseded ends the write
            #   edgeless.
            # - A successor of this write that a later fact of the same
            #   write retired re-entered the cache at its insert; it
            #   leaves again here, or the next row of the drain builds
            #   semantic edges onto a superseded row.
            for target_id in superseded_in_batch:
                backend.edges.delete_by_node(target_id)
                embed_cache.pop(target_id, None)
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
        insights_by_id: dict[str, Insight],
        embed_cache: dict[str, list[float]],
        superseded_in_batch: set[str],
        llm_client: Any,
        metadata_llm_client: Any,
        ec: Any,
        backend: Backend,
        executor: ThreadPoolExecutor,
        ) -> tuple[list[FactPlan], int]:
    """Plan a single fact without touching the DB.

    Parameters
    ----------
    fact : dict[str, Any]
        One extracted fact: `text`, `category`, `entities`.
    parent : Insight
        The queued write the fact came from; its metadata is inherited.
    replaced_id : str
        A `replace` target, or `''`.
    no_reconcile : bool
        True skips every reconcile stage and stores the fact verbatim.
    cat_explicit : bool
        True keeps the parent's category over the extractor's.
    insights_by_id : dict[str, Insight]
        The drain-scope snapshot of current rows.
    embed_cache : dict[str, list[float]]
        The drain-scope vectors of those rows.
    superseded_in_batch : set[str]
        Rows an earlier fact of this write already retired; they leave
        the shortlist, so no later fact can fork their chain.
    llm_client : Any
        The slow canonical client for the three reconcile stages.
    metadata_llm_client : Any
        The slow metadata client for enrichment and causal inference.
    ec : Any
        The store's bound embed provider.
    backend : Backend
        Open store, read only here.
    executor : ThreadPoolExecutor
        The drain's two-worker executor for enrichment and causal.

    Returns
    -------
    tuple[list[FactPlan], int]
        The plans and the LLM calls made. One plan for an add, a
        replace or a skip; one plan per retiring target when the
        verdicts link several rows, each with its own merge text.

    Notes
    -----
    - Stage 1 screens every shortlisted row in parallel on an executor
      sized to the shortlist; stage 2 shows each kept row alone; stage
      3 writes one merge text per retiring target. The stage functions
      are `llm.extract.screen_memory`, `judge_memory` and
      `merge_successor`.
    - Enriched-text re-embeds are deferred to a row-level batch pass
      (`_batch_enriched_embeds`) so multiple plans in one row collapse
      into one HTTP round-trip.
    """
    calls = 0
    fact_text = fact['text']
    fact_category = (parent.category if cat_explicit
                     else fact.get('category', parent.category))
    fact_entities = fact.get('entities', []) + list(parent.entities)

    def new_row(content: str) -> Insight:
        return Insight(
            id=str(uuid.uuid4()), content=content,
            category=fact_category, importance=parent.importance,
            entities=list(fact_entities), source=parent.source,
            access_count=parent.access_count,
            created_at=parent.created_at, updated_at=parent.updated_at,
            session_id=parent.session_id, queue_uuid=parent.queue_uuid)

    fact_vec = None
    try:
        fact_vec = ec.embed(fact_text)
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

    action = 'ADD'
    targets: list[tuple[str, str]] = []
    candidates: list[Candidate] = []
    similar: list[tuple[str, str]] = []
    clauses_by_id: dict[str, list[str]] = {}

    if replaced_id:
        action = 'REPLACE'
        targets = [(replaced_id, 'replace')]
    elif not no_reconcile:
        snapshot = list(insights_by_id.values())
        keyword_hits = keyword_search(
            snapshot, fact_text, limit=KEYWORD_HITS_LIMIT)
        seen: set[str] = set()

        for hit_ins, score in keyword_hits:
            if hit_ins.id in seen or hit_ins.id in superseded_in_batch:
                continue
            similar.append((hit_ins.id, hit_ins.content))
            candidates.append(Candidate(hit_ins.id, 'keyword', float(score)))
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
            for sim, cid, ccontent in cosine_cands:
                if len(similar) >= MAX_SIMILAR_FOR_RECONCILE:
                    break
                similar.append((cid, ccontent))
                candidates.append(Candidate(cid, 'cosine', float(sim)))
                seen.add(cid)

        if similar:
            # Exact-match rung: byte-identical content (modulo case
            # and whitespace) needs no LLM judgment when exactly ONE
            # stored row matches. Two identical stored rows mean the
            # store is already inconsistent, and which one to merge
            # into is exactly the judgment worth an LLM call. Full
            # normalized equality only -- `in` would swallow every
            # superset fact.
            normalized = _WS_COLLAPSE_RE.sub(
                ' ', fact_text).strip().lower()
            exact_ids = [
                sid for sid, scontent in similar
                if _WS_COLLAPSE_RE.sub(' ', scontent).strip().lower()
                == normalized]
            if len(exact_ids) == 1:
                return [FactPlan(
                    action='skipped',
                    fact_text=fact_text,
                    fact_insight=new_row(fact_text),
                    targets=[(exact_ids[0], 'none')],
                    candidates=candidates,
                    # Carry the already-computed vector so a target
                    # soft-deleted between planning and apply can
                    # degrade to an embedded add at no extra cost.
                    embed_vec=fact_vec,
                    skip_reason='exact duplicate',
                    )], calls

            # Notes:
            # - One worker per shortlisted row: a screen call ends in
            #   seconds and twenty of them serialized on the drain's
            #   two-worker executor would take a minute per fact.
            # - The same pool serves stage 2, whose rows are a subset.
            with ThreadPoolExecutor(max_workers=len(similar)) as stage_pool:
                screened = list(stage_pool.map(
                    lambda row: llm_extract.screen_memory(
                        llm_client, fact_text, row),
                    similar))
                calls += len(similar)
                relation_by_id = {
                    row[0]: relation
                    for row, (relation, _clauses) in zip(similar, screened)}
                clauses_by_id = {
                    row[0]: clauses
                    for row, (_relation, clauses) in zip(similar, screened)}
                kept = screened_rows(similar, relation_by_id)
                kept_ids = {row_id for row_id, _content in kept}
                for candidate in candidates:
                    candidate.relation = relation_by_id[candidate.id]
                    candidate.screened = candidate.id in kept_ids
                trace.event(
                    'reconcile_screen', rows=len(similar), kept=len(kept),
                    relations=dict(Counter(relation_by_id.values())))

                if kept:
                    verdicts = list(stage_pool.map(
                        lambda row: llm_extract.judge_memory(
                            llm_client, fact_text, row),
                        kept))
                    calls += len(kept)
                    verdict_by_id = {
                        row[0]: verdict for row, verdict in zip(kept, verdicts)}
                    for candidate in candidates:
                        candidate.verdict = verdict_by_id.get(candidate.id)
                    action, targets = assemble_verdicts(kept, verdict_by_id)

    if action == 'NONE':
        # Carry the target the model named, and the vector alongside
        # it for the same reason the exact-match rung does: a target
        # soft-deleted between planning and apply degrades to an add,
        # which reads `plan.embed_vec`.
        return [FactPlan(
            action='skipped',
            fact_text=fact_text,
            fact_insight=new_row(fact_text),
            targets=targets,
            candidates=candidates,
            embed_vec=fact_vec,
            skip_reason='already captured',
            )], calls

    if action in {'UPDATE', 'SUPERSEDE'}:
        # Notes:
        # - Stage 3, one call per retiring target, in parallel: the
        #   body lists that target alone with the clauses the screen
        #   quoted for it, or `(none)` for an update target, and each
        #   successor stores the text written for its own predecessor.
        # - A merge that returns no text falls back to the fact, which
        #   the apply phase marks `(unmerged)` for that target alone.
        content_by_id = dict(similar)
        merge_targets = [
            (target_id,
             content_by_id[target_id],
             clauses_by_id.get(target_id, []) if relation == 'supersede' else [])
            for target_id, relation in targets]
        with ThreadPoolExecutor(max_workers=len(targets)) as merge_pool:
            merged = list(merge_pool.map(
                lambda target: llm_extract.merge_successor(
                    llm_client, fact_text, target),
                merge_targets))
        calls += len(targets)
        plan_specs = [([target], merged_text or fact_text)
                      for target, merged_text in zip(targets, merged)]
    else:
        plan_specs = [(targets, fact_text)]

    plans: list[FactPlan] = []
    for idx, (plan_targets, content) in enumerate(plan_specs):
        fact_insight = new_row(content)

        embed_vec = fact_vec
        if content != fact_text:
            try:
                embed_vec = ec.embed(content)
            except EmbedCredentialError:
                raise
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning(
                    f'merged embed failed; falling back to fact vector:'
                    f' {exc}')

        def _do_enrich(row: Insight = fact_insight) -> dict[str, Any]:
            return enrich_with_llm(row, metadata_llm_client)

        def _do_causal(row: Insight = fact_insight) -> list[Edge]:
            with backend.readonly_context() as ro:
                return infer_llm_causal_edges(ro, row, metadata_llm_client)

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

        plans.append(FactPlan(
            action=action.lower(),
            fact_text=fact_text,
            fact_insight=fact_insight,
            targets=plan_targets,
            candidates=candidates if idx == 0 else [],
            embed_vec=embed_vec,
            enrichment=enrichment,
            causal_edges=causal_edges,
            enriched_vec=None,
            ))
    return plans, calls


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
    - `update`, `replace` and `supersede` share one path over the
      plan's target list: each target is superseded (never deleted),
      its edges move to the successor, and the successor inherits the
      entity union and recall history of the linked targets. They
      differ only in the oplog operation name and in whether the
      corroboration count carries, which `supersede` withholds. The
      planner hands one target per plan, so the successor is the row a
      query about that one predecessor's subject would have hit.
    - A target that is not current (forgotten, or superseded by an
      earlier write) is dropped into `targets_gone`; the plan degrades
      to a plain add only when every target is gone.
    - Every plan that carried a reconcile shortlist logs it first, as
      a `reconcile-candidates` oplog row with each row's screen
      relation, whether it was shown to the verdict call and the
      verdict, so the decision can be replayed against exactly what
      each stage saw.
    """
    fact_id = plan.fact_insight.id if plan.fact_insight is not None else None
    skip_target = plan.targets[0][0] if plan.targets else None
    if plan.candidates:
        # Notes:
        # - Keyed on the row that will exist: the NONE or exact-match
        #   memory for a skip (no successor is inserted), the new row
        #   otherwise, ADD included. `fact_id` in the detail ties a skip
        #   that later degrades to an add back to the inserted row.
        # - The oplog has no foreign key on `insight_id`, so a row
        #   logged before `nodes.insert` cannot fail.
        key = skip_target if plan.action == 'skipped' else fact_id
        if key is not None:
            backend.oplog.log(
                operation='reconcile-candidates', insight_id=key,
                detail=json.dumps({
                    'fact_id': fact_id,
                    'fact': plan.fact_text[:200],
                    'candidates': [
                        {**asdict(candidate),
                         'score': round(candidate.score, 4)}
                        for candidate in plan.candidates],
                    }))

    corroborate_degraded = False
    if plan.action == 'skipped':
        skip_fi = plan.fact_insight
        # The exact-match rung and the reconciler's NONE verdict both
        # name a target; the dedup-sibling and target-deleted skips
        # carry none.
        corroborated = False
        already_counted = (
            corroborated_ids is not None
            and skip_target in corroborated_ids)
        if skip_target and not already_counted:
            corroborated = backend.nodes.increment_corroboration(
                skip_target,
                queue_uuid=skip_fi.queue_uuid if skip_fi else None)
            if corroborated:
                if corroborated_ids is not None:
                    corroborated_ids.add(skip_target)
                backend.oplog.log(
                    operation='reconcile-corroborate',
                    insight_id=skip_target,
                    detail=f'restated by: {plan.fact_text[:200]}')
        if not skip_target or already_counted or corroborated:
            return {
                'id': skip_fi.id if skip_fi else str(uuid.uuid4()),
                'content': (skip_fi.content if skip_fi
                            else plan.fact_text),
                'action': 'skipped',
                'reason': plan.skip_reason,
                'target_id': skip_target,
                }
        # The exact-match target was soft-deleted between planning
        # and apply (an external forget); a skip here
        # would store the fact nowhere, so fall through to a plain
        # add carrying the vector computed before the rung. Mark the
        # dead target counted so a duplicate fact in the same row
        # skips against the copy this add inserts.
        corroborate_degraded = True
        if corroborated_ids is not None and skip_target:
            corroborated_ids.add(skip_target)
        logger.warning(
            f'corroborate target {skip_target} already deleted;'
            ' degrading to add')

    assert plan.fact_insight is not None, (
        'non-skipped FactPlan must carry a fact_insight')
    fi = plan.fact_insight

    linking = plan.action in {'update', 'replace', 'supersede'} and bool(plan.targets)
    linked_targets: list[tuple[str, str]] = []
    targets_gone: list[dict[str, str | None]] = []
    carried: list[tuple[str, list[Edge]]] = []
    predecessors: list[tuple[str, str, Insight]] = []
    if linking:
        for target_id, relation in plan.targets:
            before_target = backend.nodes.get_include_deleted(target_id)
            # Snapshot before the pointer is written: `supersede` removes
            # the predecessor's edges, and a later snapshot would also
            # scoop up the plan's causal edges and the successor's own
            # freshly minted ones.
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
        # Notes:
        # - Every predecessor keeps its content behind `superseded_by`;
        #   what the successor copies is what the CURRENT view keeps.
        #   Entities union rather than overwrite because the extractor
        #   sees only the incoming text and would narrow the merged
        #   row's entity set on every pass; recall history carries as
        #   the max over every linked target.
        # - Corroboration carries on a refinement, not on a
        #   contradiction: it counts restatements of the claim the
        #   supersede just falsified.
        for _target_id, relation, before_target in predecessors:
            fi.entities = list(dict.fromkeys(
                list(fi.entities) + list(before_target.entities)))
            fi.access_count = max(
                fi.access_count, before_target.access_count)
            if relation != 'supersede':
                fi.corroboration_count = max(
                    fi.corroboration_count,
                    before_target.corroboration_count)
        # One oplog row per linked target, each recording the finished
        # successor rather than the partial union at its own turn.
        for target_id, relation, before_target in predecessors:
            op_name = {
                'replace': 'replace',
                'update': 'reconcile-update',
                'supersede': 'reconcile-supersede',
                }[relation]
            detail = f'replaced by {fi.id}'
            if relation != 'replace' and fi.content == plan.fact_text:
                # Notes:
                # - No merge text was stored, so the successor may have
                #   dropped clauses of the predecessor that are still
                #   true; the marker makes that rate measurable.
                # - Every reconcile relation retires its target the
                #   same way, so the marker fires on update as on
                #   supersede; a replace stores the caller's text by
                #   contract and is never a fallback.
                detail += ' (unmerged)'
                trace.event('supersede_unmerged', target_id=target_id)
            backend.oplog.log(
                operation=op_name, insight_id=target_id,
                detail=detail,
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(fi))
        if not linked_targets:
            logger.warning(
                f'{plan.action}: every target is gone; degrading to add')

    backend.nodes.insert(fi)
    stored = backend.nodes.get(fi.id)
    if stored is not None and stored.created_at is not None:
        fi.created_at = stored.created_at
        fi.updated_at = stored.updated_at

    final_vec = plan.enriched_vec or plan.embed_vec
    embedded = final_vec is not None
    if final_vec is not None:
        # The caller evicts every planned row from the cache before the
        # apply phase and the semantic-edge builder reads the new row's
        # vector from it, so the inserted row registers itself here
        # with the vector it stores.
        embed_cache[fi.id] = final_vec
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

    if linking:
        for target_id, carried_edges in carried:
            move_edges(backend, target_id, fi.id, carried_edges)
        # Sweeps the causal edges the plan itself aimed at a target,
        # planned while the target was still current; `supersede`
        # removed only the edges that existed before the plan ran, and
        # a target already superseded must stay edgeless too.
        for target_id, _relation in plan.targets:
            backend.edges.delete_by_node(target_id)

    backend.nodes.stamp_linked(fi.id)
    if plan.enrichment:
        backend.nodes.update_enrichment(
            fi.id,
            keywords=plan.enrichment.get('keywords', []),
            summary=plan.enrichment.get('summary', ''),
            semantic_facts=plan.enrichment.get('semantic_facts', []))
        backend.nodes.stamp_enriched(fi.id)

    if corroborate_degraded or (linking and not linked_targets):
        reported_action = 'add'
    elif linking:
        relations = {relation for _target, relation in linked_targets}
        reported_action = ('supersede' if 'supersede' in relations
                           else relations.pop())
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
        # target as replaced would claim a replace that never
        # happened; `target_id` still names the row that vanished.
        result['target_id'] = skip_target
    elif linking:
        # `replaced_ids` names what this write linked; `targets_gone`
        # names the rows that now hold the topic, one read away, so a
        # degraded add cannot hide them.
        if linked_targets:
            result['replaced_ids'] = [t for t, _relation in linked_targets]
        if targets_gone:
            result['targets_gone'] = targets_gone
    return result
