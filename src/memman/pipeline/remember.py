"""Remember pipeline - single entry point shared by sync CLI and worker.

Structure:

1. Quality check - advisory warnings only.
2. Planning phase - enrich the row, then embed its content once.
   **No DB writes.**
3. Apply phase - one transaction commits the replace link, insert,
   enrichment update, and stamp.

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
from memman.graph.enrichment import enrich_with_llm
from memman.llm.client import get_llm_client
from memman.search.quality import check_content_quality
from memman.store.backend import Backend
from memman.store.model import Insight, format_timestamp, insight_to_delta_dict

logger = logging.getLogger('memman')


@functools.lru_cache(maxsize=1)
def compute_prompt_version() -> str:
    """Return a 16-char SHA-256 hash of what a rebuild can replay.

    Returns
    -------
    str
        First 16 hex chars of a SHA-256 over the enrichment prompt
        and the resolved `MEMMAN_LLM_MODEL` id.

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
    - The `MEMMAN_LLM_MODEL` id IS folded in, because `link_pending`
      runs the enrichment call on it.
    - An unresolvable model hashes as the empty string, so a
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
        metadata_model = config.require(config.LLM_MODEL)
    except ConfigError:
        metadata_model = ''
    blob = f'{ENRICHMENT_SYSTEM_PROMPT}\x00{metadata_model}'
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class FactPlan:
    """Planned write for one queued write.

    Attributes
    ----------
    fact_insight : Insight
        The row the apply phase inserts.
    replaced_id : str
        The row a `replace` supersedes; '' for a plain add.
    embed_vec : list[float] or None
        Vector of the row's content; None when the embed failed.
    enrichment : dict[str, Any]
        `summary`; empty when enrichment failed.
    """

    fact_insight: Insight
    replaced_id: str = ''
    embed_vec: list[float] | None = None
    enrichment: dict[str, Any] = field(default_factory=dict)


def run_remember(
        backend: Backend,
        insight: Insight,
        content: str,
        ec: EmbeddingProvider,
        replaced_id: str = '',
        ) -> dict[str, Any]:
    """Store one write, enriched and embedded, and return the result.

    Parameters
    ----------
    backend : Backend
        The target store.
    insight : Insight
        The queued row's metadata: category, queue_uuid and author.
    content : str
        Stored as written: no model judges it, rewords it, or picks
        its category, which is `insight.category`.
    ec : EmbeddingProvider
        The store-bound embedder, from `bound_embedder`.
    replaced_id : str, default ''
        The row a `replace` supersedes; '' for a plain add.

    Returns
    -------
    dict[str, Any]
        `{'facts': [result], 'quality_warnings': [...],
        'llm_calls': int}`, where `result` is `_apply_plan`'s dict.
    """
    quality_warnings = check_content_quality(content)

    metadata_llm_client = get_llm_client()

    plan, llm_calls = _plan_fact(
        content, insight, replaced_id, metadata_llm_client, ec)
    plan.fact_insight.prompt_version = compute_prompt_version()
    plan.fact_insight.embedding_model = ec.model

    with backend.transaction():
        result = _apply_plan(backend, plan)

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
        The queued write; its category and other metadata are
        inherited by the planned row.
    replaced_id : str
        A `replace` target, or `''`.
    metadata_llm_client : Any
        The LLM client for enrichment.
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
        category=parent.category,
        created_at=parent.created_at, updated_at=parent.updated_at,
        queue_uuid=parent.queue_uuid, author=parent.author)

    calls = 0
    try:
        enrichment = enrich_with_llm(fact_insight, metadata_llm_client)
        calls += 1
    except Exception:
        enrichment = {}

    fact_vec = None
    try:
        fact_vec = ec.embed(fact_text)
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

    return FactPlan(
        fact_insight=fact_insight,
        replaced_id=replaced_id,
        embed_vec=fact_vec,
        enrichment=enrichment,
        ), calls


def _apply_plan(backend: Backend, plan: FactPlan) -> dict[str, Any]:
    """Apply one planned write. Must be invoked inside a transaction.

    Notes
    -----
    - A `replace` supersedes its target (never deletes it).
    - A target that is not current (forgotten, or superseded by an
      earlier write) is reported under `target_gone`, and the write
      degrades to a plain add.
    """
    fi = plan.fact_insight

    replaced = False
    target_gone: dict[str, str | None] | None = None
    if plan.replaced_id:
        before_target = backend.nodes.get_include_deleted(plan.replaced_id)
        linked = backend.nodes.supersede(plan.replaced_id, fi.id)
        if linked and before_target is not None:
            replaced = True
            # The predecessor keeps its content behind `superseded_by`,
            # and the successor copies nothing from it: the CLI already
            # seeded the target's category when `--cat` was omitted.
            backend.oplog.log(
                operation='replace', insight_id=plan.replaced_id,
                detail=f'replaced by {fi.id}',
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(fi))
        else:
            target_gone = {
                'id': plan.replaced_id,
                'superseded_by': (before_target.superseded_by
                                  if before_target is not None else None),
                }
            logger.warning(
                f'replace target {plan.replaced_id} is not current;'
                ' degrading to add')
            # Notes:
            # - The row is stored either way, so nothing is lost, but a
            #   caller who ran `replace` to correct one row otherwise
            #   gets a new unlinked row and no sign the correction
            #   missed.
            # - The row is filed against the SUCCESSOR, which is
            #   readable; the requested target may be gone from the
            #   table entirely, and it is named in the detail instead.
            backend.oplog.log(
                operation='target-gone', insight_id=fi.id,
                detail=f'replace target {plan.replaced_id} was not'
                ' current; stored without the link',
                after=insight_to_delta_dict(fi))

    backend.nodes.insert(fi)
    stored = backend.nodes.get(fi.id)
    if stored is not None and stored.created_at is not None:
        fi.created_at = stored.created_at
        fi.updated_at = stored.updated_at

    final_vec = plan.embed_vec
    embedded = final_vec is not None
    if final_vec is not None:
        backend.nodes.update_embedding(
            fi.id, final_vec, fi.embedding_model or '')

    backend.oplog.log(
        operation='remember', insight_id=fi.id, detail=fi.content,
        after=insight_to_delta_dict(fi))

    backend.nodes.stamp_linked(fi.id)
    if plan.enrichment:
        backend.nodes.update_enrichment(
            fi.id, summary=plan.enrichment.get('summary', ''))
    # A vectorless row stays unstamped: the stranded-row sweep selects
    # `enriched_at is null`, and it is the only path that embeds the
    # row again.
    if plan.enrichment and embedded:
        backend.nodes.stamp_enriched(fi.id)

    result: dict[str, Any] = {
        'id': fi.id,
        'content': fi.content,
        'category': fi.category,
        'action': 'replace' if replaced else 'add',
        'created_at': (
            format_timestamp(fi.created_at)
            if fi.created_at is not None else ''),
        'enrichment': {'summary': plan.enrichment.get('summary', '')},
        'embedded': embedded,
        }
    # `replaced_id` names what this write linked; `target_gone` names
    # the row that now holds the topic, one read away, so a degraded
    # add cannot hide it.
    if replaced:
        result['replaced_id'] = plan.replaced_id
    if target_gone is not None:
        result['target_gone'] = target_gone
    return result
