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
from typing import Any

import httpx
from memman.embed import EmbeddingProvider
from memman.exceptions import EmbedCredentialError
from memman.llm.client import get_llm_client
from memman.pipeline.enrich import enrich_with_llm
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
    - THE INVARIANT: this hashes exactly the inputs `enrich_pending`
      (`pipeline/enrich.py`) re-runs, and nothing else. It is both the
      value `stamp_enriched` writes and the key
      `count_stale_insights` compares, so a key covering more than
      the remedy replays reports rows stale for a change
      re-enrichment cannot address - and `enrich --stale-only`
      then clears the report by doing unrelated work, which is worse
      than having no remedy at all.
    - The `MEMMAN_LLM_MODEL` id IS folded in, because `enrich_pending`
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
    from memman.pipeline.enrich import ENRICHMENT_SYSTEM_PROMPT

    try:
        llm_model = config.require(config.LLM_MODEL)
    except ConfigError:
        llm_model = ''
    blob = f'{ENRICHMENT_SYSTEM_PROMPT}\x00{llm_model}'
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_remember(
        backend: Backend,
        insight: Insight,
        ec: EmbeddingProvider,
        replaced_id: str = '',
        ) -> dict[str, Any]:
    """Store one write, enriched and embedded, and return the result.

    Parameters
    ----------
    backend : Backend
        The target store.
    insight : Insight
        The row to store, built by the caller: its `content` is
        stored as written, no model judges it, rewords it, or picks
        its category.
    ec : EmbeddingProvider
        The store-bound embedder, from `bound_embedder`.
    replaced_id : str, default ''
        The row a `replace` supersedes; '' for a plain add.

    Returns
    -------
    dict[str, Any]
        `_apply_plan`'s result dict, plus `quality_warnings`.

    Raises
    ------
    EmbedCredentialError
        The store's embed provider has no credentials. An HTTP or
        runtime embed failure is logged instead, and the row is
        stored without a vector.
    """
    quality_warnings = check_content_quality(insight.content)

    llm_client = get_llm_client()
    try:
        enrichment = enrich_with_llm(insight, llm_client)
    except Exception:
        enrichment = {}

    embed_vec = None
    try:
        embed_vec = ec.embed(insight.content)
    except EmbedCredentialError:
        raise
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning(
            f'fact embed failed; row stored without vector: {exc}')

    insight.prompt_version = compute_prompt_version()
    insight.embedding_model = ec.model

    with backend.transaction():
        result = _apply_plan(
            backend, insight, replaced_id, embed_vec, enrichment)

    result['quality_warnings'] = quality_warnings
    return result


def _apply_plan(
        backend: Backend,
        insight: Insight,
        replaced_id: str,
        embed_vec: list[float] | None,
        enrichment: dict[str, Any],
        ) -> dict[str, Any]:
    """Store one insight. Must be invoked inside a transaction.

    Notes
    -----
    - A `replace` supersedes its target (never deletes it).
    - A target that is not current (forgotten, or superseded by an
      earlier write) is reported under `target_gone`, and the write
      degrades to a plain add.
    """
    fi = insight

    replaced = False
    target_gone: dict[str, str | None] | None = None
    if replaced_id:
        before_target = backend.nodes.get_include_deleted(replaced_id)
        linked = backend.nodes.supersede(replaced_id, fi.id)
        if linked and before_target is not None:
            replaced = True
            # The predecessor keeps its content behind `superseded_by`,
            # and the successor copies nothing from it: the CLI already
            # seeded the target's category when `--cat` was omitted.
            backend.oplog.log(
                operation='replace', insight_id=replaced_id,
                detail=f'replaced by {fi.id}',
                before=insight_to_delta_dict(before_target),
                after=insight_to_delta_dict(fi))
        else:
            target_gone = {
                'id': replaced_id,
                'superseded_by': (before_target.superseded_by
                                  if before_target is not None else None),
                }
            logger.warning(
                f'replace target {replaced_id} is not current;'
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
                detail=f'replace target {replaced_id} was not'
                ' current; stored without the link',
                after=insight_to_delta_dict(fi))

    backend.nodes.insert(fi)
    stored = backend.nodes.get(fi.id)
    if stored is not None and stored.created_at is not None:
        fi.created_at = stored.created_at
        fi.updated_at = stored.updated_at

    embedded = embed_vec is not None
    if embed_vec is not None:
        backend.nodes.update_embedding(
            fi.id, embed_vec, fi.embedding_model or '')

    backend.oplog.log(
        operation='remember', insight_id=fi.id, detail=fi.content,
        after=insight_to_delta_dict(fi))

    backend.nodes.stamp_enrich_attempted(fi.id)
    if enrichment:
        backend.nodes.update_enrichment(
            fi.id, summary=enrichment.get('summary', ''))
    # A vectorless row stays unstamped: the stranded-row sweep selects
    # `enriched_at is null`, and it is the only path that embeds the
    # row again.
    if enrichment and embedded:
        backend.nodes.stamp_enriched(fi.id)

    result: dict[str, Any] = {
        'id': fi.id,
        'content': fi.content,
        'category': fi.category,
        'action': 'replace' if replaced else 'add',
        'created_at': (
            format_timestamp(fi.created_at)
            if fi.created_at is not None else ''),
        'enrichment': {'summary': enrichment.get('summary', '')},
        'embedded': embedded,
        }
    # `replaced_id` names what this write linked; `target_gone` names
    # the row that now holds the topic, one read away, so a degraded
    # add cannot hide it.
    if replaced:
        result['replaced_id'] = replaced_id
    if target_gone is not None:
        result['target_gone'] = target_gone
    return result
