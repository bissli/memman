"""Staleness must key on exactly what the remedy can replay (X11).

`memman status` reports `stale_insights` from `count_stale_insights`,
and the remedy it points at, `graph rebuild --stale`, routes through
`link_pending` (`graph/engine.py`), which re-runs ENRICHMENT on the
`slow` client and nothing else.

The invariant these tests pin: `compute_prompt_version` hashes exactly
the inputs `link_pending` replays, and nothing else. A key covering
more than that reports rows stale for a change re-enrichment cannot
address -- and the rebuild then CLEARS the report by doing unrelated
work, so the operator pays for LLM calls and the signal reads 0.

The fix stays inside the existing `prompt_version` column on purpose.
memman carries no alter-table path (`store/db.py::_migrate`), and a
Postgres-routed store can only gain a column by migrating to SQLite on
the previous release, rebuilding, and migrating back
(`store/postgres.py`). A new column would cost a fleet-wide
cross-backend migration to fix a reporting signal.
"""

import pytest
from memman import config
from tests.conftest import make_insight

REPLAYED_PROMPTS = [
    ('memman.graph.enrichment', 'ENRICHMENT_SYSTEM_PROMPT'),
    ]


def _key():
    """Recompute the staleness key, defeating its process-lifetime cache."""
    from memman.pipeline.remember import compute_prompt_version
    compute_prompt_version.cache_clear()
    return compute_prompt_version()


@pytest.mark.parametrize(('module', 'attr'), REPLAYED_PROMPTS)
def test_key_moves_for_a_prompt_the_rebuild_replays(
        module, attr, monkeypatch):
    """Editing a prompt `link_pending` re-runs marks rows stale.

    Mutation: dropping the enrichment prompt from the key -- an edit
        then changes what every rebuilt row gets while
        `stale_insights` stays 0, so the one drift the remedy CAN fix
        is the one nobody is told about.
    Oracle: the key recomputed with that single prompt perturbed,
        against the unperturbed key.
    """
    base = _key()
    monkeypatch.setattr(f'{module}.{attr}', 'PERTURBED FOR TEST')
    assert _key() != base, (
        f'{attr} is replayed by link_pending, so it must move the key')


def test_key_moves_for_the_metadata_model(env_file):
    """The key tracks the metadata model, which link_pending replays on.

    Mutation: leaving `MEMMAN_LLM_MODEL_SLOW` out of the key,
        which would report a rebuild's own model change as nothing --
        the one drift the remedy CAN fix would then go unreported.
    Oracle: the key recomputed across a swap of the metadata model.
    """
    base = _key()
    env_file(config.LLM_MODEL_SLOW, 'anthropic/claude-other-9.9')
    assert _key() != base, (
        'the metadata model produces exactly what a rebuild replays,'
        ' so swapping it must move the key')


def test_enrich_stamp_preserves_the_content_model(backend):
    """Re-enriching a row leaves `model_id` alone.

    `prompt_version` IS the enrichment key, so a rebuild rewriting it
    is correct. `model_id` is the model that produced the row's
    CONTENT, which a rebuild never touches.

    Read through `provenance_distribution`, not `nodes.get`:
    `_INSIGHT_COLUMNS` (`store/node.py`) omits `prompt_version`,
    `model_id` and `embedding_model`, so a row fetched through the
    model layer reports all three as None whatever is stored. The
    distribution is also the exact surface this defect corrupts,
    since `doctor.check_provenance_drift` is its only consumer.

    Mutation: `stamp_enriched` writing `model_id`, which is its
        shipped form -- after any rebuild a row is attributed to
        whatever model was configured at rebuild time, so the
        provenance report names a model that never produced that
        content.
    Oracle: the (prompt_version, model_id) group read back after the
        stamp, against the pair the insert wrote.
    """
    backend.nodes.insert(make_insight(
        id='prov-1', content='a row long enough to store as content',
        prompt_version='old-enrich-key', model_id='anthropic/writer-1.0'))
    backend.nodes.stamp_enriched('prov-1', prompt_version='new-enrich-key')
    groups = {
        (g.prompt_version, g.model_id): g.count
        for g in backend.nodes.provenance_distribution()}
    assert groups == {('new-enrich-key', 'anthropic/writer-1.0'): 1}


def test_stale_predicate_ignores_model_drift(backend):
    """A model-only difference is not staleness.

    With the metadata model folded into the key, the predicate's
    separate `model_id` branch can only fire on the CONTENT model --
    drift a rebuild cannot remedy.

    Mutation: restoring a `model_id` comparison to
        `count_stale_insights` / `iter_stale_insight_ids`, resolved
        from config rather than passed in -- the plausible shape of a
        well-meant re-fix. A store whose content model was swapped
        then reports those rows stale forever, because nothing
        rewrites `model_id` any more.
    Oracle: two rows carrying the active key and differing only in
        `model_id`, one of them naming the model that wrote the row's
        content and one not, counted against zero.
    """
    active = _key()
    backend.nodes.insert(make_insight(
        id='drift-same', content='content written by the active model',
        prompt_version=active, model_id='anthropic/claude-writer-1.0'))
    backend.nodes.insert(make_insight(
        id='drift-other', content='content written by an older model',
        prompt_version=active, model_id='anthropic/claude-older-1.0'))
    assert backend.nodes.count_stale_insights(active) == 0
    assert backend.nodes.iter_stale_insight_ids(active) == []
