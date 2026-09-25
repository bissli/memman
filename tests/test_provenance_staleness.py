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

    Mutation: leaving `MEMMAN_LLM_MODEL` out of the key,
        which would report a rebuild's own model change as nothing --
        the one drift the remedy CAN fix would then go unreported.
    Oracle: the key recomputed across a swap of the metadata model.
    """
    base = _key()
    env_file(config.LLM_MODEL, 'anthropic/claude-other-9.9')
    assert _key() != base, (
        'the metadata model produces exactly what a rebuild replays,'
        ' so swapping it must move the key')
