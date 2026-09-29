"""Staleness keys on exactly what the remedy can replay.

`memman status` reports `stale_insights` from `count_stale_insights`.
The remedy it points at, `enrich --stale-only`, routes through
`enrich_pending` (`pipeline/enrich.py`), which re-runs enrichment on
the `slow` client and nothing else.

Notes
-----
- `compute_prompt_version` hashes exactly the inputs `enrich_pending`
  replays. A wider key reports rows stale for a change re-enrichment
  cannot address. The rebuild then clears the report by doing
  unrelated work, so the operator pays for LLM calls and the signal
  reads 0.
- The key lives in the existing `prompt_version` column because a new
  column needs a cross-backend migration (`store/db.py::_migrate`,
  `store/postgres.py`) for a reporting signal.
"""

import pytest
from memman import config
from memman.pipeline.remember import compute_prompt_version

REPLAYED_PROMPTS = [
    ('memman.pipeline.enrich', 'ENRICHMENT_SYSTEM_PROMPT'),
    ]


def _key():
    """Recompute the staleness key, defeating its process-lifetime cache.
    """
    compute_prompt_version.cache_clear()
    return compute_prompt_version()


@pytest.mark.parametrize(('module', 'attr'), REPLAYED_PROMPTS)
def test_key_moves_for_a_prompt_the_rebuild_replays(
        module, attr, monkeypatch):
    """Editing a prompt `enrich_pending` re-runs marks rows stale.

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
        f'{attr} is replayed by enrich_pending, so it must move the key')


def test_key_moves_for_the_llm_model(env_file):
    """The key tracks the LLM model, which enrich_pending replays on.

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
