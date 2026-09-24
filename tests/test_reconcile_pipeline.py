"""The `replace` write path retires its target and updates the drain cache.

`_apply_plan` supersedes a `replace` target, moves its edges onto the
successor, and evicts the target from the caller's `embed_cache` so a
later row in the same drain does not mint a semantic edge back onto a
row this write already retired.
"""

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight


def test_a_replaced_row_leaves_the_drain_cache(tmp_backend):
    """Verify the row a replace retires leaves the caller's embed_cache.

    Mutation: re-registering the retired row's vector in embed_cache,
        or never evicting it, so the next queue row of the same drain
        finds it as a semantic neighbor and mints an edge back onto a
        row this write already superseded.
    Oracle: the retired row's id absent from embed_cache after the
        write, against the successor's id, which the apply phase must
        register with its own vector.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='the broker is redis'))
    embed_cache = {'old-1': [1.0, 0.0]}

    fact_text = 'the broker is rabbit'
    res = run_remember(
        tmp_backend, make_insight(id='parent', content=fact_text), fact_text,
        ec=bound_embedder(tmp_backend), replaced_id='old-1',
        embed_cache=embed_cache, store_name='test')

    fact = res['facts'][0]
    assert fact['action'] == 'replace'
    assert fact['replaced_ids'] == ['old-1']
    assert 'old-1' not in embed_cache
    assert fact['id'] in embed_cache
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == fact['id']
