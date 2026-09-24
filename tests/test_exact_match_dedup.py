"""Exact-match dedup rung (F3) and corroboration count (F4).

The rung sits inside `_plan_fact`: a fact's content_hash is looked up
against the store's current rows with one indexed query, no
shortlist and no LLM call. A match skips, corroborating the oldest
match if several exist; `_apply_plan` then bumps the target's
`corroboration_count` and writes a `reconcile-corroborate` oplog row.
"""

import uuid
from datetime import datetime, timedelta, timezone

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from memman.store.model import Insight
from tests.conftest import make_insight, set_created_at


def _new_insight(content):
    now = datetime.now(timezone.utc)
    return Insight(
        id=str(uuid.uuid4()), content=content, category='fact',
        importance=3, entities=[], source='test', access_count=0,
        created_at=now, updated_at=now)


def _store(backend, content):
    iid = str(uuid.uuid4())
    backend.nodes.insert(make_insight(id=iid, content=content))
    return iid


def _run(backend, content, **kwargs):
    kwargs.setdefault('store_name', 'test')
    return run_remember(
        backend, _new_insight(content), content,
        ec=bound_embedder(backend), **kwargs)


def test_exact_match_single_hit_skips_llm(tmp_backend):
    """One byte-identical stored row skips reconcile entirely.

    Mutation: deleting the rung -- identical content reaches
        enrichment instead of skipping.
    Oracle: the fact lands as 'skipped' with zero LLM calls.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    res = _run(tmp_backend, 'Redis caches session tokens')
    assert res['facts'][0]['action'] == 'skipped'
    assert res['llm_calls'] == 0


def test_two_identical_rows_skip_onto_the_oldest(tmp_backend):
    """Two identical stored rows still make an exact-duplicate skip.

    Mutation: the exactly-one guard kept, so a store already holding
        two copies takes a third.
    Oracle: the write skips as `exact duplicate`, naming the older
        row, whose count alone moves to 1.
    """
    older = _store(tmp_backend, 'Redis caches session tokens')
    set_created_at(
        tmp_backend, older, datetime.now(timezone.utc) - timedelta(days=1))
    newer = _store(tmp_backend, 'Redis caches session tokens')
    res = _run(tmp_backend, 'Redis caches session tokens')
    fact = res['facts'][0]
    assert (fact['action'], fact.get('reason'), fact.get('target_id')) == (
        'skipped', 'exact duplicate', older)
    assert tmp_backend.nodes.get(older).corroboration_count == 1
    assert tmp_backend.nodes.get(newer).corroboration_count == 0


def test_an_identical_row_outside_any_shortlist_is_caught(tmp_backend):
    """An identical row five better keyword hits outrank still skips.

    No stored row carries a vector, so the keyword rung is the only
    shortlist, and five importance-5 rows holding every query token
    fill its five slots ahead of the identical importance-3 row.

    Mutation: the check scoped to a shortlist (mem0's shape: the hash
        compared against the top-k hits alone).
    Oracle: the write skips as `exact duplicate` naming the identical
        row.
    """
    for n in range(5):
        tmp_backend.nodes.insert(make_insight(
            id=f'superset-{n}', importance=5,
            content=f'Redis caches session tokens alongside rate limit'
            f' counters, queue offsets and feature flags for service {n}'))
    identical = _store(tmp_backend, 'Redis caches session tokens')
    res = _run(tmp_backend, 'Redis caches session tokens')
    fact = res['facts'][0]
    assert (fact['action'], fact.get('reason'), fact.get('target_id')) == (
        'skipped', 'exact duplicate', identical)


def test_exact_match_is_not_substring_match(tmp_backend):
    """A superset fact is not swallowed by its stored subset.

    Mutation: replacing the equality with `in` -- every superset fact
        would silently skip against its stored prefix.
    Oracle: the write reaches enrichment (one LLM call) and lands as
        'add'.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    res = _run(
        tmp_backend, 'Redis caches session tokens for the api gateway')
    assert res['llm_calls'] == 1
    assert res['facts'][0]['action'] == 'add'


def test_exact_match_is_whitespace_and_case_insensitive(tmp_backend):
    """Case and whitespace differences still count as exact.

    Mutation: dropping `.lower()` or the whitespace collapse from the
        normalisation.
    Oracle: differently-cased, differently-spaced content skips with
        zero LLM calls.
    """
    _store(tmp_backend, 'Redis  Caches \t Session Tokens')
    res = _run(tmp_backend, 'redis caches session tokens')
    assert res['facts'][0]['action'] == 'skipped'
    assert res['llm_calls'] == 0


def test_replace_of_identical_content_still_replaces(tmp_backend):
    """`replace` with identical content must still replace.

    A replace names its target directly and never reaches the
    exact-match lookup, so identical content cannot be intercepted
    into a skip.

    Mutation: routing a replace through the exact-match rung before
        the replace branch.
    Oracle: action is 'replace', the target row is gone, and the new
        row exists.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    res = _run(
        tmp_backend, 'Redis caches session tokens', replaced_id=tid)
    assert res['facts'][0]['action'] == 'replace'
    assert tmp_backend.nodes.get(tid) is None
    assert tmp_backend.nodes.get(res['facts'][0]['id']) is not None


def test_exact_match_skip_bumps_corroboration_on_target(tmp_backend):
    """Each exact-match skip bumps the TARGET's corroboration_count.

    Mutation: dropping the increment, or bumping the new fact's id
        instead of `plan.target_id`.
    Oracle: two identical writes leave the stored target at
        corroboration_count == 2; no other row exists to absorb a
        misdirected bump.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _run(tmp_backend, 'Redis caches session tokens')
    _run(tmp_backend, 'Redis caches session tokens')
    stored = tmp_backend.nodes.get(tid)
    assert stored.corroboration_count == 2


def test_corroborate_writes_oplog_row(tmp_backend):
    """The skip leaves a `reconcile-corroborate` oplog row.

    Mutation: dropping the `backend.oplog.log` call.
    Oracle: exactly one row with the operation name, carrying the
        target id.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _run(tmp_backend, 'Redis caches session tokens')
    rows = tmp_backend._db._query(
        'select insight_id from oplog'
        " where operation = 'reconcile-corroborate'").fetchall()
    assert [r[0] for r in rows] == [tid]


def test_corroboration_does_not_inflate_access_count(tmp_backend):
    """Corroboration never touches `access_count`.

    `access_count` means "times this row was returned" and nothing
    else, so a restatement must leave it alone -- otherwise the one
    counter that records retrieval starts recording writes too.

    Mutation: bumping `access_count` instead of (or alongside)
        `corroboration_count`.
    Oracle: after three exact-match skips the target's access_count
        is still 0 while corroboration_count reads 3.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    for _ in range(3):
        _run(tmp_backend, 'Redis caches session tokens')
    stored = tmp_backend.nodes.get(tid)
    assert stored.corroboration_count == 3
    assert stored.access_count == 0


def test_corroborate_adopts_restating_queue_uuid(tmp_backend):
    """The corroborated target adopts the restating row's queue_uuid.

    An all-skips queue row inserts nothing carrying its uuid, so a
    worker crash between the commit and `mark_done` reclaims the row
    and the replay guard (`has_active_with_queue_uuid`) finds
    nothing -- the bump repeats on every reclaim.

    Mutation: dropping the queue_uuid adoption from the bump.
    Oracle: after the skip, the target carries the restating uuid
        and the replay guard fires for it.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    parent = _new_insight('Redis caches session tokens')
    parent.queue_uuid = 'q-restate-1'
    run_remember(
        tmp_backend, parent, 'Redis caches session tokens',
        ec=bound_embedder(tmp_backend), store_name='test')
    assert tmp_backend.nodes.get(tid).queue_uuid == 'q-restate-1'
    assert tmp_backend.nodes.has_active_with_queue_uuid(
        'q-restate-1') is True


def test_corroborate_preserves_creating_rows_queue_uuid(tmp_backend):
    """A populated queue_uuid survives corroboration.

    The creating row's replay guard outranks the restating row's:
    clobbering it lets a crash-reclaimed creator re-process its row,
    and an LLM re-extraction is not guaranteed to re-hit the rung --
    it can insert the duplicate the guard exists to prevent.

    Mutation: flipping the coalesce back to adopt-over (the 0.19.0
        form, `coalesce(?, queue_uuid)`).
    Oracle: after a restatement the target still carries the
        creating row's uuid and the replay guard fires for it.
    """
    tid = str(uuid.uuid4())
    tmp_backend.nodes.insert(make_insight(
        id=tid, content='Redis caches session tokens',
        queue_uuid='q-create-1'))
    parent = _new_insight('Redis caches session tokens')
    parent.queue_uuid = 'q-restate-2'
    run_remember(
        tmp_backend, parent, 'Redis caches session tokens',
        ec=bound_embedder(tmp_backend), store_name='test')
    assert tmp_backend.nodes.get(tid).queue_uuid == 'q-create-1'
    assert tmp_backend.nodes.has_active_with_queue_uuid(
        'q-create-1') is True


def test_corroborate_dead_target_degrades_to_add(tmp_backend, monkeypatch):
    """A target soft-deleted before apply degrades to an add.

    The exact-match lookup runs at planning time; an external forget
    can soft-delete the matched row before `_apply_plan`'s
    corroborate call reaches it, in the same synchronous write.

    Mutation: returning the skip on a zero-row bump (the 0.19.0
        form) -- the restated fact is stored nowhere and a phantom
        oplog row names a dead id.
    Oracle: the fact lands as a live 'add' row, the dead target's
        counter stays 0, and no corroborate oplog row is written.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    tmp_backend.nodes.soft_delete(tid)
    monkeypatch.setattr(
        tmp_backend.nodes, 'oldest_active_by_content_hash',
        lambda digest: tid)
    res = _run(tmp_backend, 'Redis caches session tokens')
    assert res['facts'][0]['action'] == 'add'
    assert tmp_backend.nodes.get(res['facts'][0]['id']) is not None
    # The degraded add supersedes nothing: it must name the vanished
    # row as target_id, never claim a replace of it.
    assert res['facts'][0].get('replaced_id') is None
    assert res['facts'][0]['target_id'] == tid
    dead = tmp_backend.nodes.get_include_deleted(tid)
    assert dead.corroboration_count == 0
    count = tmp_backend._db._query(
        'select count(*) from oplog'
        " where operation = 'reconcile-corroborate'").fetchone()[0]
    assert count == 0


def test_degrade_evicts_the_dead_target_from_the_embed_cache(
        tmp_backend, monkeypatch):
    """A degraded add evicts the dead target from the shared embed cache.

    `_drain_queue` shares one `embed_cache` across every row of a
    drain pass; without eviction a later row in the same pass would
    read the dead target's stale vector out of the cache and mint a
    semantic edge onto a row that no longer exists.

    Mutation: dropping the `embed_cache.pop` after a degraded add.
    Oracle: the shared cache, checked for the dead target's id before
        and after the degraded write.
    """
    first = _run(tmp_backend, 'Redis caches session tokens')
    tid = first['facts'][0]['id']
    shared_cache = dict(tmp_backend.nodes.iter_embeddings_as_vecs())
    tmp_backend.nodes.soft_delete(tid)
    monkeypatch.setattr(
        tmp_backend.nodes, 'oldest_active_by_content_hash',
        lambda digest: tid)
    assert tid in shared_cache
    res = _run(
        tmp_backend, 'Redis caches session tokens', embed_cache=shared_cache)
    assert res['facts'][0]['action'] == 'add'
    assert tid not in shared_cache


def test_skip_result_carries_target_id(tmp_backend):
    """The skip result names the row that absorbed the restatement.

    The result's 'id' is a never-inserted uuid; without 'target_id'
    the corroborated row is unreachable from the response.

    Mutation: dropping 'target_id' from the skipped result dict.
    Oracle: the result's target_id equals the stored row's id.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    res = _run(tmp_backend, 'Redis caches session tokens')
    assert res['facts'][0]['action'] == 'skipped'
    assert res['facts'][0]['target_id'] == tid


def test_corroboration_count_reaches_the_json_read_path(tmp_backend):
    """The counter is visible through the full-dict serializer.

    `insight_to_full_dict` is the consumer-facing read path that
    carries this counter (`recall` and `get` both serialize through
    it; `recall --brief` projects it away deliberately); dropping the
    key makes F4 write-only while every write-side test stays green.

    Mutation: deleting the corroboration_count line from
        `insight_to_full_dict`.
    Oracle: after one restatement the serialized target carries
        corroboration_count == 1.
    """
    from memman.store.model import insight_to_full_dict
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _run(tmp_backend, 'Redis caches session tokens')
    assert insight_to_full_dict(
        tmp_backend.nodes.get(tid))['corroboration_count'] == 1
