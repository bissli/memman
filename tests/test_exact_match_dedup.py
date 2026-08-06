"""Exact-match dedup rung (F3) and corroboration count (F4).

The rung sits inside `_plan_fact`'s reconcile branch: when exactly
one shortlist row matches the fact byte-for-byte (modulo case and
whitespace), the plan skips without an LLM call and carries the
target id; `_apply_plan` then bumps the target's
`corroboration_count` and writes a `reconcile-corroborate` oplog row.
"""

import uuid
from datetime import datetime, timezone

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from memman.store.model import Insight, is_immune
from tests.conftest import make_insight


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


def _spy_reconcile(monkeypatch):
    """Replace reconcile_memories with a recording ADD stub.

    Isolates the rung from the conftest mock's overlap heuristic
    (which would return UPDATE for identical content), so a deleted
    rung shows up as action 'add' plus a recorded call, never as a
    coincidentally-identical outcome.
    """
    calls = []

    def _fake(llm_client, facts, similar):
        calls.append((facts, similar))
        return [{'fact': f['text'], 'action': 'ADD',
                 'target_id': None, 'merged_text': None} for f in facts]

    monkeypatch.setattr(
        'memman.llm.extract.reconcile_memories', _fake)
    return calls


def _run(backend, content, **kwargs):
    return run_remember(
        backend, _new_insight(content), content,
        ec=bound_embedder(backend), **kwargs)


def test_exact_match_single_hit_skips_llm(tmp_backend, monkeypatch):
    """One byte-identical stored row skips reconcile entirely.

    Mutation: deleting the rung — identical content reaches the
        reconcile LLM call.
    Oracle: the spy records zero reconcile calls and the fact lands
        as 'skipped'.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    calls = _spy_reconcile(monkeypatch)
    res = _run(tmp_backend, 'Redis caches session tokens')
    assert res['facts'][0]['action'] == 'skipped'
    assert calls == []


def test_exact_match_two_hits_escalates_to_llm(
        tmp_backend, monkeypatch):
    """Two identical stored rows fall through to the LLM.

    With two identical rows the store is already inconsistent, and
    which one to merge into is exactly the judgement worth an LLM
    call.

    Mutation: flipping `== 1` to `>= 1`.
    Oracle: the spy records exactly one reconcile call.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    _store(tmp_backend, 'Redis caches session tokens')
    calls = _spy_reconcile(monkeypatch)
    res = _run(tmp_backend, 'Redis caches session tokens')
    assert len(calls) == 1
    assert res['facts'][0]['action'] == 'add'


def test_exact_match_is_not_substring_match(tmp_backend, monkeypatch):
    """A superset fact is not swallowed by its stored subset.

    Mutation: replacing the equality with `in` — every superset fact
        would silently skip against its stored prefix.
    Oracle: the spy records one reconcile call and the fact is added.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    calls = _spy_reconcile(monkeypatch)
    res = _run(
        tmp_backend, 'Redis caches session tokens for the api gateway')
    assert len(calls) == 1
    assert res['facts'][0]['action'] == 'add'


def test_exact_match_is_whitespace_and_case_insensitive(
        tmp_backend, monkeypatch):
    """Case and whitespace differences still count as exact.

    Mutation: dropping `.lower()` or the whitespace collapse from the
        normalisation.
    Oracle: differently-cased, differently-spaced content skips with
        zero reconcile calls.
    """
    _store(tmp_backend, 'Redis  Caches \t Session Tokens')
    calls = _spy_reconcile(monkeypatch)
    res = _run(tmp_backend, 'redis caches session tokens')
    assert res['facts'][0]['action'] == 'skipped'
    assert calls == []


def test_no_reconcile_bypasses_the_rung(tmp_backend, monkeypatch):
    """`--no-reconcile` stores verbatim even for identical content.

    The documented contract is "store verbatim, no judgement", and
    many CLI tests write identical content under the flag.

    Mutation: hoisting the rung above the `not no_reconcile` guard.
    Oracle: identical content lands as 'add' under the flag.
    """
    _store(tmp_backend, 'Redis caches session tokens')
    _spy_reconcile(monkeypatch)
    res = _run(
        tmp_backend, 'Redis caches session tokens', no_reconcile=True)
    assert res['facts'][0]['action'] == 'add'


def test_replace_of_identical_content_still_replaces(
        tmp_backend, monkeypatch):
    """`replace` with identical content must still replace.

    The CLI routes replace with `no_reconcile=... or
    bool(hint_replaced_id)`, so the rung must never intercept it.

    Mutation: the same hoist, reached via the replace route.
    Oracle: action is 'replace', the target row is gone, and the new
        row exists.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _spy_reconcile(monkeypatch)
    res = _run(
        tmp_backend, 'Redis caches session tokens',
        no_reconcile=True, replaced_id=tid)
    assert res['facts'][0]['action'] == 'replace'
    assert tmp_backend.nodes.get(tid) is None
    assert tmp_backend.nodes.get(res['facts'][0]['id']) is not None


def test_exact_match_skip_bumps_corroboration_on_target(
        tmp_backend, monkeypatch):
    """Each exact-match skip bumps the TARGET's corroboration_count.

    Mutation: dropping the increment, or bumping the new fact's id
        instead of `plan.target_id`.
    Oracle: two identical writes leave the stored target at
        corroboration_count == 2; no other row exists to absorb a
        misdirected bump.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _spy_reconcile(monkeypatch)
    _run(tmp_backend, 'Redis caches session tokens')
    _run(tmp_backend, 'Redis caches session tokens')
    stored = tmp_backend.nodes.get(tid)
    assert stored.corroboration_count == 2


def test_corroborate_writes_oplog_row(tmp_backend, monkeypatch):
    """The skip leaves a `reconcile-corroborate` oplog row.

    Mutation: dropping the `backend.oplog.log` call.
    Oracle: exactly one row with the operation name, carrying the
        target id.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _spy_reconcile(monkeypatch)
    _run(tmp_backend, 'Redis caches session tokens')
    rows = tmp_backend._db._query(
        'select insight_id from oplog'
        " where operation = 'reconcile-corroborate'").fetchall()
    assert [r[0] for r in rows] == [tid]


def test_corroboration_does_not_confer_immunity(
        tmp_backend, monkeypatch):
    """Corroboration never feeds the retention-immunity criterion.

    `access_count >= 3` grants pruning immunity; "the agent said it
    three times" must not.

    Mutation: bumping `access_count` instead of (or alongside)
        `corroboration_count`.
    Oracle: after three exact-match skips the target's access_count
        is still 0 and `is_immune` stays False.
    """
    tid = _store(tmp_backend, 'Redis caches session tokens')
    _spy_reconcile(monkeypatch)
    for _ in range(3):
        _run(tmp_backend, 'Redis caches session tokens')
    stored = tmp_backend.nodes.get(tid)
    assert stored.corroboration_count == 3
    assert stored.access_count == 0
    assert is_immune(stored.importance, stored.access_count) is False
