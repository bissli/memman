"""The staged reconcile through the planner: screen, keep, verdict, merge, apply.

`_plan_fact` screens every shortlisted row in parallel, shows the kept
rows to stage 2 one per call, assembles the verdicts in shortlist
order, and fans a linking result into one plan per retiring target,
each with its own merge text. These tests drive that path with the
three stage functions stubbed and read the store back.
"""

import json

from memman.embed.fingerprint import bound_embedder
from memman.llm.extract import UNJUDGED
from memman.pipeline import remember as rem
from memman.pipeline.remember import assemble_verdicts, run_remember
from memman.pipeline.remember import screened_rows
from tests.conftest import make_insight

ROWS = [('a', 'row a'), ('b', 'row b'), ('c', 'row c'), ('d', 'row d')]


# --- the keep rule ---


def test_keep_rule_shows_contradicts_restates_and_unjudged():
    """Verify the kept set is CONTRADICTS, RESTATES and UNJUDGED, in order.

    Mutation: dropping UNJUDGED (a screen failure then hides the row
        from stage 2), or keeping REFINES beside them (half of all
        candidates, the batch's volume back).
    Oracle: the measured keep set over four hand-labeled rows.
    """
    relations = {
        'a': 'REFINES', 'b': 'RESTATES', 'c': UNJUDGED, 'd': 'CONTRADICTS'}
    assert screened_rows(ROWS, relations) == [
        ('b', 'row b'), ('c', 'row c'), ('d', 'row d')]


def test_keep_rule_falls_back_to_the_first_refines_row_alone():
    """Verify an empty kept set shows the first REFINES row and nothing else.

    Mutation: showing every REFINES row, or none (the one UPDATE
        candidate then never reaches the contract).
    Oracle: the measured composed rule: the first REFINES row in
        shortlist order, alone, only when nothing else is kept.
    """
    relations = {'a': 'UNRELATED', 'b': 'REFINES', 'c': 'REFINES', 'd': 'UNRELATED'}
    assert screened_rows(ROWS, relations) == [('b', 'row b')]
    assert screened_rows(ROWS, {k: 'UNRELATED' for k, _ in ROWS}) == []


# --- the assembly ---


def test_assembly_folds_a_none_row_to_update_beside_a_supersede():
    """Verify a restating row folds into the write beside a superseded row.

    Mutation: leaving the NONE row current (the v3 text's own rule says
        the restating memory takes UPDATE beside a contradicted one), or
        taking it as the action over the supersede.
    Oracle: `perpair_dispose.py`'s case-rep rule on hand-set verdicts.
    """
    verdicts = {'a': 'none', 'b': 'supersede', 'c': 'keep', 'd': 'supersede'}
    assert assemble_verdicts(ROWS, verdicts) == (
        'SUPERSEDE', [('b', 'supersede'), ('d', 'supersede'), ('a', 'update')])


def test_assembly_takes_the_update_slot_in_shortlist_order():
    """Verify at most one update target, the first in shortlist order.

    Mutation: linking every UPDATE row, or taking the slot in the
        verdict dict's order.
    Oracle: two UPDATE rows handed in reverse dict order, the earlier
        shortlist row linked alone.
    """
    verdicts = {'d': 'keep', 'c': 'update', 'b': 'update', 'a': 'keep'}
    assert assemble_verdicts(ROWS, verdicts) == ('UPDATE', [('b', 'update')])


def test_assembly_reads_none_alone_and_all_keep_as_add():
    """Verify NONE alone names the first restating row and all-keep is ADD.

    Mutation: NONE on a later row, or ADD carrying a target.
    Oracle: the first NONE row alone; the empty target list on keep.
    """
    verdicts = {'a': 'keep', 'b': 'none', 'c': 'none', 'd': 'keep'}
    assert assemble_verdicts(ROWS, verdicts) == ('NONE', [('b', 'none')])
    assert assemble_verdicts(ROWS, {k: 'keep' for k, _ in ROWS}) == ('ADD', [])


# --- the planner end to end ---


class _FixedEmbedder:
    """An embed provider returning one fixed vector for every text."""

    model = 'fixed'

    def __init__(self, vec):
        self.vec = vec

    def available(self):
        return False

    def embed(self, text):
        return list(self.vec)


def _plant(backend, *rows):
    """Store `(id, content)` rows every cosine rung finds at 1.0, in order."""
    insights_by_id, embed_cache = {}, {}
    for rid, content in rows:
        ins = make_insight(id=rid, content=content)
        backend.nodes.insert(ins)
        insights_by_id[rid] = ins
        embed_cache[rid] = [1.0, 0.0]
    return insights_by_id, embed_cache


def _stub_stages(monkeypatch, screen, judge, merge):
    """Install the three stage stubs and return their call logs."""
    calls = {'screen': [], 'judge': [], 'merge': []}

    def _screen(client, fact_text, memory):
        calls['screen'].append(memory[0])
        return screen(memory[0])

    def _judge(client, fact_text, memory):
        calls['judge'].append(memory[0])
        return judge(memory[0])

    def _merge(client, fact_text, target):
        calls['merge'].append(target)
        return merge(target[0])

    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': content, 'category': 'fact', 'entities': []}])
    monkeypatch.setattr('memman.llm.extract.screen_memory', _screen)
    monkeypatch.setattr('memman.llm.extract.judge_memory', _judge)
    monkeypatch.setattr('memman.llm.extract.merge_successor', _merge)
    return calls


def _run(backend, fact_text, insights_by_id, embed_cache):
    return run_remember(
        backend, make_insight(id='parent', content=fact_text), fact_text,
        ec=_FixedEmbedder([1.0, 0.0]), embed_cache=embed_cache,
        insights_by_id=insights_by_id, store_name='test')


def _oplog(backend, operation):
    return [(e.insight_id, e.detail) for e in backend.oplog.recent(limit=50)
            if e.operation == operation]


def test_all_unrelated_screen_plans_add_with_no_verdict_call(tmp_backend, monkeypatch):
    """Verify an all-UNRELATED screen stores the fact with no stage-2 call.

    Mutation: showing every shortlisted row to stage 2 (the batch
        again), or calling stage 2 on an empty kept set.
    Oracle: the judge spy's empty log and the plain `add`.
    """
    caches = _plant(tmp_backend, ('old-1', 'the sky is blue'), ('old-2', 'tea is hot'))
    calls = _stub_stages(
        monkeypatch, screen=lambda rid: ('UNRELATED', []),
        judge=lambda rid: 'keep', merge=lambda rid: None)

    res = _run(tmp_backend, 'the broker is redis', *caches)

    assert [f['action'] for f in res['facts']] == ['add']
    assert sorted(calls['screen']) == ['old-1', 'old-2']
    assert calls['judge'] == []
    assert calls['merge'] == []


def test_an_unjudged_row_reaches_stage_two(tmp_backend, monkeypatch):
    """Verify a row the screen could not judge is shown to stage 2.

    Mutation: treating UNJUDGED as UNRELATED, so a screen outage drops
        every contradicted row silently.
    Oracle: the judge spy called with the UNJUDGED row and not the
        UNRELATED one.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the sky is blue'), ('old-2', 'tea is hot'))
    calls = _stub_stages(
        monkeypatch,
        screen=lambda rid: (UNJUDGED, []) if rid == 'old-2' else ('UNRELATED', []),
        judge=lambda rid: 'keep', merge=lambda rid: None)

    _run(tmp_backend, 'the broker is redis', *caches)

    assert calls['judge'] == ['old-2']


def test_a_contradicted_row_screened_in_is_superseded_with_the_merge_text(
        tmp_backend, monkeypatch):
    """Verify the screened, confirmed row is superseded with the merge text.

    Mutation: storing the fact text over the merge text, or feeding
        stage 3 the target without the clauses the screen quoted.
    Oracle: the predecessor's pointer, the successor's content equal to
        the stub's merge text, and the clauses on the merge spy's call.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the broker is kombu and the queue is durable'))
    calls = _stub_stages(
        monkeypatch, screen=lambda rid: ('CONTRADICTS', ['the broker is kombu']),
        judge=lambda rid: 'supersede',
        merge=lambda rid: 'the broker is redis and the queue is durable')

    res = _run(tmp_backend, 'the broker is redis', *caches)

    fact = res['facts'][0]
    assert (fact['action'], fact['replaced_ids']) == ('supersede', ['old-1'])
    assert fact['content'] == 'the broker is redis and the queue is durable'
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == fact['id']
    assert calls['merge'] == [(
        'old-1', 'the broker is kombu and the queue is durable',
        ['the broker is kombu'])]
    assert _oplog(tmp_backend, 'reconcile-supersede') == [
        ('old-1', f'replaced by {fact["id"]}')]


def test_one_successor_per_retiring_target(tmp_backend, monkeypatch):
    """Verify a fact retiring two rows writes two successors, one merge each.

    Mutation: one successor for N predecessors (the whole-body merge
        of 0.34.0), which stores a text the judges never read.
    Oracle: two `supersede` results with disjoint `replaced_ids`, each
        predecessor pointing at its own successor whose content is that
        target's merge text, and one `reconcile-candidates` row.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the broker is kombu'),
        ('old-2', 'the queue is durable'))
    _stub_stages(
        monkeypatch, screen=lambda rid: ('CONTRADICTS', [f'clause of {rid}']),
        judge=lambda rid: 'supersede', merge=lambda rid: f'merged for {rid}')

    res = _run(tmp_backend, 'nothing is durable and the broker is redis', *caches)

    by_target = {f['replaced_ids'][0]: f for f in res['facts']}
    assert set(by_target) == {'old-1', 'old-2'}
    for rid, fact in by_target.items():
        assert fact['action'] == 'supersede'
        assert fact['content'] == f'merged for {rid}'
        assert tmp_backend.nodes.get_include_deleted(rid).superseded_by == fact['id']
    assert len({f['id'] for f in res['facts']}) == 2
    assert len(_oplog(tmp_backend, 'reconcile-candidates')) == 1


def test_merge_failure_stores_the_fact_and_marks_unmerged_per_target(
        tmp_backend, monkeypatch):
    """Verify the fallback and its marker are decided per target.

    Mutation: one merge outcome for the whole fact, so a failure on one
        target marks or unmarks its sibling.
    Oracle: the target whose merge returned None stores the fact text
        and its oplog row ends `(unmerged)`; the other stores its text
        and carries no marker.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the broker is kombu'),
        ('old-2', 'the queue is durable'))
    _stub_stages(
        monkeypatch, screen=lambda rid: ('CONTRADICTS', []),
        judge=lambda rid: 'supersede',
        merge=lambda rid: 'merged for old-1' if rid == 'old-1' else None)

    res = _run(tmp_backend, 'the fact', *caches)

    by_target = {f['replaced_ids'][0]: f for f in res['facts']}
    assert by_target['old-1']['content'] == 'merged for old-1'
    assert by_target['old-2']['content'] == 'the fact'
    details = dict(_oplog(tmp_backend, 'reconcile-supersede'))
    assert details['old-1'] == f'replaced by {by_target["old-1"]["id"]}'
    assert details['old-2'] == f'replaced by {by_target["old-2"]["id"]} (unmerged)'


def test_a_none_verdict_beside_a_supersede_is_folded_as_an_update(
        tmp_backend, monkeypatch):
    """Verify the folded row is updated through its own merge under `(none)`.

    Mutation: leaving the restating row current beside the successor,
        or handing stage 3 the clauses the screen quoted for a row the
        verdict did not supersede.
    Oracle: `reconcile-update` on the NONE row, `reconcile-supersede`
        on the other, and the merge spy's empty clause list for the
        folded target although its screen quoted a clause.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the broker is kombu'),
        ('old-2', 'redis is the broker'))
    calls = _stub_stages(
        monkeypatch, screen=lambda rid: ('CONTRADICTS', [f'clause of {rid}']),
        judge=lambda rid: 'supersede' if rid == 'old-1' else 'none',
        merge=lambda rid: f'merged for {rid}')

    _run(tmp_backend, 'the broker is redis', *caches)

    assert [t for t, _ in _oplog(tmp_backend, 'reconcile-supersede')] == ['old-1']
    assert [t for t, _ in _oplog(tmp_backend, 'reconcile-update')] == ['old-2']
    assert ('old-2', 'redis is the broker', []) in calls['merge']


def test_candidates_row_carries_relation_screened_and_verdict(tmp_backend, monkeypatch):
    """Verify the replay row records what each stage saw and decided per row.

    Mutation: logging id, rung and score alone, so the F2 2x2 cannot
        tell a row the screen dropped from one stage 2 kept.
    Oracle: the planted rows' relation, screened flag and verdict read
        back from the one `reconcile-candidates` detail.
    """
    caches = _plant(
        tmp_backend, ('old-1', 'the broker is kombu'), ('old-2', 'tea is hot'),
        ('old-3', 'the queue is durable'))
    relations = {'old-1': ('CONTRADICTS', ['kombu']), 'old-2': ('UNRELATED', []),
                 'old-3': ('REFINES', [])}
    _stub_stages(
        monkeypatch, screen=lambda rid: relations[rid],
        judge=lambda rid: 'supersede', merge=lambda rid: 'merged')

    _run(tmp_backend, 'the broker is redis', *caches)

    rows = _oplog(tmp_backend, 'reconcile-candidates')
    assert len(rows) == 1
    by_id = {c['id']: c for c in json.loads(rows[0][1])['candidates']}
    seen = {(cid, c['relation'], c['screened'], c['verdict'])
            for cid, c in by_id.items()}
    assert seen == {
        ('old-1', 'CONTRADICTS', True, 'supersede'),
        ('old-2', 'UNRELATED', False, None),
        ('old-3', 'REFINES', False, None),
        }


def test_stage_executor_is_sized_to_the_shortlist(tmp_backend, monkeypatch):
    """Verify the per-fact executor takes one worker per shortlisted row.

    Mutation: the drain's two-worker executor for the screen, which
        serializes twenty 3 s calls into a minute.
    Oracle: the recorded `max_workers` equal to the planted shortlist,
        which is three.
    """
    sizes = []
    real = rem.ThreadPoolExecutor

    class _Recording(real):
        def __init__(self, max_workers=None, **kwargs):
            sizes.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(rem, 'ThreadPoolExecutor', _Recording)
    caches = _plant(tmp_backend, ('old-1', 'a'), ('old-2', 'b'), ('old-3', 'c'))
    _stub_stages(
        monkeypatch, screen=lambda rid: ('UNRELATED', []),
        judge=lambda rid: 'keep', merge=lambda rid: None)

    _run(tmp_backend, 'the fact', *caches)

    assert 3 in sizes


def test_run_remember_reports_one_result_per_successor(tmp_backend, monkeypatch):
    """Verify the drain's result lists every successor a fact fanned into.

    Mutation: reporting the first plan alone, so `queue_done` and the
        CLI hide the second retire.
    Oracle: two entries for one extracted fact, both `supersede`, with
        the real store bound embedder.
    """
    tmp_backend.nodes.insert(make_insight(id='old-1', content='the broker is kombu'))
    tmp_backend.nodes.insert(make_insight(id='old-2', content='the broker is rabbit'))
    _stub_stages(
        monkeypatch, screen=lambda rid: ('CONTRADICTS', []),
        judge=lambda rid: 'supersede', merge=lambda rid: f'merged for {rid}')

    res = run_remember(
        tmp_backend, make_insight(id='parent', content='the broker is redis'),
        'the broker is redis', ec=bound_embedder(tmp_backend), store_name='test')

    assert [f['action'] for f in res['facts']] == ['supersede', 'supersede']
    assert sorted(f['replaced_ids'][0] for f in res['facts']) == ['old-1', 'old-2']


def test_a_successor_retired_in_the_same_write_leaves_the_drain_cache(
        tmp_backend, monkeypatch):
    """Verify a row retired later in the same write leaves both caches.

    Mutation: re-registering every inserted row in the drain cache and
        never evicting the ones the write itself superseded, so the next
        queue row of the drain builds semantic edges onto a retired row.
    Oracle: two facts, the second contradicting the first's successor;
        after the write the first successor is in neither drain cache
        and the second is in both.
    """
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': 'the broker is redis', 'category': 'fact', 'entities': []},
            {'text': 'the broker is rabbit', 'category': 'fact', 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: (
            ('CONTRADICTS', []) if 'rabbit' in fact_text else ('UNRELATED', [])))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)
    embed_cache, insights_by_id = {}, {}

    res = run_remember(
        tmp_backend, make_insight(id='parent', content='the broker'), 'the broker',
        ec=_FixedEmbedder([1.0, 0.0]), embed_cache=embed_cache,
        insights_by_id=insights_by_id, store_name='test')

    first, second = res['facts']
    assert (first['action'], second['action']) == ('add', 'supersede')
    assert second['replaced_ids'] == [first['id']]
    assert first['id'] not in embed_cache
    assert first['id'] not in insights_by_id
    assert second['id'] in embed_cache
    assert second['id'] in insights_by_id
