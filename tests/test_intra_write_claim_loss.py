"""Two ways one write discarded a claim it was told it had stored.

Both defects sit in the fifteen-line window at
`pipeline/remember.py` where `assemble_verdicts` hands back an action
and its targets.

- A SIBLING - a fact the same write authored moments earlier - was
  registered in the shortlist caches so later facts could match it,
  but nothing barred it from being chosen as a SUPERSEDE target. The
  second fact then retired the first, and a two-claim write stored one
  claim.
- The NONE branch skipped the write whenever the verdict named any
  target, whatever the screen had said about it. A fact the screen
  itself called REFINES - "the fact adds compatible detail" - was
  discarded whole with `skip_reason='already captured'`, so a
  correction died against the row it was correcting.

The two are not independent: both mutate `action` and `targets` in
that one window, and the sibling filter has to run first or the NONE
guard indexes a list the filter may have emptied.
"""

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight
from tests.test_supersession_pipeline import _parent


def _two_unrelated_facts(llm_client, content):
    """Extract two facts that share no claim."""
    return [
        {'text': 'the cache eviction policy is lru', 'category': 'fact',
         'importance': 3, 'entities': []},
        {'text': 'the broker is redis now', 'category': 'fact',
         'importance': 3, 'entities': []},
        ]


def test_a_sibling_of_one_write_is_never_superseded(
        tmp_backend, monkeypatch):
    """Verify the second fact of a write cannot retire the first.

    Mutation: dropping the sibling filter on the supersede targets,
        so the second fact of one write retires the first and a
        two-claim write stores one claim. This is the live defect,
        written down as the thing that was actually wrong.
    Oracle: the two hand-named fact texts handed to the extractor,
        both of which must be present in the store afterward. The
        comparand is the extraction input, independent of any
        reconcile stage.
    """
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts', _two_unrelated_facts)
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('CONTRADICTS', []))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('two things changed'), 'two things changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert [f['action'] for f in res['facts']] == ['add', 'add']
    assert sorted(i.content for i in tmp_backend.nodes.get_all_active()) == [
        'the broker is redis now', 'the cache eviction policy is lru']


def test_a_stored_row_is_still_superseded(tmp_backend, monkeypatch):
    """Verify the sibling filter does not bar a STORED supersede target.

    The paired control for the test above: a filter that simply never
    superseded anything would pass that one.

    Mutation: barring every supersede target rather than only a
        sibling the same write authored, which would strand every
        contradicted row live forever.
    Oracle: the stored row's own `superseded_by` pointer, which must
        name the new fact.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': 'the broker is redis now', 'category': 'fact',
             'importance': 3, 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: (
            'CONTRADICTS' if memory[0] == 'old-1' else 'UNRELATED', []))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'supersede')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'supersede'
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == (
        res['facts'][0]['id'])


def test_a_sibling_of_one_write_is_never_updated(tmp_backend, monkeypatch):
    """Verify an `update` verdict cannot retire a sibling either.

    `update` is supersede-plus-pointer, so it retires its target down
    the same path. The merge stage was thought to keep the sibling's
    clauses alive on that path, but a merge returning no text falls
    back to the bare fact, and then the sibling's claim is stored
    nowhere.

    Mutation: filtering the sibling targets on `supersede` alone, so
        an `update` verdict retires the first fact of a write and a
        two-claim write stores one claim. This is the live defect,
        written down as the thing that was actually wrong.
    Oracle: the two hand-named fact texts handed to the extractor,
        both of which must be present in the store afterward. The
        merge stub returns None, which is the shipped fallback, so the
        successor text cannot carry the retired sibling's clause.
    """
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts', _two_unrelated_facts)
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('CONTRADICTS', []))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'update')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('two things changed'), 'two things changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert [f['action'] for f in res['facts']] == ['add', 'add']
    assert sorted(i.content for i in tmp_backend.nodes.get_all_active()) == [
        'the broker is redis now', 'the cache eviction policy is lru']


def test_a_stored_row_is_still_updated(tmp_backend, monkeypatch):
    """Verify the sibling filter does not bar a STORED update target.

    The paired control: a filter that barred every update target would
    pass the test above and leave no row ever refined.

    Mutation: barring every update target rather than only a sibling
        the same write authored.
    Oracle: the stored row's own `superseded_by` pointer, which must
        name the new fact.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': 'the broker is redis now', 'category': 'fact',
             'importance': 3, 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: (
            'CONTRADICTS' if memory[0] == 'old-1' else 'UNRELATED', []))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'update')
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('the broker changed'), 'the broker changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'update'
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == (
        res['facts'][0]['id'])


def test_a_mixed_target_list_keeps_the_stored_half(
        tmp_backend, monkeypatch):
    """Verify a sibling target is dropped without the stored one going too.

    One verdict can name both a stored row and a sibling. The filter
    has to remove the sibling alone; discarding the whole list instead
    strands the contradicted stored row live forever, and no other
    test in this file supplies a mixed list.

    Mutation: emptying the target list whenever ANY target is a
        sibling, rather than dropping the sibling entries. The whole
        suite stays green under it.
    Oracle: the seeded row's own `superseded_by` pointer, which must
        name the second fact, together with both sibling texts, which
        must both stay active.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

    monkeypatch.setattr(
        'memman.llm.extract.extract_facts', _two_unrelated_facts)
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('CONTRADICTS', []))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: (
            'supersede' if memory[0] == 'old-1' else 'none'))
    monkeypatch.setattr(
        'memman.llm.extract.merge_successor',
        lambda client, fact_text, target: None)

    res = run_remember(
        tmp_backend, _parent('two things changed'), 'two things changed',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == (
        res['facts'][1]['id'])
    assert sorted(i.content for i in tmp_backend.nodes.get_all_active()) == [
        'the broker is redis now', 'the cache eviction policy is lru']


def _one_fact(llm_client, content):
    """Extract a single fact carrying a clause no stored row holds."""
    return [{'text': 'the parser handles slash swaps and wires'
                     ' conversion_ratio into parity',
             'category': 'fact', 'importance': 3, 'entities': []}]


def _screen_as(relation):
    """Stub the screen to answer `relation` for the seeded row."""
    def _screen(client, fact_text, memory):
        return (relation, []) if memory[0] == 'old-1' else ('UNRELATED', [])
    return _screen


def test_none_verdict_on_a_refining_target_stores_the_fact(
        tmp_backend, monkeypatch):
    """Verify a NONE verdict on a REFINES target still stores the fact.

    Mutation: the NONE branch returning its skip unconditionally, so
        a fact the screen itself called REFINES - "the fact adds
        compatible detail" - is discarded whole with
        `skip_reason='already captured'`.
    Oracle: the screen prompt's own definition of REFINES at
        `llm/extract.py:205` as the boundary, paired with the
        RESTATES case below which must still skip. A one-sided test
        would pass on an unconditional add.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the parser handles slash swaps'))
    monkeypatch.setattr('memman.llm.extract.extract_facts', _one_fact)
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory', _screen_as('REFINES'))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'none')

    res = run_remember(
        tmp_backend, _parent('a parser correction'), 'a parser correction',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'add'
    stored = sorted(i.content for i in tmp_backend.nodes.get_all_active())
    assert stored == [
        'the parser handles slash swaps',
        ('the parser handles slash swaps and wires conversion_ratio'
         ' into parity')]


def test_none_verdict_on_a_restating_target_still_skips(
        tmp_backend, monkeypatch):
    """Verify a NONE verdict on a RESTATES target still skips and corroborates.

    The other side of the REFINES boundary: RESTATES is the one
    relation whose own prompt text says the memory already carries
    every claim the fact makes, so its skip is correct and must
    survive the fix.

    Mutation: turning every NONE verdict into an add, which would
        store a duplicate row for every restatement and stop the
        corroboration counter moving.
    Oracle: the skip action plus the seeded row's own
        `corroboration_count`, read back off the store.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the parser handles slash swaps'))
    monkeypatch.setattr('memman.llm.extract.extract_facts', _one_fact)
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory', _screen_as('RESTATES'))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'none')

    res = run_remember(
        tmp_backend, _parent('a parser restatement'), 'a parser restatement',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'skipped'
    assert tmp_backend.nodes.get('old-1').corroboration_count == 1
