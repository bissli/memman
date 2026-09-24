"""A verdict on a stored row must not discard the write's own claim.

The NONE branch of `assemble_verdicts` (`pipeline/remember.py`)
skipped the write whenever the verdict named any target, whatever the
screen had said about it. A fact the screen itself called REFINES -
"the fact adds compatible detail" - was discarded whole with
`skip_reason='already captured'`, so a correction died against the
row it was correcting.
"""

from memman.embed.fingerprint import bound_embedder
from memman.pipeline.remember import run_remember
from tests.conftest import make_insight
from tests.test_supersession_pipeline import _parent


def test_a_stored_row_is_still_superseded(tmp_backend, monkeypatch):
    """Verify a supersede verdict still retires a stored target.

    Mutation: barring every supersede target rather than acting on
        one, which would strand every contradicted row live forever.
    Oracle: the stored row's own `superseded_by` pointer, which must
        name the new fact.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

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
        tmp_backend, _parent('the broker is redis now'),
        'the broker is redis now',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'supersede'
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == (
        res['facts'][0]['id'])


def test_a_stored_row_is_still_updated(tmp_backend, monkeypatch):
    """Verify an update verdict still retires a stored target.

    Mutation: barring every update target rather than acting on one,
        which would leave no row ever refined.
    Oracle: the stored row's own `superseded_by` pointer, which must
        name the new fact.
    """
    tmp_backend.nodes.insert(make_insight(
        id='old-1', content='the broker is kombu'))

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
        tmp_backend, _parent('the broker is redis now'),
        'the broker is redis now',
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'update'
    assert tmp_backend.nodes.get_include_deleted('old-1').superseded_by == (
        res['facts'][0]['id'])


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
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory', _screen_as('REFINES'))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'none')

    fact_text = ('the parser handles slash swaps and wires'
                 ' conversion_ratio into parity')
    res = run_remember(
        tmp_backend, _parent(fact_text), fact_text,
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
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory', _screen_as('RESTATES'))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'none')

    fact_text = ('the parser handles slash swaps and wires'
                 ' conversion_ratio into parity')
    res = run_remember(
        tmp_backend, _parent(fact_text), fact_text,
        ec=bound_embedder(tmp_backend), store_name='test')

    assert res['facts'][0]['action'] == 'skipped'
    assert tmp_backend.nodes.get('old-1').corroboration_count == 1
