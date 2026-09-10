"""Tests for `pipeline.remember`'s planning and prompt pinning.

Covers the two invariants the drain's write path cannot express in
its own output: the reconcile candidate list must carry the
strongest near-duplicate rather than the first ones the cache
happened to yield, and `compute_prompt_version` must not move
except deliberately.
"""


def test_reconcile_candidates_ranked_by_similarity(monkeypatch):
    """The strongest near-duplicate must reach the reconcile candidate list.

    Regression: the cosine candidates were appended in embed_cache order
    and capped at MAX_SIMILAR_FOR_RECONCILE, so a high-cosine insight that
    sorts last could be crowded out by weaker earlier ones.
    """
    import math
    from unittest.mock import MagicMock

    from memman.llm import extract as llm_extract
    from memman.pipeline import remember as rem
    from tests.conftest import make_insight

    screened = []

    def _screen(client, fact_text, memory):
        screened.append(memory[0])
        return 'UNRELATED', []

    monkeypatch.setattr(llm_extract, 'screen_memory', _screen)

    fact_vec = [1.0, 0.0]
    med = [0.6, math.sqrt(1 - 0.6 * 0.6)]
    top = [0.95, math.sqrt(1 - 0.95 * 0.95)]

    insights_by_id = {}
    embed_cache = {}
    for i in range(10):
        ins = make_insight(id=f'dec{i}', content=f'decoy body number {i}')
        insights_by_id[ins.id] = ins
        embed_cache[ins.id] = list(med)
    topins = make_insight(id='TOP', content='topmost candidate body')
    insights_by_id[topins.id] = topins
    embed_cache[topins.id] = list(top)

    fact = {'text': 'zzqq alpha brandnew', 'category': 'fact',
            'importance': 3, 'entities': []}
    parent = make_insight(id='parent', content='zzqq alpha brandnew')
    ec = MagicMock()
    ec.embed.return_value = fact_vec

    rem._plan_fact(
        fact, parent, '', False, False,
        insights_by_id, embed_cache, set(),
        MagicMock(), MagicMock(), ec, MagicMock(), MagicMock(), 'teststore')

    assert 'TOP' in screened, f'top-cosine insight crowded out; screened={screened}'


def test_prompt_version_unchanged_by_length_caps():
    """The length caps live post-parse; the prompt hash is pinned.

    The pin is a tripwire, not a constant: any deliberate change to a
    hashed input moves it, and re-pinning is the right answer once the
    author has weighed the cost. That cost is what the tripwire
    surfaces -- every stored row in every store goes stale at once,
    and only a `graph rebuild --stale` clears it.

    Two inputs now move this value and neither is a length cap: the
    enrichment prompt and the causal prompt. So does the configured
    `MEMMAN_LLM_MODEL_SLOW_METADATA`, which the key folds in and which
    the suite seeds from `INSTALL_DEFAULTS` -- changing that default
    re-pins this test, deliberately.

    Mutation: "fixing" the length caps inside a system prompt, or any
        other incidental edit to a hashed input -- the hash moves and
        every stored row goes stale for a change nobody intended.
    Oracle: the hash of the two replayable prompts plus the seeded
        metadata model, pinned.
    """
    from memman.pipeline.remember import compute_prompt_version
    assert compute_prompt_version() == '6a60ef0080b1ab9f'


class _FixedEmbedder:
    """An embed provider returning one fixed vector for every text."""

    model = 'fixed'

    def __init__(self, vec):
        self.vec = vec

    def available(self):
        return False

    def embed(self, text):
        return list(self.vec)


def _plant_shortlist(backend):
    """Two stored rows: one the keyword rung finds, one only cosine finds.

    Returns the drain-scope caches `run_remember` takes, so the cosine
    rung reads vectors this test controls rather than the mock embedder's.
    """
    from tests.conftest import make_insight
    kw = make_insight(id='kw-1', content='zulu yankee xray whiskey victor uniform')
    cos = make_insight(id='cos-1', content='gardening tulips bloom in spring soil')
    backend.nodes.insert(kw)
    backend.nodes.insert(cos)
    insights_by_id = {kw.id: kw, cos.id: cos}
    embed_cache = {'kw-1': [0.0, 1.0], 'cos-1': [1.0, 0.0]}
    return insights_by_id, embed_cache


def _candidates_rows(backend):
    import json
    return [(e.insight_id, json.loads(e.detail))
            for e in backend.oplog.recent(limit=20)
            if e.operation == 'reconcile-candidates']


def test_reconcile_candidates_are_logged(tmp_backend, monkeypatch):
    """Verify a write logs the reconciler's shortlist with each row's rung.

    Mutation: dropping the `reconcile-candidates` row, or logging ids
        without their rung, so the F2 replay cannot tell a keyword hit
        from a cosine one.
    Oracle: one row keyed on the inserted insight whose detail lists the
        planted keyword row as `keyword` and the planted cosine row as
        `cosine`, against the rows this test planted.
    """
    from memman.pipeline.remember import run_remember
    from tests.conftest import make_insight

    insights_by_id, embed_cache = _plant_shortlist(tmp_backend)
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': content, 'category': 'fact', 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('UNRELATED', []))

    fact_text = 'zulu yankee xray whiskey victor'
    res = run_remember(
        tmp_backend, make_insight(id='parent', content=fact_text),
        fact_text, ec=_FixedEmbedder([1.0, 0.0]),
        embed_cache=embed_cache, insights_by_id=insights_by_id, store_name='test')

    new_id = res['facts'][0]['id']
    rows = _candidates_rows(tmp_backend)
    assert [key for key, _ in rows] == [new_id]
    detail = rows[0][1]
    assert detail['fact_id'] == new_id
    assert {(c['id'], c['rung']) for c in detail['candidates']} == {
        ('kw-1', 'keyword'), ('cos-1', 'cosine')}


def test_reconcile_candidates_are_logged_for_a_none_skip(tmp_backend, monkeypatch):
    """Verify a NONE skip logs the shortlist keyed on the row it corroborated.

    Mutation: logging on the write path only, so every NONE and
        exact-match case is missing from the replay's 2x2.
    Oracle: the row read back by the NONE target's id, listing both
        planted candidates.
    """
    from memman.pipeline.remember import run_remember
    from tests.conftest import make_insight

    insights_by_id, embed_cache = _plant_shortlist(tmp_backend)
    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': content, 'category': 'fact', 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: (
            ('RESTATES', []) if memory[0] == 'cos-1' else ('UNRELATED', [])))
    monkeypatch.setattr(
        'memman.llm.extract.judge_memory',
        lambda client, fact_text, memory: 'none')

    fact_text = 'zulu yankee xray whiskey victor'
    res = run_remember(
        tmp_backend, make_insight(id='parent', content=fact_text),
        fact_text, ec=_FixedEmbedder([1.0, 0.0]),
        embed_cache=embed_cache, insights_by_id=insights_by_id, store_name='test')

    assert res['facts'][0]['action'] == 'skipped'
    rows = _candidates_rows(tmp_backend)
    assert [key for key, _ in rows] == ['cos-1']
    assert {c['id'] for c in rows[0][1]['candidates']} == {'kw-1', 'cos-1'}


def _plant_cosine_rows(count, top=0.95, step=0.01):
    """Rows `row0..row{count-1}` at cosine `top - i * step` to the fact `[1, 0]`."""
    import math

    from tests.conftest import make_insight

    insights_by_id = {}
    embed_cache = {}
    for i in range(count):
        ins = make_insight(id=f'row{i}', content=f'candidate body number {i}')
        insights_by_id[ins.id] = ins
        cos = top - i * step
        embed_cache[ins.id] = [cos, math.sqrt(1 - cos * cos)]
    return insights_by_id, embed_cache


def _plan_shortlist(monkeypatch, insights_by_id, embed_cache,
                    fact_text='zzqq alpha brandnew'):
    """Run `_plan_fact` with the screen stubbed UNRELATED; return the candidates."""
    from unittest.mock import MagicMock

    from memman.llm import extract as llm_extract
    from memman.pipeline import remember as rem
    from tests.conftest import make_insight

    monkeypatch.setattr(
        llm_extract, 'screen_memory',
        lambda client, fact_text, memory: ('UNRELATED', []))
    fact = {'text': fact_text, 'category': 'fact', 'entities': []}
    ec = MagicMock()
    ec.embed.return_value = [1.0, 0.0]
    plans, _calls = rem._plan_fact(
        fact, make_insight(id='parent', content=fact_text),
        '', False, False, insights_by_id, embed_cache, set(),
        MagicMock(), MagicMock(), ec, MagicMock(), MagicMock(), 'teststore')
    return plans[0].candidates


def test_shortlist_fills_to_twenty_rows(monkeypatch):
    """Verify the two scored rungs fill the shortlist to twenty rows.

    Mutation: the cap left at ten, or the slots the keyword rung leaves
        empty not passed to the rerank rung, so the list stops short.
    Oracle: twenty-five planted rows and no keyword hit under the
        passthrough reranker, against the cap of twenty: nine cosine
        rows then eleven rerank rows.
    """
    from memman.pipeline import remember as rem

    insights_by_id, embed_cache = _plant_cosine_rows(25)

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    assert [c.id for c in candidates] == [f'row{i}' for i in range(20)]
    assert [c.rung for c in candidates] == (
        ['cosine'] * rem.RECONCILE_COSINE_SLOTS
        + ['rerank'] * (20 - rem.RECONCILE_COSINE_SLOTS))


def test_shortlist_rerank_slots_follow_the_reranker(monkeypatch):
    """Verify the rerank slots take the reranker's order, skipping taken rows.

    Mutation: the rerank slots filled in cosine order, or the cosine rows
        already taken not skipped so they appear twice.
    Oracle: a stub reranker that inverts the cosine order over thirty
        rows; hand-computed: row0..row8 by cosine, then row29 down to
        row19 by rerank score.
    """
    from memman.pipeline import remember as rem

    insights_by_id, embed_cache = _plant_cosine_rows(30)
    monkeypatch.setattr(
        'memman.rerank.voyage.Client.rerank',
        lambda self, query, documents, top_k=None: [
            (i, i / 100) for i in reversed(range(len(documents)))])

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    cosine_ids = [f'row{i}' for i in range(rem.RECONCILE_COSINE_SLOTS)]
    rerank_ids = [f'row{i}' for i in range(29, 18, -1)]
    assert [c.id for c in candidates] == cosine_ids + rerank_ids
    assert [c.rung for c in candidates][rem.RECONCILE_COSINE_SLOTS:] == ['rerank'] * 11
    assert [round(c.score, 2) for c in candidates][rem.RECONCILE_COSINE_SLOTS:] == [
        i / 100 for i in range(29, 18, -1)]


def test_shortlist_rerank_pool_is_the_top_hundred_by_cosine(monkeypatch):
    """Verify the reranker sees the top 100 rows by cosine, floor or none.

    Mutation: the pool left unbounded (every row sent), the 0.5 floor
        kept (rows below it never sent), the keyword row left out of the
        pool, or the keyword row counted against the cosine quota.
    Oracle: 130 planted rows whose cosine falls below 0.5 from row82 on,
        plus one keyword-hit row at cosine 0.999; the stub records the
        documents it receives; one keyword, nine cosine, ten rerank.
    """
    from memman.pipeline import remember as rem
    from tests.conftest import make_insight

    insights_by_id, embed_cache = _plant_cosine_rows(130, top=0.99, step=0.006)
    kw = make_insight(id='kw-hit', content='zzqq alpha brandnew as stored')
    insights_by_id[kw.id] = kw
    embed_cache[kw.id] = [0.999, (1 - 0.999 ** 2) ** 0.5]
    received = []
    monkeypatch.setattr(
        'memman.rerank.voyage.Client.rerank',
        lambda self, query, documents, top_k=None: received.append(list(documents)) or [
            (i, 1.0 - i / 200) for i in range(len(documents))])

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    assert len(received) == 1
    assert len(received[0]) == 100
    assert received[0] == (
        [kw.content] + [f'candidate body number {i}' for i in range(99)])
    assert embed_cache['row82'][0] < 0.5 < embed_cache['row81'][0]
    ids = [c.id for c in candidates]
    assert ids[0] == 'kw-hit'
    assert candidates[0].rung == 'keyword'
    assert len(ids) == len(set(ids)) == 20
    rungs = [c.rung for c in candidates]
    assert rungs.count('cosine') == rem.RECONCILE_COSINE_SLOTS
    assert rungs.count('rerank') == 20 - 1 - rem.RECONCILE_COSINE_SLOTS


def test_shortlist_falls_back_to_cosine_when_the_reranker_fails(monkeypatch):
    """Verify a failed rerank call leaves a cosine-ordered shortlist.

    Mutation: the exception propagating and losing the fact, or the
        rerank slots left empty so the screen sees nine rows.
    Oracle: a stub reranker that raises; twenty cosine rows in cosine
        order, every rung `cosine`.
    """
    def _boom(self, query, documents, top_k=None):
        raise RuntimeError('Voyage rerank returned status 500')

    insights_by_id, embed_cache = _plant_cosine_rows(25)
    monkeypatch.setattr('memman.rerank.voyage.Client.rerank', _boom)

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    assert [c.id for c in candidates] == [f'row{i}' for i in range(20)]
    assert {c.rung for c in candidates} == {'cosine'}


def test_shortlist_skips_the_reranker_when_the_store_disables_rerank(
        monkeypatch, env_file):
    """Verify the per-store rerank toggle governs the write path too.

    Mutation: the toggle read under the wrong key, or ignored, so a store
        an operator switched off still pays the rerank call; the rung's
        failure fallback would hide that, so the spy is the oracle.
    Oracle: `MEMMAN_RERANK_ENABLED_teststore=false` in the env file and a
        spy reranker that records every call; the list is cosine only.
    """
    calls = []

    def _spy(self, query, documents, top_k=None):
        calls.append(len(documents))
        return [(i, 1.0) for i in range(len(documents))]

    env_file('MEMMAN_RERANK_ENABLED_teststore', 'false')
    insights_by_id, embed_cache = _plant_cosine_rows(25)
    monkeypatch.setattr('memman.rerank.voyage.Client.rerank', _spy)

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    assert calls == []
    assert [c.id for c in candidates] == [f'row{i}' for i in range(20)]
    assert {c.rung for c in candidates} == {'cosine'}


def test_shortlist_pool_keeps_positive_cosines_only(monkeypatch):
    """Verify the sign boundary: rows at cosine <= 0 never reach the reranker.

    Mutation: the sign boundary dropped, so orthogonal and anti-correlated
        rows are sent, ranked, and screened.
    Oracle: twelve rows above zero, one at zero and five below; the stub
        records the documents it receives; the list holds the twelve.
    """
    import math

    from tests.conftest import make_insight

    insights_by_id, embed_cache = _plant_cosine_rows(12, top=0.6, step=0.05)
    for i, cos in enumerate([0.0, -0.1, -0.2, -0.3, -0.4, -0.5]):
        ins = make_insight(id=f'neg{i}', content=f'unrelated body number {i}')
        insights_by_id[ins.id] = ins
        embed_cache[ins.id] = [cos, math.sqrt(1 - cos * cos)]
    received = []

    def _record(self, query, documents, top_k=None):
        received.append(list(documents))
        return [(i, 1.0 - i / 100) for i in range(len(documents))]

    monkeypatch.setattr('memman.rerank.voyage.Client.rerank', _record)

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    assert received == [[f'candidate body number {i}' for i in range(12)]]
    assert [c.id for c in candidates] == [f'row{i}' for i in range(12)]


def test_shortlist_rerank_ties_order_by_id_descending(monkeypatch):
    """Verify equal rerank scores order by id descending, not provider order.

    Mutation: the tie rule reduced to the score alone, so Python's stable
        sort keeps the provider's order and a provider that reorders ties
        changes the shortlist run to run.
    Oracle: a stub that scores every document 0.5; hand-computed over the
        untaken ids as strings: row9 first, then row24 down to row15.
    """
    from memman.pipeline import remember as rem

    insights_by_id, embed_cache = _plant_cosine_rows(25)
    monkeypatch.setattr(
        'memman.rerank.voyage.Client.rerank',
        lambda self, query, documents, top_k=None: [
            (i, 0.5) for i in range(len(documents))])

    candidates = _plan_shortlist(monkeypatch, insights_by_id, embed_cache)

    cosine_ids = [f'row{i}' for i in range(rem.RECONCILE_COSINE_SLOTS)]
    assert [c.id for c in candidates] == cosine_ids + ['row9'] + [
        f'row{i}' for i in range(24, 14, -1)]


def test_shortlist_quotas_match_the_measured_plateau():
    """Pin the cosine slot count and the rerank pool to their measured values.

    Mutation: a quota moved without re-measuring the plateau, or the
        write path's pool drifting from the read path's shortlist.
    Oracle: the zero-call measurement of 2026-09-10 over 64 cases, where
        every cosine slot count from 6 to 12 reaches 6 of the 8 missed
        targets with all 59 kept and 9 is its middle; the read path's
        `RERANK_SHORTLIST`.
    """
    from memman.pipeline import remember as rem
    from memman.search import recall

    assert rem.RECONCILE_COSINE_SLOTS == 9
    assert rem.RERANK_POOL == recall.RERANK_SHORTLIST == 100


def test_apply_never_links_a_planned_row_before_it_is_inserted(tmp_backend, monkeypatch):
    """Verify a write of two near-identical facts commits.

    Mutation: leaving every planned row's vector in the drain cache
        through the apply phase, so the first row's semantic-edge step
        aims an edge at the second row before its insert and the
        transaction fails on the foreign key.
    Oracle: two stored rows with a semantic edge between them, on an
        embedder that returns one vector for every text.
    """
    from memman.pipeline.remember import run_remember
    from tests.conftest import make_insight

    monkeypatch.setattr(
        'memman.llm.extract.extract_facts',
        lambda client, content: [
            {'text': 'the broker is redis', 'category': 'fact', 'entities': []},
            {'text': 'redis is the broker', 'category': 'fact', 'entities': []}])
    monkeypatch.setattr(
        'memman.llm.extract.screen_memory',
        lambda client, fact_text, memory: ('UNRELATED', []))

    res = run_remember(
        tmp_backend, make_insight(id='parent', content='the broker'),
        'the broker', ec=_FixedEmbedder([1.0, 0.0]), store_name='test')

    ids = [f['id'] for f in res['facts']]
    assert [f['action'] for f in res['facts']] == ['add', 'add']
    linked = {e.target_id for e in tmp_backend.edges.by_node(ids[0])
              if e.edge_type == 'semantic'}
    assert ids[1] in linked
