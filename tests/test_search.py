"""Tests for memman.search -- keyword, intent, and recall."""

import pytest
from memman.embed.fingerprint import stored_fingerprint
from memman.search.intent import detect_intent, get_weights, intent_from_string
from memman.search.keyword import keyword_search, tokenize
from memman.search.recall import _RERANK_WEIGHTS_RAW, RERANK_WEIGHTS
from memman.search.recall import get_traversal_params, intent_aware_recall
from memman.store.model import Insight
from tests.conftest import _vec as _vec_512
from tests.conftest import make_edge, make_insight


class TestKeywordSearch:
    """Tokenization and keyword search ranking."""

    def test_tokenize_english(self):
        """English words are lowercased and split."""
        tokens = tokenize('Go uses SQLite for persistent storage')
        assert 'go' in tokens
        assert 'sqlite' in tokens
        assert 'persistent' in tokens
        assert 'storage' in tokens
        assert 'for' not in tokens

    def test_tokenize_stopwords(self):
        """Common stopwords are filtered out."""
        tokens = tokenize('the quick fox is very fast')
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'very' not in tokens
        assert 'quick' in tokens
        assert 'fox' in tokens
        assert 'fast' in tokens

    def test_tokenize_empty(self):
        """Empty string produces empty set."""
        assert len(tokenize('')) == 0

    def test_tokenize_all_stopwords(self):
        """All-stopword input produces empty set."""
        assert len(tokenize('the is a an')) == 0

    def test_keyword_search_ranking(self):
        """Best match ranks first."""
        insights = [
            Insight(id='1', content='Go language for building CLI tools', importance=3),
            Insight(id='2', content='SQLite database for Go applications', importance=3),
            Insight(id='3', content='Python machine learning framework', importance=3),
            ]
        results = keyword_search(insights, 'Go CLI tools', 10)
        assert len(results) >= 2
        assert results[0][0].id == '1'
        for i in range(1, len(results)):
            assert results[i][1] <= results[i - 1][1]

    def test_keyword_search_limit(self):
        """Limit caps the result count."""
        words = ['common', 'shared', 'words', 'alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'theta']
        insights = [
            Insight(id=str(i),
                    content=' '.join(words[:3 + (i % len(words))]),
                    importance=i + 1)
            for i in range(20)
            ]
        results = keyword_search(insights, 'common shared words', 5)
        assert len(results) <= 5

    def test_keyword_search_importance_tiebreak(self):
        """Higher importance wins on score tie."""
        insights = [
            Insight(id='low', content='Go memory graph', importance=1),
            Insight(id='high', content='Go memory graph', importance=5),
            ]
        results = keyword_search(insights, 'Go memory graph', 10)
        assert len(results) >= 2
        assert results[0][0].id == 'high'

    def test_keyword_search_empty_query(self):
        """Empty query returns empty results."""
        insights = [Insight(id='1', content='some content')]
        results = keyword_search(insights, '', 10)
        assert len(results) == 0

    def test_keyword_search_entities(self):
        """Entities contribute to matching."""
        insights = [
            Insight(id='1', content='something unrelated',
                    entities=['SQLite']),
            ]
        results = keyword_search(insights, 'SQLite', 10)
        assert len(results) > 0


class TestIntentRouting:
    """Intent detection and weight dispatch."""

    def test_detect_why(self):
        """Why-related queries detect WHY intent."""
        for q in ['why did we choose SQLite',
                  'the reason we chose Go because of motivation']:
            assert detect_intent(q) == 'WHY'

    def test_detect_when(self):
        """Time-related queries detect WHEN intent."""
        for q in ['when was the database migrated',
                  'timeline of changes',
                  'what happened before the release']:
            assert detect_intent(q) == 'WHEN'

    def test_detect_entity(self):
        """Entity-related queries detect ENTITY intent."""
        for q in ['what is MAGMA',
                  'who is responsible for the API',
                  'tell me about the graph engine']:
            assert detect_intent(q) == 'ENTITY'

    def test_detect_general(self):
        """Non-specific queries detect GENERAL intent."""
        for q in ['SQLite performance tuning',
                  'graph traversal algorithm']:
            assert detect_intent(q) == 'GENERAL'

    def test_intent_from_string_valid(self):
        """Valid intent strings parse correctly."""
        assert intent_from_string('WHY') == 'WHY'
        assert intent_from_string('why') == 'WHY'
        assert intent_from_string(' When ') == 'WHEN'
        assert intent_from_string('ENTITY') == 'ENTITY'
        assert intent_from_string('general') == 'GENERAL'

    def test_intent_from_string_invalid(self):
        """Invalid intent string raises ValueError."""
        with pytest.raises(ValueError):
            intent_from_string('BOGUS')

    def test_get_weights_known(self):
        """All intents have weights summing to ~1.0."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            w = get_weights(intent)
            assert len(w) > 0
            total = sum(w.values())
            assert 0.99 < total < 1.01

    def test_get_weights_why_prioritizes_causal(self):
        """WHY intent has highest causal weight."""
        w = get_weights('WHY')
        assert w['causal'] > w['temporal']
        assert w['causal'] > w['semantic']
        assert w['causal'] > w['entity']

    def test_get_weights_when_prioritizes_temporal(self):
        """WHEN intent has highest temporal weight."""
        w = get_weights('WHEN')
        assert w['temporal'] > w['causal']
        assert w['temporal'] > w['semantic']
        assert w['temporal'] > w['entity']

    def test_get_weights_entity_prioritizes_entity(self):
        """ENTITY intent has highest entity weight."""
        w = get_weights('ENTITY')
        assert w['entity'] > w['temporal']
        assert w['entity'] > w['causal']

    def test_get_weights_unknown_fallback(self):
        """Unknown intent falls back to GENERAL weights."""
        w = get_weights('NONEXISTENT')
        general = get_weights('GENERAL')
        for k, v in general.items():
            assert w[k] == v

    def test_detect_intent_tie_returns_general(self):
        """Tied WHY and ENTITY scores fall through to GENERAL."""
        assert detect_intent('describe why') == 'GENERAL'


class TestRecallRanking:
    """Beam search, traversal params, reranking, and meta fields."""

    def test_get_traversal_params_known(self):
        """All known intents have valid params."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            beam_width, max_depth, max_visited = get_traversal_params(intent)
            assert beam_width > 0
            assert max_depth > 0
            assert max_visited > 0

    def test_get_traversal_params_why_larger_beam(self):
        """WHY has larger beam width than GENERAL."""
        why_beam, _why_depth, _why_vis = get_traversal_params('WHY')
        gen_beam, _gen_depth, _gen_vis = get_traversal_params('GENERAL')
        assert why_beam > gen_beam

    def test_get_traversal_params_unknown_fallback(self):
        """Unknown intent falls back to GENERAL = (10, 4, 500)."""
        assert get_traversal_params('UNKNOWN') == (10, 4, 500)

    def test_rerank_weights_all_intents_present(self):
        """Weight dict covers all four intents."""
        for intent in ['WHY', 'WHEN', 'ENTITY', 'GENERAL']:
            assert intent in RERANK_WEIGHTS

    def test_rerank_weights_are_pinned_to_the_measured_table(self):
        """The raw weight table is exactly these values.

        Mutation: any silent retune of any weight -- WHY sim 0.45 ->
            0.05, WHEN graph 0.30 -> 0.01, an intent row dropped -- all
            of which the sum and direction checks below still accept,
            because normalization hides a row's scale.
        Oracle: the literal table. `experiments/recall_ablation` is the
            arbiter of these constants ("measured or not shipped"), so
            a deliberate change lands here and in that record together.
        """
        assert _RERANK_WEIGHTS_RAW == {
            'WHY':     (0.15, 0.45, 0.30),
            'WHEN':    (0.20, 0.40, 0.30),
            'ENTITY':  (0.20, 0.35, 0.10),
            'GENERAL': (0.25, 0.45, 0.15),
            }

    def test_rerank_weights_are_a_convex_combination(self):
        """Every intent's weights are positive and sum to 1.0.

        Mutation: normalizing by anything but the row's own sum --
            `max(row)`, or one shared constant across all four rows --
            which still leaves every weight positive and every row's
            direction intact, so the direction check below passes it;
            or a signal retired by zeroing its weight rather than
            deleting it, leaving an inert term in the blend.
        Oracle: 1.0 within a float ulp -- WHEN lands one low, since
            0.20 + 0.40 + 0.30 is not exact in binary. Catches
            `max(row)` and any single shared constant, since no one
            constant normalizes row sums of 0.90/0.90/0.65/0.85 at
            once. It does NOT catch a per-element divisor, which the
            direction test below does.
        """
        for intent, w in RERANK_WEIGHTS.items():
            assert all(x > 0.0 for x in w), f'{intent} has a dead signal'
            assert sum(w) == pytest.approx(1.0, abs=1e-9), \
                f'{intent} sum={sum(w)}'

    def test_rerank_weights_preserve_raw_row_direction(self):
        """Normalization rescales each row without turning it.

        Mutation: a transposition inside the comprehension -- `sim`
            and `gr` swapped on the way out. Every row still sums to
            1.0 and every weight stays positive, so the convex-
            combination check above passes it through untouched.
        Oracle: the raw row's own cross-ratios, compared as products
            to avoid a division. Ratios survive positive scaling only
            to a few ulp -- WHY's 3.0 comes back 3.0000000000000004 --
            hence `rel=1e-12` rather than equality. A swap breaks it
            in every row, since no row has `w_sim == w_gr`.
        """
        for intent, raw in _RERANK_WEIGHTS_RAW.items():
            scaled = RERANK_WEIGHTS[intent]
            for i in range(3):
                for j in range(3):
                    assert scaled[i] * raw[j] == pytest.approx(
                        scaled[j] * raw[i], rel=1e-12), \
                        f'{intent} turned at ({i}, {j})'

    def test_rerank_why_emphasizes_similarity(self):
        """WHY intent weights similarity score highest."""
        w_kw, w_sim, w_gr = RERANK_WEIGHTS['WHY']
        assert w_sim > w_gr
        assert w_sim > w_kw

    def test_rerank_general_similarity_highest(self):
        """GENERAL intent weights similarity highest."""
        w_kw, w_sim, w_gr = RERANK_WEIGHTS['GENERAL']
        assert w_sim > max(w_kw, w_gr)

    def test_hint_field_by_intent(self, backend):
        """Each intent produces its expected hint string."""
        expected_hints = {
            'WHY': 'Trace the causal chain: earlier results cause later ones',
            'WHEN': 'Results are newest-first: reconstruct the timeline',
            'ENTITY': 'Describe the entity using evidence across these memories',
            'GENERAL': 'Synthesize key points across these related memories',
            }
        backend.nodes.insert(make_insight(
            id='meta-any',
            content='test content for recall meta fields'))
        for intent, expected in expected_hints.items():
            result = intent_aware_recall(
                backend, query='test content recall',
                query_vec=None,
                limit=5, intent_override=intent,
                fingerprint=stored_fingerprint(backend))
            assert result['meta']['hint'] == expected

    def test_ordering_field_by_intent(self, backend):
        """Ordering field matches intent-specific sort strategy."""
        backend.nodes.insert(make_insight(
            id='meta-ord',
            content='test content for recall ordering'))
        expected = {
            'WHY': 'causal_topological',
            'WHEN': 'chronological',
            'ENTITY': 'score',
            'GENERAL': 'score',
            }
        for intent, ordering in expected.items():
            result = intent_aware_recall(
                backend, query='test content recall',
                query_vec=None,
                limit=5, intent_override=intent,
                fingerprint=stored_fingerprint(backend))
            assert result['meta']['ordering'] == ordering

    def test_sparse_flag_present(self, backend):
        """Sparse flag set when results are below half the requested limit."""
        result = intent_aware_recall(
            backend, query='nonexistent query xyz',
            query_vec=None,
            limit=10, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert result['meta']['sparse'] is True

    def test_sparse_flag_absent(self, backend):
        """Sparse flag absent when result count meets threshold."""
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'sparse-{i}',
                content=f'common keyword topic alpha {i}'))
        result = intent_aware_recall(
            backend, query='common keyword topic alpha',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert 'sparse' not in result['meta']

    def test_sparse_fires_on_full_irrelevant_result_set(self, backend):
        """Sparse fires when a FULL result set carries no relevance.

        Mutation: dropping the relevance clause from `sparse`, leaving
            only the `len(results) < limit // 2` count test.
        Oracle: the returned row count, asserted at or above the count
            arm's own threshold, so only the relevance arm can fire.
        """
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'irrelevant-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert len(result['results']) == 5
        assert all(r['signals']['keyword'] == 0.0
                   for r in result['results'])
        assert result['meta']['sparse'] is True

    def test_sparse_reads_keyword_only_not_similarity(self, backend):
        """A full set at similarity 1.0 still fires when keyword is 0.

        Mutation: conjoining `sim_score == 0.0` onto the relevance
            clause, which is the rule the defect ledger specified and
            which measurement showed fires on nothing once a store has
            embeddings (0 of 20 nonsense queries).
        Oracle: every row embedded parallel to the query vector, so
            similarity is exactly 1.0 while no query token appears in
            any row; the sim-conjoined rule cannot fire here and the
            shipped rule must.
        """
        vec = _vec_512(1.0, 0.0)
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'simhigh-{i}',
                content=f'saffron marmalade zeppelin {i}'))
            backend.nodes.update_embedding(f'simhigh-{i}', vec, 'fake')
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=vec,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert len(result['results']) >= 5 // 2
        assert all(r['signals']['keyword'] == 0.0
                   for r in result['results'])
        assert min(r['signals']['similarity']
                   for r in result['results']) > 0.99
        assert result['meta']['sparse'] is True

    def test_sparse_absent_when_one_row_is_relevant(self, backend):
        """One relevant row in a full set keeps `sparse` off.

        Mutation: inverting the pool test to `all(kw > 0.0)`, which
            demands every row match and so fires on this mostly-
            irrelevant pool.
        Oracle: a pool built with exactly one token-matching row, with
            both the matching and non-matching signals asserted.
        """
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'mixed-irrelevant-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        backend.nodes.insert(make_insight(
            id='mixed-relevant',
            content='quantum tungsten harpsichord resonance'))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert len(result['results']) >= 5 // 2
        signals = [r['signals']['keyword'] for r in result['results']]
        assert max(signals) > 0.0
        assert min(signals) == 0.0
        assert 'sparse' not in result['meta']

    def test_sparse_absent_when_the_filter_hid_the_matching_row(
            self, backend):
        """A category filter that hides the match is not "no match".

        Mutation: reading the keyword evidence off the returned rows
            instead of the pre-filter candidate pool, which calls a
            working graph-mediated filtered recall irrelevant and tells
            the agent to discard it.
        Oracle: a pool whose ONLY token-matching row is the one the
            `--cat` filter removes, with every returned row asserted at
            keyword 0.0 so the survivors alone would fire the arm.
        """
        backend.nodes.insert(make_insight(
            id='hidden-match', category='fact',
            content='quantum tungsten harpsichord resonance'))
        for i in range(4):
            backend.nodes.insert(make_insight(
                id=f'child-{i}', category='decision',
                content=f'saffron marmalade zeppelin {i}'))
            backend.edges.upsert(make_edge(
                source_id=f'child-{i}', target_id='hidden-match',
                edge_type='causal', weight=1.0))
            backend.edges.upsert(make_edge(
                source_id='hidden-match', target_id=f'child-{i}',
                edge_type='causal', weight=1.0))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=4, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend),
            category='decision')
        ids = {r['insight'].id for r in result['results']}
        assert 'hidden-match' not in ids
        assert len(result['results']) >= 4 // 2
        assert all(r['signals']['keyword'] == 0.0
                   for r in result['results'])
        assert 'sparse' not in result['meta']

    def test_min_score_thresholds_relevance_not_blended_score(
            self, backend):
        """The floor reads `kw + sim`, never the blended score.

        Mutation: thresholding on the blended `score` instead of the
            keyword and similarity sum.
        Oracle: min-max normalization hands the top graph row
            `graph == 1.0`, so its blended score is `w_gr` (0.1765 for
            GENERAL) -- above the 0.1 floor -- while its relevance sum
            is 0.0. A blended-score floor keeps that row; the shipped
            floor drops it.
        """
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'floor-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        baseline = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert max(r['signals']['graph']
                   for r in baseline['results']) == 1.0
        floored = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend),
            min_score=0.1)
        assert floored['results'] == []

    def test_min_score_sums_similarity_into_the_floor(self, backend):
        """Similarity carries a row over a floor keyword cannot reach.

        Mutation: dropping `sim_score` from the floor, leaving it a
            keyword-only threshold while four doc surfaces and the
            docstring all state it sums both signals.
        Oracle: rows embedded parallel to the query vector, so keyword
            is 0.0 and similarity is 1.0; a floor of 0.5 is
            unreachable by keyword alone and cleared by the sum.
        """
        vec = _vec_512(1.0, 0.0)
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'simfloor-{i}',
                content=f'saffron marmalade zeppelin {i}'))
            backend.nodes.update_embedding(f'simfloor-{i}', vec, 'fake')
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=vec,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend),
            min_score=0.5)
        assert len(result['results']) == 5
        assert all(r['signals']['keyword'] == 0.0
                   for r in result['results'])
        assert min(r['signals']['similarity']
                   for r in result['results']) > 0.99

    def test_sparse_count_arm_fires_on_a_short_matching_page(
            self, backend):
        """Too few rows is sparse even when the query matched.

        Mutation: deleting the `len(results) < limit // 2` arm, which
            no other test covers -- the empty-store cases fire all
            three arms at once and so cannot isolate it.
        Oracle: three rows returned against a limit of 10, one of them
            a keyword match, so the pool-match arm is provably quiet
            and only the count arm can set the flag.
        """
        backend.nodes.insert(make_insight(
            id='short-match',
            content='quantum tungsten harpsichord resonance'))
        for i in range(2):
            backend.nodes.insert(make_insight(
                id=f'short-filler-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=10, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert len(result['results']) == 3
        assert max(r['signals']['keyword']
                   for r in result['results']) > 0.0
        assert result['meta']['sparse'] is True

    def test_min_score_default_keeps_zero_relevance_rows(self, backend):
        """The default floor is off: zero-relevance rows still return.

        Mutation: defaulting `min_score` above 0.0, or dropping the
            `min_score > 0.0` gate so the filter runs on every call.
        Oracle: the same pool the 0.1 floor empties comes back
            populated when no floor is passed.
        """
        for i in range(6):
            backend.nodes.insert(make_insight(
                id=f'nofloor-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend))
        assert len(result['results']) > 0
        assert all(r['signals']['keyword'] == 0.0
                   and r['signals']['similarity'] == 0.0
                   for r in result['results'])

    def test_min_score_keeps_rows_at_or_above_the_floor(self, backend):
        """A relevant row survives a floor its relevance sum clears.

        Mutation: an off-by-one comparison (`>` for `>=`) at the exact
            floor, or inverting the filter to drop the rows it keeps.
        Oracle: a single row whose keyword score is exactly 1.0 (every
            query token present) against five rows scoring 0.0, with
            the floor set to that row's own value.
        """
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'keep-irrelevant-{i}',
                content=f'saffron marmalade zeppelin {i}'))
        backend.nodes.insert(make_insight(
            id='keep-relevant',
            content='quantum tungsten harpsichord resonance'))
        result = intent_aware_recall(
            backend, query='quantum tungsten harpsichord',
            query_vec=None,
            limit=5, intent_override='GENERAL',
            fingerprint=stored_fingerprint(backend),
            min_score=1.0)
        assert [r['insight'].id for r in result['results']] == [
            'keep-relevant']
        assert result['results'][0]['signals']['keyword'] == 1.0

    def test_rerank_weights_override_shipped_matches_default(self, backend):
        """Passing `RERANK_WEIGHTS` as override produces identical scores.

        Establishes that the override path with the default values is a
        no-op against the production path.
        """
        for i in range(5):
            backend.nodes.insert(make_insight(
                id=f'ovr-shipped-{i}',
                content=f'common keyword topic alpha {i}'))
        baseline = intent_aware_recall(
            backend, query='common keyword topic alpha',
            query_vec=None,
            limit=5, intent_override='WHEN',
            fingerprint=stored_fingerprint(backend))
        with_override = intent_aware_recall(
            backend, query='common keyword topic alpha',
            query_vec=None,
            limit=5, intent_override='WHEN',
            fingerprint=stored_fingerprint(backend),
            rerank_weights_override=dict(RERANK_WEIGHTS))
        baseline_order = [r['insight'].id for r in baseline['results']]
        override_order = [r['insight'].id for r in with_override['results']]
        assert baseline_order == override_order
        for a, b in zip(baseline['results'], with_override['results']):
            assert abs(a['score'] - b['score']) < 1e-9

    def test_rerank_weights_override_is_consulted(self, backend):
        """An override's keyword weight scales the final score exactly.

        Mutation: reading `RERANK_WEIGHTS` instead of the override, so
            the argument is accepted and ignored -- which no ordering
            assertion catches when the override happens to rank the
            same way. Also `rerank_table.get('GENERAL', ...)` in place
            of `.get(intent, ...)`, a hardcoded row that a
            same-tuple-per-intent override cannot see.
        Oracle: hand-computed. The row matches every query token, so
            `kw_score` is 1.0 and the other two signals are 0.0 with no
            query vector and a single-row pool; `final` is therefore the
            looked-up keyword weight itself -- 1.0, then 0.4, then 0.7
            read from a DIFFERENT intent's row than GENERAL.
        """
        backend.nodes.insert(make_insight(
            id='ovr-a', content='alpha beta gamma topic',
            importance=1))

        def _score(w_kw, intent='GENERAL', others=0.0):
            override = dict.fromkeys(RERANK_WEIGHTS, (others, 0.0, 0.0))
            override[intent] = (w_kw, 0.0, 0.0)
            resp = intent_aware_recall(
                backend, query='alpha beta gamma topic',
                query_vec=None,
                limit=5, intent_override=intent,
                fingerprint=stored_fingerprint(backend),
                rerank_weights_override=override)
            return {r['insight'].id: r['score']
                    for r in resp['results']}['ovr-a']

        assert _score(1.0) == pytest.approx(1.0)
        assert _score(0.4) == pytest.approx(0.4)
        # A row other than GENERAL, holding a value no other row holds,
        # so a hardcoded lookup key reads the wrong weight and fails.
        assert _score(0.7, intent='WHEN', others=0.1) == pytest.approx(0.7)
