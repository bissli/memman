"""Tests for memman.search -- keyword and recall.
"""

import pytest
from memman.search.keyword import insight_tokens, keyword_search, tokenize
from memman.search.recall import _RERANK_WEIGHTS_RAW, RERANK_WEIGHTS
from memman.store.model import Insight


def _counts_for(insights: list[Insight], query: str) -> dict[str, int]:
    """Distinct query-token overlap per insight id, computed in Python.

    Stands in for the index probe `keyword_search` requires, since
    these tests exercise ranking over an in-memory pool with no
    index behind it.
    """
    query_tokens = tokenize(query)
    return {
        ins.id: sum(1 for t in query_tokens if t in insight_tokens(ins))
        for ins in insights
        }


class TestKeywordSearch:
    """Tokenization and keyword search ranking.
    """

    def test_tokenize_english(self):
        """English words are lowercased and split.

        Mutation: dropping the `.lower()` before matching, or leaving
        a stopword such as `for` in the token set.
        Oracle: hand-listed tokens of a mixed-case sentence.
        """
        tokens = tokenize('Go uses SQLite for persistent storage')
        assert 'go' in tokens
        assert 'sqlite' in tokens
        assert 'persistent' in tokens
        assert 'storage' in tokens
        assert 'for' not in tokens

    def test_tokenize_stopwords(self):
        """Common stopwords are filtered out.

        Mutation: skipping the `STOPWORDS` test in `tokenize`, or
        filtering on a truncated stopword list.
        Oracle: hand-picked stopwords absent and content words present.
        """
        tokens = tokenize('the quick fox is very fast')
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'very' not in tokens
        assert 'quick' in tokens
        assert 'fox' in tokens
        assert 'fast' in tokens

    def test_tokenize_empty(self):
        """Empty string produces empty set.

        Mutation: `tokenize` returning a set holding the empty string.
        Oracle: length 0.
        """
        assert len(tokenize('')) == 0

    def test_tokenize_all_stopwords(self):
        """All-stopword input produces empty set.

        Mutation: a stopword filter that keeps a sentence's first or
        only word.
        Oracle: length 0 for four hand-picked stopwords.
        """
        assert len(tokenize('the is a an')) == 0

    def test_keyword_search_ranking(self):
        """Best match ranks first.

        Mutation: `keyword_search` returning hits in input order, or
        ascending by score.
        Oracle: the hand-checked best row (id 1, all three query
        tokens) first, with scores non-increasing after it.
        """
        insights = [
            Insight(id='1', content='Go language for building CLI tools'),
            Insight(id='2', content='SQLite database for Go applications'),
            Insight(id='3', content='Python machine learning framework'),
            ]
        results = keyword_search(
            insights, 'Go CLI tools', 10, _counts_for(insights, 'Go CLI tools'))
        assert len(results) >= 2
        assert results[0][0].id == '1'
        for i in range(1, len(results)):
            assert results[i][1] <= results[i - 1][1]

    def test_keyword_search_limit(self):
        """Limit caps the result count.

        Mutation: the heap pushing past `limit`, so every matching row
        comes back.
        Oracle: 20 rows that all match, and a limit of 5.
        """
        words = ['common', 'shared', 'words', 'alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'theta']
        insights = [
            Insight(id=str(i),
                    content=' '.join(words[:3 + (i % len(words))]))
            for i in range(20)
            ]
        results = keyword_search(
            insights, 'common shared words', 5,
            _counts_for(insights, 'common shared words'))
        assert len(results) <= 5

    def test_keyword_search_empty_query(self):
        """Empty query returns empty results.

        Mutation: dropping the `if not query_tokens` guard, which
        divides by zero.
        Oracle: an empty list for an empty query over a matching row.
        """
        insights = [Insight(id='1', content='some content')]
        results = keyword_search(insights, '', 10, {})
        assert len(results) == 0


class TestRecallRanking:
    """Rerank signal weights.
    """

    def test_rerank_weights_are_pinned_to_the_measured_table(self):
        """The raw weight table is exactly the measured GENERAL row.

        Mutation: any silent retune of a weight -- 0.45 -> 0.05 on the
            similarity term, or a term dropped -- which the sum and
            direction checks below still accept, because
            normalization hides a row's scale.
        Oracle: the literal table, so a deliberate retune edits
            this line in the same change.
        """
        assert _RERANK_WEIGHTS_RAW == (0.25, 0.45, 0.15)

    def test_rerank_weights_are_a_convex_combination(self):
        """The weights are positive and sum to 1.0.

        Mutation: normalizing by anything but the row's own sum --
            `max(row)`, or a signal retired by zeroing its weight
            rather than deleting it, leaving an inert term in the
            blend.
        Oracle: 1.0 within a float ulp -- the raw row's own sum is not
            exact in binary.
        """
        assert all(x > 0.0 for x in RERANK_WEIGHTS), 'a signal is dead'
        assert sum(RERANK_WEIGHTS) == pytest.approx(1.0, abs=1e-9)

    def test_rerank_weights_preserve_raw_row_direction(self):
        """Normalization rescales the row without turning it.

        Mutation: a transposition inside the comprehension -- `sim`
            and `gr` swapped on the way out. The row still sums to 1.0
            and every weight stays positive, so the convex-
            combination check above passes it through untouched.
        Oracle: the raw row's own cross-ratios, compared as products
            to avoid a division. Ratios survive positive scaling only
            to a few ulp, hence `rel=1e-12` rather than equality. A
            swap breaks it, since no two raw weights are equal.
        """
        for i in range(3):
            for j in range(3):
                assert RERANK_WEIGHTS[i] * _RERANK_WEIGHTS_RAW[j] == \
                    pytest.approx(
                        RERANK_WEIGHTS[j] * _RERANK_WEIGHTS_RAW[i],
                        rel=1e-12), f'turned at ({i}, {j})'

    def test_rerank_general_similarity_highest(self):
        """The surviving GENERAL row weights similarity highest.

        Mutation: retuning the raw row so keyword or anchor outweighs
        similarity.
        Oracle: `w_sim` compared against the larger of the other two.
        """
        w_kw, w_sim, w_anchor = RERANK_WEIGHTS
        assert w_sim > max(w_kw, w_anchor)
