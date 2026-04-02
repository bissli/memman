"""Tests for memman.search.keyword -- tokenization and keyword search."""

from memman.model import Insight
from memman.search.keyword import keyword_search, tokenize


def test_tokenize_english():
    """English words are lowercased and split."""
    tokens = tokenize('Go uses SQLite for persistent storage')
    assert 'go' in tokens
    assert 'sqlite' in tokens
    assert 'persistent' in tokens
    assert 'storage' in tokens
    assert 'for' not in tokens


def test_tokenize_stopwords():
    """Common stopwords are filtered out."""
    tokens = tokenize('the quick fox is very fast')
    assert 'the' not in tokens
    assert 'is' not in tokens
    assert 'very' not in tokens
    assert 'quick' in tokens
    assert 'fox' in tokens
    assert 'fast' in tokens


def test_tokenize_empty():
    """Empty string produces empty set."""
    assert len(tokenize('')) == 0


def test_tokenize_all_stopwords():
    """All-stopword input produces empty set."""
    assert len(tokenize('the is a an')) == 0


def test_keyword_search_ranking():
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


def test_keyword_search_limit():
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


def test_keyword_search_importance_tiebreak():
    """Higher importance wins on score tie."""
    insights = [
        Insight(id='low', content='Go memory graph', importance=1),
        Insight(id='high', content='Go memory graph', importance=5),
    ]
    results = keyword_search(insights, 'Go memory graph', 10)
    assert len(results) >= 2
    assert results[0][0].id == 'high'


def test_keyword_search_empty_query():
    """Empty query returns empty results."""
    insights = [Insight(id='1', content='some content')]
    results = keyword_search(insights, '', 10)
    assert len(results) == 0


def test_keyword_search_tags_entities():
    """Tags and entities contribute to matching."""
    insights = [
        Insight(id='1', content='something unrelated',
                tags=['database'], entities=['SQLite']),
    ]
    results = keyword_search(insights, 'SQLite database', 10)
    assert len(results) > 0
