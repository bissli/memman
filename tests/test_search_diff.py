"""Tests for mnemon.search.diff -- duplicate/conflict detection."""

from mnemon.model import Insight
from mnemon.search.antonyms import ANTONYM_PAIRS
from mnemon.search.diff import classify_suggestion, diff


def test_classify_add():
    """Low similarity classifies as ADD."""
    assert classify_suggestion(0.3, 'completely new', 'existing') == 'ADD'


def test_classify_duplicate():
    """High similarity classifies as DUPLICATE."""
    assert classify_suggestion(0.95, 'very similar', 'very similar indeed') == 'DUPLICATE'


def test_classify_update():
    """Medium similarity classifies as UPDATE."""
    assert classify_suggestion(0.7, 'Go uses SQLite', 'Go uses PostgreSQL') == 'UPDATE'


def test_classify_conflict_negation():
    """Negation words with medium similarity classify as CONFLICT."""
    cases = [
        ('do not use Redis', 'use Redis for caching'),
        ('no longer supports Python 2', 'supports Python 2'),
        ('replaced Flask with FastAPI', 'uses Flask for API'),
    ]
    for new_text, existing in cases:
        assert classify_suggestion(0.7, new_text, existing) == 'CONFLICT'


def test_classify_boundary():
    """Boundary values: 0.55 not ADD, 0.9 not DUPLICATE."""
    got = classify_suggestion(0.55, 'some content', 'other content')
    assert got != 'ADD'
    got = classify_suggestion(0.9, 'some content', 'other content')
    assert got != 'DUPLICATE'


def test_classify_below_new_threshold():
    """Similarity below 0.55 classifies as ADD."""
    assert classify_suggestion(0.5, 'some text', 'other text') == 'ADD'
    assert classify_suggestion(0.54, 'some text', 'other text') == 'ADD'
    assert classify_suggestion(0.6, 'some text', 'other text') == 'UPDATE'


def test_diff_token_only():
    """Diff finds matches via token similarity."""
    insights = [
        Insight(id='1', content='Go uses SQLite for persistent memory storage'),
        Insight(id='2', content='Python machine learning with TensorFlow'),
        Insight(id='3', content='Go uses SQLite for memory persistence'),
    ]
    result = diff(insights, 'Go uses SQLite for persistent memory storage')
    assert result['suggestion'] != 'ADD'
    assert len(result['matches']) > 0
    assert result['matches'][0]['id'] == '1'


def test_diff_no_matches():
    """No matching content returns ADD."""
    insights = [
        Insight(id='1', content='something about cooking recipes'),
    ]
    result = diff(insights, 'Go database library benchmarks')
    assert result['suggestion'] == 'ADD'


def test_diff_duplicate_overrides():
    """DUPLICATE in any match overrides overall suggestion."""
    insights = [
        Insight(id='1', content='Go uses SQLite for storage', importance=5),
        Insight(
            id='2', importance=3,
            content='Go uses SQLite for storage exactly'
                    ' the same content repeated verbatim'),
    ]
    result = diff(insights, 'Go uses SQLite for storage')
    assert result['suggestion'] == 'DUPLICATE'


def test_diff_limit_default():
    """Default limit caps matches at 5."""
    words = ['shared', 'words', 'database', 'memory', 'alpha', 'beta',
             'gamma', 'delta', 'epsilon', 'zeta']
    insights = [
        Insight(id=str(i),
                content=' '.join(words[:4 + (i % len(words))]),
                importance=i + 1)
        for i in range(20)
    ]
    result = diff(insights, 'shared words database memory')
    assert len(result['matches']) <= 5


def test_classify_conflict_antonym_tech():
    """Tech antonym pairs trigger CONFLICT classification."""
    cases = [
        ('use synchronous calls', 'use asynchronous calls'),
        ('the API is stateful', 'the API is stateless'),
        ('data is mutable', 'data is immutable'),
        ('use blocking IO', 'use non-blocking IO'),
        ('centralized architecture', 'decentralized architecture'),
        ('schema is normalized', 'schema is denormalized'),
    ]
    for new_text, existing in cases:
        result = classify_suggestion(0.7, new_text, existing)
        assert result == 'CONFLICT', (
            f'Expected CONFLICT for {new_text!r} vs {existing!r}, got {result}')


def test_classify_conflict_antonym_general():
    """General antonym pairs trigger CONFLICT classification."""
    cases = [
        ('the change is reversible', 'the change is irreversible'),
        ('this field is required', 'this field is optional'),
        ('uses explicit configuration', 'uses implicit configuration'),
        ('the system is stable', 'the system is unstable'),
    ]
    for new_text, existing in cases:
        result = classify_suggestion(0.7, new_text, existing)
        assert result == 'CONFLICT', (
            f'Expected CONFLICT for {new_text!r} vs {existing!r}, got {result}')


def test_antonym_pairs_minimum_term_length():
    """All antonym terms are at least 4 characters to avoid false matches."""
    for term_a, term_b in ANTONYM_PAIRS:
        assert len(term_a) >= 4, f'Term too short: {term_a!r}'
        assert len(term_b) >= 4, f'Term too short: {term_b!r}'


def test_antonym_pairs_no_duplicates():
    """No duplicate pairs in ANTONYM_PAIRS."""
    normalized = set()
    for a, b in ANTONYM_PAIRS:
        pair = tuple(sorted([a.lower(), b.lower()]))
        assert pair not in normalized, f'Duplicate pair: {pair}'
        normalized.add(pair)


def test_diff_best_match_idx_present():
    """Diff result always includes best_match_idx."""
    insights = [Insight(id='1', content='Go uses SQLite')]
    result = diff(insights, 'Go uses SQLite for storage')
    assert 'best_match_idx' in result


def test_diff_cosine_conflict_overrides_keyword_add():
    """Cosine-only CONFLICT overrides keyword ADD overall."""
    kw_insight = Insight(
        id='kw', content='alpha bravo charlie delta echo foxtrot')
    cos_insight = Insight(
        id='cos', content='not using Redis cache anymore')
    insights = [kw_insight, cos_insight]

    new_vec = [1.0] + [0.0] * 767
    kw_vec = [0.0] * 384 + [1.0] * 384
    cos_vec = [0.95] + [0.0] * 767
    existing_embed = [
        ('kw', kw_vec),
        ('cos', cos_vec),
        ]
    result = diff(
        insights, 'using Redis cache for sessions',
        new_embedding=new_vec, existing_embed=existing_embed)
    assert result['suggestion'] == 'CONFLICT'
    best_idx = result['best_match_idx']
    assert result['matches'][best_idx]['id'] == 'cos'


def test_diff_same_priority_prefers_higher_similarity():
    """Within same priority tier, higher similarity wins."""
    ins_a = Insight(id='a', content='chose SQLite because embedded database')
    ins_b = Insight(
        id='b', content='chose SQLite because embedded and simple')
    insights = [ins_a, ins_b]
    result = diff(
        insights, 'chose SQLite because embedded database works')
    if result['suggestion'] != 'ADD':
        best_idx = result['best_match_idx']
        best = result['matches'][best_idx]
        for m in result['matches']:
            if m['suggestion'] == best['suggestion']:
                assert m['similarity'] <= best['similarity']
