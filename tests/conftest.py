"""Shared fixtures for memman tests.

Dual-mode API mocking: mocked by default, real APIs with --live flag.

    pytest                    # fast, mocked LLM + embeddings
    pytest --live             # real Haiku + Voyage APIs (slow, needs keys)

Mock mode patches `LLMClient.complete` and `voyage.Client.embed` at the
HTTP layer, so all extraction/reconciliation/expansion logic still runs
with realistic canned responses. This exercises the real code paths.
"""

import hashlib
import json
import re
import struct
from datetime import datetime, timezone

import pytest
from memman.model import Edge, Insight

EMBEDDING_DIM = 512


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --live flag for real API calls."""
    parser.addoption(
        '--live', action='store_true', default=False,
        help='Use real Haiku LLM and Voyage embedding APIs')


@pytest.fixture(autouse=True)
def _mock_apis(request, monkeypatch):
    """Mock LLM and embedding HTTP calls unless --live is set.

    Patches at the HTTP layer: LLMClient.complete returns realistic
    JSON that the real extract/reconcile/expand code parses. Voyage
    embed returns a deterministic content-hash vector.
    """
    if request.config.getoption('--live'):
        return

    monkeypatch.setattr(
        'memman.llm.client.LLMClient.complete', _mock_llm_complete)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _mock_embed)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.available', lambda self: True)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'mock-key-for-testing')


def _mock_llm_complete(self: object, system: str, user: str) -> str:
    """Route to appropriate mock based on system prompt content."""
    if 'Extract atomic facts' in system:
        return _mock_fact_extraction(user)
    if 'Compare new facts' in system:
        return _mock_reconciliation(user)
    if 'Expand a search query' in system:
        return _mock_query_expansion(user)
    if 'keyword' in system.lower() and 'enrichment' in system.lower():
        return _mock_enrichment(user)
    if 'causal' in system.lower():
        return _mock_causal(user)
    return json.dumps({'facts': [{'text': user, 'category': 'fact',
                                  'importance': 3, 'entities': []}]})


def _mock_fact_extraction(content: str) -> str:
    """Generate realistic fact extraction response.

    Extracts entities by finding capitalized words. Preserves
    content as-is in a single fact — mimics Haiku behavior for
    substantive content.
    """
    entities = _extract_mock_entities(content)
    category = _infer_category(content)
    importance = _infer_importance(content)
    return json.dumps({
        'facts': [{
            'text': content,
            'category': category,
            'importance': importance,
            'entities': entities,
            }],
        'skip_reason': None,
        })


def _mock_reconciliation(prompt: str) -> str:
    """Generate realistic reconciliation response.

    Parses the structured prompt to find existing memories and new
    facts. If a new fact is very similar to an existing memory,
    returns UPDATE; otherwise ADD.
    """
    existing = {}
    new_facts = []
    in_existing = False
    in_new = False
    for line in prompt.split('\n'):
        if line.startswith('EXISTING MEMORIES:'):
            in_existing = True
            in_new = False
            continue
        if line.startswith('NEW FACTS:'):
            in_new = True
            in_existing = False
            continue
        if in_existing:
            m = re.match(r'\[(\d+)\]\s+(.*)', line)
            if m:
                existing[m.group(1)] = m.group(2)
        if in_new and line.startswith('- '):
            new_facts.append(line[2:])

    actions = []
    for fact in new_facts:
        fact_lower = fact.lower()
        best_id = None
        best_overlap = 0
        for eid, econtent in existing.items():
            words_f = set(fact_lower.split())
            words_e = set(econtent.lower().split())
            overlap = len(words_f & words_e) / max(len(words_f | words_e), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = eid

        if best_overlap > 0.6:
            actions.append({
                'fact': fact,
                'action': 'UPDATE',
                'target_id': best_id,
                'merged_text': fact,
                'reason': 'similar content, updating',
                })
        elif best_overlap > 0.4:
            actions.append({
                'fact': fact,
                'action': 'NONE',
                'target_id': best_id,
                'merged_text': None,
                'reason': 'already captured',
                })
        else:
            actions.append({
                'fact': fact,
                'action': 'ADD',
                'target_id': None,
                'merged_text': None,
                'reason': 'new information',
                })

    return json.dumps({'actions': actions})


def _mock_query_expansion(query: str) -> str:
    """Generate realistic query expansion response."""
    words = query.split()
    entities = [w for w in words if w[0:1].isupper()]
    return json.dumps({
        'expanded_query': query,
        'keywords': words,
        'entities': entities,
        'intent': 'GENERAL',
        })


def _mock_enrichment(content: str) -> str:
    """Generate realistic enrichment response."""
    return json.dumps({
        'keywords': content.lower().split()[:5],
        'summary': content[:100],
        'entities': _extract_mock_entities(content),
        })


def _mock_causal(content: str) -> str:
    """Generate realistic causal analysis response."""
    return json.dumps({'causal_links': []})


def _extract_mock_entities(text: str) -> list[str]:
    """Extract entities from text using capitalized words heuristic."""
    stopwords = {'The', 'A', 'An', 'In', 'On', 'For', 'With', 'And',
                 'Or', 'Is', 'Are', 'Was', 'Were', 'To', 'From', 'By',
                 'At', 'Of', 'But', 'Not', 'It', 'This', 'That', 'If',
                 'As', 'So', 'We', 'I', 'My', 'No', 'Yes', 'Chose',
                 'Switched', 'Store', 'Production', 'Configured',
                 'Infrastructure', 'Deployed', 'Critical', 'Uses',
                 'Using', 'Based', 'After', 'Before'}
    entities = []
    for word in re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', text):
        if word not in stopwords and word not in entities:
            entities.append(word)
    return entities[:5]


def _infer_category(text: str) -> str:
    """Infer category from content keywords."""
    lower = text.lower()
    if any(w in lower for w in ['chose', 'decided', 'picked', 'switched',
                                'migrated', 'prefer']):
        return 'decision'
    if any(w in lower for w in ['like', 'prefer', 'always', 'never',
                                'love', 'hate']):
        return 'preference'
    return 'fact'


def _infer_importance(text: str) -> int:
    """Infer importance from content keywords."""
    lower = text.lower()
    if any(w in lower for w in ['critical', 'outage', 'production',
                                'architecture']):
        return 4
    return 3


def _mock_embed(self: object, text: str) -> list[float]:
    """Deterministic embedding from content hash.

    Produces a 512-dim unit vector seeded by content, so identical
    text gives identical vectors. Different text gives different
    vectors with low cosine similarity.
    """
    digest = hashlib.sha256(text.encode()).digest()
    floats = list(struct.unpack(
        f'<{len(digest) // 4}f', digest))
    while len(floats) < EMBEDDING_DIM:
        extra = hashlib.sha256(
            digest + len(floats).to_bytes(4, 'little')).digest()
        floats.extend(struct.unpack(f'<{len(extra) // 4}f', extra))
    floats = floats[:EMBEDDING_DIM]
    norm = sum(x * x for x in floats) ** 0.5
    if norm > 0:
        floats = [x / norm for x in floats]
    return floats


@pytest.fixture
def tmp_db(tmp_path):
    """Fresh SQLite database in temp directory."""
    from memman.store.db import open_db
    db = open_db(str(tmp_path))
    yield db
    db.close()


@pytest.fixture
def populated_db(tmp_db):
    """DB pre-loaded with 5 insights for query/graph tests."""
    from memman.store.node import insert_insight
    insights = [
        make_insight(id='pop-1', content='Go uses SQLite for storage',
                     importance=3, tags=['go', 'sqlite'],
                     entities=['Go', 'SQLite']),
        make_insight(id='pop-2', content='Python web framework comparison',
                     importance=2, category='decision'),
        make_insight(id='pop-3', content='Graph traversal algorithm for knowledge',
                     importance=4, tags=['graph'],
                     entities=['MAGMA']),
        make_insight(id='pop-4', content='Docker deployment strategy',
                     importance=5, category='preference',
                     entities=['Docker']),
        make_insight(id='pop-5', content='Go concurrency patterns',
                     importance=3, entities=['Go']),
        ]
    for ins in insights:
        insert_insight(tmp_db, ins)
    return tmp_db


def make_insight(**overrides) -> Insight:
    """Factory for test Insight instances."""
    now = datetime.now(timezone.utc)
    defaults = {
        'id': 'test-id',
        'content': 'test content',
        'category': 'fact',
        'importance': 3,
        'tags': [],
        'entities': [],
        'source': 'test',
        'access_count': 0,
        'created_at': now,
        'updated_at': now,
        'deleted_at': None,
        'last_accessed_at': None,
        'effective_importance': 0.0,
        }
    defaults.update(overrides)
    if 'tags' in overrides and overrides['tags'] is None:
        defaults['tags'] = []
    if 'entities' in overrides and overrides['entities'] is None:
        defaults['entities'] = []
    return Insight(**defaults)


def make_edge(**overrides) -> Edge:
    """Factory for test Edge instances."""
    now = datetime.now(timezone.utc)
    defaults = {
        'source_id': 'src',
        'target_id': 'tgt',
        'edge_type': 'semantic',
        'weight': 0.5,
        'metadata': {},
        'created_at': now,
        }
    defaults.update(overrides)
    return Edge(**defaults)
