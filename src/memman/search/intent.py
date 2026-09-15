"""Query intent detection and intent-specific edge type weights.

Notes
-----
- Every vector sums to 1.0. The reason is not cross-intent
  comparison, which never happens: one recall carries one intent and
  `graph_score` is min-max normalized over that call's own pool. The
  sum fixes the STRUCTURAL term's scale against the two terms it is
  summed with inside a call - the anchor's RRF score and
  `LAMBDA2 * semantic` - neither of which these weights touch.
- That balance is load-bearing, not cosmetic. Because a traversal
  score is `anchor_rrf + LAMBDA1 * structural + LAMBDA2 * semantic`,
  scaling only the middle term is not a transform min-max undoes: it
  re-ranks rows, and it changes which nodes the beam keeps at its
  cut. Changing a vector here changes retrieved order.
- The types weighted here are exactly the edge types the store
  writes. A vector naming a type nothing mints contributes nothing
  while still consuming the 1.0 budget, which silently shrinks every
  other type's share.
"""

import re

WHY_PATTERN = re.compile(
    r'(?i)\b(why|reason|because|cause|motivation|rationale)\b')
WHEN_PATTERN = re.compile(
    r'(?i)\b(when|time|date|before|after|during|timeline|history|sequence)\b')
ENTITY_PATTERN = re.compile(
    r'(?i)\b(what is|who is|tell me about|describe|about)\b')

INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    'WHY': {
        'temporal': 0.666, 'entity': 0.167, 'semantic': 0.167,
        },
    'WHEN': {
        'temporal': 0.764, 'entity': 0.118, 'semantic': 0.118,
        },
    'ENTITY': {
        'entity': 0.611, 'semantic': 0.333, 'temporal': 0.056,
        },
    'GENERAL': {
        'temporal': 0.334, 'semantic': 0.333, 'entity': 0.333,
        },
    }


def intent_from_string(s: str) -> str:
    """Parse a user-provided intent string into a valid intent value."""
    upper = s.strip().upper()
    if upper in {'WHY', 'WHEN', 'ENTITY', 'GENERAL'}:
        return upper
    raise ValueError(
        f'unknown intent {s!r}; valid: WHY, WHEN, ENTITY, GENERAL')


def detect_intent(query: str) -> str:
    """Analyze a query string and return the detected intent."""
    q = query.lower()
    why_score = len(WHY_PATTERN.findall(q))
    when_score = len(WHEN_PATTERN.findall(q))
    entity_score = len(ENTITY_PATTERN.findall(q))

    if why_score > when_score and why_score > entity_score and why_score > 0:
        return 'WHY'
    if (when_score > why_score and when_score > entity_score
            and when_score > 0):
        return 'WHEN'
    if (entity_score > why_score and entity_score > when_score
            and entity_score > 0):
        return 'ENTITY'
    return 'GENERAL'


def get_weights(intent: str) -> dict[str, float]:
    """Return edge type weights for the given intent."""
    return INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS['GENERAL'])
