"""Token-based keyword search.

The recall pipeline counts matches in the store instead of here.
"""

import heapq
import re

from memman.store.model import Insight

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'that',
    'this', 'it', 'its', 'or', 'and', 'but', 'if', 'not', 'no', 'so',
    'up', 'out', 'than', 'then', 'too', 'very', 'just', 'also', 'more',
    'some', 'any', 'all', 'each', 'i', 'me', 'my', 'we', 'you', 'your',
    'he', 'she', 'they', 'them', 'his', 'her', 'our', 'their', 'what',
    'which', 'who', 'how', 'when', 'where',
    }

_WORD_RE = re.compile(r'[a-zA-Z0-9]+')


def tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens with stopword filtering."""
    tokens: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        if word not in STOPWORDS:
            tokens.add(word)
    return tokens


def insight_tokens(ins: Insight) -> set[str]:
    """Return the token set of an insight's content."""
    return tokenize(ins.content)


def keyword_search(
        insights: list[Insight], query: str,
        limit: int,
        counts: dict[str, int],
        ) -> list[tuple[Insight, float]]:
    """Score insights by token overlap with query.

    Parameters
    ----------
    insights : list[Insight]
        Rows to rank. Order matters: an exact tie on `score` at the
        limit is resolved in favor of whichever row reached the heap
        first.
    query : str
        Search text, tokenized here to fix the score denominator.
    limit : int
        Keep at most this many hits; `limit <= 0` keeps all.
    counts : dict[str, int]
        Distinct query tokens present per insight id, from an index
        probe. An id absent from the dict scores 0.

    Returns
    -------
    list[tuple[Insight, float]]
        `(insight, score)` descending, `score` in [0, 1].

    Notes
    -----
    - score = (distinct query tokens present) / (query tokens), which
      also fills `signals.keyword`.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    heap_list: list[tuple[float, str, Insight]] = []
    for ins in insights:
        intersection = counts.get(ins.id, 0)
        if intersection == 0:
            continue
        score = intersection / len(query_tokens)

        entry = (score, ins.id, ins)
        if limit <= 0 or len(heap_list) < limit:
            heapq.heappush(heap_list, entry)
        elif score > heap_list[0][0]:
            heapq.heapreplace(heap_list, entry)

    result = []
    while heap_list:
        score, _id, ins = heapq.heappop(heap_list)
        result.append((ins, score))
    result.reverse()
    return result
