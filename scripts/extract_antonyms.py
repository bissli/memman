#!/usr/bin/env python3
"""One-time WordNet antonym extraction for contradiction detection.

Requires: pip install nltk
Then: python -c "import nltk; nltk.download('wordnet')"

Usage: python scripts/extract_antonyms.py > antonyms.json
"""

import json
import sys

from nltk.corpus import wordnet as wn


def extract_antonym_pairs(min_length: int = 4) -> list[tuple[str, str]]:
    """Extract adjective and adverb antonym pairs from WordNet.
    """
    pairs: set[tuple[str, str]] = set()

    for synset in wn.all_synsets():
        if synset.pos() not in ('a', 's', 'r'):
            continue
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                a = lemma.name().lower().replace('_', ' ')
                b = ant.name().lower().replace('_', ' ')
                if len(a) < min_length or len(b) < min_length:
                    continue
                pair = tuple(sorted([a, b]))
                pairs.add(pair)

    return sorted(pairs)


def main() -> None:
    """Extract and print antonym pairs as JSON."""
    pairs = extract_antonym_pairs(min_length=4)
    json.dump(pairs, sys.stdout, indent=2)
    sys.stdout.write('\n')
    print(f'# {len(pairs)} pairs extracted', file=sys.stderr)


if __name__ == '__main__':
    main()
