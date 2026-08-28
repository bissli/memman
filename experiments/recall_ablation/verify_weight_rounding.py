"""Differential check that a `RERANK_WEIGHTS` rescale preserves order.

Runs every labeled query through `intent_aware_recall` once per weight
table and diffs the returned id sequences against a baseline table.
Rerank is off, so the weighted sum IS the returned order and any
reorder a rescale can cause on the blend has to show here.

Arms
----
- `control`: the baseline table repeated verbatim. Any movement here
  is nondeterminism and invalidates every other arm.
- `<N>dp`: the baseline normalized to sum 1.0 and rounded to N places.
- `computed`: the baseline divided by each row's own sum, which is
  what `RERANK_WEIGHTS` ships.

Usage
-----
    python verify_weight_rounding.py \\
        --data-dir /tmp/ablation_sandbox --store search \\
        --labels ../eval/data/search_store/queries_labeled.jsonl \\
        --limit 30

Notes
-----
- Rounding is not a pure rescale: no quotient here has an exact float
  literal, so a rounded row is turned as well as scaled. The point of
  this script is to price that turn in returned positions rather than
  argue it from the arithmetic.
- Movement concentrates in the deep tail, where adjacent blended
  scores are separated by less than the rounding error, so raise
  `--limit` to make the instrument stricter.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from memman.embed.fingerprint import bound_embedder, stored_fingerprint
from memman.search.recall import _RERANK_WEIGHTS_RAW, RERANK_WEIGHTS
from memman.search.recall import intent_aware_recall
from memman.store.db import open_read_only, store_dir
from memman.store.sqlite import SqliteBackend


def rounded_to(row: tuple[float, float, float],
               places: int) -> tuple[float, float, float]:
    """Row scaled to sum 1.0 and rounded, nudged back onto 1.0.

    Parameters
    ----------
    row : tuple[float, float, float]
        Raw `(w_kw, w_sim, w_gr)`, any positive row sum.
    places : int
        Decimal places to round each normalized weight to.

    Returns
    -------
    tuple[float, float, float]
        Rounded row summing to 1.0 at `places`. The residual is
        absorbed by the weight that rounding moved furthest, which is
        how a person writing the table by hand would balance it.
    """
    total = sum(row)
    exact = [w / total for w in row]
    out = [round(w, places) for w in exact]
    residual = round(1.0 - sum(out), places)
    if residual:
        worst = max(range(len(out)), key=lambda i: abs(exact[i] - out[i]))
        out[worst] = round(out[worst] + residual, places)
    return (out[0], out[1], out[2])


def main() -> None:
    """Diff each arm's returned ids against the raw-table baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--places', default='4,5,6,8')
    args = ap.parse_args()

    backend = SqliteBackend(open_read_only(store_dir(args.data_dir, args.store)))
    fingerprint = stored_fingerprint(backend)
    embedder = bound_embedder(backend)

    queries = []
    with Path(args.labels).open() as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line)['query'])

    arms: list[tuple[str, dict]] = [('control', dict(_RERANK_WEIGHTS_RAW))]
    for places in (int(p) for p in args.places.split(',')):
        arms.append((f'{places}dp', {
            intent: rounded_to(row, places)
            for intent, row in _RERANK_WEIGHTS_RAW.items()
            }))
    arms.append(('computed', dict(RERANK_WEIGHTS)))

    moved: Counter = Counter()
    slots = 0
    for query in queries:
        qvec = embedder.embed(query)
        baseline = [r['insight'].id for r in intent_aware_recall(
            backend, query, qvec, args.limit, fingerprint=fingerprint,
            rerank_weights_override=dict(_RERANK_WEIGHTS_RAW))['results']]
        slots += len(baseline)
        for name, table in arms:
            ids = [r['insight'].id for r in intent_aware_recall(
                backend, query, qvec, args.limit, fingerprint=fingerprint,
                rerank_weights_override=table)['results']]
            moved[name] += (sum(1 for a, b in zip(baseline, ids) if a != b)
                            + abs(len(baseline) - len(ids)))

    print(f'{args.store}: {len(queries)} queries, {slots} returned slots '
          f'at limit {args.limit}, rerank off')
    for name, _table in arms:
        print(f'  {name:9} positions moved vs the raw table: {moved[name]}')


if __name__ == '__main__':
    main()
