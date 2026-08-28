r"""Price each candidate `meta.sparse` rule against labeled queries.

`sparse` is meant to tell a caller "nothing relevant is stored" apart
from "here are the five nearest unrelated memories". Recall always
returns rows -- a recency channel seeds the newest insights as
traversal anchors whether or not they match -- so the rule has to read
the relevance signals, not the row count.

This script scores each candidate rule on two query populations and
reports the two error rates that decide it.

Populations
-----------
- `labeled`: the graded queries for the store. Every one has relevant
  insights by construction, so a rule that fires here is a FALSE
  POSITIVE and would tell the caller to discard a good answer.
- `nonsense`: fixed word-salad queries that match nothing. A rule that
  stays quiet here is a FALSE NEGATIVE and leaves the reported defect
  in place.

Rules
-----
- `exact_zero`: every row at `kw == 0` AND `sim == 0`. The rule the
  defect ledger specified.
- `kw_only`: every row at `kw == 0`. Parameter-free, so it needs
  validating rather than calibrating.
- `sim_floor_<t>`: every row at `kw == 0` and `sim < t`. Carries a
  tunable, reported here only to price what a calibrated rule would
  buy over `kw_only`.
- `shipped_meta`: `meta.sparse` as returned, which is the rule that
  ships. It reads the keyword evidence off the pool BEFORE any
  `category`/`source` filter and ORs in the empty and count arms, so
  it is not identical to `kw_only` and is the ground truth here.

Usage
-----
    python verify_sparse_rule.py \\
        --data-dir /tmp/ablation_sandbox --store search \\
        --labels ../eval/data/search_store/queries_labeled.jsonl \\
        --limit 10

Notes
-----
- Rerank is off. The cross-encoder overwrites `score` and adds
  `signals.rerank`, but leaves `signals.keyword` and
  `signals.similarity` untouched, so it cannot move any rule here.
- `sim` is never negative: `recall.py` stores a cosine only when it is
  strictly positive, so an absent entry reads back as exactly 0.0.
  That is why `exact_zero` is testable with `==` rather than an
  epsilon, and also why it almost never fires on an embedded store.
"""
import argparse
import json
from pathlib import Path

from memman.embed.fingerprint import bound_embedder, stored_fingerprint
from memman.search.recall import intent_aware_recall
from memman.store.db import open_read_only, store_dir
from memman.store.sqlite import SqliteBackend

# Word salad built from rare and cross-language tokens, fixed rather
# than generated so a rerun scores the same negatives.
NONSENSE_QUERIES = [
    'saffron marmalade zeppelin harpsichord',
    'tungsten quokka vestibule marzipan',
    'clavichord petunia bauxite flamingo',
    'obsidian kumquat trebuchet lanolin',
    'zither pomegranate basalt yodel',
    'narwhal chiffon quasar linoleum',
    'meerkat tapioca gantry filigree',
    'sextant rhubarb wombat cornice',
    'alabaster ferret sitar mulch',
    'gecko brocade thimble plinth',
    'walrus origami cistern paprika',
    'lychee mandolin gargoyle turnip',
    'axolotl velveteen dulcimer scree',
    'okapi tamarind spandrel gouache',
    'ptarmigan cassava barouche loam',
    'wallaby semolina escarpment tulle',
    'ibex marzipan carillon shale',
    'lemur damask portcullis sorghum',
    'tapir chenille bastion quinoa',
    'dugong tarragon architrave gneiss',
    ]

SIM_FLOORS = (0.15, 0.20, 0.25)


def rule_verdicts(rows: list[dict]) -> dict[str, bool]:
    """Evaluate every candidate sparse rule against one result set.

    Parameters
    ----------
    rows : list[dict]
        `results` from `intent_aware_recall`, each carrying a
        `signals` dict with `keyword` and `similarity` entries.

    Returns
    -------
    dict[str, bool]
        Rule name to whether that rule calls this set sparse. An empty
        set is sparse under every rule, which matches the shipped
        `not results` arm.

    Notes
    -----
    - Every rule requires `kw == 0` on all rows, so the rules form a
      chain: `exact_zero` implies `sim_floor_t` implies `kw_only`.
      Their false-positive rates are therefore ordered too, and the
      question is only where the useful firing rate is bought.
    """
    if not rows:
        return dict(
            {'exact_zero': True, 'kw_only': True},
            **{f'sim_floor_{t}': True for t in SIM_FLOORS})
    kws = [r['signals']['keyword'] for r in rows]
    sims = [r['signals']['similarity'] for r in rows]
    kw_zero = all(k == 0.0 for k in kws)
    out = {
        'exact_zero': kw_zero and all(s == 0.0 for s in sims),
        'kw_only': kw_zero,
        }
    for t in SIM_FLOORS:
        out[f'sim_floor_{t}'] = kw_zero and all(s < t for s in sims)
    return out


def main() -> None:
    """Report each rule's false-positive and firing rate."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    backend = SqliteBackend(
        open_read_only(store_dir(args.data_dir, args.store)))
    fingerprint = stored_fingerprint(backend)
    embedder = bound_embedder(backend)

    labeled = []
    with Path(args.labels).open() as f:
        labeled.extend(json.loads(line)['query'] for line in f if line.strip())

    populations = [('labeled', labeled), ('nonsense', NONSENSE_QUERIES)]
    names = list(rule_verdicts([]).keys()) + ['shipped_meta']
    fired = {pop: dict.fromkeys(names, 0) for pop, _ in populations}
    sim_tops = {pop: [] for pop, _ in populations}
    counts = {}

    for pop, queries in populations:
        counts[pop] = len(queries)
        for query in queries:
            qvec = embedder.embed(query)
            resp = intent_aware_recall(
                backend, query, qvec, args.limit,
                fingerprint=fingerprint, rerank=False)
            rows = resp['results']
            fired[pop]['shipped_meta'] += int(
                resp['meta'].get('sparse', False))
            if rows:
                sim_tops[pop].append(
                    max(r['signals']['similarity'] for r in rows))
            for name, hit in rule_verdicts(rows).items():
                fired[pop][name] += int(hit)

    print(f'{args.store}: limit {args.limit}, rerank off')
    for pop, _q in populations:
        tops = sorted(sim_tops[pop])
        if tops:
            mid = tops[len(tops) // 2]
            print(f'  {pop}: n={counts[pop]}, top-row sim '
                  f'min {tops[0]:.4f} / median {mid:.4f} / '
                  f'max {tops[-1]:.4f}')
    print(f'  {"rule":16} {"fires on labeled":>18} '
          f'{"fires on nonsense":>18}')
    for name in names:
        fp = fired['labeled'][name]
        tp = fired['nonsense'][name]
        print(f'  {name:16} {fp:>7}/{counts["labeled"]:<10} '
              f'{tp:>7}/{counts["nonsense"]:<10}')


if __name__ == '__main__':
    main()
