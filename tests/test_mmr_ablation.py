"""MMR diversity rerank (F6) and the repaired ablation harness.

The harness is loaded via importlib (experiments/ is not a package).
MMR tests
monkeypatch `MMR_LAMBDA`/`MMR_POOL` so they are independent of the
shipped (measured) values.
"""

import csv
import importlib.util
import json
import sys
import uuid
from pathlib import Path

from memman.search import recall as recall_mod
from memman.search.recall import intent_aware_recall
from tests.conftest import make_insight

_SPEC = importlib.util.spec_from_file_location(
    'run_ablation',
    Path(__file__).parent.parent / 'experiments' / 'recall_ablation'
    / 'run_ablation.py')
ablation = importlib.util.module_from_spec(_SPEC)
sys.modules['run_ablation'] = ablation
_SPEC.loader.exec_module(ablation)


def _seed_vec(backend, iid, content, vec):
    backend.nodes.insert(make_insight(id=iid, content=content))
    backend.nodes.update_embedding(iid, vec, 'test-model')


def _abc_pool(backend):
    """A + near-duplicate B ranked above diverse C by pure relevance."""
    _seed_vec(backend, 'mmr-a', 'anchor row body a', [1.0, 0.0, 0.0])
    _seed_vec(backend, 'mmr-b', 'anchor row body b', [0.995, 0.0999, 0.0])
    _seed_vec(backend, 'mmr-c', 'anchor row body c', [0.5, 0.0, 0.866])


def _recall_ids(backend, **kwargs):
    qv = [1.0, 0.0, 0.0]
    resp = intent_aware_recall(
        backend, 'zzz unmatched query', qv, 0,
        intent_override='GENERAL', **kwargs)
    return [r['insight'].id for r in resp['results']]


def test_mmr_demotes_near_duplicate(tmp_backend, monkeypatch):
    """A near-duplicate loses to a diverse row under the MMR term.

    Mutation: flipping the diversity term's sign (`+` for `-`), or
        misaligning the penalty index against the pool.
    Oracle: by pure relevance the order is A, B, C (B is a
        near-duplicate of A; C is diverse but less similar to the
        query); at lambda 0.5 the pool-max penalty (~0.995 for A and
        B vs ~0.5 for C) must rank C above B.
    """
    _abc_pool(tmp_backend)
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 1.0)
    baseline = _recall_ids(tmp_backend)
    assert baseline.index('mmr-b') < baseline.index('mmr-c')
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 0.5)
    ids = _recall_ids(tmp_backend)
    assert ids.index('mmr-c') < ids.index('mmr-b')


def test_mmr_pool_exceeds_rerank_shortlist():
    """The MMR pool is strictly larger than the rerank shortlist.

    MMR before the cross-encoder only matters if it can change
    shortlist MEMBERSHIP; a pool equal to `RERANK_SHORTLIST` makes it
    a provable no-op whenever rerank is on (the default).

    Mutation: setting the pool to `RERANK_SHORTLIST`.
    Oracle: constant relation, no store needed.
    """
    assert recall_mod.MMR_POOL > recall_mod.RERANK_SHORTLIST


def test_mmr_pool_is_bounded(tmp_backend, monkeypatch):
    """Rows beyond `MMR_POOL` are exempt from the diversity re-sort.

    Mutation: removing the pool slice -- the gram matrix goes O(n^2)
        over the whole store and the tail reorders.
    Oracle: with the pool capped at 2, the diverse row C (rank 3 by
        relevance) must stay third even though an uncapped MMR at
        lambda 0.5 would lift it to first; the capped pair A/B carry
        equal penalties, so their relative order is unchanged.
    """
    _abc_pool(tmp_backend)
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 0.5)
    monkeypatch.setattr(recall_mod, 'MMR_POOL', 2)
    ids = _recall_ids(tmp_backend)
    assert ids.index('mmr-c') == 2
    assert ids.index('mmr-a') < ids.index('mmr-b')


def test_ablation_overridden_restores_mmr_lambda():
    """`overridden()` restores `MMR_LAMBDA` after each config.

    Mutation: a monkey-patch leak -- one config's lambda contaminates
        every later config in the same sweep.
    Oracle: the module constant returns to its prior value after the
        context exits, including on the no-override path.
    """
    saved = recall_mod.MMR_LAMBDA
    with ablation.overridden(None, None, mmr_lambda=0.5):
        assert recall_mod.MMR_LAMBDA == 0.5
    assert recall_mod.MMR_LAMBDA == saved
    with ablation.overridden(60, None):
        assert recall_mod.MMR_LAMBDA == saved
    assert recall_mod.MMR_LAMBDA == saved


def test_ablation_harness_runs(tmp_path, monkeypatch):
    """One config runs end-to-end and produces a non-empty results.csv.

    `main()` swallows per-config exceptions and still writes the csv
    with exit 0, so process success is vacuous -- the row count is the
    only falsifiable oracle.

    Mutation: the `TypeError` regression (positional `fingerprint`,
        raw `DB` instead of a `Backend`) -- every config errors, the
        csv is empty, and the process still exits 0.
    """
    from memman.embed import get_client
    from memman.embed.fingerprint import seed_default_fingerprint
    from memman.embed.fingerprint import write_fingerprint
    from memman.store.db import open_db, store_dir
    from memman.store.sqlite import SqliteBackend

    data_dir = tmp_path / 'abl'
    sdir = store_dir(str(data_dir), 'abstore')
    db = open_db(sdir)
    backend = SqliteBackend(db)
    write_fingerprint(backend, seed_default_fingerprint())
    ec = get_client()
    for i in range(5):
        iid = f'ab-{i}-{uuid.uuid4().hex[:6]}'
        content = f'alpha ablation row body {i}'
        backend.nodes.insert(make_insight(id=iid, content=content))
        backend.nodes.update_embedding(iid, ec.embed(content), ec.model)
    db.close()

    queries = tmp_path / 'queries.json'
    queries.write_text(json.dumps(
        [{'query': 'alpha ablation row', 'kind': 'smoke',
          'expected_intent': 'GENERAL'}]))
    out = tmp_path / 'results.csv'
    monkeypatch.setattr(sys, 'argv', [
        'run_ablation.py', '--store', 'abstore',
        '--data-dir', str(data_dir),
        '--queries', str(queries), '--out', str(out),
        '--configs', 'baseline', '--limit', '5'])
    ablation.main()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, 'harness wrote an empty results.csv'
    assert all(r['config'] == 'baseline' for r in rows)


def test_mmr_unembedded_rows_hold_position(tmp_backend, monkeypatch):
    """A vector-less row is exempt from the MMR re-sort.

    An unembedded row is exactly the degraded case (failed embed,
    dim-mismatch drop); scoring it with a zero penalty hands it the
    maximum diversity bonus and floats it to the head at any
    lambda < 1.

    Mutation: defaulting the penalty of an unembedded row to 0.0 and
        re-sorting it with the pool.
    Oracle: an unembedded row ranked last by relevance stays last at
        lambda 0.5 while the embedded rows reorder around it (C still
        rises above B).
    """
    _abc_pool(tmp_backend)
    tmp_backend.nodes.insert(
        make_insight(id='mmr-z', content='unembedded filler row'))
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 1.0)
    baseline = _recall_ids(tmp_backend)
    assert baseline.index('mmr-z') == 3
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 0.5)
    ids = _recall_ids(tmp_backend)
    assert ids.index('mmr-z') == 3
    assert ids.index('mmr-c') < ids.index('mmr-b')


def test_mmr_mixed_dim_rows_hold_position(tmp_backend, monkeypatch):
    """An off-modal-dim vector joins the unembedded exemption.

    A mid-model-swap store or a partial reembed leaves rows at two
    dims, and a gram matrix cannot be built over ragged vectors.

    Mutation: dropping the modal-dim filter -- `np.array` raises
        ValueError straight out of `intent_aware_recall`.
    Oracle: recall completes, the off-dim row keeps its relevance
        slot (last), and the embedded rows still reorder around it
        (C above B) at lambda 0.5.
    """
    _abc_pool(tmp_backend)
    _seed_vec(tmp_backend, 'mmr-z', 'off dim filler row', [0.1, 0.2])
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 1.0)
    baseline = _recall_ids(tmp_backend)
    assert baseline.index('mmr-z') == 3
    monkeypatch.setattr(recall_mod, 'MMR_LAMBDA', 0.5)
    ids = _recall_ids(tmp_backend)
    assert ids.index('mmr-z') == 3
    assert ids.index('mmr-c') < ids.index('mmr-b')
