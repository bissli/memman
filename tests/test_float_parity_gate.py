"""Float32-vs-float64 ranking parity gate.

For each of 20 query vectors against a synthetic corpus, the top-5
set on Postgres (float32 via pgvector) must intersect the top-5 set
on SQLite (float64 numpy) at >= 4/5. Validates that float-precision
differences do not visibly perturb retrieval rank.

The corpus is structured (topic centers x insights per topic) rather
than random, so the matching topic's cosines are high and the top-5
is decided by the vector channel rather than by which near-orthogonal
rows happen to cross a boundary. A random 512-dim corpus gives cosines
near 0, where float32 and float64 legitimately disagree on rank.

Notes
-----
- Agreement of the two backends at the sign boundary of the vector
  channel is uncovered. Reaching it needs a corpus built at cosine
  near 1e-8, where float32 and float64 disagree on sign. On this
  corpus the boundary sits at the median and the keyword channel fills
  the page, so no test on it can have teeth.
"""

import random

import pytest
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.search.recall import run_recall
from memman.store.model import Insight
from memman.store.sqlite import drop_sqlite_store, open_sqlite_backend
from tests.conftest import EMBEDDING_DIM

N_TOPICS = 12
INSIGHTS_PER_TOPIC = 5
N_QUERIES = 20
CORPUS_NOISE = 0.02
QUERY_NOISE = 0.05
PARITY_FLOOR = 4

pytestmark = pytest.mark.postgres


def _unit(vec: list[float]) -> list[float]:
    """Normalize to unit length.
    """
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def _gaussian_unit(seed: int) -> list[float]:
    """Deterministic 512-dim unit Gaussian vector.
    """
    rng = random.Random(seed)
    return _unit([rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)])


def _perturb(vec: list[float], seed: int, scale: float) -> list[float]:
    """Add Gaussian noise then re-normalize.
    """
    rng = random.Random(seed)
    noisy = [x + rng.gauss(0.0, scale) for x in vec]
    return _unit(noisy)


def _populate(backend, topic_centers: list[list[float]]) -> None:
    """Insert INSIGHTS_PER_TOPIC perturbed corpus vectors per topic.
    """
    for t_idx, center in enumerate(topic_centers):
        for k in range(INSIGHTS_PER_TOPIC):
            ins_id = f't{t_idx:02d}-i{k}'
            ins = Insight(
                id=ins_id,
                content=f'topic {t_idx} insight {k} alpha bravo charlie',
                category='fact',
                created_at=None,
                updated_at=None,
                deleted_at=None)
            backend.nodes.insert(ins)
            vec = _perturb(center, seed=t_idx * 100 + k, scale=CORPUS_NOISE)
            backend.nodes.update_embedding(ins_id, vec, 'voyage-3-lite')


def _top5_ids(backend, qvec) -> set[str]:
    """Return the top-5 ids by intent-aware recall on the given backend.
    """
    result = run_recall(
        backend, query='topic insight',
        query_vec=qvec,
        limit=5)
    return {r['insight'].id for r in result['results'][:5]}


def test_float32_float64_top5_intersection_geq_4_across_20_queries(
        tmp_path, pg_dsn):
    """Verify sqlite and postgres top-5 sets overlap by 4 or more on every query.

    Mutation: a postgres recall path whose float32 ranking diverges from the
        float64 sqlite ranking, such as a distance operator that changes
        the vector-channel order.
    Oracle: the sqlite top-5 for the same corpus and query vector, with
        PARITY_FLOOR of 4 of 5.
    """
    from memman.store.postgres import drop_postgres_store
    from memman.store.postgres import open_postgres_backend

    topic_centers = [_gaussian_unit(seed=i) for i in range(N_TOPICS)]

    sqlite_data_dir = str(tmp_path / 'memman')
    sqlite_backend = open_sqlite_backend('parity', sqlite_data_dir)
    sqlite_backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
    _populate(sqlite_backend, topic_centers)

    drop_postgres_store('parity_test', pg_dsn)
    postgres_backend = open_postgres_backend('parity_test', pg_dsn)
    postgres_backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
    _populate(postgres_backend, topic_centers)

    try:
        failures = []
        for q in range(N_QUERIES):
            qvec = _perturb(
                topic_centers[q % N_TOPICS],
                seed=10000 + q, scale=QUERY_NOISE)
            sqlite_top = _top5_ids(sqlite_backend, qvec)
            postgres_top = _top5_ids(postgres_backend, qvec)
            intersection = len(sqlite_top & postgres_top)
            if intersection < PARITY_FLOOR:
                failures.append(
                    f'query {q}: intersection={intersection}/5, '
                    f'sqlite={sorted(sqlite_top)}, '
                    f'postgres={sorted(postgres_top)}')
        assert not failures, (
            f'{len(failures)}/{N_QUERIES} queries below parity floor '
            f'{PARITY_FLOOR}/5:\n' + '\n'.join(failures))
    finally:
        sqlite_backend.close()
        drop_sqlite_store('parity', sqlite_data_dir)
        postgres_backend.close()
        drop_postgres_store('parity_test', pg_dsn)
