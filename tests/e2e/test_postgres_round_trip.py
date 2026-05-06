"""Enqueue -> drain -> recall round-trip on Postgres.

Exercises the basic happy path of the queue + backend integration on
Postgres without invoking the LLM/embedding pipeline:

1. `PostgresQueueBackend.enqueue` puts a row into `queue.queue`.
2. `claim_batch` -> `mark_done` simulates a worker drain.
3. The drained payload turns into an `Insight` + embedding insert
   via the Backend Protocol.
4. `recall_session().vector_anchors()` returns that insight when
   queried with its own embedding.

This test does not call any external API; embeddings are synthetic
512-dim vectors so the round-trip is reproducible offline.
"""

from __future__ import annotations

import json

import pytest
from memman.store.model import Insight
from memman.store.postgres import PostgresCluster
from memman.store.postgres import PostgresQueueBackend
from tests.e2e.conftest import _pg_vec as _vec
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_enqueue_drain_recall_round_trip(pg_dsn, request):
    """Full round-trip: enqueue, drain, insert+embed, recall finds it.

    Asserts the queue state machine moves through pending -> claimed
    -> done, that the resulting Insight is recallable via
    `vector_anchors`, and that the score is finite cosine.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass

    backend = cluster.open(store=store, data_dir='')
    queue = PostgresQueueBackend(dsn=pg_dsn)

    try:
        payload = json.dumps({'id': 'rt-1', 'content': 'round-trip probe'})
        queued_id = queue.enqueue(
            store=store, op='ingest', payload=payload)
        assert queued_id > 0

        batch = queue.claim_batch(limit=10)
        store_rows = [row for row in batch if row.store == store]
        assert len(store_rows) == 1
        claimed = store_rows[0]
        assert claimed.id == queued_id
        assert claimed.op == 'ingest'

        decoded = json.loads(claimed.payload)
        backend.nodes.insert(Insight(
            id=decoded['id'],
            content=decoded['content'],
            importance=3,
            source='user'))
        backend.nodes.update_embedding(
            decoded['id'], _vec(7), 'voyage-3-lite')
        backend._conn.commit()

        queue.mark_done([claimed.id])

        with backend.recall_session() as session:
            results = session.vector_anchors(_vec(7), k=5)
        assert results, 'vector_anchors returned no hits'
        ids = [hit[0] for hit in results]
        assert 'rt-1' in ids, (
            f'recall did not surface the round-trip insight; ids={ids}')
        score = next(s for hid, s in results if hid == 'rt-1')
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6
    finally:
        backend.close()
        cluster.drop_store(store=store, data_dir='')
