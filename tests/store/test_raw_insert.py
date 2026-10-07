"""Raw row copy: `nodes.get_raw`, `nodes.insert_raw` and Migrator.apply.
"""

from datetime import datetime, timedelta, timezone

import pytest
from memman.embed.fingerprint import seed_default_fingerprint
from memman.migrate import MigrateInsight, MigrationPayload
from memman.store.sqlite import SqliteMigrator, open_sqlite_backend
from tests.conftest import _vec

CREATED = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
UPDATED = datetime(2025, 3, 5, 6, 7, 8, tzinfo=timezone.utc)
ENRICHED = datetime(2025, 3, 4, 5, 7, 0, tzinfo=timezone.utc)


def _row(row_id='row-a', content='grackle migrates north', **kwargs):
    fields = {
        'id': row_id, 'content': content, 'summary': 'a summary',
        'embedding': _vec(0.5, 0.25, -0.75),
        'enrich_attempted_at': ENRICHED, 'enriched_at': ENRICHED,
        'created_at': CREATED, 'updated_at': UPDATED, 'deleted_at': None,
        'embedding_model': 'model-x',
        'queue_uuid': 'qu-a', 'replaced_by': None, 'author': 'alice',
        'summary_model': 'summary-model-y',
        }
    fields.update(kwargs)
    return MigrateInsight(**fields)


def test_raw_insert_keeps_every_column(backend):
    """Verify a raw-inserted row reads back with every column as given.

    Mutation: insert_raw routed through nodes.insert, which stamps the
        current time and drops the embedding, summary and chain.
    Oracle: the hand-built row, with float values exact in float4.
    """
    row = _row(replaced_by='row-b', deleted_at=UPDATED)

    assert backend.nodes.insert_raw(row) is True

    assert backend.nodes.get_raw('row-a') == row


def test_raw_insert_rolls_back_with_the_callers_transaction(backend):
    """Verify a raise inside the caller's transaction leaves no row.

    Mutation: insert_raw writing through a connection of its own, as
        Migrator.apply does, which commits outside the caller's
        transaction.
    Oracle: the store held no row before the transaction.
    """
    with pytest.raises(RuntimeError), backend.transaction():
        backend.nodes.insert_raw(_row())
        raise RuntimeError('fault after the insert')

    assert backend.nodes.get_raw('row-a') is None


def test_raw_insert_ignores_an_existing_id(backend):
    """Verify a second raw insert of one id changes nothing.

    Mutation: an upsert in place of insert-or-ignore, which lets a
        resumed merge overwrite a row the parent changed since.
    Oracle: the first row, inserted by hand.
    """
    backend.nodes.insert_raw(_row())

    assert backend.nodes.insert_raw(_row(content='other text')) is False
    assert backend.nodes.get_raw('row-a').content == 'grackle migrates north'


def test_raw_inserted_current_row_scores_a_keyword_hit(backend):
    """Verify a raw-inserted current row is reachable by keyword.

    Mutation: the Postgres raw insert leaving `kw_tokens` empty, so
        the keyword channel never sees a copied or merged row.
    Oracle: the row's content holds both query tokens.
    """
    backend.nodes.insert_raw(_row())

    with backend.recall_session() as session:
        counts = session.keyword_counts({'grackle', 'north'})

    assert counts == {'row-a': 2}


def test_raw_insert_keeps_the_instant_of_a_non_utc_timestamp(backend):
    """Verify a timestamp in another zone reads back as the same instant.

    Mutation: the SQLite insert formatting the wall clock with a `Z`
        suffix and no UTC conversion, as psycopg hands back timestamptz
        in the session zone.
    Oracle: the same instant written at UTC-5.
    """
    eastern = CREATED.astimezone(timezone(timedelta(hours=-5)))

    backend.nodes.insert_raw(_row(created_at=eastern, updated_at=eastern))

    row = backend.nodes.get_raw('row-a')
    assert (row.created_at, row.updated_at) == (CREATED, CREATED)


def test_sqlite_apply_keeps_the_instant_of_a_non_utc_timestamp(tmp_path):
    """Verify SqliteMigrator.apply stores a non-UTC timestamp as its instant.

    Mutation: apply formatting the wall clock with a `Z` suffix and no
        UTC conversion, which shifts a row migrated from Postgres.
    Oracle: the same instant written at UTC-5.
    """
    data_dir = str(tmp_path / 'memman')
    eastern = CREATED.astimezone(timezone(timedelta(hours=-5)))
    fingerprint = seed_default_fingerprint()
    payload = MigrationPayload(
        fingerprint=fingerprint, embedding_dim=fingerprint.dim,
        insights=[_row(created_at=eastern, updated_at=eastern)],
        oplog=[], meta={})

    SqliteMigrator(data_dir).apply('target', payload)

    with open_sqlite_backend('target', data_dir) as backend:
        row = backend.nodes.get_raw('row-a')
    assert (row.created_at, row.updated_at) == (CREATED, CREATED)


def test_get_raw_returns_none_for_a_missing_id(backend):
    """Verify get_raw answers None for an id the store lacks.

    Mutation: get_raw raising on a miss, which breaks the overlay's
        lookup of a branch-only id in the parent.
    Oracle: an empty store.
    """
    assert backend.nodes.get_raw('absent') is None
