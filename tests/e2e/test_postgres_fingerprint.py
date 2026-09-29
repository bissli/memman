"""Embed fingerprint mismatch refusal on Postgres.

The fingerprint contract: every store stamps `meta.embed_fingerprint`
with `{provider, model, dim}` of the embedding client used at seed
time. On reopen, a different active client surfaces a mismatch.

The tests exercise the contract at the Backend Protocol layer
(`backend.meta.set` / `.get`) and compare to a `Fingerprint`
parsed from the stored JSON, the same shape `assert_consistent`
checks once the application layer loads the row.
"""

from __future__ import annotations

import pytest
from memman.embed.fingerprint import META_KEY, EmbedFingerprintError
from memman.embed.fingerprint import Fingerprint
from memman.store.postgres import drop_postgres_store, open_postgres_backend
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_stored_fingerprint_round_trips_through_backend_meta(
        pg_dsn, request):
    """Verify a Fingerprint survives Postgres meta set and get.

    Mutation: the meta store truncating, re-encoding, or dropping
        the JSON value, or `Fingerprint.from_json` losing a field.
    Oracle: the `Fingerprint` written equals the one parsed back.
    """
    store = _safe(request.node.name)
    drop_postgres_store(store, pg_dsn)
    backend = open_postgres_backend(store, pg_dsn)
    try:
        target = Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512)
        backend.meta.set(META_KEY, target.to_json())
        backend._conn.commit()

        raw = backend.meta.get(META_KEY)
        assert raw is not None
        recovered = Fingerprint.from_json(raw)
        assert recovered == target
    finally:
        backend.close()
        drop_postgres_store(store, pg_dsn)


def test_active_vs_stored_fingerprint_mismatch_is_observable(
        pg_dsn, request):
    """Verify a stored fingerprint differs from a different active one.

    Mutation: the stored fingerprint reading back as the active
        one (wrong key, stale value), or `Fingerprint` equality
        ignoring model or dim.
    Oracle: the seeded `voyage-3-lite` 512 compared with a
        hand-built `voyage-large` 1024.
    """
    store = _safe(request.node.name)
    drop_postgres_store(store, pg_dsn)
    backend = open_postgres_backend(store, pg_dsn)
    try:
        seeded = Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512)
        backend.meta.set(META_KEY, seeded.to_json())
        backend._conn.commit()

        active = Fingerprint(
            provider='voyage', model='voyage-large', dim=1024)
        stored_raw = backend.meta.get(META_KEY)
        assert stored_raw is not None
        stored = Fingerprint.from_json(stored_raw)
        assert stored != active, (
            'stored vs active fingerprints must compare unequal'
            ' to drive the refusal path')
        assert stored.model == 'voyage-3-lite'
        assert active.model == 'voyage-large'
    finally:
        backend.close()
        drop_postgres_store(store, pg_dsn)


def test_corrupt_fingerprint_json_raises(pg_dsn, request):
    """Verify a corrupt fingerprint meta value raises on parse.

    Mutation: `Fingerprint.from_json` swallowing a decode error and
        returning a default, so a corrupt store is silently
        misindexed.
    Oracle: `EmbedFingerprintError` for the value `{not valid json`
        read back from Postgres.
    """
    store = _safe(request.node.name)
    drop_postgres_store(store, pg_dsn)
    backend = open_postgres_backend(store, pg_dsn)
    try:
        backend.meta.set(META_KEY, '{not valid json')
        backend._conn.commit()
        raw = backend.meta.get(META_KEY)
        assert raw == '{not valid json'
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json(raw)
    finally:
        backend.close()
        drop_postgres_store(store, pg_dsn)
