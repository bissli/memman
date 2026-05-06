"""Embed fingerprint mismatch refusal on Postgres.

The fingerprint contract: every store stamps `meta.embed_fingerprint`
with `{provider, model, dim}` of the embedding client used at seed
time. On reopen, a different active client surfaces a mismatch.

The e2e check exercises the contract at the Backend Protocol layer
(`backend.meta.set` / `.get`) and compares to a `Fingerprint`
parsed from the stored JSON -- the same shape `assert_consistent`
checks once the application-layer loads the row.
"""

from __future__ import annotations

import pytest
from memman.embed.fingerprint import META_KEY, Fingerprint
from memman.store.postgres import PostgresCluster
from tests.e2e.conftest import _safe

pytestmark = [pytest.mark.postgres, pytest.mark.e2e_container]


def test_stored_fingerprint_round_trips_through_backend_meta(
        pg_dsn, request):
    """Backend.meta.set + .get round-trips a Fingerprint JSON value."""
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend = cluster.open(store=store, data_dir='')
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
        cluster.drop_store(store=store, data_dir='')


def test_active_vs_stored_fingerprint_mismatch_is_observable(
        pg_dsn, request):
    """A different active fingerprint compares unequal to the stored.

    Drives the production refusal path in spirit: when the active
    client is `voyage:voyage-3-lite:512` and the store carries
    `voyage:voyage-large:1024`, the equality check that powers
    `assert_consistent` returns False and a refusal is warranted.
    """
    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend = cluster.open(store=store, data_dir='')
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
        cluster.drop_store(store=store, data_dir='')


def test_corrupt_fingerprint_json_raises(pg_dsn, request):
    """A meta row whose value is not valid Fingerprint JSON raises.

    The hard failure on corruption is part of the contract:
    `Fingerprint.from_json` raises `EmbedFingerprintError`, which
    `assert_consistent` propagates so the operator runs
    `memman embed reembed` rather than silently misindexing.
    """
    from memman.embed.fingerprint import EmbedFingerprintError

    store = _safe(request.node.name)
    cluster = PostgresCluster(dsn=pg_dsn)
    try:
        cluster.drop_store(store=store, data_dir='')
    except Exception:
        pass
    backend = cluster.open(store=store, data_dir='')
    try:
        backend.meta.set(META_KEY, '{not valid json')
        backend._conn.commit()
        raw = backend.meta.get(META_KEY)
        assert raw == '{not valid json'
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json(raw)
    finally:
        backend.close()
        cluster.drop_store(store=store, data_dir='')
