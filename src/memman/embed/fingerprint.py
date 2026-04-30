"""Embed-fingerprint canonical state for the DB.

A `Fingerprint` records which provider/model/dim produced the
vectors stored in a memman DB. The canonical value lives in
`meta.embed_fingerprint`. On every code path that reads or writes
embeddings, the active client's fingerprint is compared to the
stored one; mismatch (or missing) raises `EmbedFingerprintError`
with a remediation message.

`seed_if_fresh` writes the meta row on first contact when the DB
has no fingerprint AND no insights, so any DB created lazily by a
CLI command behaves as if `memman install` had pre-seeded it. The
existing `memman embed reembed` and `setup.claude._init_default_store`
paths route through `seed_if_fresh` for a single seed implementation.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memman.exceptions import EmbedFingerprintError
from memman.store.db import DB, get_meta, set_meta

if TYPE_CHECKING:
    from memman.embed import EmbeddingProvider

META_KEY = 'embed_fingerprint'


@dataclass(frozen=True)
class Fingerprint:
    """Canonical (provider, model, dim) tuple for embedded vectors.
    """

    provider: str
    model: str
    dim: int

    def to_json(self) -> str:
        """Serialize to a stable JSON string.
        """
        return json.dumps(
            {'provider': self.provider,
             'model': self.model,
             'dim': self.dim},
            sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> 'Fingerprint':
        """Parse from the JSON string written by `to_json`.

        Malformed JSON, missing keys, and bad types raise
        `EmbedFingerprintError` so the caller surfaces a clean
        operator-facing error rather than a stdlib traceback.
        """
        try:
            d = json.loads(s)
            return cls(
                provider=str(d['provider']),
                model=str(d['model']),
                dim=int(d['dim']))
        except (json.JSONDecodeError, KeyError, TypeError,
                ValueError) as exc:
            raise EmbedFingerprintError(
                f'corrupt embed_fingerprint meta value: {exc}'
                f" -- run 'memman embed reembed' to reset"
                ) from exc

    @classmethod
    def from_client(
            cls, client: 'EmbeddingProvider') -> 'Fingerprint':
        """Build from any embed client exposing name/model/dim.
        """
        return cls(
            provider=str(client.name),
            model=str(client.model),
            dim=int(client.dim))


def active_fingerprint() -> Fingerprint:
    """Return the fingerprint of the env-resolved active client.
    """
    from memman.embed import get_client
    return Fingerprint.from_client(get_client())


def stored_fingerprint(db: DB) -> Fingerprint | None:
    """Return the fingerprint stored in `meta.embed_fingerprint`.
    """
    raw = get_meta(db, META_KEY)
    if raw is None:
        return None
    return Fingerprint.from_json(raw)


def write_fingerprint(db: DB, fp: Fingerprint) -> None:
    """Atomically write the fingerprint into `meta.embed_fingerprint`.
    """
    set_meta(db, META_KEY, fp.to_json())


def seed_if_fresh(db: DB) -> bool:
    """Seed `meta.embed_fingerprint` when the store is genuinely fresh.

    Writes the active client's fingerprint when both: (a) no
    fingerprint is stored, and (b) the `insights` table is empty.
    Idempotent. Returns True if a seed was written.

    Establishes the invariant that any DB containing data also has a
    fingerprint, so `assert_consistent` can rely on `stored is None`
    meaning fresh, not corrupted. The insights count must be checked
    BEFORE the embed client is constructed -- a DB with insights but
    no fingerprint is corruption and must surface the existing hard
    error from `assert_consistent`, not a misleading missing-key one.
    """
    if stored_fingerprint(db) is not None:
        return False
    row = db._query('SELECT COUNT(*) FROM insights').fetchone()
    if row and int(row[0]) > 0:
        return False
    from memman.embed import get_client
    ec = get_client()
    if not ec.available():
        raise EmbedFingerprintError(ec.unavailable_message())
    target = Fingerprint.from_client(ec)
    if target.dim <= 0:
        raise EmbedFingerprintError(
            f'embed provider {target.provider} returned'
            f' dim={target.dim}; cannot seed fingerprint')
    write_fingerprint(db, target)
    return True


def assert_consistent(db: DB) -> None:
    """Raise `EmbedFingerprintError` when the active client does not
    match the stored fingerprint, or when no fingerprint is stored.

    Callers that open a DB through the CLI (`_open_db`,
    `_StoreContext`, `store_create`, `_init_default_store`) call
    `seed_if_fresh` first so that a missing fingerprint here means
    real corruption (data without provenance), not an uninitialized
    store.
    """
    active = active_fingerprint()
    stored = stored_fingerprint(db)
    if stored is None:
        raise EmbedFingerprintError(
            f"DB has no embed fingerprint. Active is"
            f" {active.provider}:{active.model}:{active.dim}."
            f" Run 'memman embed reembed' to initialize this store.")
    if stored != active:
        raise EmbedFingerprintError(
            f"Embed fingerprint mismatch: active is"
            f" {active.provider}:{active.model}:{active.dim},"
            f" stored is {stored.provider}:{stored.model}:{stored.dim}."
            f" Run 'memman scheduler stop && memman embed reembed'"
            f" to converge.")
