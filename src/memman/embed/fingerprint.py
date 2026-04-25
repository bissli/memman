"""Embed-fingerprint canonical state for the DB.

A `Fingerprint` records which provider/model/dim produced the
vectors stored in a memman DB. The canonical value lives in
`meta.embed_fingerprint`. On every code path that reads or writes
embeddings, the active client's fingerprint is compared to the
stored one; mismatch (or missing) raises `EmbedFingerprintError`
with a remediation message.

The only code paths that *write* the meta row are
`memman embed reembed` and `setup.claude._init_default_store`.
There is no scan-on-open helper and no first-write seeding.
"""

import json
from dataclasses import dataclass

from memman.exceptions import EmbedFingerprintError
from memman.store.db import DB, get_meta, set_meta

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
        """
        d = json.loads(s)
        return cls(
            provider=str(d['provider']),
            model=str(d['model']),
            dim=int(d['dim']))

    @classmethod
    def from_client(cls, client: object) -> 'Fingerprint':
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


def assert_consistent(db: DB) -> None:
    """Raise `EmbedFingerprintError` when the active client does not
    match the stored fingerprint, or when no fingerprint is stored.

    The only path that writes the meta row is `memman embed reembed`
    (and `setup.claude._init_default_store` for fresh installs). A
    DB without it must run `embed reembed` once before any
    embedding-touching operation.
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
