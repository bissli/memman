"""Embed-fingerprint canonical state for a store.

`Fingerprint` records which provider/model/dim produced the vectors
stored in a memman DB. The canonical value lives in
`meta.embed_fingerprint`; reads and writes compare the active
client's fingerprint to the stored one and raise
`EmbedFingerprintError` on drift.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memman.exceptions import EmbedFingerprintError

if TYPE_CHECKING:
    from memman.embed import EmbeddingProvider
    from memman.store.backend import Backend

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


def stored_fingerprint(backend: 'Backend') -> Fingerprint | None:
    """Return the fingerprint stored in `meta.embed_fingerprint`.
    """
    raw = backend.meta.get(META_KEY)
    if raw is None:
        return None
    return Fingerprint.from_json(raw)


def write_fingerprint(backend: 'Backend', fp: Fingerprint) -> None:
    """Atomically write the fingerprint into `meta.embed_fingerprint`.
    """
    backend.meta.set(META_KEY, fp.to_json())


def seed_if_fresh(
        backend: 'Backend',
        ec: 'EmbeddingProvider') -> bool:
    """Seed `meta.embed_fingerprint` when the store is genuinely fresh.

    Writes `ec`'s fingerprint when both: (a) no fingerprint is
    stored, and (b) the `insights` table is empty. Idempotent.
    Returns True if a seed was written.
    """
    if stored_fingerprint(backend) is not None:
        return False
    if backend.nodes.count_total() > 0:
        return False
    if not ec.available():
        raise EmbedFingerprintError(ec.unavailable_message())
    target = Fingerprint.from_client(ec)
    if target.dim <= 0:
        raise EmbedFingerprintError(
            f'embed provider {target.provider} returned'
            f' dim={target.dim}; cannot seed fingerprint')
    write_fingerprint(backend, target)
    return True


def assert_consistent(
        backend: 'Backend',
        ec: 'EmbeddingProvider') -> None:
    """Raise `EmbedFingerprintError` when `ec` does not match the
    stored fingerprint, or when no fingerprint is stored.
    """
    active = Fingerprint.from_client(ec)
    stored = stored_fingerprint(backend)
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
