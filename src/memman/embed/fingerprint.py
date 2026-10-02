"""Embed-fingerprint canonical state for a store.

`Fingerprint` records which model and dim produced the vectors
stored in a memman DB. The canonical value lives in
`meta.embed_fingerprint`; reads and writes compare the active
client's fingerprint to the stored one and raise
`EmbedFingerprintError` on drift.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memman import config
from memman.embed import registry as _ec_registry
from memman.exceptions import EmbedFingerprintError

if TYPE_CHECKING:
    from memman.embed import EmbeddingProvider
    from memman.store.backend import Backend

META_KEY = 'embed_fingerprint'


@dataclass(frozen=True)
class Fingerprint:
    """Canonical (model, dim) pair for embedded vectors.
    """

    model: str
    dim: int

    def to_json(self) -> str:
        """Serialize to a stable JSON string.
        """
        return json.dumps(
            {'model': self.model, 'dim': self.dim}, sort_keys=True)

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
        """Build from any embed client exposing model/dim.
        """
        return cls(
            model=str(client.model),
            dim=int(client.dim))


def swap_command(store: str) -> str:
    """Command that re-embeds `store` and records its fingerprint.

    Parameters
    ----------
    store : str
        Store name, or a `<store>` placeholder where the caller has none.

    Returns
    -------
    str
        A command line that runs on SQLite and Postgres alike.
    """
    return f'memman --store {store} embed swap --to <model>'


def seed_default_fingerprint() -> Fingerprint:
    """Env-active client's fingerprint, for seeding a fresh store.

    Seeds a brand-new store's `meta.embed_fingerprint` (via
    `seed_if_fresh`) or a fresh Postgres `vector(N)` column. For an
    existing store, resolve via `stored_fingerprint` / `bound_embedder`
    instead: each store keeps its own embedder. The registry client has
    probed its dim, so a fresh `vector(N)` column matches the model.
    """
    return Fingerprint.from_client(
        _ec_registry.get_for(config.require(config.EMBED_MODEL)))


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

    Parameters
    ----------
    backend : Backend
        The store to seed.
    ec : EmbeddingProvider
        The embedder whose fingerprint is written.

    Returns
    -------
    bool
        True if a seed was written.

    Raises
    ------
    EmbedFingerprintError
        When the store is fresh but `ec` is unavailable or reports a
        non-positive dimension.
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
            f'embed model {target.model!r} returned'
            f' dim={target.dim}; cannot seed fingerprint')
    write_fingerprint(backend, target)
    return True


def bound_embedder(backend: 'Backend') -> 'EmbeddingProvider':
    """Return the embed client cached for this store's stored fingerprint.

    Resolves `meta.embed_fingerprint` and dispatches to
    `embed.registry.get_for(model, dim)`. Raises
    `EmbedFingerprintError` if the store has no fingerprint yet --
    callers that may face a fresh store must run `seed_if_fresh`
    first.
    """
    fp = stored_fingerprint(backend)
    if fp is None:
        raise EmbedFingerprintError(
            'store has no embed fingerprint;'
            f' run `{swap_command("<store>")}` to re-embed it.')
    return _ec_registry.get_for(fp.model, fp.dim)
